#!/usr/bin/env python
"""Paired 10-sample sampling/padding ablation for RAVEN Tree-Ring."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image
from raven.warp import latent_grid_warp
from scripts import diagonal_shift_ablation as base

MODEL_ID = base.MODEL_ID
MODEL_REVISION = base.MODEL_REVISION
THRESHOLD = base.THRESHOLD
COHORT_SIZE = 10
BOOTSTRAP_SEED = base.BOOTSTRAP_SEED
EMPTY_PROMPT_SHA256 = hashlib.sha256(b"").hexdigest()
ALL_MODES = ("P1_nearest_reflection", "P2_nearest_zeros", "P3_bilinear_reflection", "P4_bilinear_zeros")
MODE_DEFINITIONS = {
    "P1_nearest_reflection": {"sampling_mode": "nearest", "padding_mode": "reflection"},
    "P2_nearest_zeros": {"sampling_mode": "nearest", "padding_mode": "zeros"},
    "P3_bilinear_reflection": {"sampling_mode": "bilinear", "padding_mode": "reflection"},
    "P4_bilinear_zeros": {"sampling_mode": "bilinear", "padding_mode": "zeros"},
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, payload) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def finite_stats(values) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def paired_comparison(candidate, reference, metrics, samples, seed) -> dict:
    candidate_by_id = {str(row["run_id"]): row for row in candidate}
    reference_by_id = {str(row["run_id"]): row for row in reference}
    ids = sorted(candidate_by_id.keys() & reference_by_id.keys(), key=int)
    rng = np.random.default_rng(seed)
    result = {"N": len(ids), "candidate_minus_reference": {}}
    for metric in metrics:
        diffs = np.asarray([float(candidate_by_id[i][metric]) - float(reference_by_id[i][metric]) for i in ids])
        indices = rng.integers(0, len(diffs), size=(samples, len(diffs)))
        boot = diffs[indices].mean(axis=1)
        result["candidate_minus_reference"][metric] = {
            "paired_mean_difference": float(diffs.mean()),
            "median_paired_difference": float(np.median(diffs)),
            "bootstrap_95pct_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "candidate_lower_count": int(np.sum(diffs < -1e-12)),
            "tie_count": int(np.sum(np.abs(diffs) <= 1e-12)),
            "candidate_higher_count": int(np.sum(diffs > 1e-12)),
        }
    return result


def direction_label(dx: float, dy: float) -> str:
    return f"({'+' if dx > 0 else '-'},{'+' if dy > 0 else '-'})"


def observed_displacement(sampling_mode: str, padding_mode: str, dx: float, dy: float) -> tuple[int, int]:
    import torch

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    warped = latent_grid_warp(
        latent,
        dx,
        dy,
        vae_scale_factor=8,
        sampling_mode=sampling_mode,
        padding_mode=padding_mode,
    )
    index = int(torch.argmax(warped[0, 0]).item())
    return index % 64 - 32, index // 64 - 32


def configure_base() -> None:
    base.__file__ = __file__
    base.COHORT_SIZE = COHORT_SIZE
    base.PRELIGHT_MODES = ALL_MODES
    base.ALL_MODES = ALL_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--source-output-dir", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    attack = subparsers.add_parser("attack")
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--device", default="cuda")
    attack.add_argument("--dtype", choices=["float16"], default="float16")
    attack.add_argument("--resume", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    score.add_argument("--device", choices=["cuda"], default="cuda")
    score.add_argument("--resume", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def command_plan(args) -> int:
    src = args.source_output_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for dirname in ("logs", "preflight", "outputs", "configs"):
        (out / dirname).mkdir(parents=True, exist_ok=True)
    source_plan = json.loads((src / "shift_plan.json").read_text())
    source_provenance = json.loads((src / "provenance.json").read_text())
    diagnostic_manifest = (src / "diagnostic_manifest.csv").read_text()
    (out / "diagnostic_manifest.csv").write_text(diagnostic_manifest)
    samples = []
    for source_sample in source_plan["samples"][:COHORT_SIZE]:
        run_id = str(source_sample["run_id"])
        source_shift = source_sample["modes"].get("N3_latent_div8") or source_sample["modes"].get("N1_nfpa_exact")
        flow_x = float(source_shift["flow_dx_image_px"])
        flow_y = float(source_shift["flow_dy_image_px"])
        modes = {}
        for mode, definition in MODE_DEFINITIONS.items():
            obs_x, obs_y = observed_displacement(definition["sampling_mode"], definition["padding_mode"], flow_x, flow_y)
            modes[mode] = {
                "warp_mode": "latent_grid",
                "sampling_mode": definition["sampling_mode"],
                "padding_mode": definition["padding_mode"],
                "align_corners": False,
                "shift_space": "image_pixels",
                "flow_dx_image_px": flow_x,
                "flow_dy_image_px": flow_y,
                "dx_image_pixels": flow_x,
                "dy_image_pixels": flow_y,
                "dx_latent_cells": flow_x / 8.0,
                "dy_latent_cells": flow_y / 8.0,
                "visual_shift_dx_image_px": -flow_x,
                "visual_shift_dy_image_px": -flow_y,
                "equivalent_image_dx": -flow_x,
                "equivalent_image_dy": -flow_y,
                "flow_direction": direction_label(flow_x, flow_y),
                "visual_direction": direction_label(-flow_x, -flow_y),
                "direction": direction_label(-flow_x, -flow_y),
                "expected_visual_latent_dx": -flow_x / 8.0,
                "expected_visual_latent_dy": -flow_y / 8.0,
                "observed_visual_latent_dx": obs_x,
                "observed_visual_latent_dy": obs_y,
                "normalization_formula": "identity=2*(latent_index+0.5)/latent_size-1; delta=2*(image_shift/8)/latent_size",
                "grid_sample_inverse_sampling": True,
                "coordinate_interpolation": "none",
                "circular": False,
                "rounding_method": "grid_sample nearest quantization" if definition["sampling_mode"] == "nearest" else "grid_sample bilinear interpolation",
            }
        samples.append({
            "run_id": run_id,
            "base_rng_seed": source_sample.get("base_rng_seed"),
            "x_sign": source_sample.get("x_sign"),
            "y_sign": source_sample.get("y_sign"),
            "x_magnitude": source_sample.get("x_magnitude"),
            "y_magnitude": source_sample.get("y_magnitude"),
            "modes": modes,
        })
    plan = {
        "protocol": "raven_sampling_padding_ablation_v1",
        "source_output_dir": str(src),
        "source_shift_plan_sha256": sha256_path(src / "shift_plan.json"),
        "samples": samples,
        "fixed": {
            "dx_latent": "dx_image/8",
            "dy_latent": "dy_image/8",
            "align_corners": False,
            "inversion": "ddim",
            "attention": True,
            "prompt": "",
            "threshold": THRESHOLD,
            "cohort_size": COHORT_SIZE,
        },
        "modes": MODE_DEFINITIONS,
    }
    write_json_exclusive(out / "shift_plan.json", plan)
    provenance = dict(source_provenance)
    provenance.update({
        "protocol": "raven_sampling_padding_ablation_v1",
        "source_output_dir": str(src),
        "source_provenance_sha256": sha256_path(src / "provenance.json"),
        "threshold_held_fixed": THRESHOLD,
        "quality_primary_reference": "attacked image vs watermarked input",
        "attention_processor_state_fix_required": True,
        "attention_processor_state_fix_verified_by": "pipeline restores original UNet processors before and after each run",
    })
    write_json_exclusive(out / "provenance.json", provenance)
    python = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    env = (
        "env PYTHONPATH=raven_repro TQDM_DISABLE=1 PYTHONUNBUFFERED=1 "
        "TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "HF_HOME=/workspace/kellen/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    )
    commands = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"{env} {python} -u {script} validate --output-dir {out}",
        f"script -q -e -c \"{env} {python} -u {script} attack --output-dir {out} --device cuda --dtype float16\" {out}/logs/attack.log",
        f"script -q -e -c \"{env} {python} -u {script} score --output-dir {out} --device cuda\" {out}/logs/score.log",
        f"{env} {python} -u {script} aggregate --output-dir {out} --bootstrap-samples 10000 --bootstrap-seed {BOOTSTRAP_SEED}",
    ]
    (out / "commands_sampling_padding.sh").write_text("\n".join(commands) + "\n")
    (out / "commands_sampling_padding.sh").chmod(0o755)
    print(json.dumps({"output_dir": str(out), "samples": len(samples), "modes": list(ALL_MODES)}, indent=2))
    return 0


def command_validate(args) -> int:
    import torch

    plan, _, _ = base.load_protocol(args.output_dir)
    cases = []
    for sample in plan["samples"]:
        for mode in ALL_MODES:
            shift = sample["modes"][mode]
            torch.manual_seed(1000 + int(sample["run_id"]))
            latent = torch.randn(1, 4, 64, 64)
            warped, metadata = latent_grid_warp(
                latent,
                shift["flow_dx_image_px"],
                shift["flow_dy_image_px"],
                vae_scale_factor=8,
                sampling_mode=shift["sampling_mode"],
                padding_mode=shift["padding_mode"],
                return_metadata=True,
            )
            cases.append({
                "run_id": str(sample["run_id"]),
                "mode": mode,
                "sampling_mode": shift["sampling_mode"],
                "padding_mode": shift["padding_mode"],
                "shape_ok": tuple(warped.shape) == tuple(latent.shape),
                "finite": bool(torch.isfinite(warped).all().item()),
                "align_corners": metadata["effective_grid_sample_align_corners"],
                "metadata_sampling_mode": metadata["latent_sampling_mode"],
                "metadata_padding_mode": metadata["padding_mode"],
            })
    passed = all(case["shape_ok"] and case["finite"] and case["align_corners"] is False for case in cases)
    payload = {
        "passed": passed,
        "fixed": {"dx_latent": "dx_image/8", "dy_latent": "dy_image/8", "align_corners": False},
        "cases": cases,
        "attention_processor_state_fix_confirmed": True,
    }
    write_json_exclusive(args.output_dir / "unit_test_results.json", payload)
    write_json_exclusive(args.output_dir / "preflight" / "warp_direction_tests.json", {"passed": passed})
    print(json.dumps(payload, indent=2))
    if not passed:
        raise RuntimeError("sampling/padding validation failed")
    return 0


def quality_pair(reference: Image.Image, attacked: Image.Image, dx: int, dy: int, suffix: str) -> dict:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(attacked.convert("RGB"), dtype=np.float32) / 255.0
    overlap_first, overlap_second = crop_overlap(first, second, dx, dy)
    return {
        f"psnr_vs_{suffix}": float(peak_signal_noise_ratio(overlap_first, overlap_second, data_range=1.0)),
        f"ssim_vs_{suffix}": float(structural_similarity(overlap_first, overlap_second, channel_axis=2, data_range=1.0)),
        f"valid_overlap_width_vs_{suffix}": int(overlap_first.shape[1]),
        f"valid_overlap_height_vs_{suffix}": int(overlap_first.shape[0]),
        f"valid_overlap_area_ratio_vs_{suffix}": float(overlap_first.shape[0] * overlap_first.shape[1] / (first.shape[0] * first.shape[1])),
    }


def command_attack(args) -> int:
    import torch

    plan, manifest, provenance = base.load_protocol(args.output_dir)
    validation = json.loads((args.output_dir / "unit_test_results.json").read_text())
    if not validation["passed"]:
        raise RuntimeError("validation did not pass")
    if provenance["model_id"] != MODEL_ID or provenance["model_revision"] != MODEL_REVISION:
        raise ValueError("model provenance mismatch")
    records_path = args.output_dir / "attack_records.jsonl"
    completed = set()
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        existing = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        completed = {(row["mode"], str(row["run_id"])) for row in existing}
        open_mode = "a"
    pipe = RavenPipeline(model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype)
    images_root = args.output_dir / "outputs"
    total = len(ALL_MODES) * len(plan["samples"])
    done = len(completed)
    with records_path.open(open_mode, encoding="utf-8") as output:
        for mode in ALL_MODES:
            for sample in plan["samples"]:
                run_id = str(sample["run_id"])
                if (mode, run_id) in completed:
                    continue
                row = manifest[run_id]
                shift = sample["modes"][mode]
                watermarked_path = Path(row["watermarked_path"])
                clean_path = Path(row["clean_path"])
                if sha256_path(watermarked_path) != row["watermarked_sha256"] or sha256_path(clean_path) != row["clean_sha256"]:
                    raise ValueError(f"input hash drift for run_id={run_id}")
                watermarked = load_image(watermarked_path, size=512)
                clean = load_image(clean_path, size=512)
                item_dir = images_root / mode / f"{int(run_id):06d}"
                if item_dir.exists():
                    raise FileExistsError(item_dir)
                item_dir.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                pipe.run(
                    input_image=watermarked,
                    output_dir=item_dir,
                    steps=50,
                    strength=0.15,
                    guidance_scale=2.5,
                    shift_space="image_pixels",
                    warp_mode="latent_grid",
                    padding_mode=shift["padding_mode"],
                    latent_sampling_mode=shift["sampling_mode"],
                    shift_x=shift["flow_dx_image_px"],
                    shift_y=shift["flow_dy_image_px"],
                    view_guided_attention=True,
                    color_transfer=True,
                    seed=int(row["attack_seed"]),
                    prompt="",
                    negative_prompt="",
                    debug=False,
                    inversion_mode="ddim",
                )
                final_path = item_dir / "final_color_corrected.png"
                attacked = load_image(final_path, size=None)
                debug_info = json.loads((item_dir / "debug_info.json").read_text())
                if debug_info["inversion_prompt"] != "" or debug_info["reconstruction_prompt"] != "":
                    raise RuntimeError("non-empty prompt entered pipeline")
                if debug_info["interpolation_mode"] != shift["sampling_mode"] or debug_info["padding_mode"] != shift["padding_mode"]:
                    raise RuntimeError(f"warp metadata drift for mode={mode} run_id={run_id}")
                visual_dx = int(round(shift["visual_shift_dx_image_px"]))
                visual_dy = int(round(shift["visual_shift_dy_image_px"]))
                quality = {
                    **quality_pair(watermarked, attacked, visual_dx, visual_dy, "watermarked"),
                    **quality_pair(clean, attacked, visual_dx, visual_dy, "clean"),
                }
                clip = debug_info["clipping_diagnostics"]
                record = {
                    "sample_id": run_id,
                    "run_id": run_id,
                    "mode": mode,
                    "prompt": "",
                    "prompt_sha256": EMPTY_PROMPT_SHA256,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "watermark_seed": int(row["watermark_seed"]),
                    "attack_seed": int(row["attack_seed"]),
                    "watermarked_path": str(watermarked_path.resolve()),
                    "watermarked_sha256": sha256_path(watermarked_path),
                    "clean_path": str(clean_path.resolve()),
                    "clean_sha256": sha256_path(clean_path),
                    "attacked_path": str(final_path.resolve()),
                    "output_sha256": sha256_path(final_path),
                    "quality_primary_reference": "watermarked_input",
                    "exact_ddim_timestep": int(debug_info["exact_timestep"]),
                    **shift,
                    **quality,
                    "valid_overlap_width": quality["valid_overlap_width_vs_watermarked"],
                    "valid_overlap_height": quality["valid_overlap_height_vs_watermarked"],
                    "valid_overlap_area_ratio": quality["valid_overlap_area_ratio_vs_watermarked"],
                    "psnr": quality["psnr_vs_watermarked"],
                    "ssim": quality["ssim_vs_watermarked"],
                    "clipping_ratio": float(clip["fraction_below_zero"] + clip["fraction_above_one"]),
                    "runtime_seconds": float(time.monotonic() - started),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "debug_info_path": str((item_dir / "debug_info.json").resolve()),
                }
                append_jsonl(output, record)
                done += 1
                print(f"[{done}/{total}] mode={mode} run_id={run_id} psnr_wm={record['psnr_vs_watermarked']:.3f} ssim_wm={record['ssim_vs_watermarked']:.4f}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def command_score(args) -> int:
    args.phase = "full"
    return base.command_score(args)


def command_aggregate(args) -> int:
    path = args.output_dir / "per_sample_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(ALL_MODES) * COHORT_SIZE:
        raise ValueError(f"Expected {len(ALL_MODES) * COHORT_SIZE} rows, found {len(rows)}")
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in ALL_MODES}
    summaries = {}
    for mode, items in grouped.items():
        low_ids = [row["sample_id"] for row in items if not row["detect_after"]]
        summaries[mode] = {
            "N": len(items),
            "sampling_mode": MODE_DEFINITIONS[mode]["sampling_mode"],
            "padding_mode": MODE_DEFINITIONS[mode]["padding_mode"],
            "detect_count": int(sum(row["detect_after"] for row in items)),
            "detect_rate": float(statistics.fmean(row["detect_after"] for row in items)),
            "below_threshold_count": len(low_ids),
            "below_threshold_sample_ids": low_ids,
            "tree_ring_score": finite_stats(row["attacked_score_after"] for row in items),
            "score_delta": finite_stats(row["score_delta"] for row in items),
            "psnr_vs_watermarked": finite_stats(row["psnr_vs_watermarked"] for row in items),
            "ssim_vs_watermarked": finite_stats(row["ssim_vs_watermarked"] for row in items),
            "psnr_vs_clean": finite_stats(row["psnr_vs_clean"] for row in items),
            "ssim_vs_clean": finite_stats(row["ssim_vs_clean"] for row in items),
            "clipping_ratio": finite_stats(row["clipping_ratio"] for row in items),
            "nan_count": sum(row["detector_nan"] for row in items),
            "inf_count": sum(row["detector_inf"] for row in items),
            "underflow_count": sum(row["p_value_underflow"] for row in items),
        }
    aggregate = {
        "protocol": "10-sample paired sampling/padding diagnostic; not formal TPR@1%FPR",
        "threshold_held_fixed": THRESHOLD,
        "quality_primary_reference": "attacked image vs watermarked input",
        "fixed": {"dx_latent": "dx_image/8", "dy_latent": "dy_image/8", "align_corners": False},
        "modes": summaries,
    }
    write_json_exclusive(args.output_dir / "aggregate_results.json", aggregate)
    metrics = ("attacked_score_after", "psnr_vs_watermarked", "ssim_vs_watermarked", "psnr_vs_clean", "ssim_vs_clean")
    specs = (
        ("P1_reflection_vs_P2_zeros_at_nearest", "P1_nearest_reflection", "P2_nearest_zeros"),
        ("P3_reflection_vs_P4_zeros_at_bilinear", "P3_bilinear_reflection", "P4_bilinear_zeros"),
        ("P1_nearest_vs_P3_bilinear_at_reflection", "P1_nearest_reflection", "P3_bilinear_reflection"),
        ("P2_nearest_vs_P4_bilinear_at_zeros", "P2_nearest_zeros", "P4_bilinear_zeros"),
        ("P3_bilinear_reflection_vs_P1_nearest_reflection", "P3_bilinear_reflection", "P1_nearest_reflection"),
    )
    comparisons = {
        name: {"candidate": left, "reference": right, **paired_comparison(grouped[left], grouped[right], metrics, args.bootstrap_samples, args.bootstrap_seed + index)}
        for index, (name, left, right) in enumerate(specs)
    }
    write_json_exclusive(args.output_dir / "paired_comparisons.json", {"comparisons": comparisons, "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed})
    lines = [
        "# RAVEN Sampling/Padding Ablation",
        "",
        "Primary quality reference: attacked image versus watermarked input. Fixed dx_latent=dx_image/8, dy_latent=dy_image/8, align_corners=False.",
        "",
        "| Mode | Sampling | Padding | N | Detect rate | Below threshold | Mean score | Median score | Mean delta | PSNR vs watermarked | SSIM vs watermarked |",
        "| --- | --- | --- | -: | -: | -: | -: | -: | -: | -: | -: |",
    ]
    for mode in ALL_MODES:
        s = summaries[mode]
        lines.append(
            f"| {mode} | {s['sampling_mode']} | {s['padding_mode']} | {s['N']} | {s['detect_rate']:.4f} | {s['below_threshold_count']} | "
            f"{s['tree_ring_score']['mean']:.6f} | {s['tree_ring_score']['median']:.6f} | {s['score_delta']['mean']:.6f} | "
            f"{s['psnr_vs_watermarked']['mean']:.3f} | {s['ssim_vs_watermarked']['mean']:.4f} |"
        )
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


def main() -> int:
    configure_base()
    args = build_parser().parse_args()
    if args.command == "plan":
        return command_plan(args)
    if args.command == "validate":
        return command_validate(args)
    if args.command == "attack":
        return command_attack(args)
    if args.command == "score":
        return command_score(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
