#!/usr/bin/env python
"""Paired 10-sample NFPA normalization-only ablation for RAVEN Tree-Ring."""

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
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image
from raven.warp import translate_latent
from scripts import diagonal_shift_ablation as base

MODEL_ID = base.MODEL_ID
MODEL_REVISION = base.MODEL_REVISION
THRESHOLD = base.THRESHOLD
COHORT_SIZE = 10
PLAN_SEED = base.PLAN_SEED
BOOTSTRAP_SEED = base.BOOTSTRAP_SEED
EMPTY_PROMPT_SHA256 = hashlib.sha256(b"").hexdigest()
ALL_MODES = ("N1_nfpa_exact", "N2_pixel_center", "N3_latent_div8")
MODE_DEFINITIONS = {
    "N1_nfpa_exact": {
        "warp_mode": "nfpa_exact",
        "normalization": "2*(image_index+flow)/image_size-1",
        "pixel_center_offset": 0.0,
    },
    "N2_pixel_center": {
        "warp_mode": "nfpa_pixel_center",
        "normalization": "2*(image_index+flow+0.5)/image_size-1",
        "pixel_center_offset": 0.5,
    },
    "N3_latent_div8": {
        "warp_mode": "latent_grid_nearest_reflection",
        "normalization": "identity=2*(latent_index+0.5)/latent_size-1; delta=2*(flow/8)/latent_size",
        "pixel_center_offset": "latent-grid identity center",
    },
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction_label(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        return "(0,0)"
    return f"({'+' if dx > 0 else '-'},{'+' if dy > 0 else '-'})"


def observed_displacement(mode: str, dx: float, dy: float) -> tuple[int, int]:
    import torch

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    warped = translate_latent(
        latent, dx, dy, shift_space="image_pixels", vae_scale_factor=8,
        padding_mode="reflection", warp_mode=MODE_DEFINITIONS[mode]["warp_mode"],
    )
    index = int(torch.argmax(warped[0, 0]).item())
    return index % 64 - 32, index // 64 - 32


def mode_shift(mode: str, x_mag: int, y_mag: int, x_sign: int, y_sign: int, common_sign: int) -> dict:
    del common_sign
    flow_x, flow_y = x_sign * x_mag, y_sign * y_mag
    observed_x, observed_y = observed_displacement(mode, flow_x, flow_y)
    definition = MODE_DEFINITIONS[mode]
    return {
        **definition,
        "sign_rule": "independent x/y signs; inherited paired shift plan",
        "magnitude_rule": "independent UniformInteger[24,32] image pixels",
        "shift_space": "image_pixels",
        "flow_dx_image_px": float(flow_x),
        "flow_dy_image_px": float(flow_y),
        "dx_image_pixels": float(flow_x),
        "dy_image_pixels": float(flow_y),
        "dx_latent_cells": float(flow_x / 8.0),
        "dy_latent_cells": float(flow_y / 8.0),
        "visual_shift_dx_image_px": float(-flow_x),
        "visual_shift_dy_image_px": float(-flow_y),
        "equivalent_image_dx": float(-flow_x),
        "equivalent_image_dy": float(-flow_y),
        "flow_direction": direction_label(flow_x, flow_y),
        "visual_direction": direction_label(-flow_x, -flow_y),
        "direction": direction_label(-flow_x, -flow_y),
        "expected_visual_latent_dx": float(-flow_x / 8.0),
        "expected_visual_latent_dy": float(-flow_y / 8.0),
        "observed_visual_latent_dx": observed_x,
        "observed_visual_latent_dy": observed_y,
        "coordinate_interpolation": "bilinear" if mode != "N3_latent_div8" else "none",
        "interpolation_mode": "nearest",
        "padding_mode": "reflection",
        "align_corners": False,
        "circular": False,
        "rounding_method": "nearest grid_sample quantization",
    }


def configure_base() -> None:
    base.__file__ = __file__
    base.COHORT_SIZE = COHORT_SIZE
    base.PRELIGHT_MODES = ALL_MODES
    base.ALL_MODES = ALL_MODES
    base.MODE_DEFINITIONS = MODE_DEFINITIONS
    base.mode_shift = mode_shift


def write_json_exclusive(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, payload) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--baseline-records", type=Path, required=True)
    plan.add_argument("--calibrated-metrics", type=Path, required=True)
    plan.add_argument("--source-shift-plan", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--count", type=int, default=COHORT_SIZE)
    plan.add_argument("--plan-seed", type=int, default=PLAN_SEED)
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
    if not args.source_shift_plan.is_file():
        raise FileNotFoundError(args.source_shift_plan)
    result = base.command_plan(args)
    plan_path = args.output_dir / "shift_plan.json"
    plan = json.loads(plan_path.read_text())
    source = json.loads(args.source_shift_plan.read_text())
    source_by_id = {str(sample["run_id"]): sample for sample in source["samples"]}
    for sample in plan["samples"]:
        run_id = str(sample["run_id"])
        source_shift = source_by_id[run_id]["modes"]["nfpa_independent"]
        for mode in ALL_MODES:
            shift = sample["modes"][mode]
            if shift["flow_dx_image_px"] != source_shift["flow_dx_image_px"] or shift["flow_dy_image_px"] != source_shift["flow_dy_image_px"]:
                raise ValueError(f"paired shift drift for run_id={run_id} mode={mode}")
    plan["protocol"] = "raven_nfpa_normalization_ablation_v1"
    plan["source_shift_plan"] = str(args.source_shift_plan.resolve())
    plan["source_shift_plan_sha256"] = sha256_path(args.source_shift_plan)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    root = args.output_dir.resolve()
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
        f"{env} {python} -u {script} validate --output-dir {root}",
        f"script -q -e -c \"{env} {python} -u {script} attack --output-dir {root} --device cuda --dtype float16\" {root}/logs/attack.log",
        f"script -q -e -c \"{env} {python} -u {script} score --output-dir {root} --device cuda\" {root}/logs/score.log",
        f"{env} {python} -u {script} aggregate --output-dir {root} --bootstrap-samples 10000 --bootstrap-seed {BOOTSTRAP_SEED}",
    ]
    (args.output_dir / "commands_normalization.sh").write_text("\n".join(commands) + "\n")
    (args.output_dir / "commands_normalization.sh").chmod(0o755)
    return result


def command_validate(args) -> int:
    import torch

    plan, _, _ = base.load_protocol(args.output_dir)
    cases = []
    all_equal = True
    for sample in plan["samples"]:
        dx = sample["modes"]["N1_nfpa_exact"]["flow_dx_image_px"]
        dy = sample["modes"]["N1_nfpa_exact"]["flow_dy_image_px"]
        torch.manual_seed(1000 + int(sample["run_id"]))
        latent = torch.randn(1, 4, 64, 64)
        outputs = {
            mode: translate_latent(
                latent, dx, dy, shift_space="image_pixels", vae_scale_factor=8,
                padding_mode="reflection", warp_mode=MODE_DEFINITIONS[mode]["warp_mode"],
            )
            for mode in ALL_MODES
        }
        equal_n2_n3 = torch.equal(outputs["N2_pixel_center"], outputs["N3_latent_div8"])
        all_equal = all_equal and equal_n2_n3
        cases.append({
            "run_id": str(sample["run_id"]), "flow_dx_image_px": dx, "flow_dy_image_px": dy,
            "N2_equals_N3": equal_n2_n3,
            "N1_vs_N2_max_abs_difference": float((outputs["N1_nfpa_exact"] - outputs["N2_pixel_center"]).abs().max()),
            "observed_displacement": {
                mode: [sample["modes"][mode]["observed_visual_latent_dx"], sample["modes"][mode]["observed_visual_latent_dy"]]
                for mode in ALL_MODES
            },
        })
    payload = {
        "passed": all_equal,
        "fixed_sampling": {"mode": "nearest", "padding_mode": "reflection", "align_corners": False},
        "N2_equals_N3_all_samples": all_equal,
        "cases": cases,
    }
    write_json_exclusive(args.output_dir / "unit_test_results.json", payload)
    write_json_exclusive(args.output_dir / "observed_displacement.json", {
        "modes": {mode: [case["observed_displacement"][mode] for case in cases] for mode in ALL_MODES}
    })
    write_json_exclusive(args.output_dir / "preflight" / "warp_direction_tests.json", {"passed": payload["passed"]})
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("normalization isolation validation failed")
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
                    input_image=watermarked, output_dir=item_dir, steps=50, strength=0.15,
                    guidance_scale=2.5, shift_space="image_pixels", warp_mode=shift["warp_mode"],
                    padding_mode="reflection", shift_x=shift["flow_dx_image_px"], shift_y=shift["flow_dy_image_px"],
                    view_guided_attention=True, color_transfer=True, seed=int(row["attack_seed"]),
                    prompt="", negative_prompt="", debug=False, inversion_mode="ddim",
                )
                final_path = item_dir / "final_color_corrected.png"
                attacked = load_image(final_path, size=None)
                debug_info = json.loads((item_dir / "debug_info.json").read_text())
                if debug_info["inversion_prompt"] != "" or debug_info["reconstruction_prompt"] != "":
                    raise RuntimeError("non-empty prompt entered pipeline")
                visual_dx = int(round(shift["visual_shift_dx_image_px"]))
                visual_dy = int(round(shift["visual_shift_dy_image_px"]))
                quality = {
                    **quality_pair(watermarked, attacked, visual_dx, visual_dy, "watermarked"),
                    **quality_pair(clean, attacked, visual_dx, visual_dy, "clean"),
                }
                clip = debug_info["clipping_diagnostics"]
                record = {
                    "sample_id": run_id, "run_id": run_id, "mode": mode,
                    "prompt": "", "prompt_sha256": EMPTY_PROMPT_SHA256,
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "watermark_seed": int(row["watermark_seed"]), "attack_seed": int(row["attack_seed"]),
                    "watermarked_path": str(watermarked_path.resolve()), "watermarked_sha256": sha256_path(watermarked_path),
                    "clean_path": str(clean_path.resolve()), "clean_sha256": sha256_path(clean_path),
                    "attacked_path": str(final_path.resolve()), "output_sha256": sha256_path(final_path),
                    "quality_primary_reference": "watermarked_input",
                    "exact_ddim_timestep": int(debug_info["exact_timestep"]),
                    **shift, **quality,
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


def finite_stats(values) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(values)), "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
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
            "win_lower": int(np.sum(diffs < -1e-12)), "tie": int(np.sum(np.abs(diffs) <= 1e-12)),
            "loss_higher": int(np.sum(diffs > 1e-12)),
        }
    return result


def command_aggregate(args) -> int:
    path = args.output_dir / "per_sample_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(ALL_MODES) * COHORT_SIZE:
        raise ValueError(f"Expected 30 rows, found {len(rows)}")
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in ALL_MODES}
    summaries = {}
    for mode, items in grouped.items():
        summaries[mode] = {
            "N": len(items), "detect_count": sum(row["detect_after"] for row in items),
            "detect_rate": float(statistics.fmean(row["detect_after"] for row in items)),
            "tree_ring_score": finite_stats(row["attacked_score_after"] for row in items),
            "psnr_vs_watermarked": finite_stats(row["psnr_vs_watermarked"] for row in items),
            "ssim_vs_watermarked": finite_stats(row["ssim_vs_watermarked"] for row in items),
            "psnr_vs_clean": finite_stats(row["psnr_vs_clean"] for row in items),
            "ssim_vs_clean": finite_stats(row["ssim_vs_clean"] for row in items),
            "observed_latent_displacements": [[row["observed_visual_latent_dx"], row["observed_visual_latent_dy"]] for row in items],
            "nan_count": sum(row["detector_nan"] for row in items),
            "inf_count": sum(row["detector_inf"] for row in items),
            "underflow_count": sum(row["p_value_underflow"] for row in items),
        }
    aggregate = {
        "protocol": "10-sample paired normalization-only diagnostic; not formal TPR@1%FPR",
        "threshold_held_fixed": THRESHOLD,
        "quality_primary_reference": "attacked image vs watermarked input",
        "fixed_sampling": {"mode": "nearest", "padding_mode": "reflection", "align_corners": False},
        "modes": summaries,
    }
    write_json_exclusive(args.output_dir / "aggregate_results.json", aggregate)
    metrics = ("attacked_score_after", "psnr_vs_watermarked", "ssim_vs_watermarked", "psnr_vs_clean", "ssim_vs_clean")
    specs = (("N1_vs_N2", "N1_nfpa_exact", "N2_pixel_center"), ("N1_vs_N3", "N1_nfpa_exact", "N3_latent_div8"), ("N2_vs_N3", "N2_pixel_center", "N3_latent_div8"))
    comparisons = {
        name: {"candidate": left, "reference": right, **paired_comparison(grouped[left], grouped[right], metrics, args.bootstrap_samples, args.bootstrap_seed + index)}
        for index, (name, left, right) in enumerate(specs)
    }
    write_json_exclusive(args.output_dir / "paired_comparisons.json", {"comparisons": comparisons, "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed})
    lines = [
        "# NFPA Normalization Ablation", "",
        "Primary quality reference: attacked image versus watermarked input.", "",
        "| Mode | Detect rate | Mean score | Median score | PSNR vs watermarked | SSIM vs watermarked | PSNR vs clean | SSIM vs clean |",
        "| --- | -: | -: | -: | -: | -: | -: | -: |",
    ]
    for mode in ALL_MODES:
        s = summaries[mode]
        lines.append(f"| {mode} | {s['detect_rate']:.4f} | {s['tree_ring_score']['mean']:.6f} | {s['tree_ring_score']['median']:.6f} | {s['psnr_vs_watermarked']['mean']:.3f} | {s['ssim_vs_watermarked']['mean']:.4f} | {s['psnr_vs_clean']['mean']:.3f} | {s['ssim_vs_clean']['mean']:.4f} |")
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


def main() -> int:
    configure_base()
    args = build_parser().parse_args()
    if args.command == "plan": return command_plan(args)
    if args.command == "validate": return command_validate(args)
    if args.command == "attack": return command_attack(args)
    if args.command == "score": return command_score(args)
    if args.command == "aggregate": return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
