#!/usr/bin/env python
"""ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT. Historical P1 workflow."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap_inverse_warp, detection_rate, roc_auc
from raven.pairing_provenance import (
    audit_pairing_rows,
    build_attack_config_sha256,
)
from raven.pipeline_raven import RavenPipeline, require_effective_source_flow
from raven.utils import load_image
MODEL_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
THRESHOLD = 1.6372738343020807
EMPTY_PROMPT_SHA256 = hashlib.sha256(b"").hexdigest()
PLAN_SEED = 2026071401
VAE_SCALE_FACTOR = 8
P1_MODE = "RAVEN_paper_NFPA_gap_fill_nearest"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_text(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd or Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def direction_label(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        return "(0,0)"
    return f"({'+' if dx > 0 else '-'},{'+' if dy > 0 else '-'})"


def finite_stats(values) -> dict:
    values = [float(v) for v in values]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        raise ValueError("no finite values")
    return {
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def histogram(values, bins: int = 40) -> dict:
    values = np.asarray([float(v) for v in values], dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def quality_pair(
    reference: Image.Image,
    attacked: Image.Image,
    effective_source_dx: float,
    effective_source_dy: float,
    suffix: str,
) -> dict:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(attacked.convert("RGB"), dtype=np.float32) / 255.0
    overlap_first, overlap_second = crop_overlap_inverse_warp(
        first, second, effective_source_dx, effective_source_dy
    )
    return {
        f"psnr_vs_{suffix}": float(peak_signal_noise_ratio(overlap_first, overlap_second, data_range=1.0)),
        f"ssim_vs_{suffix}": float(structural_similarity(overlap_first, overlap_second, channel_axis=2, data_range=1.0)),
        f"valid_overlap_width_vs_{suffix}": int(overlap_first.shape[1]),
        f"valid_overlap_height_vs_{suffix}": int(overlap_first.shape[0]),
        f"valid_overlap_area_ratio_vs_{suffix}": float(overlap_first.shape[0] * overlap_first.shape[1] / (first.shape[0] * first.shape[1])),
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_dirs(output_dir: Path) -> None:
    for name in ("configs", "logs", "pids", "outputs", "state"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def build_shift(run_id: str, index: int, seed_base: int) -> dict:
    rng = random.Random(seed_base + int(run_id))
    x_mag = rng.randint(24, 32)
    y_mag = rng.randint(24, 32)
    # Independent signs, deterministic under the global seed. Gate is stratified for coverage.
    if index < 30:
        signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))[index % 4]
        x_sign, y_sign = signs
    else:
        x_sign = rng.choice([-1, 1])
        y_sign = rng.choice([-1, 1])
    flow_x = float(x_sign * x_mag)
    flow_y = float(y_sign * y_mag)
    return {
        "mode": P1_MODE,
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "sampling_mode": "nearest",
        "padding_mode": "reflection",
        "align_corners": False,
        "shift_space": "image_pixels",
        "x_magnitude": x_mag,
        "y_magnitude": y_mag,
        "x_sign": x_sign,
        "y_sign": y_sign,
        "flow_dx_image_px": flow_x,
        "flow_dy_image_px": flow_y,
        "dx_image_px": flow_x,
        "dy_image_px": flow_y,
        "dx_latent_equivalent": flow_x / VAE_SCALE_FACTOR,
        "dy_latent_equivalent": flow_y / VAE_SCALE_FACTOR,
        "visual_shift_dx_image_px": -flow_x,
        "visual_shift_dy_image_px": -flow_y,
        "visual_dx_image_px": -flow_x,
        "visual_dy_image_px": -flow_y,
        "flow_direction": direction_label(flow_x, flow_y),
        "visual_content_direction": direction_label(-flow_x, -flow_y),
        "normalization_formula": "x_norm = 2*(x+dx)/W - 1; y_norm = 2*(y+dy)/H - 1; coordinate grid resized bilinear to latent grid",
        "grid_sample_inverse_sampling": True,
    }


def command_plan_dataset(args) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = load_csv_rows(args.manifest)
    if args.count is not None:
        rows = rows[:args.count]
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} rows, got {len(rows)}")
    pairing_audit = audit_pairing_rows(
        rows, expected_count=args.expected_count, verify_files=True
    )
    if pairing_audit["model_revision"] != MODEL_REVISION:
        raise ValueError(
            f"paired generation model revision {pairing_audit['model_revision']} "
            f"does not match formal attack revision {MODEL_REVISION}"
        )
    if {row["model_id"] for row in rows} != {MODEL_ID}:
        raise ValueError("paired generation model_id does not match formal attack model_id")
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_id in manifest slice")
    stage_dirs(output_dir)
    manifest_rows = []
    plan_rows = []
    for index, row in enumerate(rows):
        run_id = str(row["run_id"])
        clean = Path(row["clean_path"]).resolve()
        watermarked = Path(row["watermarked_path"]).resolve()
        for path, expected in ((clean, row["clean_sha256"]), (watermarked, row["watermarked_sha256"])):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_path(path)
            if expected and actual != expected:
                raise ValueError(f"SHA mismatch for {path}: expected {expected}, got {actual}")
            with Image.open(path) as image:
                if image.convert("RGB").size != (512, 512):
                    raise ValueError(f"unexpected image size for {path}: {image.size}")
        shift = build_shift(run_id, index, args.plan_seed)
        plan_rows.append({
            "cohort_index": index,
            "sample_id": run_id,
            "run_id": run_id,
            "base_rng_seed": args.plan_seed + int(run_id),
            "attack_seed": int(row.get("attack_seed") or (42 + int(run_id))),
            "shift": shift,
        })
        manifest_rows.append({
            "dataset": row["dataset"],
            "method": "TR",
            "cohort_index": index,
            "sample_id": run_id,
            "run_id": run_id,
            "prompt_id": row.get("prompt_id", ""),
            "prompt": row.get("prompt", ""),
            "source_prompt": row.get("prompt", ""),
            "raven_prompt": "",
            "prompt_sha256": row["prompt_sha256"],
            "raven_prompt_sha256": EMPTY_PROMPT_SHA256,
            "source": row.get("source", ""),
            "clean_path": str(clean),
            "clean_sha256": sha256_path(clean),
            "watermarked_path": str(watermarked),
            "watermarked_sha256": sha256_path(watermarked),
            "protocol": row["protocol"],
            "base_latent_seed": int(row["base_latent_seed"]),
            "base_latent_sha256": row["base_latent_sha256"],
            "clean_base_latent_sha256": row["clean_base_latent_sha256"],
            "watermarked_base_latent_sha256": row["watermarked_base_latent_sha256"],
            "watermarked_latent_sha256": row["watermarked_latent_sha256"],
            "watermark_target_sha256": row["watermark_target_sha256"],
            "watermark_mask_sha256": row["watermark_mask_sha256"],
            "generation_config_sha256": row["generation_config_sha256"],
            "watermark_config_sha256": row["watermark_config_sha256"],
            "pairing_sha256": row["pairing_sha256"],
            "injection_only_difference_verified": row["injection_only_difference_verified"],
            "injection_max_abs_error": row["injection_max_abs_error"],
            "generation_seed": row.get("generation_seed", ""),
            "attack_seed": int(row.get("attack_seed") or (42 + int(run_id))),
            "watermark_seed": int(row.get("w_seed") or 999999),
            "w_seed": row.get("w_seed") or "999999",
            "w_channel": row.get("w_channel") or "3",
            "w_pattern": row.get("w_pattern") or "ring",
            "w_mask_shape": row.get("w_mask_shape") or "circle",
            "w_radius": row.get("w_radius") or "10",
            "w_measurement": row.get("w_measurement") or "l1_complex",
            "w_injection": row.get("w_injection") or "complex",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        })
    with (output_dir / "diagnostic_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_json(output_dir / "shift_plan.json", {
        "protocol": "raven_p1_full_v1",
        "dataset": args.dataset,
        "plan_seed": args.plan_seed,
        "shift_rule": "x/y magnitude independent; x/y sign independent; four diagonal directions allowed",
        "fixed_mode": {
            "warp_sampling": "nearest",
            "padding_mode": "reflection",
            "align_corners": False,
            "shift_unit": "image-space pixels",
            "latent_equivalent": "image shift / 8",
            "strength": 0.15,
            "guidance_scale": 2.5,
            "steps": 50,
            "prompt": "",
            "attention": "all UNet self-attention processors, every active denoising step",
        },
        "samples": plan_rows,
    })
    fixed = {
        "dataset": args.dataset,
        "expected_count": args.expected_count,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_path(args.manifest),
        "baseline_records": str(args.baseline_records.resolve()) if args.baseline_records else "",
        "baseline_records_sha256": sha256_path(args.baseline_records) if args.baseline_records else "",
        "threshold": THRESHOLD,
        "threshold_source": "DiffusionDB calibrated threshold from outputs/verification_v2/metrics/TR_diffusiondb_1001_20260713T074340Z.json",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pairing_audit": pairing_audit,
        "python": sys.executable,
    }
    write_json(output_dir / "configs" / "fixed_conditions.json", fixed)
    repo = Path(__file__).resolve().parents[2]
    diff_text = run_text(["git", "diff"], cwd=repo)
    (output_dir / "git_diff.patch").write_text(diff_text + ("\n" if diff_text else ""), encoding="utf-8")
    provenance = {
        **fixed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "git_status_short": run_text(["git", "status", "--short"], cwd=repo).splitlines(),
        "git_diff_sha256": hashlib.sha256(diff_text.encode()).hexdigest(),
        "python_version": run_text([sys.executable, "--version"]),
        "package_versions": package_versions(),
        "nvidia_smi": run_text(["nvidia-smi"]),
    }
    write_json(output_dir / "provenance.json", provenance)
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"{sys.executable} -u {Path(__file__).resolve()} attack --output-dir {output_dir}",
        f"{sys.executable} -u {Path(__file__).resolve()} score --output-dir {output_dir}",
        f"{sys.executable} -u {Path(__file__).resolve()} aggregate --output-dir {output_dir}",
    ]
    commands_path = output_dir / "commands.sh"
    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)
    print(json.dumps({"output_dir": str(output_dir), "dataset": args.dataset, "samples": len(manifest_rows)}, indent=2))
    return 0


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "numpy", "scipy", "pandas", "skimage", "PIL", "tqdm"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


def load_protocol(output_dir: Path) -> tuple[dict, dict[str, dict[str, str]], dict]:
    plan = json.loads((output_dir / "shift_plan.json").read_text())
    with (output_dir / "diagnostic_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(rows, expected_count=len(plan["samples"]), verify_files=True)
    manifest = {row["run_id"]: row for row in rows}
    provenance = json.loads((output_dir / "provenance.json").read_text())
    if provenance.get("pairing_audit") != audit:
        raise ValueError("stored pairing audit does not match current source provenance")
    return plan, manifest, provenance


def command_attack(args) -> int:
    import torch

    plan, manifest, provenance = load_protocol(args.output_dir)
    records_path = args.output_dir / "attack_records.jsonl"
    completed: dict[tuple[str, str], dict] = {}
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        for row in (json.loads(line) for line in records_path.read_text().splitlines() if line.strip()):
            key = (row["mode"], str(row["run_id"]))
            attacked_path = Path(row["attacked_path"])
            if not attacked_path.is_file() or sha256_path(attacked_path) != row["attacked_sha256"]:
                raise ValueError(f"completed output hash mismatch for {key}")
            if row.get("attack_config_sha256") != build_attack_config_sha256(row):
                raise ValueError(f"completed attack config hash mismatch for {key}")
            completed[key] = row
        open_mode = "a"
    pipe = RavenPipeline(model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype)
    total = len(plan["samples"])
    done = len(completed)
    with records_path.open(open_mode, encoding="utf-8") as output:
        for sample in plan["samples"]:
            run_id = str(sample["run_id"])
            key = (P1_MODE, run_id)
            if key in completed:
                continue
            row = manifest[run_id]
            shift = sample["shift"]
            watermarked_path = Path(row["watermarked_path"])
            clean_path = Path(row["clean_path"])
            if sha256_path(watermarked_path) != row["watermarked_sha256"]:
                raise ValueError(f"watermarked SHA drift run_id={run_id}")
            if sha256_path(clean_path) != row["clean_sha256"]:
                raise ValueError(f"clean SHA drift run_id={run_id}")
            watermarked = load_image(watermarked_path, size=512)
            clean = load_image(clean_path, size=512)
            item_dir = args.output_dir / "outputs" / f"{int(run_id):06d}"
            if item_dir.exists():
                raise FileExistsError(item_dir)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            pipe.run(
                input_image=watermarked,
                output_dir=item_dir,
                steps=50,
                strength=0.15,
                guidance_scale=2.5,
                shift_space="image_pixels",
                warp_mode="raven_paper_nfpa_gap_fill",
                padding_mode="reflection",
                latent_sampling_mode="nearest",
                shift_x=shift["flow_dx_image_px"],
                shift_y=shift["flow_dy_image_px"],
                view_guided_attention=True,
                color_transfer=False,
                seed=int(row["attack_seed"]),
                prompt="",
                negative_prompt="",
                debug=args.debug,
                inversion_mode="ddim",
            )
            final_path = item_dir / "final.png"
            attacked = load_image(final_path, size=None)
            debug_info_path = item_dir / "debug_info.json"
            debug_info = json.loads(debug_info_path.read_text())
            if debug_info["inversion_prompt"] != "" or debug_info["reconstruction_prompt"] != "":
                raise RuntimeError(f"non-empty prompt run_id={run_id}")
            if debug_info["warp_mode"] != "raven_paper_nfpa_gap_fill" or debug_info["padding_mode"] != "reflection" or debug_info["interpolation_mode"] != "nearest":
                raise RuntimeError(f"P1 metadata drift run_id={run_id}")
            effective_dx, effective_dy = require_effective_source_flow(debug_info)
            quality = {
                **quality_pair(
                    watermarked, attacked, effective_dx, effective_dy, "watermarked"
                ),
                **quality_pair(clean, attacked, effective_dx, effective_dy, "clean"),
            }
            clip = debug_info["clipping_diagnostics"]
            attention = debug_info.get("attention_debug", {})
            record = {
                "dataset": provenance["dataset"],
                "sample_id": run_id,
                "run_id": run_id,
                "mode": P1_MODE,
                "source_image_path": row["clean_path"],
                "clean_path": row["clean_path"],
                "clean_sha256": row["clean_sha256"],
                "watermarked_path": row["watermarked_path"],
                "watermarked_sha256": row["watermarked_sha256"],
                "pairing_sha256": row["pairing_sha256"],
                "base_latent_seed": int(row["base_latent_seed"]),
                "base_latent_sha256": row["base_latent_sha256"],
                "watermark_target_sha256": row["watermark_target_sha256"],
                "generation_config_sha256": row["generation_config_sha256"],
                "watermark_config_sha256": row["watermark_config_sha256"],
                "attacked_path": str(final_path.resolve()),
                "attacked_sha256": sha256_path(final_path),
                "seed": int(row["attack_seed"]),
                "generation_seed": row.get("generation_seed", ""),
                "watermark_seed": int(row["watermark_seed"]),
                **shift,
                "prompt": "",
                "inversion_prompt": debug_info["inversion_prompt"],
                "reconstruction_prompt": debug_info["reconstruction_prompt"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "exact_ddim_timestep": int(debug_info["exact_timestep"]),
                "active_denoising_steps": len(debug_info["timesteps"]),
                "steps": 50,
                "strength": 0.15,
                "guidance_scale": 2.5,
                "inversion_mode": debug_info["inversion_mode"],
                "warp_mode": debug_info["warp_mode"],
                "sampling_mode": debug_info["interpolation_mode"],
                "padding_mode": debug_info["padding_mode"],
                "normalization_formula": debug_info["normalized_coordinate_formula"],
                "color_transfer_mode": debug_info["color_transfer_mode"],
                "transform_config_hash": debug_info.get("transform_config_hash"),
                "effective_source_flow_dx_image_px": effective_dx,
                "effective_source_flow_dy_image_px": effective_dy,
                **quality,
                "quality_primary_reference": "watermarked_input",
                "runtime_seconds": float(time.monotonic() - started),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "clipping_ratio": float(clip["fraction_below_zero"] + clip["fraction_above_one"]),
                "attention_processor_count": int(debug_info.get("attention_processor_count") or 0),
                "attention_self_processor_count": int(attention.get("self_processor_count") or 0),
                "attention_processors_with_calls": int(attention.get("processors_with_calls") or 0),
                "attention_total_calls": int(attention.get("total_calls") or 0),
                "attention_expected_total_calls": int((attention.get("self_processor_count") or 0) * len(debug_info["timesteps"])),
                "debug_info_path": str(debug_info_path.resolve()),
            }
            record["attack_config_sha256"] = build_attack_config_sha256(record)
            append_jsonl(output, record)
            done += 1
            print(f"[{done}/{total}] dataset={record['dataset']} run_id={run_id} score_pending psnr_wm={record['psnr_vs_watermarked']:.3f} ssim_wm={record['ssim_vs_watermarked']:.4f} gpu={record['peak_gpu_memory_bytes']/2**30:.2f}GiB", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def load_baseline(path: str) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {str(row["run_id"]): row for row in csv.DictReader(handle)}


def tr_diagnostic(result: dict) -> dict:
    diagnostic = (result.get("p_value_diagnostics") or [{}])[0]
    return {
        "log_p": diagnostic.get("log_p"),
        "sigma": diagnostic.get("sigma"),
        "lambda": diagnostic.get("lambda"),
        "statistic": diagnostic.get("statistic"),
        "df": diagnostic.get("df"),
        "p_underflow": bool(diagnostic.get("p_underflow", False)),
    }


def score_one(torch, provider, pipe, path: Path, raw_score, canonical_score, evaluate_image) -> dict:
    result = evaluate_image(torch, provider, pipe, path, 50)
    raw = raw_score("TR", result)
    canonical = canonical_score("TR", raw, result)
    diag = tr_diagnostic(result)
    return {"raw": raw, "canonical": canonical, **diag}


def baseline_stage(base: dict[str, str], stage: str) -> dict:
    return {
        "raw": float(base[f"{stage}_raw_score"]),
        "canonical": float(base[f"{stage}_canonical_score"]),
        "log_p": float(base[f"{stage}_tr_log_p"]),
        "sigma": float(base[f"{stage}_tr_sigma"]),
        "lambda": float(base[f"{stage}_tr_lambda"]),
        "statistic": float(base[f"{stage}_tr_statistic"]),
        "df": int(float(base[f"{stage}_tr_df"])),
        "p_underflow": str(base.get(f"{stage}_tr_p_underflow", "")).lower() == "true",
    }


def command_score(args) -> int:
    import torch
    if not (args.eval_repo / "utils" / "pipe" / "pipe_utils.py").is_file():
        raise FileNotFoundError(args.eval_repo)
    sys.path.insert(0, str(args.eval_repo.resolve()))
    from scripts.extract_verification_scores import canonical_score, evaluate_image, provider_class, provider_kwargs, raw_score
    from raven.resource_guard import limit_cpu_threads
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    plan, manifest, provenance = load_protocol(args.output_dir)
    baseline = load_baseline(provenance.get("baseline_records", ""))
    attacks = [json.loads(line) for line in (args.output_dir / "attack_records.jsonl").read_text().splitlines() if line.strip()]
    if len(attacks) != len(plan["samples"]):
        raise ValueError(f"expected {len(plan['samples'])} attack records, found {len(attacks)}")
    records_path = args.output_dir / "per_sample_results.jsonl"
    completed: dict[str, dict] = {}
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        for row in (json.loads(line) for line in records_path.read_text().splitlines() if line.strip()):
            completed[str(row["run_id"])] = row
        open_mode = "a"
    device = torch.device(args.device)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=MODEL_ID,
        resolution=512,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        revision=MODEL_REVISION,
    )
    first = manifest[attacks[0]["run_id"]]
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **provider_kwargs("TR", first),
    )
    attack_by_id = {str(row["run_id"]): row for row in attacks}
    scored = list(completed.values())
    with records_path.open(open_mode, encoding="utf-8") as output:
        for index, sample in enumerate(plan["samples"], start=1):
            run_id = str(sample["run_id"])
            if run_id in completed:
                continue
            row = manifest[run_id]
            attack = attack_by_id[run_id]
            if sha256_path(Path(attack["attacked_path"])) != attack["attacked_sha256"]:
                raise ValueError(f"attacked hash drift run_id={run_id}")
            if run_id in baseline:
                clean_score = baseline_stage(baseline[run_id], "clean")
                watermarked_score = baseline_stage(baseline[run_id], "watermarked")
            else:
                clean_score = score_one(torch, provider, pipe, Path(row["clean_path"]), raw_score, canonical_score, evaluate_image)
                watermarked_score = score_one(torch, provider, pipe, Path(row["watermarked_path"]), raw_score, canonical_score, evaluate_image)
            attacked_score = score_one(torch, provider, pipe, Path(attack["attacked_path"]), raw_score, canonical_score, evaluate_image)
            record = {
                **attack,
                "clean_raw_score": clean_score["raw"],
                "clean_canonical_score": clean_score["canonical"],
                "clean_tr_log_p": clean_score["log_p"],
                "clean_tr_statistic": clean_score["statistic"],
                "clean_tr_p_underflow": clean_score["p_underflow"],
                "watermarked_raw_score": watermarked_score["raw"],
                "watermarked_score_before": watermarked_score["canonical"],
                "watermarked_canonical_score": watermarked_score["canonical"],
                "watermarked_tr_log_p": watermarked_score["log_p"],
                "watermarked_tr_statistic": watermarked_score["statistic"],
                "watermarked_tr_p_underflow": watermarked_score["p_underflow"],
                "attacked_raw_score": attacked_score["raw"],
                "attacked_score_after": attacked_score["canonical"],
                "attacked_canonical_score": attacked_score["canonical"],
                "attacked_tr_log_p": attacked_score["log_p"],
                "attacked_tr_statistic": attacked_score["statistic"],
                "attacked_tr_p_underflow": attacked_score["p_underflow"],
                "score_delta": attacked_score["canonical"] - watermarked_score["canonical"],
                "calibrated_threshold": THRESHOLD,
                "detect_clean": clean_score["canonical"] >= THRESHOLD,
                "detect_before": watermarked_score["canonical"] >= THRESHOLD,
                "detect_after": attacked_score["canonical"] >= THRESHOLD,
                "tree_ring_score_definition": "-log10(p), higher means more watermark",
                "detector_nan": any(math.isnan(float(x["canonical"])) for x in (clean_score, watermarked_score, attacked_score)),
                "detector_inf": any(math.isinf(float(x["canonical"])) for x in (clean_score, watermarked_score, attacked_score)),
                "p_value_underflow": bool(clean_score["p_underflow"] or watermarked_score["p_underflow"] or attacked_score["p_underflow"]),
            }
            if record["detector_nan"] or record["detector_inf"]:
                raise ValueError(f"non-finite detector score run_id={run_id}")
            append_jsonl(output, record)
            scored.append(record)
            print(f"[{index}/{len(plan['samples'])}] dataset={record['dataset']} run_id={run_id} before={record['watermarked_score_before']:.6f} after={record['attacked_score_after']:.6f}", flush=True)
    scored.sort(key=lambda x: int(x["run_id"]))
    csv_path = args.output_dir / "per_sample_results.csv"
    if not csv_path.exists():
        write_csv(csv_path, scored)
    del provider, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def summarize_dataset(rows: list[dict]) -> dict:
    clean = [float(row["clean_canonical_score"]) for row in rows]
    before = [float(row["watermarked_score_before"]) for row in rows]
    attacked = [float(row["attacked_score_after"]) for row in rows]
    clean_fpr = detection_rate(clean, THRESHOLD)
    before_tpr = detection_rate(before, THRESHOLD)
    attacked_tpr = detection_rate(attacked, THRESHOLD)
    directions = {}
    for direction in sorted({row["visual_content_direction"] for row in rows}):
        items = [row for row in rows if row["visual_content_direction"] == direction]
        directions[direction] = {
            "N": len(items),
            "attacked_tpr": detection_rate([float(row["attacked_score_after"]) for row in items], THRESHOLD),
            "mean_attacked_score": finite_stats(row["attacked_score_after"] for row in items)["mean"],
            "psnr_vs_watermarked_mean": finite_stats(row["psnr_vs_watermarked"] for row in items)["mean"],
            "ssim_vs_watermarked_mean": finite_stats(row["ssim_vs_watermarked"] for row in items)["mean"],
        }
    return {
        "N_expected": len(rows),
        "N_completed": len(rows),
        "N_failed": 0,
        "threshold": THRESHOLD,
        "clean_actual_FPR_at_fixed_threshold": clean_fpr,
        "watermarked_TPR_before_attack": before_tpr,
        "attacked_TPR_at_fixed_threshold": attacked_tpr,
        "attack_success_rate": 1.0 - attacked_tpr,
        "ROC_AUC": {"before": roc_auc(before, clean), "attacked": roc_auc(attacked, clean)},
        "score": {
            "clean": finite_stats(clean),
            "before": finite_stats(before),
            "attacked": finite_stats(attacked),
        },
        "score_histogram": {
            "clean": histogram(clean),
            "before": histogram(before),
            "attacked": histogram(attacked),
        },
        "quality": {
            "primary_reference": "watermarked_input",
            "psnr_vs_watermarked": finite_stats(row["psnr_vs_watermarked"] for row in rows),
            "ssim_vs_watermarked": finite_stats(row["ssim_vs_watermarked"] for row in rows),
            "psnr_vs_clean": finite_stats(row["psnr_vs_clean"] for row in rows),
            "ssim_vs_clean": finite_stats(row["ssim_vs_clean"] for row in rows),
        },
        "numeric_diagnostics": {
            "nan_count": sum(bool(row["detector_nan"]) for row in rows),
            "inf_count": sum(bool(row["detector_inf"]) for row in rows),
            "underflow_count": sum(bool(row["p_value_underflow"]) for row in rows),
        },
        "runtime": {
            "total_attack_runtime_seconds": float(sum(float(row["runtime_seconds"]) for row in rows)),
            "mean_attack_runtime_seconds": finite_stats(row["runtime_seconds"] for row in rows)["mean"],
            "peak_gpu_memory_bytes": int(max(int(row["peak_gpu_memory_bytes"]) for row in rows)),
            "peak_cpu_rss_kib": int(max(int(row["peak_cpu_rss_kib"]) for row in rows)),
        },
        "shift_direction_analysis": directions,
        "below_threshold_sample_ids": [row["run_id"] for row in rows if not row["detect_after"]],
        "failed_sample_ids": [],
    }


def command_aggregate(args) -> int:
    rows = [json.loads(line) for line in (args.output_dir / "per_sample_results.jsonl").read_text().splitlines() if line.strip()]
    plan, _, provenance = load_protocol(args.output_dir)
    if len(rows) != len(plan["samples"]):
        raise ValueError(f"expected {len(plan['samples'])} rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate scored run_id")
    aggregate = {
        "protocol": "raven_p1_full_v1",
        "dataset": provenance["dataset"],
        "metric_naming": (
            "TPR@1%FPR" if abs(detection_rate([float(row["clean_canonical_score"]) for row in rows], THRESHOLD) - 0.01) <= 0.002
            else "TPR at fixed threshold; clean FPR is reported explicitly"
        ),
        "summary": summarize_dataset(rows),
    }
    write_json(args.output_dir / "aggregate_results.json", aggregate)
    lines = [
        f"# RAVEN P1 Full - {provenance['dataset']}",
        "",
        f"Threshold fixed at `{THRESHOLD}`; no recalibration on attacked outputs.",
        "",
        "| N | Clean FPR | Before TPR | Attacked TPR | Attack success | ROC-AUC attacked | PSNR vs WM | SSIM vs WM |",
        "| -: | -: | -: | -: | -: | -: | -: | -: |",
    ]
    s = aggregate["summary"]
    lines.append(
        f"| {s['N_completed']} | {s['clean_actual_FPR_at_fixed_threshold']:.6f} | {s['watermarked_TPR_before_attack']:.6f} | "
        f"{s['attacked_TPR_at_fixed_threshold']:.6f} | {s['attack_success_rate']:.6f} | {s['ROC_AUC']['attacked']:.6f} | "
        f"{s['quality']['psnr_vs_watermarked']['mean']:.3f} | {s['quality']['ssim_vs_watermarked']['mean']:.4f} |"
    )
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


def command_determinism_check(args) -> int:
    import torch

    plan, manifest, _ = load_protocol(args.output_dir)
    sample = plan["samples"][0]
    row = manifest[str(sample["run_id"])]
    shift = sample["shift"]
    out = args.output_dir / "determinism_check"
    if out.exists():
        raise FileExistsError(out)
    pipe = RavenPipeline(model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype)
    shas = []
    for idx in (1, 2):
        item_dir = out / f"replay_{idx}"
        torch.cuda.reset_peak_memory_stats()
        pipe.run(
            input_image=load_image(Path(row["watermarked_path"]), size=512),
            output_dir=item_dir,
            steps=50,
            strength=0.15,
            guidance_scale=2.5,
            shift_space="image_pixels",
            warp_mode="raven_paper_nfpa_gap_fill",
            padding_mode="reflection",
            latent_sampling_mode="nearest",
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
        shas.append(sha256_path(item_dir / "final_color_corrected.png"))
    payload = {"passed": shas[0] == shas[1], "sha256": shas, "run_id": str(sample["run_id"])}
    write_json(args.output_dir / "determinism_check.json", payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("determinism check failed")
    return 0


def command_validate_gate(args) -> int:
    rows = [json.loads(line) for line in (args.output_dir / "per_sample_results.jsonl").read_text().splitlines() if line.strip()]
    aggregate = json.loads((args.output_dir / "aggregate_results.json").read_text())
    determinism = json.loads((args.output_dir / "determinism_check.json").read_text())
    attention_ok = all(
        row["attention_self_processor_count"] == 16
        and row["attention_processors_with_calls"] == 16
        and row["attention_total_calls"] == row["attention_expected_total_calls"]
        for row in rows
    )
    prompts_ok = all(row["inversion_prompt"] == "" and row["reconstruction_prompt"] == "" for row in rows)
    timestep_ok = len({row["exact_ddim_timestep"] for row in rows}) == 1
    payload = {
        "passed": len(rows) == 30 and aggregate["summary"]["N_failed"] == 0 and attention_ok and prompts_ok and timestep_ok and determinism["passed"] and aggregate["summary"]["numeric_diagnostics"]["nan_count"] == 0 and aggregate["summary"]["numeric_diagnostics"]["inf_count"] == 0 and aggregate["summary"]["numeric_diagnostics"]["underflow_count"] == 0,
        "N": len(rows),
        "attention_16_of_16_every_active_step": attention_ok,
        "empty_prompts": prompts_ok,
        "exact_ddim_timestep_unique": sorted({row["exact_ddim_timestep"] for row in rows}),
        "determinism": determinism,
        "quality_primary_reference": "watermarked_input",
        "peak_gpu_memory_bytes": aggregate["summary"]["runtime"]["peak_gpu_memory_bytes"],
    }
    write_json(args.output_dir / "gate_validation.json", payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("gate validation failed")
    return 0


def run_stage(args, argv: list[str]) -> None:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *argv]
    print("$ " + " ".join(str(x) for x in command), flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def state_path(root: Path) -> Path:
    return root / "run_state.json"


def load_state(root: Path) -> dict:
    path = state_path(root)
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_stages": []}


def save_state(root: Path, state: dict) -> None:
    write_json(state_path(root), state, exclusive=False)


def mark_done(root: Path, state: dict, stage: str) -> None:
    if stage not in state["completed_stages"]:
        state["completed_stages"].append(stage)
    save_state(root, state)


def command_run_sequential(args) -> int:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = load_state(root)
    if "created_utc" not in state:
        state["created_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        state["gate_dir"] = str((root / "gate" / state["created_utc"]).resolve())
        state["diffusiondb_dir"] = str((root / "diffusiondb" / state["created_utc"]).resolve())
        state["mscoco_dir"] = str((root / "mscoco" / state["created_utc"]).resolve())
        save_state(root, state)
    stages = state["completed_stages"]
    env_summary = {"python": sys.executable, "argv": sys.argv, "cwd": str(Path.cwd())}
    write_json(root / "orchestrator_provenance.json", env_summary, exclusive=not (root / "orchestrator_provenance.json").exists())

    gate_dir = Path(state["gate_dir"])
    db_dir = Path(state["diffusiondb_dir"])
    coco_dir = Path(state["mscoco_dir"])
    if "gate_plan" not in stages:
        run_stage(args, ["plan-dataset", "--dataset", "diffusiondb_gate", "--manifest", str(args.diffusiondb_manifest), "--baseline-records", str(args.diffusiondb_baseline_records), "--output-dir", str(gate_dir), "--expected-count", "30", "--count", "30", "--plan-seed", str(args.plan_seed)])
        mark_done(root, state, "gate_plan")
    if "gate_attack" not in stages:
        run_stage(args, ["attack", "--output-dir", str(gate_dir), "--device", "cuda", "--dtype", "float16", "--debug"])
        mark_done(root, state, "gate_attack")
    if "gate_score" not in stages:
        run_stage(args, ["score", "--output-dir", str(gate_dir), "--device", "cuda"])
        mark_done(root, state, "gate_score")
    if "gate_aggregate" not in stages:
        run_stage(args, ["aggregate", "--output-dir", str(gate_dir)])
        mark_done(root, state, "gate_aggregate")
    if "gate_determinism" not in stages:
        run_stage(args, ["determinism-check", "--output-dir", str(gate_dir), "--device", "cuda", "--dtype", "float16"])
        mark_done(root, state, "gate_determinism")
    if "gate_validate" not in stages:
        run_stage(args, ["validate-gate", "--output-dir", str(gate_dir)])
        mark_done(root, state, "gate_validate")

    if "diffusiondb_plan" not in stages:
        run_stage(args, ["plan-dataset", "--dataset", "diffusiondb", "--manifest", str(args.diffusiondb_manifest), "--baseline-records", str(args.diffusiondb_baseline_records), "--output-dir", str(db_dir), "--expected-count", "1001", "--plan-seed", str(args.plan_seed)])
        mark_done(root, state, "diffusiondb_plan")
    if "diffusiondb_attack" not in stages:
        run_stage(args, ["attack", "--output-dir", str(db_dir), "--device", "cuda", "--dtype", "float16"])
        mark_done(root, state, "diffusiondb_attack")
    if "diffusiondb_score" not in stages:
        run_stage(args, ["score", "--output-dir", str(db_dir), "--device", "cuda"])
        mark_done(root, state, "diffusiondb_score")
    if "diffusiondb_aggregate" not in stages:
        run_stage(args, ["aggregate", "--output-dir", str(db_dir)])
        mark_done(root, state, "diffusiondb_aggregate")

    if "mscoco_plan" not in stages:
        run_stage(args, ["plan-dataset", "--dataset", "mscoco", "--manifest", str(args.mscoco_manifest), "--output-dir", str(coco_dir), "--expected-count", "1000", "--plan-seed", str(args.plan_seed)])
        mark_done(root, state, "mscoco_plan")
    if "mscoco_attack" not in stages:
        run_stage(args, ["attack", "--output-dir", str(coco_dir), "--device", "cuda", "--dtype", "float16"])
        mark_done(root, state, "mscoco_attack")
    if "mscoco_score" not in stages:
        run_stage(args, ["score", "--output-dir", str(coco_dir), "--device", "cuda"])
        mark_done(root, state, "mscoco_score")
    if "mscoco_aggregate" not in stages:
        run_stage(args, ["aggregate", "--output-dir", str(coco_dir)])
        mark_done(root, state, "mscoco_aggregate")
    if "combined" not in stages:
        write_combined(root, db_dir, coco_dir)
        mark_done(root, state, "combined")
    print(json.dumps(load_state(root), indent=2), flush=True)
    return 0


def write_combined(root: Path, db_dir: Path, coco_dir: Path) -> None:
    db = json.loads((db_dir / "aggregate_results.json").read_text())
    coco = json.loads((coco_dir / "aggregate_results.json").read_text())
    old = json.loads(Path("outputs/verification_v2/metrics/TR_diffusiondb_1001_20260713T074340Z.json").read_text())
    rows = []
    for name, agg in (("DiffusionDB", db), ("MS-COCO", coco)):
        s = agg["summary"]
        rows.append({
            "Dataset": name,
            "N": s["N_completed"],
            "Clean FPR": s["clean_actual_FPR_at_fixed_threshold"],
            "Before TPR": s["watermarked_TPR_before_attack"],
            "Attacked TPR": s["attacked_TPR_at_fixed_threshold"],
            "Attack success": s["attack_success_rate"],
            "ROC-AUC": s["ROC_AUC"]["attacked"],
            "PSNR vs WM": s["quality"]["psnr_vs_watermarked"]["mean"],
            "SSIM vs WM": s["quality"]["ssim_vs_watermarked"]["mean"],
        })
    summary = {
        "protocol": "raven_p1_full_combined_v1",
        "threshold": THRESHOLD,
        "diffusiondb_output": str(db_dir.resolve()),
        "mscoco_output": str(coco_dir.resolve()),
        "datasets": rows,
        "old_diffusiondb_pipeline_reference": {
            "source": "outputs/verification_v2/metrics/TR_diffusiondb_1001_20260713T074340Z.json",
            "attacked_TPR_old_pipeline": old["metric"]["calibrated_TPR_at_1pct_FPR"],
            "note": "legacy/old RAVEN attacked images, not P1 result",
        },
        "comparisons": {
            "suppression_diff_attacked_tpr_coco_minus_db": rows[1]["Attacked TPR"] - rows[0]["Attacked TPR"],
            "quality_diff_psnr_coco_minus_db": rows[1]["PSNR vs WM"] - rows[0]["PSNR vs WM"],
            "quality_diff_ssim_coco_minus_db": rows[1]["SSIM vs WM"] - rows[0]["SSIM vs WM"],
            "p1_diffusiondb_attacked_tpr_minus_old_pipeline": rows[0]["Attacked TPR"] - old["metric"]["calibrated_TPR_at_1pct_FPR"],
        },
    }
    write_json(root / "combined_summary.json", summary, exclusive=False)
    lines = [
        "# RAVEN P1 Combined Summary",
        "",
        "| Dataset | N | Clean FPR | Before TPR | Attacked TPR | Attack success | ROC-AUC | PSNR vs WM | SSIM vs WM |",
        "| ------- | -: | --------: | ---------: | -----------: | -------------: | ------: | ---------: | ---------: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['N']} | {row['Clean FPR']:.6f} | {row['Before TPR']:.6f} | {row['Attacked TPR']:.6f} | "
            f"{row['Attack success']:.6f} | {row['ROC-AUC']:.6f} | {row['PSNR vs WM']:.3f} | {row['SSIM vs WM']:.4f} |"
        )
    lines.extend([
        "",
        f"Old DiffusionDB pipeline attacked TPR reference: `{old['metric']['calibrated_TPR_at_1pct_FPR']}`. This is not a P1 result.",
    ])
    (root / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan-dataset")
    plan.add_argument("--dataset", required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--baseline-records", type=Path, default=None)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--expected-count", type=int, required=True)
    plan.add_argument("--count", type=int, default=None)
    plan.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    attack = sub.add_parser("attack")
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--device", default="cuda")
    attack.add_argument("--dtype", choices=["float16"], default="float16")
    attack.add_argument("--resume", action="store_true")
    attack.add_argument("--debug", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    score.add_argument("--device", choices=["cuda"], default="cuda")
    score.add_argument("--resume", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    determinism = sub.add_parser("determinism-check")
    determinism.add_argument("--output-dir", type=Path, required=True)
    determinism.add_argument("--device", default="cuda")
    determinism.add_argument("--dtype", choices=["float16"], default="float16")
    gate = sub.add_parser("validate-gate")
    gate.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run-sequential")
    run.add_argument("--root", type=Path, default=Path("outputs/tr/diffusiondb/p1_full"))
    run.add_argument("--diffusiondb-manifest", type=Path, required=True)
    run.add_argument("--diffusiondb-baseline-records", type=Path, required=True)
    run.add_argument("--mscoco-manifest", type=Path, required=True)
    run.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan-dataset":
        return command_plan_dataset(args)
    if args.command == "attack":
        return command_attack(args)
    if args.command == "score":
        return command_score(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    if args.command == "determinism-check":
        return command_determinism_check(args)
    if args.command == "validate-gate":
        return command_validate_gate(args)
    if args.command == "run-sequential":
        return command_run_sequential(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
