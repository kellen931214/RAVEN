#!/usr/bin/env python
"""N=10 effective-flow warp and color-transfer diagnostics."""

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
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAVEN_ROOT = ROOT / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.metrics import pair_quality_metrics
from raven.pipeline_raven import RavenPipeline
from raven.tree_ring_official import score_image
from scripts.tree_ring_official_raven_eval import (
    DEFAULT_MODEL_SOURCE,
    load_official_pipeline,
    sha256_path,
)

COUNT = 10
BASE_ATTACK = {
    "steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "inversion_mode": "ddim",
    "prompt": "",
    "negative_prompt": "",
    "shift_space": "image_pixels",
    "padding_mode": "reflection",
    "view_guided_attention": True,
}
EXP1_MODES = {
    "NFPA_exact_nearest": {
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "latent_sampling_mode": "nearest",
        "color_transfer": True,
    },
    "latent_grid_nearest": {
        "warp_mode": "latent_grid",
        "latent_sampling_mode": "nearest",
        "color_transfer": True,
    },
    "latent_grid_bilinear": {
        "warp_mode": "latent_grid",
        "latent_sampling_mode": "bilinear",
        "color_transfer": True,
    },
}
EXP2_MODES = {
    "DDIM_shift_no_color": {
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "latent_sampling_mode": "nearest",
        "color_transfer": False,
        "use_shift": True,
    },
    "DDIM_shift_with_color": {
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "latent_sampling_mode": "nearest",
        "color_transfer": True,
        "use_shift": True,
    },
    "DDIM_no_shift_no_color": {
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "latent_sampling_mode": "nearest",
        "color_transfer": False,
        "use_shift": False,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    fields: list[str] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        flat_rows.append(flat)
        for key in flat:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
        handle.flush()
        os.fsync(handle.fileno())


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def numeric_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("numeric_stats requires finite non-empty values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def quality(reference: Path, attacked: Path, debug: dict[str, Any]) -> dict[str, Any]:
    requested_dx = float(debug["flow_dx_image_px"])
    requested_dy = float(debug["flow_dy_image_px"])
    metadata = debug.get("nfpa_warp_metadata") or {}
    effective_dx = float(metadata["effective_flow_dx_image_px"])
    effective_dy = float(metadata["effective_flow_dy_image_px"])
    sampling = str(debug["interpolation_mode"])
    alignment = "fractional_grid_sample" if sampling == "bilinear" else "integer_crop"
    reference_image = Image.open(reference).convert("RGB")
    attacked_image = Image.open(attacked).convert("RGB")
    corrected = pair_quality_metrics(
        reference_image,
        attacked_image,
        effective_dx,
        effective_dy,
        alignment_mode=alignment,
    )
    legacy = pair_quality_metrics(
        reference_image,
        attacked_image,
        requested_dx,
        requested_dy,
        alignment_mode="integer_crop",
    )
    return {
        "raw_full_psnr": corrected["raw_full_psnr"],
        "raw_full_ssim": corrected["raw_full_ssim"],
        "corrected_overlap_psnr": corrected["overlap_psnr"],
        "corrected_overlap_ssim": corrected["overlap_ssim"],
        "corrected_overlap_protocol": corrected["overlap_protocol"],
        "legacy_requested_overlap_psnr": legacy["overlap_psnr"],
        "legacy_requested_overlap_ssim": legacy["overlap_ssim"],
        "legacy_protocol": "requested_flow_integer_crop_diagnostic_only",
        "requested_dx_image_px": requested_dx,
        "requested_dy_image_px": requested_dy,
        "effective_dx_image_px": effective_dx,
        "effective_dy_image_px": effective_dy,
        "effective_dx_latent_cells": float(metadata["effective_source_dx_latent_cells"]),
        "effective_dy_latent_cells": float(metadata["effective_source_dy_latent_cells"]),
        "valid_overlap_width": corrected["valid_overlap_width"],
        "valid_overlap_height": corrected["valid_overlap_height"],
        "valid_overlap_area_ratio": corrected["valid_overlap_area_ratio"],
    }


def validate_debug(debug: dict[str, Any], config: dict[str, Any], dx: int, dy: int) -> None:
    expected = {
        "inversion_mode": "ddim",
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "warp_mode": config["warp_mode"],
        "padding_mode": "reflection",
        "interpolation_mode": config["latent_sampling_mode"],
        "color_transfer_mode": "paper_exact_two_stage" if config["color_transfer"] else "none",
        "warp_input_stage": "ddim_inversion.noisy_latents_z_tau",
        "warp_input_is_inversion_noisy_latents": True,
        "decoded_output_branch": "view_branch_index_1",
    }
    for key, value in expected.items():
        if debug.get(key) != value:
            raise RuntimeError(f"config drift: {key}={debug.get(key)!r}, expected {value!r}")
    if float(debug["flow_dx_image_px"]) != dx or float(debug["flow_dy_image_px"]) != dy:
        raise RuntimeError("shift plan drift")
    attention = debug.get("attention_debug") or {}
    active_steps = len(debug.get("timesteps") or [])
    if int(attention.get("self_processor_count", 0)) != 16:
        raise RuntimeError("expected 16 self-attention processors")
    if int(attention.get("processors_with_calls", 0)) != 16:
        raise RuntimeError("not all self-attention processors were invoked")
    if int(attention.get("total_calls", 0)) != 16 * active_steps:
        raise RuntimeError("self-attention call count mismatch")
    metadata = debug.get("nfpa_warp_metadata") or {}
    required = {
        "effective_flow_dx_image_px",
        "effective_flow_dy_image_px",
        "effective_source_dx_latent_cells",
        "effective_source_dy_latent_cells",
    }
    if not required.issubset(metadata):
        raise RuntimeError(f"missing effective-flow metadata: {sorted(required - set(metadata))}")


def run_one(
    pipeline: RavenPipeline,
    input_path: Path,
    item_dir: Path,
    config: dict[str, Any],
    dx: int,
    dy: int,
    seed: int,
) -> dict[str, Any]:
    if item_dir.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output: {item_dir}")
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    pipeline.run(
        input_image=Image.open(input_path).convert("RGB"),
        output_dir=item_dir,
        steps=BASE_ATTACK["steps"],
        strength=BASE_ATTACK["strength"],
        guidance_scale=BASE_ATTACK["guidance_scale"],
        shift_space=BASE_ATTACK["shift_space"],
        warp_mode=config["warp_mode"],
        padding_mode=BASE_ATTACK["padding_mode"],
        latent_sampling_mode=config["latent_sampling_mode"],
        shift_x=dx,
        shift_y=dy,
        view_guided_attention=True,
        color_transfer=bool(config["color_transfer"]),
        seed=seed,
        prompt="",
        negative_prompt="",
        debug=False,
        inversion_mode="ddim",
    )
    runtime = time.monotonic() - started
    debug_path = item_dir / "debug_info.json"
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    validate_debug(debug, config, dx, dy)
    output_name = "final_color_corrected.png" if config["color_transfer"] else "final.png"
    output_path = item_dir / output_name
    pre_color_path = item_dir / "view_guided_output.png"
    if not output_path.is_file() or not pre_color_path.is_file():
        raise FileNotFoundError(f"missing pipeline output: {item_dir}")
    return {
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_path(output_path),
        "pre_color_path": str(pre_color_path.resolve()),
        "pre_color_sha256": sha256_path(pre_color_path),
        "debug_path": str(debug_path.resolve()),
        "debug": debug,
        "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def aggregate_modes(rows: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        if len(subset) != COUNT:
            raise ValueError(f"{mode}: expected {COUNT}, got {len(subset)}")
        output[mode] = {
            "n": len(subset),
            "detect_count": sum(bool(row["detect_after"]) for row in subset),
            "detect_rate": float(np.mean([row["detect_after"] for row in subset])),
            "attack_success": 1.0 - float(np.mean([row["detect_after"] for row in subset])),
            "score": numeric_stats([row["detector_score"] for row in subset]),
            "corrected_overlap_psnr": numeric_stats([row["corrected_overlap_psnr"] for row in subset]),
            "corrected_overlap_ssim": numeric_stats([row["corrected_overlap_ssim"] for row in subset]),
            "raw_full_psnr": numeric_stats([row["raw_full_psnr"] for row in subset]),
            "raw_full_ssim": numeric_stats([row["raw_full_ssim"] for row in subset]),
            "legacy_requested_overlap_psnr": numeric_stats([row["legacy_requested_overlap_psnr"] for row in subset]),
            "legacy_requested_overlap_ssim": numeric_stats([row["legacy_requested_overlap_ssim"] for row in subset]),
            "requested_displacements": sorted({
                (row["requested_dx_image_px"], row["requested_dy_image_px"]) for row in subset
            }),
            "effective_displacements": sorted({
                (row["effective_dx_image_px"], row["effective_dy_image_px"]) for row in subset
            }),
            "unique_effective_displacement_count": len({
                (row["effective_dx_image_px"], row["effective_dy_image_px"]) for row in subset
            }),
        }
    return output


def paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_rows = {int(row["run_id"]): row for row in rows if row["mode"] == left}
    right_rows = {int(row["run_id"]): row for row in rows if row["mode"] == right}
    if set(left_rows) != set(right_rows) or len(left_rows) != COUNT:
        raise ValueError(f"unpaired comparison {left} vs {right}")
    result: dict[str, Any] = {"left": left, "right": right, "difference": "left_minus_right"}
    for field in (
        "detector_score",
        "corrected_overlap_psnr",
        "corrected_overlap_ssim",
        "raw_full_psnr",
        "raw_full_ssim",
    ):
        values = [left_rows[index][field] - right_rows[index][field] for index in sorted(left_rows)]
        result[field] = numeric_stats(values)
        result[field]["win_tie_loss"] = {
            "positive": sum(value > 1e-12 for value in values),
            "tie": sum(abs(value) <= 1e-12 for value in values),
            "negative": sum(value < -1e-12 for value in values),
        }
    result["per_sample"] = [
        {
            "run_id": index,
            "score_difference": left_rows[index]["detector_score"] - right_rows[index]["detector_score"],
            "psnr_difference": left_rows[index]["corrected_overlap_psnr"] - right_rows[index]["corrected_overlap_psnr"],
            "ssim_difference": left_rows[index]["corrected_overlap_ssim"] - right_rows[index]["corrected_overlap_ssim"],
        }
        for index in sorted(left_rows)
    ]
    return result


def markdown_summary(exp1: dict[str, Any], exp2: dict[str, Any]) -> str:
    lines = [
        "# Effective-flow N=10 Diagnostic Ablations",
        "",
        "These are paired diagnostics only; no statistical-significance claim is made.",
        "Requested-flow overlap metrics are legacy diagnostics only.",
        "",
        "## Experiment 1: Warp sampling",
        "",
        "| Mode | N | Detect rate | Attack success | Mean score | Median score | PSNR mean | PSNR median | SSIM mean | SSIM median | Unique effective shifts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, value in exp1.items():
        lines.append(
            f"| {mode} | {value['n']} | {value['detect_rate']:.4f} | {value['attack_success']:.4f} | "
            f"{value['score']['mean']:.6f} | {value['score']['median']:.6f} | "
            f"{value['corrected_overlap_psnr']['mean']:.4f} | {value['corrected_overlap_psnr']['median']:.4f} | "
            f"{value['corrected_overlap_ssim']['mean']:.6f} | {value['corrected_overlap_ssim']['median']:.6f} | "
            f"{value['unique_effective_displacement_count']} |"
        )
    lines += [
        "",
        "## Experiment 2: Shift and color transfer",
        "",
        "| Mode | N | Detect rate | Attack success | Mean score | Median score | PSNR mean | PSNR median | SSIM mean | SSIM median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, value in exp2.items():
        lines.append(
            f"| {mode} | {value['n']} | {value['detect_rate']:.4f} | {value['attack_success']:.4f} | "
            f"{value['score']['mean']:.6f} | {value['score']['median']:.6f} | "
            f"{value['corrected_overlap_psnr']['mean']:.4f} | {value['corrected_overlap_psnr']['median']:.4f} | "
            f"{value['corrected_overlap_ssim']['mean']:.6f} | {value['corrected_overlap_ssim']['median']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=COUNT, choices=[COUNT])
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    (output / "experiment1" / "images").mkdir(parents=True)
    (output / "experiment2" / "images").mkdir(parents=True)

    parity = json.loads((source / "scores" / "official_detector_parity.json").read_text())
    if not parity.get("passed"):
        raise RuntimeError("source official detector parity did not pass")
    cohort = read_jsonl(source / "paired_latent_manifest.jsonl")[:COUNT]
    scores = {int(row["run_id"]): row for row in read_jsonl(source / "score_records.jsonl")[:COUNT]}
    plan_payload = json.loads((source / "shift_plan.json").read_text())
    plan = {int(row["run_id"]): row for row in plan_payload["samples"][:COUNT]}
    summary = json.loads((source / "summary.json").read_text())
    threshold_l1 = float(summary["before"]["complex_l1_threshold_equivalent"])
    if len(cohort) != COUNT or len(scores) != COUNT or len(plan) != COUNT:
        raise ValueError("source cohort/score/shift plan is incomplete")
    write_json(output / "shift_plan.json", {"source": str((source / "shift_plan.json").resolve()), "samples": list(plan.values())})
    write_json(output / "configs" / "base_attack.json", BASE_ATTACK)
    write_json(output / "configs" / "experiment1.json", EXP1_MODES)
    write_json(output / "configs" / "experiment2.json", EXP2_MODES)
    (output / "commands.sh").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    pipeline = RavenPipeline(model_id=str(args.model_source.resolve()), device=args.device, dtype="float16")
    attack_runs: dict[tuple[str, int], dict[str, Any]] = {}
    exp1_rows: list[dict[str, Any]] = []
    exp2_rows: list[dict[str, Any]] = []

    for position, row in enumerate(cohort, start=1):
        run_id = int(row["run_id"])
        input_path = Path(row["watermarked_image_path"])
        if sha256_path(input_path) != row["watermarked_image_sha256"]:
            raise RuntimeError(f"input SHA mismatch run_id={run_id}")
        shift = plan[run_id]
        requested_dx = int(shift["flow_dx_image_px"])
        requested_dy = int(shift["flow_dy_image_px"])
        seed = int(shift["attack_seed"])
        for mode, mode_config in EXP1_MODES.items():
            config = {**BASE_ATTACK, **mode_config, "mode": mode, "requested_dx": requested_dx, "requested_dy": requested_dy}
            attack = run_one(
                pipeline, input_path, output / "experiment1" / "images" / mode / f"{run_id:06d}",
                mode_config, requested_dx, requested_dy, seed,
            )
            attack_runs[(mode, run_id)] = attack
            debug = attack.pop("debug")
            post_quality = quality(input_path, Path(attack["output_path"]), debug)
            pre_quality = quality(input_path, Path(attack["pre_color_path"]), debug)
            exp1_rows.append({
                "experiment": 1,
                "mode": mode,
                "run_id": run_id,
                "input_path": str(input_path.resolve()),
                "input_sha256": row["watermarked_image_sha256"],
                "seed": seed,
                "prompt": "",
                "requested_dx_image_px": post_quality["requested_dx_image_px"],
                "requested_dy_image_px": post_quality["requested_dy_image_px"],
                "effective_dx_image_px": post_quality["effective_dx_image_px"],
                "effective_dy_image_px": post_quality["effective_dy_image_px"],
                "effective_dx_latent_cells": post_quality["effective_dx_latent_cells"],
                "effective_dy_latent_cells": post_quality["effective_dy_latent_cells"],
                "corrected_overlap_protocol": post_quality["corrected_overlap_protocol"],
                "config_hash": config_hash(config),
                "exact_ddim_timestep": debug["exact_timestep"],
                "output_path": attack["output_path"],
                "output_sha256": attack["output_sha256"],
                "pre_color_path": attack["pre_color_path"],
                "pre_color_sha256": attack["pre_color_sha256"],
                "runtime_seconds": attack["runtime_seconds"],
                "peak_gpu_memory_bytes": attack["peak_gpu_memory_bytes"],
                "peak_cpu_rss_kib": attack["peak_cpu_rss_kib"],
                "before_watermarked_l1": float(scores[run_id]["watermarked_l1"]),
                "clipping_diagnostics": debug.get("clipping_diagnostics"),
                "color_transfer_diagnostics": debug.get("color_transfer_diagnostics"),
                **post_quality,
                "pre_color_corrected_overlap_psnr": pre_quality["corrected_overlap_psnr"],
                "pre_color_corrected_overlap_ssim": pre_quality["corrected_overlap_ssim"],
            })
        print(f"[attack experiment1 {position}/{COUNT}] run_id={run_id}", flush=True)

    for position, row in enumerate(cohort, start=1):
        run_id = int(row["run_id"])
        input_path = Path(row["watermarked_image_path"])
        shift = plan[run_id]
        for mode in ("DDIM_shift_no_color", "DDIM_no_shift_no_color"):
            mode_config = EXP2_MODES[mode]
            dx = int(shift["flow_dx_image_px"]) if mode_config["use_shift"] else 0
            dy = int(shift["flow_dy_image_px"]) if mode_config["use_shift"] else 0
            seed = int(shift["attack_seed"])
            attack = run_one(
                pipeline, input_path, output / "experiment2" / "images" / mode / f"{run_id:06d}",
                mode_config, dx, dy, seed,
            )
            attack_runs[(mode, run_id)] = attack
        print(f"[attack experiment2 {position}/{COUNT}] run_id={run_id}", flush=True)

    del pipeline
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    detector = load_official_pipeline(args.model_source.resolve(), args.device)
    target = torch.load(source / "configs" / "watermark_target.pt", map_location=args.device, weights_only=True)
    mask = torch.load(source / "configs" / "watermark_mask.pt", map_location=args.device, weights_only=True)
    score_cache: dict[str, float] = {}

    def official_score(path_value: str) -> float:
        path = Path(path_value)
        digest = sha256_path(path)
        if digest not in score_cache:
            value, _ = score_image(detector, Image.open(path).convert("RGB"), mask, target, steps=50)
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite detector score: {path}")
            score_cache[digest] = float(value)
        return score_cache[digest]

    for index, row in enumerate(exp1_rows, start=1):
        row["detector_score"] = official_score(row["output_path"])
        row["detect_after"] = row["detector_score"] <= threshold_l1
        row["attack_success"] = not row["detect_after"]
        print(f"[score experiment1 {index}/{len(exp1_rows)}]", flush=True)

    for row in cohort:
        run_id = int(row["run_id"])
        input_path = Path(row["watermarked_image_path"])
        for mode in EXP2_MODES:
            if mode == "DDIM_shift_with_color":
                attack = attack_runs[("NFPA_exact_nearest", run_id)]
                debug = json.loads(Path(attack["debug_path"]).read_text())
                output_path = Path(attack["output_path"])
                pre_color_path = Path(attack["pre_color_path"])
                runtime = attack["runtime_seconds"]
                mode_config = EXP2_MODES[mode]
                dx = int(plan[run_id]["flow_dx_image_px"])
                dy = int(plan[run_id]["flow_dy_image_px"])
            else:
                attack = attack_runs[(mode, run_id)]
                debug = json.loads(Path(attack["debug_path"]).read_text())
                output_path = Path(attack["output_path"])
                pre_color_path = Path(attack["pre_color_path"])
                runtime = attack["runtime_seconds"]
                mode_config = EXP2_MODES[mode]
                dx = int(plan[run_id]["flow_dx_image_px"]) if mode_config["use_shift"] else 0
                dy = int(plan[run_id]["flow_dy_image_px"]) if mode_config["use_shift"] else 0
            final_quality = quality(input_path, output_path, debug)
            pre_quality = quality(input_path, pre_color_path, debug)
            score = official_score(str(output_path))
            config = {**BASE_ATTACK, **mode_config, "mode": mode, "requested_dx": dx, "requested_dy": dy}
            exp2_rows.append({
                "experiment": 2,
                "mode": mode,
                "run_id": run_id,
                "input_path": str(input_path.resolve()),
                "input_sha256": row["watermarked_image_sha256"],
                "seed": int(plan[run_id]["attack_seed"]),
                "prompt": "",
                "config_hash": config_hash(config),
                "exact_ddim_timestep": debug["exact_timestep"],
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_path(output_path),
                "pre_color_path": str(pre_color_path.resolve()),
                "pre_color_sha256": sha256_path(pre_color_path),
                "runtime_seconds": runtime,
                "peak_gpu_memory_bytes": attack["peak_gpu_memory_bytes"],
                "detector_score": score,
                "before_watermarked_l1": float(scores[run_id]["watermarked_l1"]),
                "detect_after": score <= threshold_l1,
                "attack_success": score > threshold_l1,
                "pre_color_corrected_overlap_psnr": pre_quality["corrected_overlap_psnr"],
                "pre_color_corrected_overlap_ssim": pre_quality["corrected_overlap_ssim"],
                "post_color_corrected_overlap_psnr": (final_quality["corrected_overlap_psnr"] if mode_config["color_transfer"] else None),
                "post_color_corrected_overlap_ssim": (final_quality["corrected_overlap_ssim"] if mode_config["color_transfer"] else None),
                "clipping_diagnostics": debug.get("clipping_diagnostics"),
                "color_transfer_diagnostics": debug.get("color_transfer_diagnostics"),
                **final_quality,
            })
        print(f"[score experiment2 run_id={run_id}]", flush=True)

    del detector, target, mask
    gc.collect()
    torch.cuda.empty_cache()

    exp1_aggregate = aggregate_modes(exp1_rows, list(EXP1_MODES))
    exp2_aggregate = aggregate_modes(exp2_rows, list(EXP2_MODES))
    paired_results = {
        "experiment1": [
            paired(exp1_rows, "NFPA_exact_nearest", "latent_grid_nearest"),
            paired(exp1_rows, "NFPA_exact_nearest", "latent_grid_bilinear"),
            paired(exp1_rows, "latent_grid_nearest", "latent_grid_bilinear"),
        ],
        "experiment2": [
            paired(exp2_rows, "DDIM_shift_with_color", "DDIM_shift_no_color"),
            paired(exp2_rows, "DDIM_shift_no_color", "DDIM_no_shift_no_color"),
            paired(exp2_rows, "DDIM_shift_with_color", "DDIM_no_shift_no_color"),
        ],
    }
    write_jsonl(output / "experiment1" / "per_sample_results.jsonl", exp1_rows)
    write_csv(output / "experiment1" / "per_sample_results.csv", exp1_rows)
    write_json(output / "experiment1" / "aggregate_results.json", exp1_aggregate)
    write_jsonl(output / "experiment2" / "per_sample_results.jsonl", exp2_rows)
    write_csv(output / "experiment2" / "per_sample_results.csv", exp2_rows)
    write_json(output / "experiment2" / "aggregate_results.json", exp2_aggregate)
    write_json(output / "paired_comparisons.json", paired_results)
    write_json(output / "aggregate_results.json", {
        "n": COUNT,
        "diagnostic_only": True,
        "fixed_before_complex_l1_threshold": threshold_l1,
        "threshold_source": str((source / "summary.json").resolve()),
        "experiment1": exp1_aggregate,
        "experiment2": exp2_aggregate,
    })
    (output / "aggregate_results.md").write_text(
        markdown_summary(exp1_aggregate, exp2_aggregate), encoding="utf-8"
    )
    write_json(output / "provenance.json", {
        "source_root": str(source),
        "source_shift_plan_sha256": sha256_path(source / "shift_plan.json"),
        "source_cohort_manifest_sha256": sha256_path(source / "paired_latent_manifest.jsonl"),
        "source_score_records_sha256": sha256_path(source / "score_records.jsonl"),
        "fixed_before_complex_l1_threshold": threshold_l1,
        "threshold_recalibrated": False,
        "count": COUNT,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "score_cache_unique_images": len(score_cache),
    })
    print(json.dumps({"output_dir": str(output), "experiment1": exp1_aggregate, "experiment2": exp2_aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
