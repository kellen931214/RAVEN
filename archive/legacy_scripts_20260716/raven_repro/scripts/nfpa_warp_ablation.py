#!/usr/bin/env python
"""Paired NFPA-compatible latent-warp ablation for RAVEN Tree-Ring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.warp import coords_grid, create_nfpa_translation_flow, nfpa_warp_single_latent
from scripts import diagonal_shift_ablation as base

MODEL_ID = base.MODEL_ID
MODEL_REVISION = base.MODEL_REVISION
THRESHOLD = base.THRESHOLD
VAE_SCALE_FACTOR = base.VAE_SCALE_FACTOR
COHORT_SIZE = 10
PLAN_SEED = base.PLAN_SEED
BOOTSTRAP_SEED = base.BOOTSTRAP_SEED
EMPTY_PROMPT_SHA256 = hashlib.sha256(b"").hexdigest()
DIRECTIONS = base.DIRECTIONS
ALL_MODES = (
    "nfpa_independent",
    "nfpa_sign_bound",
    "nfpa_strict_diagonal",
    "integer_zero_pad",
    "direct_latent",
    "no_shift",
)
PREFLIGHT_MODES = ALL_MODES

MODE_DEFINITIONS = {
    "nfpa_independent": {
        "unit_interpretation": "NFPA image-coordinate flow",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "independent x/y UniformInteger[24,32] image pixels",
        "warp_mode": "nfpa_exact",
        "coordinate_interpolation": "bilinear",
        "interpolation_mode": "nearest",
        "padding_mode": "reflection",
        "rounding_method": "none",
    },
    "nfpa_sign_bound": {
        "unit_interpretation": "NFPA image-coordinate flow",
        "sign_rule": "common sign",
        "magnitude_rule": "independent x/y UniformInteger[24,32] image pixels",
        "warp_mode": "nfpa_exact",
        "coordinate_interpolation": "bilinear",
        "interpolation_mode": "nearest",
        "padding_mode": "reflection",
        "rounding_method": "none",
    },
    "nfpa_strict_diagonal": {
        "unit_interpretation": "NFPA image-coordinate flow",
        "sign_rule": "common sign",
        "magnitude_rule": "shared scalar UniformInteger[24,32] image pixels; dx=dy",
        "warp_mode": "nfpa_exact",
        "coordinate_interpolation": "bilinear",
        "interpolation_mode": "nearest",
        "padding_mode": "reflection",
        "rounding_method": "none",
    },
    "integer_zero_pad": {
        "unit_interpretation": "image pixels rounded to latent cells",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "paired image magnitudes mapped to 3/4 latent cells",
        "warp_mode": "integer",
        "coordinate_interpolation": "none",
        "interpolation_mode": "none",
        "padding_mode": "zeros",
        "rounding_method": "half_up floor(image_pixels/8 + 0.5)",
    },
    "direct_latent": {
        "unit_interpretation": "direct_latent_ambiguity_ablation",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "independent x/y UniformInteger[24,32] latent cells",
        "warp_mode": "integer",
        "coordinate_interpolation": "none",
        "interpolation_mode": "none",
        "padding_mode": "zeros",
        "rounding_method": "integer sampling",
    },
    "no_shift": {
        "unit_interpretation": "no shift control",
        "sign_rule": "none",
        "magnitude_rule": "dx=dy=0",
        "warp_mode": "integer",
        "coordinate_interpolation": "none",
        "interpolation_mode": "none",
        "padding_mode": "zeros",
        "rounding_method": "none",
    },
}


def half_up_latent(image_magnitude: int) -> int:
    return math.floor(image_magnitude / VAE_SCALE_FACTOR + 0.5)


def direction_label(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        return "(0,0)"
    return f"({'+' if dx > 0 else '-'},{'+' if dy > 0 else '-'})"


def observed_nfpa_displacement(flow_x: float, flow_y: float) -> tuple[int, int]:
    import torch

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    flow = create_nfpa_translation_flow(
        flow_x, flow_y, device=torch.device("cpu"), dtype=torch.float32
    )
    warped = nfpa_warp_single_latent(latent, flow)
    observed_x, observed_y = impulse_position(warped)
    return observed_x - 32, observed_y - 32


def mode_shift(mode: str, x_mag: int, y_mag: int, x_sign: int, y_sign: int, common_sign: int) -> dict:
    definition = MODE_DEFINITIONS[mode]
    if mode == "nfpa_independent":
        flow_x, flow_y = x_sign * x_mag, y_sign * y_mag
        latent_x, latent_y = flow_x / VAE_SCALE_FACTOR, flow_y / VAE_SCALE_FACTOR
    elif mode == "nfpa_sign_bound":
        flow_x, flow_y = common_sign * x_mag, common_sign * y_mag
        latent_x, latent_y = flow_x / VAE_SCALE_FACTOR, flow_y / VAE_SCALE_FACTOR
    elif mode == "nfpa_strict_diagonal":
        flow_x = flow_y = common_sign * x_mag
        latent_x = latent_y = flow_x / VAE_SCALE_FACTOR
    elif mode == "integer_zero_pad":
        latent_x = x_sign * half_up_latent(x_mag)
        latent_y = y_sign * half_up_latent(y_mag)
        flow_x, flow_y = latent_x * VAE_SCALE_FACTOR, latent_y * VAE_SCALE_FACTOR
    elif mode == "direct_latent":
        latent_x, latent_y = x_sign * x_mag, y_sign * y_mag
        flow_x, flow_y = latent_x * VAE_SCALE_FACTOR, latent_y * VAE_SCALE_FACTOR
    elif mode == "no_shift":
        flow_x = flow_y = latent_x = latent_y = 0
    else:
        raise ValueError(mode)

    inverse_sampling = definition["warp_mode"] == "nfpa_exact"
    visual_x = -flow_x if inverse_sampling else flow_x
    visual_y = -flow_y if inverse_sampling else flow_y
    expected_latent_x = -flow_x / VAE_SCALE_FACTOR if inverse_sampling else latent_x
    expected_latent_y = -flow_y / VAE_SCALE_FACTOR if inverse_sampling else latent_y
    if inverse_sampling:
        observed_latent_x, observed_latent_y = observed_nfpa_displacement(flow_x, flow_y)
    else:
        observed_latent_x, observed_latent_y = int(latent_x), int(latent_y)
    before_norm = {
        "x_min": float(flow_x), "x_max": float(511 + flow_x),
        "y_min": float(flow_y), "y_max": float(511 + flow_y),
    } if inverse_sampling else None
    normalized_minmax = {
        key: float(2.0 * value / 512.0 - 1.0)
        for key, value in before_norm.items()
    } if inverse_sampling else None
    return {
        **definition,
        "dx_sign": 0 if flow_x == 0 else (1 if flow_x > 0 else -1),
        "dy_sign": 0 if flow_y == 0 else (1 if flow_y > 0 else -1),
        "flow_direction": direction_label(flow_x, flow_y),
        "visual_direction": direction_label(visual_x, visual_y),
        "direction": direction_label(visual_x, visual_y),
        "flow_dx_image_px": float(flow_x),
        "flow_dy_image_px": float(flow_y),
        "visual_shift_dx_image_px": float(visual_x),
        "visual_shift_dy_image_px": float(visual_y),
        "dx_image_pixels": float(flow_x),
        "dy_image_pixels": float(flow_y),
        "dx_latent_cells": float(latent_x),
        "dy_latent_cells": float(latent_y),
        "equivalent_image_dx": float(visual_x),
        "equivalent_image_dy": float(visual_y),
        "shift_space": "image_pixels" if inverse_sampling else "latent_pixels",
        "align_corners": False if inverse_sampling else None,
        "normalized_coordinate_formula": (
            "x_norm = 2*x_pixel/W - 1; y_norm = 2*y_pixel/H - 1"
            if inverse_sampling else None
        ),
        "normalized_flow_dx": float(2.0 * flow_x / 512.0) if inverse_sampling else None,
        "normalized_flow_dy": float(2.0 * flow_y / 512.0) if inverse_sampling else None,
        "expected_visual_latent_dx": float(expected_latent_x),
        "expected_visual_latent_dy": float(expected_latent_y),
        "observed_visual_latent_dx": int(observed_latent_x),
        "observed_visual_latent_dy": int(observed_latent_y),
        "coordinate_grid_minmax_before_norm": before_norm,
        "coordinate_grid_minmax_normalized": normalized_minmax,
        "coordinates_exceed_unit_range": (
            any(value < -1.0 or value > 1.0 for value in normalized_minmax.values())
            if inverse_sampling else None
        ),
        "coordinate_grid_input_shape": [1, 2, 512, 512] if inverse_sampling else None,
        "coordinate_grid_output_shape": [1, 2, 64, 64] if inverse_sampling else None,
        "nfpa_source_omitted_align_corners": True if inverse_sampling else None,
        "effective_align_corners": False if inverse_sampling else None,
        "grid_sample_inverse_sampling": inverse_sampling,
        "half_pixel_offset": False if inverse_sampling else None,
        "circular": False,
    }


def source_audit_text() -> str:
    return """# NFPA Source Audit

Only the coordinate-flow and sampling convention below is ported into RAVEN. NFPA checkpoint, 10 steps, `xy=40`, timestep-0 noising, and detector calibration are not imported.

| NFPA source | Lines | Observed behavior | Local implementation |
| --- | ---: | --- | --- |
| `NFPA/utils.py::CrossFrameAttnProcessor` | 61-123 | Self-attention replaces K/V with first frame; cross-attention unchanged | Existing `raven/attention.py`; unchanged in this ablation |
| `NFPA/utils.py::coords_grid` | 126-132 | RAFT-style `(x,y)` image pixel grid | `raven/warp.py::coords_grid` |
| `NFPA/utils.py::warp_single_latent` | 135-159 | `coords+flow`, divide by W/H, `*2-1`, bilinear coordinate resize, nearest/reflection grid sample | `raven/warp.py::nfpa_warp_single_latent` |
| `NFPA/utils.py::create_motion_field` | 162-184 | Constant 512x512 flow per frame | `raven/warp.py::create_nfpa_translation_flow` |
| `NFPA/utils.py::create_motion_field_and_warp_latents_xy` | 228-241 | One flow is used to warp each selected latent | `RavenPipeline.run` applies one explicit flow after DDIM inversion |
| `NFPA/utils.py::MyStableDiffusionPipeline` | 314 onward | Installs cross-frame processor; notebook supplies warped latents | RAVEN attention path retained; only warp changes |
| `NFPA/nfp_main.ipynb::inversion_latents_fun` | notebook cell 0 | Empty prompt and DDIM inverse scheduler | RAVEN true DDIM inversion with empty prompt |
| `NFPA/nfp_main.ipynb::nfp_attack` | notebook cell 0 | Empty reconstruction prompt; NFPA-specific 10 steps and xy=40 | Empty prompt retained; RAVEN remains 50 steps, strength 0.15, guidance 2.5 |
"""


def configure_base() -> None:
    base.__file__ = __file__
    base.COHORT_SIZE = COHORT_SIZE
    base.PRELIGHT_MODES = PREFLIGHT_MODES
    base.ALL_MODES = ALL_MODES
    base.MODE_DEFINITIONS = MODE_DEFINITIONS
    base.mode_shift = mode_shift


def command_plan(args) -> int:
    result = base.command_plan(args)
    plan_path = args.output_dir / "shift_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["protocol"] = "raven_nfpa_warp_ablation_v1"
    plan["nfpa_inverse_sampling_sign_rule"] = "visual_shift = -flow for nfpa_exact"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "nfpa_source_audit.md").write_text(source_audit_text())
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
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"{env} {python} -u {script} validate-warp --output-dir {root}",
        f"script -q -e -c \"{env} {python} -u {script} attack --output-dir {root} --phase preflight --device cuda --dtype float16\" {root}/logs/preflight_attack.log",
        f"script -q -e -c \"{env} {python} -u {script} score --output-dir {root} --phase preflight --device cuda\" {root}/logs/preflight_score.log",
        f"script -q -e -c \"{env} {python} -u {script} attack --output-dir {root} --phase full --device cuda --dtype float16\" {root}/logs/full_attack.log",
        f"script -q -e -c \"{env} {python} -u {script} score --output-dir {root} --phase full --device cuda\" {root}/logs/full_score.log",
        f"{env} {python} -u {script} aggregate --output-dir {root} --bootstrap-samples 10000 --bootstrap-seed {BOOTSTRAP_SEED}",
    ]
    commands_path = args.output_dir / "commands.sh"
    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)
    return result


def impulse_position(tensor):
    import torch

    index = int(torch.argmax(tensor[0, 0]).item())
    width = tensor.shape[-1]
    return index % width, index // width


def command_validate_warp(args) -> int:
    import torch

    tests = []
    grid = coords_grid(2, 3, 4, torch.device("cpu"), dtype=torch.float32)
    tests.append({
        "name": "coordinate_grid_shape_and_order",
        "passed": list(grid.shape) == [2, 2, 3, 4]
        and float(grid[0, 0, 0, 3]) == 3.0
        and float(grid[0, 1, 2, 0]) == 2.0,
        "shape": list(grid.shape),
    })

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    normalization_flow = create_nfpa_translation_flow(24, -32, device=latent.device, dtype=latent.dtype)
    _, normalization_meta = nfpa_warp_single_latent(latent, normalization_flow, return_metadata=True)
    tests.append({
        "name": "normalization_formula",
        "passed": abs(normalization_meta["normalized_flow_dx"] - 0.09375) < 1e-7
        and abs(normalization_meta["normalized_flow_dy"] + 0.125) < 1e-7,
        "metadata": normalization_meta,
    })

    directions = []
    for dx, dy in ((24, 24), (24, -24), (-24, 24), (-24, -24)):
        flow = create_nfpa_translation_flow(dx, dy, device=latent.device, dtype=latent.dtype)
        shifted = nfpa_warp_single_latent(latent, flow)
        observed_x, observed_y = impulse_position(shifted)
        expected_x, expected_y = 32 - dx // 8, 32 - dy // 8
        item = {
            "flow_direction": direction_label(dx, dy),
            "visual_direction": direction_label(-dx, -dy),
            "observed": [observed_x, observed_y],
            "expected": [expected_x, expected_y],
            "passed": (observed_x, observed_y) == (expected_x, expected_y),
        }
        directions.append(item)
    tests.append({"name": "four_direction_inverse_sampling", "passed": all(x["passed"] for x in directions), "cases": directions})

    ones = torch.ones(1, 1, 64, 64)
    reflected = nfpa_warp_single_latent(
        ones, create_nfpa_translation_flow(320, -320, device=ones.device, dtype=ones.dtype)
    )
    tests.append({
        "name": "reflection_padding",
        "passed": bool(torch.equal(reflected, ones)),
        "minimum": float(reflected.min()),
        "maximum": float(reflected.max()),
    })

    values = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    nearest = nfpa_warp_single_latent(
        values, create_nfpa_translation_flow(28, -29, device=values.device, dtype=values.dtype)
    )
    tests.append({
        "name": "nearest_sampling",
        "passed": bool(torch.allclose(nearest, nearest.round())),
        "maximum_fractional_part": float((nearest - nearest.round()).abs().max()),
    })

    displacement = []
    for dx in (24, 28, 32):
        shifted = nfpa_warp_single_latent(
            latent, create_nfpa_translation_flow(dx, 0, device=latent.device, dtype=latent.dtype)
        )
        observed_x, observed_y = impulse_position(shifted)
        displacement.append({
            "flow_dx_image_px": dx,
            "expected_continuous_visual_latent_dx": -dx / 8.0,
            "observed_visual_latent_dx": observed_x - 32,
            "observed_y": observed_y,
            "quantization": "nearest",
        })
    tests.append({
        "name": "effective_displacement",
        "passed": [x["observed_visual_latent_dx"] for x in displacement] == [-3, -3, -4],
        "cases": displacement,
    })

    finite_output = nfpa_warp_single_latent(
        torch.randn(1, 4, 64, 64),
        create_nfpa_translation_flow(31, -27, device=torch.device("cpu"), dtype=torch.float32),
    )
    tests.append({
        "name": "no_nan_inf",
        "passed": bool(torch.isfinite(finite_output).all()),
        "nan_count": int(torch.isnan(finite_output).sum()),
        "inf_count": int(torch.isinf(finite_output).sum()),
    })

    if torch.cuda.is_available():
        torch.manual_seed(1)
        cpu_latent = torch.randn(1, 4, 64, 64)
        cpu_flow = create_nfpa_translation_flow(27, -31, device=torch.device("cpu"), dtype=torch.float32)
        cpu_output = nfpa_warp_single_latent(cpu_latent, cpu_flow)
        gpu_output = nfpa_warp_single_latent(cpu_latent.cuda(), cpu_flow.cuda()).cpu()
        max_difference = float((cpu_output - gpu_output).abs().max())
        tests.append({"name": "cpu_gpu_consistency", "passed": max_difference <= 1e-6, "max_abs_difference": max_difference})
    else:
        tests.append({"name": "cpu_gpu_consistency", "passed": False, "reason": "CUDA unavailable"})

    payload = {
        "passed": all(test["passed"] for test in tests),
        "nfpa_source_omitted_align_corners": True,
        "effective_align_corners": False,
        "tests": tests,
    }
    base.write_json_exclusive(args.output_dir / "unit_test_results.json", payload)
    base.write_json_exclusive(args.output_dir / "effective_displacement.json", {
        "coordinate_grid": "512x512 image pixels",
        "latent_grid": "64x64",
        "sampling": "nearest",
        "padding": "reflection",
        "measurements": displacement,
    })
    base.write_json_exclusive(args.output_dir / "preflight" / "warp_direction_tests.json", {
        "passed": payload["passed"],
        "positive_flow_semantics": "positive flow samples farther right/down; visual content moves left/up",
        "directions": directions,
    })
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("NFPA warp unit validation failed")
    return 0


def summarize_rows(rows: list[dict]) -> dict:
    score = base.finite_stats(row["attacked_score_after"] for row in rows)
    delta = base.finite_stats(row["score_delta"] for row in rows)
    psnr = base.finite_stats(row["psnr"] for row in rows)
    ssim = base.finite_stats(row["ssim"] for row in rows)
    overlap = base.finite_stats(row["valid_overlap_area_ratio"] for row in rows)
    low_ids = [row["sample_id"] for row in rows if not row["detect_after"]]
    return {
        "N": len(rows),
        "direction_counts": dict(Counter(row["visual_direction"] for row in rows)),
        "attacked_detect_count": sum(row["detect_after"] for row in rows),
        "attacked_detection_rate": float(statistics.fmean(row["detect_after"] for row in rows)),
        "attack_success_rate": float(statistics.fmean(not row["detect_after"] for row in rows)),
        "attacked_score": score,
        "score_delta": delta,
        "psnr": psnr,
        "ssim": ssim,
        "valid_overlap": overlap,
        "below_threshold_count": len(low_ids),
        "below_threshold_sample_ids": low_ids,
        "nan_count": sum(row["detector_nan"] for row in rows),
        "inf_count": sum(row["detector_inf"] for row in rows),
        "p_value_underflow_count": sum(row["p_value_underflow"] for row in rows),
    }


def command_aggregate(args) -> int:
    result_path, _ = base.result_paths(args.output_dir, "full")
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    expected = len(ALL_MODES) * COHORT_SIZE
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} scored rows, found {len(rows)}")
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in ALL_MODES}
    if any(len(items) != COHORT_SIZE for items in grouped.values()):
        raise ValueError("Mode sample count mismatch")

    aggregate = {
        "protocol": "10-sample paired diagnostic; not formal TPR@1%FPR",
        "calibrated_threshold_held_fixed": THRESHOLD,
        "score_definition": "-log10(p), higher means more watermark",
        "modes": {mode: summarize_rows(items) for mode, items in grouped.items()},
    }
    base.write_json_exclusive(args.output_dir / "aggregate_results.json", aggregate)

    comparisons_spec = (
        ("nfpa_independent_vs_integer_zero_pad", "nfpa_independent", "integer_zero_pad"),
        ("nfpa_independent_vs_nfpa_sign_bound", "nfpa_independent", "nfpa_sign_bound"),
        ("nfpa_independent_vs_nfpa_strict_diagonal", "nfpa_independent", "nfpa_strict_diagonal"),
        ("direct_latent_vs_nfpa_independent", "direct_latent", "nfpa_independent"),
        *((f"{mode}_vs_no_shift", mode, "no_shift") for mode in ALL_MODES if mode != "no_shift"),
    )
    comparisons = {
        name: {
            "candidate": candidate,
            "reference": reference,
            **base.bootstrap_comparison(
                grouped[candidate], grouped[reference], args.bootstrap_samples,
                args.bootstrap_seed + index,
            ),
        }
        for index, (name, candidate, reference) in enumerate(comparisons_spec)
    }
    base.write_json_exclusive(args.output_dir / "paired_comparisons.json", {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed_base": args.bootstrap_seed,
        "interpretation": "negative attacked-score difference favors candidate",
        "comparisons": comparisons,
    })

    direction = {}
    for mode in ("nfpa_independent", "integer_zero_pad", "direct_latent"):
        direction[mode] = {}
        for label in ("(+,+)", "(+,-)", "(-,+)", "(-,-)"):
            selected = [row for row in grouped[mode] if row["visual_direction"] == label]
            direction[mode][label] = summarize_rows(selected) if selected else {"N": 0}
    base.write_json_exclusive(args.output_dir / "direction_analysis.json", direction)

    lines = [
        "# RAVEN NFPA Warp Ablation",
        "",
        "10-sample paired diagnostic only. The calibrated threshold is held fixed and is not recalibrated.",
        "",
        "| Warp mode | Direction rule | N | Detect rate | Mean score after | Mean delta | PSNR | SSIM | Valid overlap |",
        "| --- | --- | -: | -: | -: | -: | -: | -: | -: |",
    ]
    for mode in ALL_MODES:
        definition = MODE_DEFINITIONS[mode]
        summary = aggregate["modes"][mode]
        lines.append(
            f"| {mode} | {definition['sign_rule']} | {summary['N']} | "
            f"{summary['attacked_detection_rate']:.4f} | {summary['attacked_score']['mean']:.6f} | "
            f"{summary['score_delta']['mean']:.6f} | {summary['psnr']['mean']:.3f} | "
            f"{summary['ssim']['mean']:.4f} | {summary['valid_overlap']['mean']:.4f} |"
        )
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "aggregate": str(args.output_dir / "aggregate_results.json")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create deterministic 10-sample plan and output structure")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--baseline-records", type=Path, required=True)
    plan.add_argument("--calibrated-metrics", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--count", type=int, default=COHORT_SIZE)
    plan.add_argument("--plan-seed", type=int, default=PLAN_SEED)

    validate = subparsers.add_parser("validate-warp", help="Run NFPA coordinate, sampling, and displacement tests")
    validate.add_argument("--output-dir", type=Path, required=True)

    attack = subparsers.add_parser("attack", help="Generate 2-sample preflight or 10-sample attacks")
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--phase", choices=["preflight", "full"], required=True)
    attack.add_argument("--device", default="cuda")
    attack.add_argument("--dtype", choices=["float16"], default="float16")
    attack.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score", help="Score attacks with the fixed Tree-Ring detector")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--phase", choices=["preflight", "full"], required=True)
    score.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    score.add_argument("--device", choices=["cuda"], default="cuda")
    score.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate the completed 10-sample paired cohort")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> int:
    configure_base()
    args = build_parser().parse_args()
    if args.command == "plan":
        return command_plan(args)
    if args.command == "validate-warp":
        return command_validate_warp(args)
    if args.command == "attack":
        return base.command_attack(args)
    if args.command == "score":
        return base.command_score(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
