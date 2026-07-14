#!/usr/bin/env python
"""Paired RAVEN diagonal-shift interpretation ablation for Tree-Ring."""

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
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image
from raven.warp import translate_latent


MODEL_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
THRESHOLD = 1.6372738343020807
VAE_SCALE_FACTOR = 8
COHORT_SIZE = 30
PLAN_SEED = 20260714
BOOTSTRAP_SEED = 20260714
PRELIGHT_MODES = ("A", "B", "C", "D", "E", "G", "I")
ALL_MODES = tuple("ABCDEFGHI")
EMPTY_PROMPT_SHA256 = hashlib.sha256(b"").hexdigest()
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

MODE_DEFINITIONS = {
    "A": {
        "unit_interpretation": "image_pixels",
        "sign_rule": "common sign",
        "magnitude_rule": "independent x/y UniformInteger[24,32]",
        "warp_mode": "grid_sample",
        "interpolation_mode": "bilinear",
        "rounding_method": "none",
    },
    "B": {
        "unit_interpretation": "image_pixels",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "independent x/y UniformInteger[24,32]",
        "warp_mode": "grid_sample",
        "interpolation_mode": "bilinear",
        "rounding_method": "none",
    },
    "C": {
        "unit_interpretation": "image_pixels",
        "sign_rule": "common sign",
        "magnitude_rule": "shared scalar UniformInteger[24,32], dx=dy",
        "warp_mode": "grid_sample",
        "interpolation_mode": "bilinear",
        "rounding_method": "none",
    },
    "D": {
        "unit_interpretation": "direct_latent_cells",
        "sign_rule": "common sign",
        "magnitude_rule": "independent x/y UniformInteger[24,32]",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "integer sampling",
    },
    "E": {
        "unit_interpretation": "direct_latent_cells",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "independent x/y UniformInteger[24,32]",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "integer sampling",
    },
    "F": {
        "unit_interpretation": "direct_latent_cells",
        "sign_rule": "common sign",
        "magnitude_rule": "shared scalar UniformInteger[24,32], dx=dy",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "integer sampling",
    },
    "G": {
        "unit_interpretation": "integer_latent_3_or_4_cells",
        "sign_rule": "independent x/y signs, deterministic stratified",
        "magnitude_rule": "independent x/y image magnitudes mapped to 3/4 cells",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "half_up floor(image_pixels/8 + 0.5)",
    },
    "H": {
        "unit_interpretation": "integer_latent_3_or_4_cells",
        "sign_rule": "common sign",
        "magnitude_rule": "independent x/y image magnitudes mapped to 3/4 cells",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "half_up floor(image_pixels/8 + 0.5)",
    },
    "I": {
        "unit_interpretation": "no_shift",
        "sign_rule": "none",
        "magnitude_rule": "dx=dy=0",
        "warp_mode": "integer",
        "interpolation_mode": "none",
        "rounding_method": "none",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create deterministic 30-sample plan and output structure")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--baseline-records", type=Path, required=True)
    plan.add_argument("--calibrated-metrics", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--count", type=int, default=COHORT_SIZE)
    plan.add_argument("--plan-seed", type=int, default=PLAN_SEED)

    warp = subparsers.add_parser("validate-warp", help="Run four-direction bright-point warp tests")
    warp.add_argument("--output-dir", type=Path, required=True)

    attack = subparsers.add_parser("attack", help="Generate preflight or full ablation attacks")
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--phase", choices=["preflight", "full"], required=True)
    attack.add_argument("--device", default="cuda")
    attack.add_argument("--dtype", choices=["float16"], default="float16")
    attack.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score", help="Score preflight or full attacks with Tree-Ring")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--phase", choices=["preflight", "full"], required=True)
    score.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    score.add_argument("--device", choices=["cuda"], default="cuda")
    score.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate the completed 30-sample cohort")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip()


def half_up_latent(image_magnitude: int) -> int:
    return math.floor(image_magnitude / VAE_SCALE_FACTOR + 0.5)


def direction_label(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        return "(0,0)"
    return f"({'+' if dx > 0 else '-'},{'+' if dy > 0 else '-'})"


def mode_shift(mode: str, x_mag: int, y_mag: int, x_sign: int, y_sign: int, common_sign: int) -> dict:
    definition = MODE_DEFINITIONS[mode]
    if mode == "A":
        ix, iy = common_sign * x_mag, common_sign * y_mag
        lx, ly = ix / VAE_SCALE_FACTOR, iy / VAE_SCALE_FACTOR
    elif mode == "B":
        ix, iy = x_sign * x_mag, y_sign * y_mag
        lx, ly = ix / VAE_SCALE_FACTOR, iy / VAE_SCALE_FACTOR
    elif mode == "C":
        ix = iy = common_sign * x_mag
        lx = ly = ix / VAE_SCALE_FACTOR
    elif mode == "D":
        lx, ly = common_sign * x_mag, common_sign * y_mag
        ix, iy = lx * VAE_SCALE_FACTOR, ly * VAE_SCALE_FACTOR
    elif mode == "E":
        lx, ly = x_sign * x_mag, y_sign * y_mag
        ix, iy = lx * VAE_SCALE_FACTOR, ly * VAE_SCALE_FACTOR
    elif mode == "F":
        lx = ly = common_sign * x_mag
        ix = iy = lx * VAE_SCALE_FACTOR
    elif mode == "G":
        lx, ly = x_sign * half_up_latent(x_mag), y_sign * half_up_latent(y_mag)
        ix, iy = lx * VAE_SCALE_FACTOR, ly * VAE_SCALE_FACTOR
    elif mode == "H":
        lx, ly = common_sign * half_up_latent(x_mag), common_sign * half_up_latent(y_mag)
        ix, iy = lx * VAE_SCALE_FACTOR, ly * VAE_SCALE_FACTOR
    elif mode == "I":
        ix = iy = lx = ly = 0
    else:
        raise ValueError(mode)
    return {
        **definition,
        "dx_sign": 0 if lx == 0 else (1 if lx > 0 else -1),
        "dy_sign": 0 if ly == 0 else (1 if ly > 0 else -1),
        "direction": direction_label(lx, ly),
        "dx_image_pixels": float(ix),
        "dy_image_pixels": float(iy),
        "dx_latent_cells": float(lx),
        "dy_latent_cells": float(ly),
        "equivalent_image_dx": float(ix),
        "equivalent_image_dy": float(iy),
        "shift_space": "image_pixels" if mode in {"A", "B", "C"} else "latent_pixels",
        "padding_mode": "zeros",
        "align_corners": True if definition["warp_mode"] == "grid_sample" else None,
        "normalized_coordinate_formula": (
            "source_grid = output_grid - (2*latent_shift/(size-1)); align_corners=True"
            if definition["warp_mode"] == "grid_sample" else None
        ),
        "half_pixel_offset": False if definition["warp_mode"] == "grid_sample" else None,
        "circular": False,
    }


def command_plan(args) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    if args.count != COHORT_SIZE:
        raise ValueError(f"This protocol requires exactly {COHORT_SIZE} samples")
    for path in (args.manifest, args.baseline_records, args.calibrated_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    calibrated = json.loads(args.calibrated_metrics.read_text())
    actual_threshold = float(calibrated["metric"]["threshold"])
    if actual_threshold != THRESHOLD:
        raise ValueError(f"Threshold drift: expected {THRESHOLD}, found {actual_threshold}")

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))[:args.count]
    if len(source_rows) != args.count:
        raise ValueError(f"Manifest has only {len(source_rows)} rows")
    if len({row["run_id"] for row in source_rows}) != args.count:
        raise ValueError("Duplicate run_id in cohort")

    args.output_dir.mkdir(parents=True)
    for relative in ("configs", "logs", "preflight", "preflight/outputs", "outputs"):
        (args.output_dir / relative).mkdir()
    plan_rows = []
    manifest_rows = []
    for index, row in enumerate(source_rows):
        run_id = str(row["run_id"])
        base_rng_seed = args.plan_seed + int(run_id)
        rng = random.Random(base_rng_seed)
        x_magnitude = rng.randint(24, 32)
        y_magnitude = rng.randint(24, 32)
        x_sign, y_sign = DIRECTIONS[index % len(DIRECTIONS)]
        common_sign = 1 if index % 2 == 0 else -1
        modes = {
            mode: mode_shift(mode, x_magnitude, y_magnitude, x_sign, y_sign, common_sign)
            for mode in ALL_MODES
        }
        plan_rows.append({
            "cohort_index": index,
            "sample_id": run_id,
            "run_id": run_id,
            "base_rng_seed": base_rng_seed,
            "attack_seed": int(row.get("attack_seed") or (42 + int(run_id))),
            "x_sign": x_sign,
            "y_sign": y_sign,
            "common_sign": common_sign,
            "x_magnitude": x_magnitude,
            "y_magnitude": y_magnitude,
            "modes": modes,
        })
        watermarked_path = Path(row["watermarked_path"]).resolve()
        clean_path = Path(row["clean_path"]).resolve()
        for path, expected in (
            (watermarked_path, row["watermarked_sha256"]),
            (clean_path, row["clean_sha256"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_path(path) != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected image size {image.size}: {path}")
        manifest_rows.append({
            "cohort_index": index,
            "sample_id": run_id,
            "run_id": run_id,
            "prompt_id": row.get("prompt_id", ""),
            "source_prompt": row.get("prompt", ""),
            "conditioning_prompt": "",
            "prompt_sha256": EMPTY_PROMPT_SHA256,
            "clean_path": str(clean_path),
            "clean_sha256": row["clean_sha256"],
            "watermarked_path": str(watermarked_path),
            "watermarked_sha256": row["watermarked_sha256"],
            "generation_seed": row.get("generation_seed", ""),
            "attack_seed": int(row.get("attack_seed") or (42 + int(run_id))),
            "watermark_seed": int(row["w_seed"]),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        })

    write_json_exclusive(args.output_dir / "shift_plan.json", {
        "protocol": "raven_diagonal_interpretation_v1",
        "plan_seed": args.plan_seed,
        "vae_scale_factor": VAE_SCALE_FACTOR,
        "direction_assignment": "stratified cycle (+,+),(+,-),(-,+),(-,-)",
        "common_sign_assignment": "alternating +,-",
        "modes": MODE_DEFINITIONS,
        "samples": plan_rows,
    })
    manifest_fields = list(manifest_rows[0])
    with (args.output_dir / "diagnostic_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    fixed = {
        "inversion_mode": "ddim",
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "negative_prompt": "",
        "attention": True,
        "steps": 50,
        "strength": 0.15,
        "guidance_scale": 2.5,
        "dtype": "float16",
        "device": "cuda",
        "image_size": 512,
        "color_transfer": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tree_ring_calibrated_threshold": THRESHOLD,
        "threshold_source": str(args.calibrated_metrics.resolve()),
        "baseline_records": str(args.baseline_records.resolve()),
    }
    write_json_exclusive(args.output_dir / "configs" / "fixed_conditions.json", fixed)
    for mode in ALL_MODES:
        write_json_exclusive(args.output_dir / "configs" / f"mode_{mode}.json", {
            "mode": mode, **MODE_DEFINITIONS[mode],
        })
    script = Path(__file__).resolve()
    python = Path(sys.executable).resolve()
    root = args.output_dir.resolve()
    env = (
        "env TQDM_DISABLE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    )
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"{env} {python} -u {script} validate-warp --output-dir {root}",
        f"{env} {python} -u {script} attack --output-dir {root} --phase preflight",
        f"{env} {python} -u {script} score --output-dir {root} --phase preflight",
        "# Run the following only after preflight/validation.json reports passed=true.",
        f"{env} {python} -u {script} attack --output-dir {root} --phase full",
        f"{env} {python} -u {script} score --output-dir {root} --phase full",
        f"{env} {python} -u {script} aggregate --output-dir {root}",
    ]
    commands_path = args.output_dir / "commands.sh"
    commands_path.write_text("\n".join(commands) + "\n")
    commands_path.chmod(0o755)
    provenance = {
        **fixed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_path(args.manifest),
        "baseline_records_sha256": sha256_path(args.baseline_records),
        "calibrated_metrics_sha256": sha256_path(args.calibrated_metrics),
        "python": str(python),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short").splitlines(),
        "git_diff_sha256": hashlib.sha256(git_value("diff").encode()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_exclusive(args.output_dir / "provenance.json", provenance)
    print(json.dumps({"output_dir": str(root), "samples": len(plan_rows), "modes": list(ALL_MODES)}, indent=2))
    return 0


def command_validate_warp(args) -> int:
    import torch

    output = args.output_dir / "preflight" / "warp_direction_tests.json"
    if output.exists():
        raise FileExistsError(output)
    tests = []
    for dx, dy in ((12.0, 20.0), (12.0, -20.0), (-12.0, 20.0), (-12.0, -20.0)):
        latent = torch.zeros(1, 1, 17, 17)
        latent[0, 0, 8, 8] = 1.0
        shifted = translate_latent(
            latent, dx=dx, dy=dy, shift_space="image_pixels",
            vae_scale_factor=8, padding_mode="zeros", warp_mode="grid_sample",
        )
        weights = shifted[0, 0]
        yy, xx = torch.meshgrid(torch.arange(17), torch.arange(17), indexing="ij")
        center_x = float((weights * xx).sum() / weights.sum())
        center_y = float((weights * yy).sum() / weights.sum())
        expected_x, expected_y = 8.0 + dx / 8.0, 8.0 + dy / 8.0
        tests.append({
            "direction": direction_label(dx, dy),
            "dx_image_pixels": dx,
            "dy_image_pixels": dy,
            "observed_center_x": center_x,
            "observed_center_y": center_y,
            "expected_center_x": expected_x,
            "expected_center_y": expected_y,
            "passed": abs(center_x - expected_x) < 1e-5 and abs(center_y - expected_y) < 1e-5,
        })
    integer = torch.zeros(1, 1, 8, 8)
    integer[0, 0, 4, 7] = 1.0
    integer_shifted = translate_latent(
        integer, dx=1, dy=0, shift_space="latent_pixels",
        padding_mode="zeros", warp_mode="integer",
    )
    payload = {
        "passed": all(test["passed"] for test in tests) and float(integer_shifted.sum()) == 0.0,
        "positive_convention": "dx>0 right; dy>0 down",
        "grid_sample": {
            "mode": "bilinear",
            "padding_mode": "zeros",
            "align_corners": True,
            "formula": "source_grid = output_grid - (2*latent_shift/(size-1))",
            "half_pixel_offset": False,
        },
        "integer_warp_no_wraparound": float(integer_shifted.sum()) == 0.0,
        "directions": tests,
    }
    write_json_exclusive(output, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("warp direction validation failed")
    return 0


def load_protocol(output_dir: Path):
    plan = json.loads((output_dir / "shift_plan.json").read_text())
    with (output_dir / "diagnostic_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {row["run_id"]: row for row in csv.DictReader(handle)}
    provenance = json.loads((output_dir / "provenance.json").read_text())
    return plan, manifest, provenance


def quality(reference: Image.Image, attacked: Image.Image, dx: int, dy: int) -> dict:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(attacked.convert("RGB"), dtype=np.float32) / 255.0
    overlap_first, overlap_second = crop_overlap(first, second, dx, dy)
    height, width = overlap_first.shape[:2]
    return {
        "psnr": float(peak_signal_noise_ratio(overlap_first, overlap_second, data_range=1.0)),
        "ssim": float(structural_similarity(overlap_first, overlap_second, channel_axis=2, data_range=1.0)),
        "valid_overlap_width": int(width),
        "valid_overlap_height": int(height),
        "valid_overlap_area_ratio": float(width * height / (first.shape[0] * first.shape[1])),
    }


def attack_paths(output_dir: Path, phase: str):
    if phase == "preflight":
        return output_dir / "preflight" / "attack_records.jsonl", output_dir / "preflight" / "outputs"
    return output_dir / "attack_records.jsonl", output_dir / "outputs"


def command_attack(args) -> int:
    import torch

    plan, manifest, provenance = load_protocol(args.output_dir)
    if provenance["model_id"] != MODEL_ID or provenance["model_revision"] != MODEL_REVISION:
        raise ValueError("model provenance mismatch")
    warp_validation = args.output_dir / "preflight" / "warp_direction_tests.json"
    if not warp_validation.is_file() or not json.loads(warp_validation.read_text())["passed"]:
        raise RuntimeError("Run validate-warp successfully before attacks")
    modes = PRELIGHT_MODES if args.phase == "preflight" else ALL_MODES
    samples = plan["samples"][:2] if args.phase == "preflight" else plan["samples"]
    records_path, images_root = attack_paths(args.output_dir, args.phase)
    completed = set()
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        completed = {
            (item["mode"], item["run_id"])
            for item in (json.loads(line) for line in records_path.read_text().splitlines() if line.strip())
        }
        open_mode = "a"

    pipe = RavenPipeline(
        model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype,
    )
    torch.cuda.reset_peak_memory_stats()
    with records_path.open(open_mode, encoding="utf-8") as output:
        total = len(modes) * len(samples)
        done = len(completed)
        for mode in modes:
            for sample in samples:
                run_id = sample["run_id"]
                if (mode, run_id) in completed:
                    continue
                row = manifest[run_id]
                shift = sample["modes"][mode]
                input_path = Path(row["watermarked_path"])
                if sha256_path(input_path) != row["watermarked_sha256"]:
                    raise ValueError(f"input hash drift: {input_path}")
                reference = load_image(input_path, size=512)
                item_dir = images_root / f"mode_{mode}" / f"{int(run_id):06d}"
                if item_dir.exists():
                    raise FileExistsError(item_dir)
                item_dir.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                shift_x = shift["dx_image_pixels"] if shift["shift_space"] == "image_pixels" else shift["dx_latent_cells"]
                shift_y = shift["dy_image_pixels"] if shift["shift_space"] == "image_pixels" else shift["dy_latent_cells"]
                pipe.run(
                    input_image=reference,
                    output_dir=item_dir,
                    steps=50,
                    strength=0.15,
                    guidance_scale=2.5,
                    shift_space=shift["shift_space"],
                    warp_mode=shift["warp_mode"],
                    padding_mode="zeros",
                    shift_x=shift_x,
                    shift_y=shift_y,
                    view_guided_attention=True,
                    color_transfer=True,
                    seed=int(row["attack_seed"]),
                    prompt="",
                    negative_prompt="",
                    debug=args.phase == "preflight",
                    inversion_mode="ddim",
                )
                final_path = item_dir / "final_color_corrected.png"
                attacked = load_image(final_path, size=None)
                debug_info = json.loads((item_dir / "debug_info.json").read_text())
                if debug_info["inversion_prompt"] != "" or debug_info["reconstruction_prompt"] != "":
                    raise RuntimeError("non-empty prompt entered the pipeline")
                if debug_info["shift_source"] != "explicit_plan":
                    raise RuntimeError("pipeline ignored explicit shift")
                image_dx = int(round(shift["equivalent_image_dx"]))
                image_dy = int(round(shift["equivalent_image_dy"]))
                metrics = quality(reference, attacked, image_dx, image_dy)
                clip = debug_info["clipping_diagnostics"]
                record = {
                    "sample_id": run_id,
                    "run_id": run_id,
                    "mode": mode,
                    "prompt": "",
                    "prompt_sha256": EMPTY_PROMPT_SHA256,
                    "source_prompt": row["source_prompt"],
                    "prompt_id": row["prompt_id"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "watermark_seed": int(row["watermark_seed"]),
                    "generation_seed": row["generation_seed"],
                    "attack_seed": int(row["attack_seed"]),
                    "input_path": str(input_path.resolve()),
                    "input_sha256": sha256_path(input_path),
                    "attacked_path": str(final_path.resolve()),
                    "output_sha256": sha256_path(final_path),
                    "exact_ddim_timestep": int(debug_info["exact_timestep"]),
                    **shift,
                    **metrics,
                    "clipping_ratio": float(clip["fraction_below_zero"] + clip["fraction_above_one"]),
                    "runtime_seconds": float(time.monotonic() - started),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "inversion_prompt": debug_info["inversion_prompt"],
                    "reconstruction_prompt": debug_info["reconstruction_prompt"],
                    "negative_prompt": debug_info["negative_prompt"],
                    "inversion_conditioning": debug_info["inversion_conditioning"],
                    "reconstruction_conditioning": debug_info["reconstruction_conditioning"],
                    "attention_processor_count": debug_info.get("attention_processor_count"),
                    "attention_debug": debug_info.get("attention_debug"),
                    "debug_info_path": str((item_dir / "debug_info.json").resolve()),
                }
                append_jsonl(output, record)
                done += 1
                print(
                    f"[{done}/{total}] phase={args.phase} mode={mode} run_id={run_id} "
                    f"shift=({shift_x},{shift_y}) psnr={record['psnr']:.3f} "
                    f"ssim={record['ssim']:.4f} gpu={record['peak_gpu_memory_bytes']/2**30:.2f}GiB",
                    flush=True,
                )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def result_paths(output_dir: Path, phase: str):
    if phase == "preflight":
        base = output_dir / "preflight"
        return base / "per_sample_results.jsonl", base / "per_sample_results.csv"
    return output_dir / "per_sample_results.jsonl", output_dir / "per_sample_results.csv"


def command_score(args) -> int:
    import torch
    if not (args.eval_repo / "utils" / "pipe" / "pipe_utils.py").is_file():
        raise FileNotFoundError(f"Invalid detector repository: {args.eval_repo}")
    sys.path.insert(0, str(args.eval_repo.resolve()))
    from scripts.extract_verification_scores import (
        canonical_score, evaluate_image, provider_class, provider_kwargs, raw_score,
    )
    from raven.resource_guard import limit_cpu_threads
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    _, manifest, provenance = load_protocol(args.output_dir)
    records_path, _ = attack_paths(args.output_dir, args.phase)
    attacks = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    expected = len(PRELIGHT_MODES) * 2 if args.phase == "preflight" else len(ALL_MODES) * COHORT_SIZE
    if len(attacks) != expected:
        raise ValueError(f"Expected {expected} attack records, found {len(attacks)}")
    with Path(provenance["baseline_records"]).open(newline="", encoding="utf-8") as handle:
        baseline = {row["run_id"]: row for row in csv.DictReader(handle)}
    calibrated = json.loads(Path(provenance["threshold_source"]).read_text())
    threshold = float(calibrated["metric"]["threshold"])
    if threshold != THRESHOLD:
        raise ValueError(f"Threshold drift: {threshold}")

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
    first = baseline[attacks[0]["run_id"]]
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **provider_kwargs("TR", first),
    )
    jsonl_path, csv_path = result_paths(args.output_dir, args.phase)
    completed = set()
    open_mode = "x"
    scored = []
    if jsonl_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {jsonl_path}")
        scored = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        completed = {(row["mode"], row["run_id"]) for row in scored}
        open_mode = "a"
    torch.cuda.reset_peak_memory_stats()
    with jsonl_path.open(open_mode, encoding="utf-8") as output:
        for index, attack in enumerate(attacks, start=1):
            key = (attack["mode"], attack["run_id"])
            if key in completed:
                continue
            base = baseline[attack["run_id"]]
            result = evaluate_image(torch, provider, pipe, Path(attack["attacked_path"]), 50)
            attacked_raw = raw_score("TR", result)
            attacked_score = canonical_score("TR", attacked_raw, result)
            diagnostic = (result.get("p_value_diagnostics") or [{}])[0]
            before_raw = float(base["watermarked_raw_score"])
            before_score = float(base["watermarked_canonical_score"])
            record = {
                **attack,
                "watermarked_raw_detector_score": before_raw,
                "watermarked_score_before": before_score,
                "attacked_raw_detector_score": attacked_raw,
                "attacked_score_after": attacked_score,
                "score_delta": attacked_score - before_score,
                "calibrated_threshold": threshold,
                "detect_before": before_score >= threshold,
                "detect_after": attacked_score >= threshold,
                "tree_ring_score_definition": "-log10(p), higher means more watermark",
                "p_value": attacked_raw,
                "log_p": diagnostic.get("log_p"),
                "chi_square_statistic": diagnostic.get("statistic"),
                "p_value_underflow": bool(diagnostic.get("p_underflow", False)),
                "detector_nan": math.isnan(attacked_score),
                "detector_inf": math.isinf(attacked_score),
            }
            if not math.isfinite(attacked_score):
                raise ValueError(f"Non-finite detector score mode={attack['mode']} run_id={attack['run_id']}")
            append_jsonl(output, record)
            scored.append(record)
            print(
                f"[{index}/{len(attacks)}] mode={attack['mode']} run_id={attack['run_id']} "
                f"before={before_score:.6f} after={attacked_score:.6f}",
                flush=True,
            )
    scored.sort(key=lambda row: (ALL_MODES.index(row["mode"]), int(row["run_id"])))
    if csv_path.exists():
        raise FileExistsError(csv_path)
    fields = [
        key for key, value in scored[0].items()
        if not isinstance(value, (dict, list))
    ]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)

    if args.phase == "preflight":
        validation_path = args.output_dir / "preflight" / "validation.json"
        metadata_required = {
            "sample_id", "mode", "prompt", "prompt_sha256", "model_id", "model_revision",
            "exact_ddim_timestep", "dx_image_pixels", "dy_image_pixels", "dx_latent_cells",
            "dy_latent_cells", "equivalent_image_dx", "equivalent_image_dy",
            "interpolation_mode", "padding_mode", "align_corners", "valid_overlap_width",
            "valid_overlap_height", "valid_overlap_area_ratio", "watermarked_score_before",
            "attacked_score_after", "score_delta", "detect_before", "detect_after", "psnr",
            "ssim", "clipping_ratio", "output_sha256", "runtime_seconds", "peak_gpu_memory_bytes",
        }
        mode_counts = Counter(row["mode"] for row in scored)
        exact_timesteps = {row["exact_ddim_timestep"] for row in scored}
        validation = {
            "passed": (
                len(scored) == expected
                and all(mode_counts[mode] == 2 for mode in PRELIGHT_MODES)
                and all(row["prompt"] == "" and row["prompt_sha256"] == EMPTY_PROMPT_SHA256 for row in scored)
                and all(metadata_required <= row.keys() for row in scored)
                and all(not row["detector_nan"] and not row["detector_inf"] for row in scored)
                and len(exact_timesteps) == 1
                and json.loads((args.output_dir / "preflight" / "warp_direction_tests.json").read_text())["passed"]
            ),
            "records": len(scored),
            "mode_counts": dict(mode_counts),
            "empty_prompt_records": sum(row["prompt"] == "" for row in scored),
            "metadata_complete_records": sum(metadata_required <= row.keys() for row in scored),
            "nan_count": sum(row["detector_nan"] for row in scored),
            "inf_count": sum(row["detector_inf"] for row in scored),
            "underflow_count": sum(row["p_value_underflow"] for row in scored),
            "exact_timesteps": sorted(exact_timesteps),
            "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in scored),
            "peak_cpu_rss_kib": max(row["peak_cpu_rss_kib"] for row in scored),
        }
        write_json_exclusive(validation_path, validation)
        print(json.dumps(validation, indent=2))
        if not validation["passed"]:
            raise RuntimeError("preflight validation failed")
    del provider, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def finite_stats(values) -> dict:
    values = [float(value) for value in values]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("No finite values")
    return {
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def summarize_rows(rows: list[dict]) -> dict:
    score = finite_stats(row["attacked_score_after"] for row in rows)
    delta = finite_stats(row["score_delta"] for row in rows)
    psnr = finite_stats(row["psnr"] for row in rows)
    ssim = finite_stats(row["ssim"] for row in rows)
    overlap = finite_stats(row["valid_overlap_area_ratio"] for row in rows)
    low_ids = [row["sample_id"] for row in rows if not row["detect_after"]]
    return {
        "N": len(rows),
        "direction_counts": dict(Counter(row["direction"] for row in rows)),
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


def bootstrap_comparison(candidate: list[dict], reference: list[dict], samples: int, seed: int) -> dict:
    candidate_by_id = {row["sample_id"]: row for row in candidate}
    reference_by_id = {row["sample_id"]: row for row in reference}
    ids = sorted(candidate_by_id.keys() & reference_by_id.keys(), key=int)
    metrics = ("attacked_score_after", "psnr", "ssim", "valid_overlap_area_ratio")
    rng = np.random.default_rng(seed)
    output = {"N": len(ids), "candidate_minus_reference": {}, "score_win_tie_loss": {}}
    for metric in metrics:
        diffs = np.asarray([
            float(candidate_by_id[sample_id][metric]) - float(reference_by_id[sample_id][metric])
            for sample_id in ids
        ])
        indices = rng.integers(0, len(diffs), size=(samples, len(diffs)))
        bootstrap_means = diffs[indices].mean(axis=1)
        output["candidate_minus_reference"][metric] = {
            "paired_mean_difference": float(diffs.mean()),
            "median_paired_difference": float(np.median(diffs)),
            "bootstrap_95pct_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
        }
        if metric == "attacked_score_after":
            tolerance = 1e-12
            output["score_win_tie_loss"] = {
                "win_lower_score": int(np.sum(diffs < -tolerance)),
                "tie": int(np.sum(np.abs(diffs) <= tolerance)),
                "loss_higher_score": int(np.sum(diffs > tolerance)),
            }
    return output


def command_aggregate(args) -> int:
    jsonl_path, _ = result_paths(args.output_dir, "full")
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    expected = len(ALL_MODES) * COHORT_SIZE
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} scored rows, found {len(rows)}")
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in ALL_MODES}
    if any(len(items) != COHORT_SIZE for items in grouped.values()):
        raise ValueError("Mode sample count mismatch")
    aggregate = {
        "protocol": "30-sample paired diagnostic; not formal TPR@1%FPR",
        "calibrated_threshold_held_fixed": THRESHOLD,
        "score_definition": "-log10(p), higher means more watermark",
        "modes": {mode: summarize_rows(items) for mode, items in grouped.items()},
    }
    write_json_exclusive(args.output_dir / "aggregate_results.json", aggregate)

    comparisons_spec = (
        ("B_vs_A", "B", "A"),
        ("B_vs_C", "B", "C"),
        ("E_vs_B", "E", "B"),
        ("G_vs_B", "G", "B"),
        *((f"{mode}_vs_I", mode, "I") for mode in ALL_MODES if mode != "I"),
    )
    comparisons = {
        name: {
            "candidate": candidate,
            "reference": reference,
            **bootstrap_comparison(
                grouped[candidate], grouped[reference], args.bootstrap_samples,
                args.bootstrap_seed + index,
            ),
        }
        for index, (name, candidate, reference) in enumerate(comparisons_spec)
    }
    write_json_exclusive(args.output_dir / "paired_comparisons.json", {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed_base": args.bootstrap_seed,
        "interpretation": "negative attacked_score difference favors candidate",
        "comparisons": comparisons,
    })
    direction = {}
    for mode in ("B", "E"):
        direction[mode] = {
            label: summarize_rows([row for row in grouped[mode] if row["direction"] == label])
            for label in ("(+,+)", "(+,-)", "(-,+)", "(-,-)")
        }
    write_json_exclusive(args.output_dir / "direction_analysis.json", direction)

    header = (
        "| Mode | Unit interpretation | Sign rule | Magnitude rule | N | Detect rate | "
        "Mean score | Median score | Mean delta | PSNR | SSIM | Valid overlap |"
    )
    lines = [
        "# RAVEN Diagonal Shift Interpretation",
        "",
        "30-sample paired diagnostic only; the calibrated threshold is held fixed and is not recalibrated.",
        "",
        header,
        "| --- | --- | --- | --- | -: | -: | -: | -: | -: | -: | -: | -: |",
    ]
    for mode in ALL_MODES:
        definition = MODE_DEFINITIONS[mode]
        summary = aggregate["modes"][mode]
        lines.append(
            f"| {mode} | {definition['unit_interpretation']} | {definition['sign_rule']} | "
            f"{definition['magnitude_rule']} | {summary['N']} | "
            f"{summary['attacked_detection_rate']:.4f} | {summary['attacked_score']['mean']:.6f} | "
            f"{summary['attacked_score']['median']:.6f} | {summary['score_delta']['mean']:.6f} | "
            f"{summary['psnr']['mean']:.3f} | {summary['ssim']['mean']:.4f} | "
            f"{summary['valid_overlap']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Direction Analysis",
        "",
        "| Mode | Direction | N | Detect rate | Mean score | PSNR | SSIM |",
        "| --- | --- | -: | -: | -: | -: | -: |",
    ])
    for mode in ("B", "E"):
        for label in ("(+,+)", "(+,-)", "(-,+)", "(-,-)"):
            summary = direction[mode][label]
            lines.append(
                f"| {mode} | {label} | {summary['N']} | {summary['attacked_detection_rate']:.4f} | "
                f"{summary['attacked_score']['mean']:.6f} | {summary['psnr']['mean']:.3f} | "
                f"{summary['ssim']['mean']:.4f} |"
            )
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "aggregate": str(args.output_dir / "aggregate_results.json")}, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        return command_plan(args)
    if args.command == "validate-warp":
        return command_validate_warp(args)
    if args.command == "attack":
        return command_attack(args)
    if args.command == "score":
        return command_score(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
