#!/usr/bin/env python
"""Run the fixed three-sample RAVEN attack diagnostic matrix."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image


CONFIGS = (
    {"name": "ddim_attn_independent_24", "inversion_mode": "ddim", "attention": True, "sampling": "independent_axes", "shift": 24},
    {"name": "ddim_noattn_independent_24", "inversion_mode": "ddim", "attention": False, "sampling": "independent_axes", "shift": 24},
    {"name": "ddim_attn_coupled_24", "inversion_mode": "ddim", "attention": True, "sampling": "coupled_diagonal", "shift": 24},
    {"name": "forward_noise_attn_independent_24", "inversion_mode": "forward_noise", "attention": True, "sampling": "independent_axes", "shift": 24},
    {"name": "ddim_attn_no_shift", "inversion_mode": "ddim", "attention": True, "sampling": "independent_axes", "shift": 0},
    {"name": "ddim_attn_independent_32", "inversion_mode": "ddim", "attention": True, "sampling": "independent_axes", "shift": 32},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(3, 11), default=3)
    parser.add_argument(
        "--config",
        action="append",
        choices=[item["name"] for item in CONFIGS],
        help="Run only the named config; repeat for multiple configs",
    )
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quality(reference: Image.Image, attacked: Image.Image, dx: int, dy: int) -> dict:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(attacked.convert("RGB"), dtype=np.float32) / 255.0
    first, second = crop_overlap(first, second, dx, dy)
    return {
        "psnr": float(peak_signal_noise_ratio(first, second, data_range=1.0)),
        "ssim": float(structural_similarity(first, second, channel_axis=2, data_range=1.0)),
        "overlap_shape": list(first.shape),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"manifest has only {len(rows)} rows, expected {args.limit}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate run_id in selected diagnostic rows")
    requested = set(args.config or ())
    configs = [item for item in CONFIGS if not requested or item["name"] in requested]
    if not configs:
        raise ValueError("no diagnostic configs selected")

    pipe = RavenPipeline(
        model_id=args.model_id,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
    )
    records_path = args.output_dir / "attack_records.jsonl"
    records = []
    with records_path.open("x", encoding="utf-8") as output:
        for config in configs:
            for row in rows:
                run_id = row["run_id"]
                input_path = Path(row["watermarked_path"]).resolve()
                reference = load_image(input_path, size=args.size)
                item_dir = args.output_dir / config["name"] / f"{int(run_id):06d}"
                torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
                started = time.monotonic()
                pipe.run(
                    input_image=reference,
                    output_dir=item_dir,
                    steps=args.steps,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    shift_min=config["shift"],
                    shift_max=config["shift"],
                    shift_sign="random",
                    shift_sampling=config["sampling"],
                    shift_space="image_pixels",
                    warp_mode="integer",
                    padding_mode="zeros",
                    view_guided_attention=config["attention"],
                    color_transfer=True,
                    seed=int(row.get("attack_seed") or (42 + int(run_id))),
                    prompt=row.get("prompt", ""),
                    negative_prompt="",
                    debug=config["attention"],
                    inversion_mode=config["inversion_mode"],
                )
                final_path = item_dir / "final_color_corrected.png"
                attacked = load_image(final_path, size=None)
                debug_info = json.loads((item_dir / "debug_info.json").read_text())
                metrics = quality(reference, attacked, int(debug_info["image_dx"]), int(debug_info["image_dy"]))
                record = {
                    "config": config["name"],
                    "mode": config["inversion_mode"],
                    "attention": config["attention"],
                    "shift_sampling": config["sampling"],
                    "run_id": run_id,
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt": row.get("prompt", ""),
                    "seed": int(row.get("attack_seed") or (42 + int(run_id))),
                    "input_path": str(input_path),
                    "input_sha256": sha256(input_path),
                    "attacked_path": str(final_path.resolve()),
                    "attacked_sha256": sha256(final_path),
                    "exact_timestep": debug_info["exact_timestep"],
                    "image_dx": debug_info["image_dx"],
                    "image_dy": debug_info["image_dy"],
                    "latent_dx": debug_info["latent_dx"],
                    "latent_dy": debug_info["latent_dy"],
                    "warp_mode": debug_info["warp_mode"],
                    "padding_mode": debug_info["padding_mode"],
                    "attention_debug": debug_info.get("attention_debug"),
                    "clipping_diagnostics": debug_info["clipping_diagnostics"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "overlap_shape": metrics["overlap_shape"],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
                    "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
                records.append(record)
                print(
                    f"{config['name']} run_id={run_id} psnr={record['psnr']:.4f} "
                    f"ssim={record['ssim']:.4f} elapsed={record['elapsed_seconds']:.2f}s",
                    flush=True,
                )

    comparisons = []
    config_names = {item["name"] for item in configs}
    if {"ddim_attn_independent_24", "ddim_noattn_independent_24"} <= config_names:
        for row in rows:
            run_id = row["run_id"]
            on = next(item for item in records if item["config"] == "ddim_attn_independent_24" and item["run_id"] == run_id)
            off = next(item for item in records if item["config"] == "ddim_noattn_independent_24" and item["run_id"] == run_id)
            on_array = np.asarray(Image.open(on["attacked_path"]).convert("RGB"), dtype=np.int16)
            off_array = np.asarray(Image.open(off["attacked_path"]).convert("RGB"), dtype=np.int16)
            difference = np.abs(on_array - off_array)
            comparisons.append({
                "run_id": run_id,
                "identical": bool(np.array_equal(on_array, off_array)),
                "mean_absolute_pixel_difference": float(difference.mean()),
                "max_absolute_pixel_difference": int(difference.max()),
            })

    summary = {
        "manifest": str(args.manifest.resolve()),
        "sample_count": len(rows),
        "attack_count": len(records),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "steps": args.steps,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "configs": configs,
        "attention_on_off_comparison": comparisons,
        "records_path": str(records_path.resolve()),
    }
    (args.output_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    del pipe
    gc.collect()
    torch.cuda.empty_cache() if args.device == "cuda" else None
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
