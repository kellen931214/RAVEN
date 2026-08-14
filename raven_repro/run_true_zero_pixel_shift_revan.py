#!/usr/bin/env python3
"""Stream a true RGB-zero-padding + pre-inversion REVAN cohort.

Each sample has two independent DDIM inversions: the original RGB reference
and the zero-padded RGB view.  This runner never calls a latent translation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dx", type=int, required=True)
    parser.add_argument("--dy", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def iter_rows(path: Path, limit: int | None):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                return
            yield row


def main() -> None:
    args = parse_args()
    # Must be set before importing RavenPipeline/torch so this job cannot
    # initialize a non-selected GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch
    from raven.experiment_io import (
        is_sample_complete,
        output_image_path,
        rebuild_records_jsonl,
        write_record,
    )
    from raven.pipeline_raven import RavenPipeline
    from raven.rgb_shift import translate_rgb_zero_padding
    from raven.shift_plan import compute_attack_seed

    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}; use --resume"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config = {
        "experiment": "true_zero_pixel_shift_revan",
        "metadata_path": str(args.metadata.resolve()),
        "shift_domain": "rgb_pixel_space",
        "transform_stage": "pre_inversion_rgb",
        "dx": args.dx,
        "dy": args.dy,
        "padding_mode": "zeros",
        "padding_value_rgb": [0, 0, 0],
        "circular": False,
        "reflection": False,
        "latent_shift": False,
        "reference_branch": "original RGB -> independent DDIM inversion",
        "view_branch": "shifted zero-padded RGB -> independent DDIM inversion",
        "steps": args.steps,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "color_transfer": False,
        "base_seed": args.seed,
    }
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise ValueError("Stored config differs from requested cohort configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for REVAN")
    pipeline = RavenPipeline(device="cuda", dtype="float16", scheduler_mode="ddim")
    completed = skipped = 0
    started = time.monotonic()
    try:
        for index, row in enumerate(iter_rows(args.metadata, args.limit), start=1):
            run_id = str(row["run_id"])
            if is_sample_complete(args.output_dir, "watermarked", run_id):
                skipped += 1
                continue
            source_path = Path(row["watermarked_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"run_id={run_id}: {source_path}")
            with Image.open(source_path) as opened:
                reference = ImageOps.exif_transpose(opened).convert("RGB")
                reference.load()
            shifted = translate_rgb_zero_padding(reference, args.dx, args.dy)
            sample_dir = args.output_dir / "samples" / "watermarked" / run_id
            attack_seed = compute_attack_seed(args.seed, run_id)
            final = pipeline.run(
                input_image=reference,
                pre_inversion_view_image=shifted,
                rgb_shift_dx=args.dx,
                rgb_shift_dy=args.dy,
                output_dir=sample_dir,
                steps=args.steps,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                shift_x=0.0,
                shift_y=0.0,
                shift_space="rgb_pixel_space",
                warp_mode="none_pre_inversion_rgb",
                padding_mode="zeros",
                view_guided_attention=True,
                color_transfer=False,
                seed=attack_seed,
                prompt=row.get("prompt", ""),
                debug=False,
                save_input_copy=True,
            )
            output_path = output_image_path(args.output_dir, "watermarked", run_id)
            final.save(output_path)
            write_record(args.output_dir, "watermarked", run_id, {
                "run_id": run_id,
                "role": "watermarked",
                "dataset": row.get("dataset", "diffusiondb"),
                "method": "TR",
                "input_path": str(source_path),
                "output_path": str(output_path),
                "prompt": row.get("prompt", ""),
                "prompt_id": row.get("prompt_id", ""),
                "attack_seed": attack_seed,
                "shift_domain": "rgb_pixel_space",
                "transform_stage": "pre_inversion_rgb",
                "pixel_shift_dx_px": args.dx,
                "pixel_shift_dy_px": args.dy,
                "padding_mode": "zeros",
                "padding_value_rgb": [0, 0, 0],
                "circular": False,
                "reflection": False,
                "latent_shift": False,
                "source_metadata": row,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            completed += 1
            del reference, shifted, final
            gc.collect()
            torch.cuda.empty_cache()
            if completed % 10 == 0 or completed == 1:
                elapsed = time.monotonic() - started
                print(
                    f"progress completed={completed} skipped={skipped} "
                    f"last_run_id={run_id} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    finally:
        rebuild_records_jsonl(args.output_dir)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"completed={completed} skipped={skipped} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
