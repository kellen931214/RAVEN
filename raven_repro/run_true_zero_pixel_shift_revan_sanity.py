#!/usr/bin/env python3
"""Run exactly one RGB-zero-padding + pre-inversion REVAN sanity sample.

This runner intentionally has no cohort/evaluation loop.  It writes the
reference image, shifted RGB view, reconstruction outputs, and provenance into
one new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="0")
    parser.add_argument("--dx", type=int, default=-32)
    parser.add_argument("--dy", type=int, default=-32)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument(
        "--model-revision", default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    return parser.parse_args()


def load_row(metadata_path: Path, run_id: str) -> dict[str, str]:
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("run_id", "")) == str(run_id):
                return row
    raise ValueError(f"run_id={run_id!r} not found in {metadata_path}")


def assert_zero_padding(source: Image.Image, shifted: Image.Image, dx: int, dy: int) -> None:
    """Verify the exact slicing contract before model execution."""
    original = np.asarray(source.convert("RGB"), dtype=np.uint8)
    actual = np.asarray(shifted.convert("RGB"), dtype=np.uint8)
    expected = np.zeros_like(original)
    height, width = original.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx0 < sx1 and sy0 < sy1:
        tx0, ty0 = max(0, dx), max(0, dy)
        expected[ty0:ty0 + sy1 - sy0, tx0:tx0 + sx1 - sx0] = original[sy0:sy1, sx0:sx1]
    if not np.array_equal(actual, expected):
        raise AssertionError("RGB translation does not match explicit zero-padding slicing")

    if dx < 0 and not np.all(actual[:, width + dx:, :] == 0):
        raise AssertionError("right padding is not all RGB(0,0,0)")
    if dx > 0 and not np.all(actual[:, :dx, :] == 0):
        raise AssertionError("left padding is not all RGB(0,0,0)")
    if dy < 0 and not np.all(actual[height + dy:, :, :] == 0):
        raise AssertionError("bottom padding is not all RGB(0,0,0)")
    if dy > 0 and not np.all(actual[:dy, :, :] == 0):
        raise AssertionError("top padding is not all RGB(0,0,0)")


def main() -> None:
    args = parse_args()
    # This is set before RavenPipeline imports torch/diffusers, preventing an
    # accidental initialization of a different host GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    from raven.pipeline_raven import RavenPipeline
    from raven.rgb_shift import translate_rgb_zero_padding

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row = load_row(args.metadata, args.run_id)
    source_path = Path(row["watermarked_path"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as opened:
        reference = ImageOps.exif_transpose(opened).convert("RGB")
        reference.load()
    shifted = translate_rgb_zero_padding(reference, args.dx, args.dy)
    assert_zero_padding(reference, shifted, args.dx, args.dy)

    config = {
        "experiment": "true_zero_pixel_shift_revan_sanity",
        "sample_count": 1,
        "run_id": str(args.run_id),
        "source_image": str(source_path),
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
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "steps": args.steps,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "color_transfer": False,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    pipeline = RavenPipeline(
        model_id=args.model_id,
        revision=args.model_revision,
        device="cuda",
        dtype="float16",
        scheduler_mode="ddim",
    )
    final = pipeline.run(
        input_image=reference,
        pre_inversion_view_image=shifted,
        rgb_shift_dx=args.dx,
        rgb_shift_dy=args.dy,
        output_dir=args.output_dir,
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
        seed=args.seed,
        prompt=row.get("prompt", ""),
        debug=True,
        save_input_copy=True,
    )
    final.save(args.output_dir / "final.png")
    print(json.dumps({
        "input": str(args.output_dir / "input.png"),
        "shifted_zero_padded": str(args.output_dir / "shifted_zero_padded.png"),
        "view_guided_output": str(args.output_dir / "view_guided_output.png"),
        "final": str(args.output_dir / "final.png"),
        "debug_info": str(args.output_dir / "debug_info.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
