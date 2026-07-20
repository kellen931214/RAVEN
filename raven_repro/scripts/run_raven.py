#!/usr/bin/env python
"""ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT. Run one diagnostic image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.gpu_utils import configure_gpu, finalize_gpu_logging, setup_run_logging, utc_timestamp, write_experiment_records
from raven.resource_guard import CpuMemoryGuard
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image, parse_bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAVEN reproduction on one image.")
    parser.add_argument("--input", required=True, help="Input watermarked image")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--model_id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision", default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--inversion_mode", choices=["ddim", "forward_noise"], default="ddim")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--shift_min", type=int, default=24)
    parser.add_argument("--shift_max", type=int, default=32)
    parser.add_argument("--shift_sign", choices=["positive", "negative", "random"], default="random")
    parser.add_argument("--shift_sampling", choices=["independent_axes", "coupled_diagonal"], default="independent_axes")
    parser.add_argument("--shift_space", choices=["image_pixels", "latent_pixels"], default="image_pixels")
    parser.add_argument("--warp_mode", choices=["integer", "grid_sample"], default="integer")
    parser.add_argument("--padding_mode", choices=["reflection", "border", "zeros"], default="zeros")
    parser.add_argument("--view_guided_attention", type=parse_bool, default=True)
    parser.add_argument("--color_transfer", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None, help="GPU id to use, or 'auto' to select the least-used GPU")
    parser.add_argument("--require_free_gpu", type=parse_bool, default=False)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--debug", type=parse_bool, default=False)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=16.0, help="Stop if system MemAvailable falls below this many GiB")
    parser.add_argument("--max_process_ram_gb", type=float, default=40.0, help="Stop if this process RSS exceeds this many GiB")
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=None, help="Warn if system MemAvailable falls below this many GiB; does not stop")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_run_logging(args.output_dir)
    started_at = utc_timestamp()
    gpu_record = configure_gpu(args.gpu, args.device, args.output_dir, require_free_gpu=args.require_free_gpu)
    memory_guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
    memory_guard.check("before loading RAVEN model")
    status = "success"
    try:
        image = load_image(args.input, size=args.size)
        pipe = RavenPipeline(model_id=args.model_id, device=args.device, dtype=args.dtype, revision=args.model_revision)
        memory_guard.check("after loading RAVEN model")
        pipe.run(
            input_image=image,
            output_dir=args.output_dir,
            steps=args.steps,
            strength=args.strength,
            inversion_mode=args.inversion_mode,
            guidance_scale=args.guidance_scale,
            shift_min=args.shift_min,
            shift_max=args.shift_max,
            shift_sign=args.shift_sign,
            shift_sampling=args.shift_sampling,
            shift_space=args.shift_space,
            padding_mode=args.padding_mode,
            warp_mode=args.warp_mode,
            view_guided_attention=args.view_guided_attention,
            color_transfer=args.color_transfer,
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            debug=args.debug,
        )
    except Exception:
        status = "failed"
        raise
    finally:
        finished_at = utc_timestamp()
        finalize_gpu_logging(args.output_dir, gpu_record)
        write_experiment_records(args.output_dir, vars(args), gpu_record, started_at, finished_at, status)


if __name__ == "__main__":
    main()
