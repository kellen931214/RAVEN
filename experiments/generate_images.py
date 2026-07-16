#!/usr/bin/env python
"""Generate images from a prompt CSV with GPU auto-selection support."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "raven_repro"))

from raven.gpu_utils import configure_gpu, finalize_gpu_logging, setup_run_logging, utc_timestamp, write_experiment_records
from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
from raven.utils import parse_bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images from prompts.")
    parser.add_argument("--prompts_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "pndm", "euler", "dpm"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None, help="GPU id to use, or 'auto' to select the least-used GPU")
    parser.add_argument("--require_free_gpu", type=parse_bool, default=False)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=16.0, help="Stop if system MemAvailable falls below this many GiB")
    parser.add_argument("--max_process_ram_gb", type=float, default=40.0, help="Stop if this process RSS exceeds this many GiB")
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=None, help="Warn if system MemAvailable falls below this many GiB; does not stop")
    return parser


def _resolve_dtype(torch, dtype: str, device: str):
    if dtype in {None, "auto"}:
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _set_scheduler(pipe, scheduler: str):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler

    schedulers = {
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "euler": EulerDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
    }
    pipe.scheduler = schedulers[scheduler].from_config(pipe.scheduler.config)


def _load_prompts(path: str, prompt_column: str | None, limit: int | None) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return []
    column = prompt_column or next((name for name in ("prompt", "caption", "text") if name in rows[0]), None)
    if column is None:
        raise ValueError("Could not infer prompt column; pass --prompt_column")
    prompts = [{"index": str(i), "prompt": row[column]} for i, row in enumerate(rows)]
    return prompts[:limit] if limit is not None else prompts


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)
    started_at = utc_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_record = configure_gpu(args.gpu, args.device, output_dir, require_free_gpu=args.require_free_gpu)
    memory_guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
    memory_guard.check("before loading generation model")
    status = "success"
    generated_files: list[str] = []

    try:
        import torch
        limit_cpu_threads(1)
        from diffusers import StableDiffusionPipeline

        dtype = _resolve_dtype(torch, args.dtype, args.device)
        prompts = _load_prompts(args.prompts_csv, args.prompt_column, args.limit)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_csv}", flush=True)
        print(f"Loading model {args.model_id} with dtype={dtype} on device={args.device}", flush=True)
        model_variant = "fp16" if dtype == torch.float16 else None
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant=model_variant,
            low_cpu_mem_usage=True,
        )
        _set_scheduler(pipe, args.scheduler)
        pipe = pipe.to(args.device)
        print("Model loaded and moved to device", flush=True)

        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        manifest = []
        for batch_start in range(0, len(prompts), args.batch_size):
            memory_guard.check(f"before generation batch {batch_start // args.batch_size + 1}")
            batch = prompts[batch_start : batch_start + args.batch_size]
            print(f"Generating batch {batch_start // args.batch_size + 1}/{(len(prompts) + args.batch_size - 1) // args.batch_size}", flush=True)
            todo = []
            for row in batch:
                image_path = output_dir / f"{int(row['index']):06d}.png"
                if image_path.exists():
                    generated_files.append(str(image_path))
                    manifest.append({"file": str(image_path), "prompt": row["prompt"], "index": int(row["index"]), "skipped_existing": True})
                    print(f"Skipping existing {image_path}", flush=True)
                else:
                    todo.append(row)
            if not todo:
                continue
            images = pipe(
                [row["prompt"] for row in todo],
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images
            for row, image in zip(todo, images):
                image_path = output_dir / f"{int(row['index']):06d}.png"
                image.save(image_path)
                generated_files.append(str(image_path))
                manifest.append({"file": str(image_path), "prompt": row["prompt"], "index": int(row["index"]), "skipped_existing": False})
                print(f"Saved {image_path}", flush=True)
        (output_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True))
        print(f"Generation complete: {len(generated_files)} images", flush=True)
    except Exception:
        status = "failed"
        raise
    finally:
        finalize_gpu_logging(output_dir, gpu_record)
        write_experiment_records(
            output_dir,
            vars(args),
            gpu_record,
            started_at,
            utc_timestamp(),
            status,
            extra_summary={"num_generated": len(generated_files), "generated_files": generated_files},
        )


if __name__ == "__main__":
    main()
