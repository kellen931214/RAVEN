#!/usr/bin/env python
"""Evaluate image quality for RAVEN outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.gpu_utils import configure_gpu, finalize_gpu_logging, setup_run_logging, utc_timestamp, write_experiment_records
from raven.metrics import crop_overlap, psnr
from raven.resource_guard import CpuMemoryGuard
from raven.utils import iter_image_files, parse_bool


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _crop_overlap(a, b, dx: int, dy: int):
    return crop_overlap(a, b, dx, dy)


def _psnr(a, b):
    return psnr(a, b)


def _ssim(a, b):
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def _find_output_image(item_dir: Path) -> Path | None:
    for name in ("final_color_corrected.png", "final.png", "view_guided_output.png"):
        path = item_dir / name
        if path.exists():
            return path
    return None


def _load_prompt_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    mapping = {}
    for index, record in enumerate(records):
        identifier = record.get("file") or record.get("filename") or record.get("image") or f"{index:06d}.png"
        prompt = record.get("prompt") or record.get("caption") or record.get("text")
        if prompt is not None:
            mapping[Path(identifier).name] = prompt
            mapping[Path(identifier).stem] = prompt
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAVEN output quality.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim"], choices=["psnr", "ssim", "clip"])
    parser.add_argument("--prompts_csv", default=None)
    parser.add_argument("--clip_model", default="ViT-bigG-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--shift_min", type=int, default=24)
    parser.add_argument("--shift_max", type=int, default=32)
    parser.add_argument("--shift_space", choices=["image_pixels", "latent_pixels"], default="image_pixels")
    parser.add_argument("--save_csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None, help="GPU id to use, or 'auto' to select the least-used GPU")
    parser.add_argument("--require_free_gpu", type=parse_bool, default=False)
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
    memory_guard.check("before quality evaluation")
    rows = []
    status = "success"
    save_path = Path(args.save_csv)
    prompt_map = _load_prompt_map(args.prompts_csv)

    try:
        for row_index, input_path in enumerate(iter_image_files(args.input_dir)):
            memory_guard.check(f"before quality item {row_index + 1}")
            item_dir = Path(args.output_dir) / input_path.stem
            output_path = _find_output_image(item_dir)
            if output_path is None:
                rows.append({"file": input_path.name, "error": "missing_output"})
                continue

            original = _load_rgb(input_path)
            generated = _load_rgb(output_path)
            dx = dy = 0
            debug_path = item_dir / "debug_info.json"
            if debug_path.exists():
                info = json.loads(debug_path.read_text())
                dx = int(info.get("dx", 0))
                dy = int(info.get("dy", 0))
            original_crop, generated_crop = _crop_overlap(original, generated, dx, dy)

            row = {"file": input_path.name, "output": str(output_path), "dx": dx, "dy": dy}
            if "psnr" in args.metrics:
                row["psnr"] = _psnr(original_crop, generated_crop)
            if "ssim" in args.metrics:
                row["ssim"] = _ssim(original_crop, generated_crop)
            if "clip" in args.metrics:
                prompt = prompt_map.get(input_path.name) or prompt_map.get(input_path.stem)
                if prompt is None:
                    raise ValueError(f"Missing prompt mapping for {input_path.name}")
                row["prompt"] = prompt
            rows.append(row)

        if "clip" in args.metrics:
            from raven.quality import openclip_text_image_scores

            valid_rows = [row for row in rows if "output" in row]
            clip_result = openclip_text_image_scores(
                [row["output"] for row in valid_rows],
                [row["prompt"] for row in valid_rows],
                device=args.device,
                model_name=args.clip_model,
                pretrained=args.clip_pretrained,
            )
            for row, score in zip(valid_rows, clip_result["scores"]):
                row["clip"] = score
                row["clip_model"] = clip_result["model_name"]
                row["clip_pretrained"] = clip_result["pretrained"]

        fieldnames = sorted({key for row in rows for key in row.keys()})
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        status = "failed"
        raise
    finally:
        finalize_gpu_logging(args.output_dir, gpu_record)
        write_experiment_records(
            args.output_dir,
            vars(args),
            gpu_record,
            started_at,
            utc_timestamp(),
            status,
            extra_summary={"num_rows": len(rows), "save_csv": str(save_path)},
        )


if __name__ == "__main__":
    main()
