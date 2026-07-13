#!/usr/bin/env python
"""Recompute protocol-correct RAVEN metrics from auditable raw records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import (
    SEMANTIC_METHODS,
    bit_accuracy,
    canonical_watermark_score,
    crop_overlap,
    mean_finite,
    psnr,
    summarize_detection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--records", type=Path, required=True, help="CSV with raw scores and/or image paths")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--expected-gs-bits", type=int, default=256)
    parser.add_argument("--compute-clip", action="store_true")
    parser.add_argument("--clip-model", default="ViT-bigG-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--fid-reference-dir", type=Path, default=None)
    parser.add_argument("--fid-attacked-dir", type=Path, default=None)
    parser.add_argument("--quality-device", default="cuda")
    return parser


def parse_float(row: dict[str, str], column: str) -> float:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"missing {column}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {column}: {value}")
    return parsed


def load_rgb(path: str):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def quality_row(row: dict[str, str]) -> dict:
    original_path = row.get("original_path") or row.get("watermarked_path")
    attacked_path = row.get("attacked_path")
    if not original_path or not attacked_path:
        return {}
    original = load_rgb(original_path)
    attacked = load_rgb(attacked_path)
    dx = int(float(row.get("dx") or 0))
    dy = int(float(row.get("dy") or 0))
    original_overlap, attacked_overlap = crop_overlap(original, attacked, dx, dy)
    result = {"overlap_psnr": psnr(original_overlap, attacked_overlap), "dx": dx, "dy": dy}
    try:
        from skimage.metrics import structural_similarity

        result["overlap_ssim"] = float(
            structural_similarity(original_overlap, attacked_overlap, channel_axis=2, data_range=1.0)
        )
    except ImportError:
        result["overlap_ssim"] = None
    return result


def evaluate_semantic(method: str, rows: list[dict[str, str]], target_fpr: float) -> dict:
    clean = [canonical_watermark_score(method, parse_float(row, "clean_raw_score")) for row in rows]
    watermarked = [canonical_watermark_score(method, parse_float(row, "watermarked_raw_score")) for row in rows]
    attacked = [canonical_watermark_score(method, parse_float(row, "attacked_raw_score")) for row in rows]
    return summarize_detection(clean, watermarked, attacked, target_fpr=target_fpr).to_dict()


def evaluate_gs(rows: list[dict[str, str]], expected_bits: int) -> tuple[dict, list[dict]]:
    audited = []
    before = []
    after = []
    for index, row in enumerate(rows):
        run_id = row.get("run_id") or str(index)
        gt = row.get("ground_truth_bits")
        before_bits = row.get("watermarked_predicted_bits")
        after_bits = row.get("attacked_predicted_bits")
        if not gt or not before_bits or not after_bits:
            raise ValueError(f"run_id={run_id}: GS requires ground_truth_bits and both predicted bit strings")
        before_result = bit_accuracy(gt, before_bits, expected_length=expected_bits)
        after_result = bit_accuracy(gt, after_bits, expected_length=expected_bits)
        before.append(before_result["accuracy"])
        after.append(after_result["accuracy"])
        audited.append({"run_id": run_id, "before": before_result, "after": after_result})
    return {
        "num_samples": len(rows),
        "num_bits_per_sample": expected_bits,
        "bit_accuracy_before": sum(before) / len(before),
        "bit_accuracy_after": sum(after) / len(after),
        "total_bits": len(rows) * expected_bits,
        "total_errors_after": sum(item["after"]["num_errors"] for item in audited),
    }, audited


def main() -> int:
    args = build_parser().parse_args()
    with args.records.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No records in {args.records}")

    method = args.method.upper()
    detailed_rows = []
    if method in SEMANTIC_METHODS:
        metric = evaluate_semantic(method, rows, args.target_fpr)
    else:
        metric, detailed_rows = evaluate_gs(rows, args.expected_gs_bits)

    quality = []
    for row in rows:
        item = quality_row(row)
        if item:
            quality.append(item)
    quality_summary = {"num_pairs": len(quality)}
    if quality:
        quality_summary["overlap_psnr"] = mean_finite(item["overlap_psnr"] for item in quality)
        ssim_values = [item["overlap_ssim"] for item in quality if item.get("overlap_ssim") is not None]
        quality_summary["overlap_ssim"] = sum(ssim_values) / len(ssim_values) if ssim_values else None
    if args.compute_clip:
        from raven.quality import openclip_text_image_scores

        image_paths = [row.get("attacked_path") for row in rows]
        prompts = [row.get("prompt") for row in rows]
        if any(not value for value in image_paths) or any(value is None for value in prompts):
            raise ValueError("CLIP requires attacked_path and prompt for every record")
        quality_summary["clip_text"] = openclip_text_image_scores(
            image_paths,
            prompts,
            device=args.quality_device,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
        )
    if args.compute_fid:
        if args.fid_reference_dir is None or args.fid_attacked_dir is None:
            raise ValueError("--compute-fid requires --fid-reference-dir and --fid-attacked-dir")
        from raven.quality import clean_fid, torchmetrics_fid

        reference_paths = sorted(args.fid_reference_dir.glob("*.png"))
        attacked_paths = sorted(args.fid_attacked_dir.glob("*.png"))
        quality_summary["fid"] = {
            "clean_fid": clean_fid(args.fid_reference_dir, args.fid_attacked_dir, device=args.quality_device),
            "torchmetrics": torchmetrics_fid(reference_paths, attacked_paths, device=args.quality_device),
            "count_reference": len(reference_paths),
            "count_attacked": len(attacked_paths),
        }

    report = {
        "protocol_version": 1,
        "method": method,
        "records": str(args.records.resolve()),
        "metric": metric,
        "quality": quality_summary,
        "notes": {
            "semantic_score_direction": "canonical higher means watermark",
            "semantic_metric": "TPR calibrated from clean negatives at requested FPR",
            "gs_metric": "per-sample, per-bit decoded accuracy",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_rows and detailed_rows:
        args.output_rows.parent.mkdir(parents=True, exist_ok=True)
        args.output_rows.write_text(json.dumps(detailed_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
