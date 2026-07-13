#!/usr/bin/env python
"""Aggregate versioned per-method RAVEN metric JSON files into CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-schemes", type=int, default=15)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def row_from_report(report: dict) -> dict:
    method = report["method"]
    metric = report["metric"]
    quality = report.get("quality", {})
    row = {
        "method": method,
        "protocol_version": report.get("protocol_version"),
        "num_pairs": quality.get("num_pairs"),
        "overlap_psnr": quality.get("overlap_psnr"),
        "overlap_ssim": quality.get("overlap_ssim"),
    }
    if method == "GS":
        row.update({
            "metric_type": "Bit Accuracy",
            "before": metric.get("bit_accuracy_before"),
            "after": metric.get("bit_accuracy_after"),
            "calibrated_fpr": None,
        })
    else:
        calibration = metric["calibration"]
        row.update({
            "metric_type": "TPR@1%FPR",
            "before": metric.get("watermarked_tpr"),
            "after": metric.get("attacked_tpr"),
            "calibrated_fpr": calibration.get("actual_fpr"),
        })
    clip = quality.get("clip_text")
    row["clip_text"] = clip.get("mean") if isinstance(clip, dict) else None
    fid = quality.get("fid")
    row["fid_clean"] = fid.get("clean_fid", {}).get("value") if isinstance(fid, dict) else None
    row["fid_torchmetrics"] = fid.get("torchmetrics", {}).get("value") if isinstance(fid, dict) else None
    return row


def display(value) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = build_parser().parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    versions = {report.get("protocol_version") for report in reports}
    methods = [report.get("method") for report in reports]
    if len(versions) != 1:
        raise ValueError(f"Cannot aggregate mixed protocol versions: {versions}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate method reports: {methods}")
    rows = [row_from_report(report) for report in reports]
    paper_comparable = len(rows) == args.expected_schemes
    for row in rows:
        row["dataset"] = args.dataset
        row["aggregation_scope"] = (
            f"paper-comparable {args.expected_schemes}-scheme set"
            if paper_comparable
            else f"diagnostic {len(rows)}-scheme subset"
        )

    fieldnames = [
        "dataset", "method", "metric_type", "before", "after", "calibrated_fpr",
        "fid_clean", "fid_torchmetrics", "clip_text", "overlap_psnr", "overlap_ssim",
        "num_pairs", "protocol_version", "aggregation_scope",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = ["WM", "Metric", "Before", "After", "Actual FPR", "FID", "CLIP-Text", "Overlap PSNR", "Overlap SSIM"]
    markdown = [
        f"# RAVEN evaluation: {args.dataset}",
        "",
        f"Aggregation scope: **{rows[0]['aggregation_scope']}**",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(display(value) for value in [
            row["method"], row["metric_type"], row["before"], row["after"],
            row["calibrated_fpr"], row["fid_clean"], row["clip_text"],
            row["overlap_psnr"], row["overlap_ssim"],
        ]) + " |")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv} and {args.output_md}; scope={rows[0]['aggregation_scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
