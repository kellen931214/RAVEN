#!/usr/bin/env python3
"""Build Markdown/CSV summary tables for RAVEN eval outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

METHODS = ["GS", "TR", "RID", "HSTR", "HSQR"]

COLUMNS = [
    ("wm_type", "WM"),
    ("completed", "N"),
    ("before_detection_rate", "Before TPR/Detect"),
    ("after_detection_rate", "After TPR/Detect"),
    ("raven_suppression_rate_on_detected", "RAVEN Suppression"),
    ("mean_before_bit_accuracy", "Bit Acc Before"),
    ("mean_after_bit_accuracy", "Bit Acc After"),
    ("fid_watermarked_vs_raven", "FID WM-vs-RAVEN"),
    ("mean_clip_score", "CLIP"),
    ("mean_overlap_psnr", "Overlap PSNR"),
    ("mean_overlap_ssim", "Overlap SSIM"),
    ("errors", "Errors"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAVEN eval summary tables")
    parser.add_argument("--eval_dir", default="/workspace/outputs/raven_eval/mscoco")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def load_summary(eval_dir: Path, method: str) -> Dict[str, Any]:
    path = eval_dir / method / "summary.json"
    if not path.exists():
        return {"wm_type": method, "error": f"missing {path}"}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("wm_type", method)
    return data


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) <= 1.0:
            return f"{value:.4f}"
        return f"{value:.3f}"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if abs(f) <= 1.0:
        return f"{f:.4f}"
    return f"{f:.3f}"


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [label for _, label in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key, _ in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key, _ in COLUMNS})


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    rows = [load_summary(eval_dir, method) for method in args.methods]
    md_path = Path(args.output_md) if args.output_md else eval_dir / "eval_summary_table.md"
    csv_path = Path(args.output_csv) if args.output_csv else eval_dir / "eval_summary_table.csv"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(rows), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
