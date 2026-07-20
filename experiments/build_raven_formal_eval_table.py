#!/usr/bin/env python3
"""Build an auditable table from validated formal RAVEN result roots only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "Dataset", "Watermark", "N", "Metric protocol", "Target FPR", "Actual FPR",
    "Before TPR", "Attacked TPR at original threshold",
    "Attacked TPR at recalibrated threshold", "Attack success rate", "ROC-AUC",
    "GS bit accuracy before", "GS bit accuracy attacked", "FID reference", "FID",
    "CLIP model", "CLIP score", "Quality reference", "Overlap protocol", "PSNR",
    "SSIM", "Manifest SHA", "Attack config hash", "Detector config hash", "Git SHA", "Status",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--formal-roots", type=Path, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def mean_jsonl(path: Path, key: str) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def table_row(root: Path) -> dict:
    validation = json.loads((root / "VALIDATED.json").read_text())
    if validation.get("status") != "validated_formal_result":
        raise ValueError(f"not a validated formal result: {root}")
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    detector = json.loads(Path(aggregate["detector_result"]).read_text())
    method = aggregate["method"]
    if method == "TR":
        metric = detector["nfpa_rounded2_protocol"]
    elif method == "GS":
        metric = detector["metric"]
    else:
        metric = detector["metric"]
    quality_path = Path(aggregate["quality_records"])
    clip = aggregate["clip_result"]
    fid = aggregate["fid_result"]
    manifest = root / "verification" / "manifest.csv"
    row = {field: "" for field in FIELDS}
    row.update({
        "Dataset": aggregate["dataset"], "Watermark": method, "N": aggregate["N"],
        "Metric protocol": aggregate["metric_protocol_version"],
        "FID reference": fid["reference_definition"],
        "FID": fid.get("value", fid.get("fid", fid.get("score", ""))),
        "CLIP model": f"{clip['clip_model_name']}/{clip['clip_pretrained']}",
        "CLIP score": clip.get("mean", clip.get("score", "")),
        "Quality reference": "watermarked input",
        "Overlap protocol": "effective source flow inverse warp",
        "PSNR": mean_jsonl(quality_path, "post_color_vs_watermarked_overlap_psnr"),
        "SSIM": mean_jsonl(quality_path, "post_color_vs_watermarked_overlap_ssim"),
        "Manifest SHA": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "Attack config hash": aggregate["attack_config_hash"],
        "Detector config hash": aggregate["detector_config_hash"],
        "Git SHA": aggregate["git_head"], "Status": validation["status"],
    })
    if method == "GS":
        row["GS bit accuracy before"] = metric["micro_bit_accuracy_before"]
        row["GS bit accuracy attacked"] = metric["micro_bit_accuracy_attacked"]
    elif method == "TR":
        row.update({
            "Target FPR": metric["target_fpr"], "Actual FPR": metric["before_actual_fpr"],
            "Before TPR": metric["before_tpr"],
            "Attacked TPR at original threshold": metric["attacked_tpr_at_original_clean_threshold"],
            "Attacked TPR at recalibrated threshold": metric["attacked_tpr_at_attacked_clean_recalibrated_threshold"],
            "Attack success rate": metric["attack_success_rate_at_recalibrated_threshold"],
            "ROC-AUC": metric["attacked_roc_auc"],
        })
    else:
        row.update({
            "Target FPR": metric["target_fpr"], "Actual FPR": metric["actual_empirical_fpr"],
            "Before TPR": metric["before_tpr"],
            "Attacked TPR at original threshold": metric["attacked_tpr_at_original_clean_threshold"],
            "Attacked TPR at recalibrated threshold": metric["attacked_tpr_at_recalibrated_threshold"],
            "Attack success rate": metric["attack_success_rate_at_original_clean_threshold"],
            "ROC-AUC": metric["attacked_roc_auc"],
        })
    return row


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [table_row(root.resolve()) for root in args.formal_roots]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
