#!/usr/bin/env python3
"""Build an auditable table from validated formal RAVEN result roots only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "Dataset", "Watermark", "Variant", "N", "Metric protocol", "Target FPR",
    "Original-clean actual FPR", "Attacked-clean recalibrated actual FPR",
    "Before TPR", "Attacked TPR at original threshold",
    "Attacked TPR at recalibrated threshold",
    "Attack success at original-clean threshold",
    "Attack success at attacked-clean recalibrated threshold", "ROC-AUC",
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
    if validation.get("status") == "validated_aligned_color_evaluation":
        return aligned_color_table_row(root, validation)
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
        "Dataset": aggregate["dataset"], "Watermark": method,
        "Variant": aggregate["attack_config"].get("variant_name", "formal_variant"),
        "N": aggregate["N"],
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
            "Target FPR": metric["target_fpr"],
            "Original-clean actual FPR": metric["original_clean_actual_fpr"],
            "Attacked-clean recalibrated actual FPR": metric["attacked_clean_actual_fpr"],
            "Before TPR": metric["before_tpr"],
            "Attacked TPR at original threshold": metric["attacked_tpr_at_original_clean_threshold"],
            "Attacked TPR at recalibrated threshold": metric["attacked_tpr_at_attacked_clean_recalibrated_threshold"],
            "Attack success at original-clean threshold": (
                1.0 - metric["attacked_tpr_at_original_clean_threshold"]
            ),
            "Attack success at attacked-clean recalibrated threshold": metric[
                "attack_success_rate_at_recalibrated_threshold"
            ],
            "ROC-AUC": metric["attacked_roc_auc"],
        })
    else:
        row.update({
            "Target FPR": metric["target_fpr"],
            "Original-clean actual FPR": metric["actual_empirical_fpr"],
            "Attacked-clean recalibrated actual FPR": metric.get(
                "attacked_clean_actual_fpr", ""
            ),
            "Before TPR": metric["before_tpr"],
            "Attacked TPR at original threshold": metric["attacked_tpr_at_original_clean_threshold"],
            "Attacked TPR at recalibrated threshold": metric["attacked_tpr_at_recalibrated_threshold"],
            "Attack success at original-clean threshold": metric[
                "attack_success_rate_at_original_clean_threshold"
            ],
            "Attack success at attacked-clean recalibrated threshold": metric.get(
                "attack_success_rate_at_recalibrated_threshold", ""
            ),
            "ROC-AUC": metric["attacked_roc_auc"],
        })
    return row



def aligned_color_table_row(root: Path, validation: dict) -> dict:
    """Adapt the retained aligned baseline without recomputing any metric."""
    aggregate = json.loads((root / "aggregate_results.json").read_text())
    source_manifest = json.loads(
        Path(aggregate["source_code_manifest_path"]).read_text(encoding="utf-8")
    )
    detector = aggregate["detector"]["nfpa_rounded2_protocol"]
    fid = aggregate["fid"]
    clip = aggregate["clip"]
    manifest = root / "verification" / "manifest.csv"
    row = {field: "" for field in FIELDS}
    row.update({
        "Dataset": "diffusiondb",
        "Watermark": "TR",
        "Variant": "nfpa_nearest_reflection_ddim_aligned_reused",
        "N": aggregate["sample_count"],
        "Metric protocol": aggregate["detector"]["protocol"],
        "Target FPR": detector["target_fpr"],
        "Original-clean actual FPR": detector["original_clean_actual_fpr"],
        "Attacked-clean recalibrated actual FPR": detector[
            "attacked_clean_actual_fpr"
        ],
        "Before TPR": detector["before_tpr"],
        "Attacked TPR at original threshold": detector[
            "attacked_tpr_at_original_clean_threshold"
        ],
        "Attacked TPR at recalibrated threshold": detector[
            "attacked_tpr_at_attacked_clean_recalibrated_threshold"
        ],
        "Attack success at original-clean threshold": (
            1.0 - detector["attacked_tpr_at_original_clean_threshold"]
        ),
        "Attack success at attacked-clean recalibrated threshold": detector[
            "attack_success_rate_at_recalibrated_threshold"
        ],
        "ROC-AUC": detector["attacked_roc_auc"],
        "FID reference": fid["reference_definition"],
        "FID": fid["value"],
        "CLIP model": f"{clip['clip_model_name']}/{clip['clip_pretrained']}",
        "CLIP score": clip["mean"],
        "Quality reference": aggregate["quality_reference"],
        "Overlap protocol": aggregate["quality_overlap"],
        "PSNR": aggregate["quality_psnr_mean"],
        "SSIM": aggregate["quality_ssim_mean"],
        "Manifest SHA": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "Attack config hash": aggregate["formal_attack_config_hash"],
        "Detector config hash": "",
        "Git SHA": source_manifest["git_head"],
        "Status": validation["status"],
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
