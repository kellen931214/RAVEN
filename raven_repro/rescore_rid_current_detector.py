#!/usr/bin/env python3
"""Re-score a fixed RID cohort with the current official-parity detector.

This tool never generates images or runs an attack.  It streams the three
already-materialized image paths from a formal verification manifest, scores
one image at a time with the current RID provider, and writes a new result
directory without touching the source cohort or historical scores.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_bench_wm"))
sys.path.insert(0, str(REPO / "raven_repro"))


COHORTS = (
    ("clean", "clean_path"),
    ("original_watermarked", "watermarked_path"),
    ("attacked_watermarked", "attacked_path"),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty score list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Pearson correlation requires equally sized 2+ vectors")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denom_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    denom_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if denom_x == 0.0 or denom_y == 0.0:
        raise ValueError("Pearson correlation is undefined for constant vectors")
    return numerator / (denom_x * denom_y)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def score_one(provider_info: dict[str, Any], record: dict[str, str], image_path: Path) -> dict[str, float]:
    """Current fourier-detector adapter semantics, retaining RID channel detail."""
    import torch
    from raven.detectors import fourier_detector

    provider = provider_info["provider"]
    scoring = provider_info["scoring_module"]
    # Same mandatory source target/mask validation as score_image().
    fourier_detector._validate_row_target_mask(
        provider, "RID", record, provider_info["_manifest"]
    )
    result = scoring.evaluate_image(torch, provider, provider_info["pipe"], image_path, steps=50)
    raw_l1 = finite_float(scoring.raw_score("RID", result), "raw_l1")
    canonical = finite_float(
        scoring.canonical_score("RID", raw_l1, result), "canonical_score"
    )
    rid_records = result.get("rid_records")
    if not isinstance(rid_records, list) or len(rid_records) != 1:
        raise RuntimeError("current RID detector did not return exactly one per-image record")
    detail = rid_records[0]
    ch0 = finite_float(detail["rid_channel_0_l1"], "rid_channel_0_l1")
    ch3 = finite_float(detail["rid_channel_3_l1"], "rid_channel_3_l1")
    channel_min = finite_float(detail["rid_channel_min_l1"], "rid_channel_min_l1")
    if channel_min != min(ch0, ch3):
        raise RuntimeError("RID channel-min score does not equal min(ch0_l1, ch3_l1)")
    if canonical != -channel_min:
        raise RuntimeError("RID canonical score is not -min(ch0_l1, ch3_l1)")
    return {
        "rid_channel_0_l1": ch0,
        "rid_channel_3_l1": ch3,
        "rid_channel_min_l1": channel_min,
        "raw_l1": raw_l1,
        "canonical_score": canonical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-scores", type=Path, required=True)
    parser.add_argument("--old-verification-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-count", type=int, default=1001)
    args = parser.parse_args()
    import torch
    if args.device != "cuda":
        raise ValueError("this verification expects CUDA; CPU fallback is intentionally disabled")
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")

    manifest_rows = load_csv(args.manifest)
    old_rows = load_csv(args.old_scores)
    if len(manifest_rows) != args.expected_count or len(old_rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} manifest and old-score rows, got "
            f"{len(manifest_rows)} and {len(old_rows)}"
        )
    manifest_ids = [str(row["run_id"]) for row in manifest_rows]
    old_ids = [str(row["run_id"]) for row in old_rows]
    if manifest_ids != old_ids:
        raise ValueError("manifest and old scores differ in run_id ordering")
    if {row.get("method") for row in manifest_rows} != {"RID"}:
        raise ValueError("manifest is not a homogeneous RID cohort")

    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "current_detector_scores.jsonl"
    progress_path = args.output_dir / "progress.json"
    current_rows: list[dict[str, Any]] = []
    cohort_values: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("rid_channel_0_l1", "rid_channel_3_l1", "rid_channel_min_l1", "canonical_score")}
        for name, _ in COHORTS
    }

    # Lazy pipe setup creates precisely one model/provider on one selected GPU.
    from raven.detectors import fourier_detector
    provider_info = fourier_detector.load_state(manifest_rows, args.device, method="RID")
    provider = provider_info["provider"]
    detector_info = {
        "method": "RID",
        "raw_detector_math_modified": False,
        "current_git_head": os.popen("git rev-parse HEAD").read().strip(),
        "score_definition": provider_info["score_definition"],
        "score_direction": "higher_is_watermarked",
        "raw_score_direction": "lower_is_watermarked",
        "pattern_sha256": provider.bundle.manifest["selected_pattern_sha256"],
        "mask_sha256": provider.bundle.manifest["mask_sha256"],
        "watermarked_channels": [0, 3],
        "channel_aggregation": "min(channel_0_complex_l1,channel_3_complex_l1)",
        "canonical_score_formula": "-min(channel_0_l1,channel_3_l1)",
        "inversion_steps": 50,
        "device": args.device,
    }

    with score_path.open("x", encoding="utf-8") as handle:
        for row_index, (record, old) in enumerate(zip(manifest_rows, old_rows), start=1):
            for cohort, path_field in COHORTS:
                image_path = Path(record[path_field])
                values = score_one(provider_info, record, image_path)
                old_key = {
                    "clean": "clean_rid_canonical_score",
                    "original_watermarked": "watermarked_rid_canonical_score",
                    "attacked_watermarked": "attacked_rid_canonical_score",
                }[cohort]
                old_score = finite_float(old[old_key], old_key)
                output = {
                    "run_id": str(record["run_id"]),
                    "cohort": cohort,
                    "image_path": str(image_path),
                    "old_canonical_score": old_score,
                    "absolute_score_difference": abs(values["canonical_score"] - old_score),
                    **values,
                }
                handle.write(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                cohort_store = cohort_values[cohort]
                for key in cohort_store:
                    cohort_store[key].append(values[key])
                current_rows.append(output)
                del values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if row_index % 10 == 0 or row_index == args.expected_count:
                progress_path.write_text(
                    json.dumps({"completed_samples": row_index, "total_samples": args.expected_count}) + "\n",
                    encoding="utf-8",
                )
                print(f"[RID re-score] {row_index}/{args.expected_count}", flush=True)

    expected_rows = args.expected_count * len(COHORTS)
    if len(current_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} score rows, got {len(current_rows)}")
    old_values = [row["old_canonical_score"] for row in current_rows]
    new_values = [row["canonical_score"] for row in current_rows]
    differences = [row["absolute_score_difference"] for row in current_rows]
    from raven.evaluation.metrics import unified_detection_report

    report = unified_detection_report(
        cohort_values["clean"]["canonical_score"],
        cohort_values["original_watermarked"]["canonical_score"],
        cohort_values["attacked_watermarked"]["canonical_score"],
        score_definition=provider_info["score_definition"],
        target_fpr=0.01,
    )
    old_verification = json.loads(args.old_verification_result.read_text(encoding="utf-8"))
    old_metric = old_verification["metric"]
    result = {
        "result_schema": "rid_current_detector_rescore_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_path(args.manifest),
        "old_scores_csv": str(args.old_scores),
        "old_scores_csv_sha256": sha256_path(args.old_scores),
        "old_verification_result": str(args.old_verification_result),
        "old_verification_result_sha256": sha256_path(args.old_verification_result),
        "expected_samples_per_cohort": args.expected_count,
        "total_rescored_images": expected_rows,
        "detector": detector_info,
        "comparison_old_vs_current": {
            "max_absolute_score_difference": max(differences),
            "mean_absolute_score_difference": statistics.fmean(differences),
            "pearson_correlation": pearson(old_values, new_values),
        },
        "historical_old_protocol": {
            "threshold": old_metric["clean_calibrated_threshold"],
            "after_tpr": old_metric["attacked_detection_rate_at_clean_calibrated_threshold"],
            "threshold_type": old_metric["threshold_type"],
        },
        "unified_evaluation": report,
        "channel_l1_statistics": {
            cohort: {
                "channel_0_l1": summary(values["rid_channel_0_l1"]),
                "channel_3_l1": summary(values["rid_channel_3_l1"]),
                "channel_min_l1": summary(values["rid_channel_min_l1"]),
            }
            for cohort, values in cohort_values.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "max_absolute_score_difference": result["comparison_old_vs_current"]["max_absolute_score_difference"],
        "mean_absolute_score_difference": result["comparison_old_vs_current"]["mean_absolute_score_difference"],
        "pearson_correlation": result["comparison_old_vs_current"]["pearson_correlation"],
        "new_threshold": report["calibrated_threshold_before"],
        "new_after_tpr": report["tpr_after"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
