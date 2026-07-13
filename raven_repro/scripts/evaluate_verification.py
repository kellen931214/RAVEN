#!/usr/bin/env python
"""Evaluate protocol-correct watermark verification from streamed score records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import bit_accuracy, summarize_detection

SEMANTIC_METHODS = {"TR", "RID", "HSTR", "HSQR"}
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--legacy-threshold", type=float, default=None)
    parser.add_argument("--expected-gs-bits", type=int, default=256)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, default=None)
    return parser


def finite_float(row: dict[str, str], column: str) -> float:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"run_id={row.get('run_id')}: missing {column}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"run_id={row.get('run_id')}: non-finite {column}: {value}")
    return parsed


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict:
    return {
        "N": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "quantiles": {f"q{int(probability * 100):02d}": quantile(values, probability) for probability in QUANTILES},
    }


def legacy_detected(method: str, raw_score: float, threshold: float) -> bool:
    if method == "TR":
        return raw_score < threshold
    if method in {"RID", "HSTR", "HSQR"}:
        return -raw_score > threshold
    raise ValueError(method)


def consistent_value(rows: list[dict[str, str]], column: str, default: str = "unspecified") -> str:
    values = {row.get(column, "").strip() for row in rows if row.get(column, "").strip()}
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f"Mixed {column} provenance: {sorted(values)}")
    return next(iter(values))


def provenance(rows: list[dict[str, str]], method: str) -> dict:
    fields = (
        "dataset", "model_id", "model_revision", "vae_id", "vae_scaling_factor",
        "scheduler", "inverse_scheduler", "steps", "resolution", "detector_dtype",
        "score_direction", "provider_parameters",
    )
    result = {field: consistent_value(rows, field) for field in fields}
    result["method"] = method
    return result


def semantic_report(method: str, rows: list[dict[str, str]], target_fpr: float, threshold_override: float | None) -> dict:
    raw = {stage: [finite_float(row, f"{stage}_raw_score") for row in rows] for stage in ("clean", "watermarked", "attacked")}
    canonical = {stage: [finite_float(row, f"{stage}_canonical_score") for row in rows] for stage in raw}
    summary = summarize_detection(canonical["clean"], canonical["watermarked"], canonical["attacked"], target_fpr)
    metric = summary.to_dict()

    recorded_thresholds = {finite_float(row, "legacy_threshold") for row in rows}
    if threshold_override is not None:
        legacy_threshold = float(threshold_override)
    elif len(recorded_thresholds) == 1:
        legacy_threshold = next(iter(recorded_thresholds))
    else:
        raise ValueError(f"Expected one legacy threshold, got {sorted(recorded_thresholds)}")
    legacy_rates = {
        stage: sum(legacy_detected(method, value, legacy_threshold) for value in values) / len(values)
        for stage, values in raw.items()
    }
    metric.update({
        "N": {stage: len(values) for stage, values in raw.items()},
        "threshold": summary.calibration.threshold,
        "target_FPR": target_fpr,
        "actual_FPR": summary.calibration.actual_fpr,
        "calibrated_TPR_at_1pct_FPR": summary.attacked_tpr if target_fpr == 0.01 else None,
        "calibrated_before_TPR": summary.watermarked_tpr,
        "calibrated_attacked_TPR": summary.attacked_tpr,
        "legacy_threshold": legacy_threshold,
        "legacy_fixed_threshold_detect_rate": legacy_rates["attacked"],
        "legacy_before_detect_rate": legacy_rates["watermarked"],
        "legacy_actual_clean_FPR": legacy_rates["clean"],
        "ROC_AUC": {"before": summary.watermarked_auc, "attacked": summary.attacked_auc},
        "score_distributions": {
            stage: {"raw": distribution(raw[stage]), "canonical": distribution(canonical[stage])}
            for stage in raw
        },
    })
    if method == "TR":
        metric["tree_ring_numeric_diagnostics"] = {
            stage: {
                "p_value_zero_count": sum(value == 0.0 for value in raw[stage]),
                "p_value_zero_rate": sum(value == 0.0 for value in raw[stage]) / len(raw[stage]),
                "reported_underflow_count": sum(str(row.get(f"{stage}_tr_p_underflow", "")).lower() == "true" for row in rows),
            }
            for stage in raw
        }
    return metric


def gs_report(rows: list[dict[str, str]], expected_bits: int) -> tuple[dict, list[dict]]:
    audited, stage_results = [], {stage: [] for stage in ("clean", "watermarked", "attacked")}
    for index, row in enumerate(rows):
        run_id = row.get("run_id") or str(index)
        ground_truth = row.get("ground_truth_bits", "")
        item = {
            "run_id": run_id,
            "key_hex": row.get("key_hex", ""),
            "nonce_hex": row.get("nonce_hex", ""),
            "offset": row.get("offset", ""),
            "ground_truth_bits": ground_truth,
        }
        for stage in stage_results:
            prediction = row.get(f"{stage}_predicted_bits", "")
            result = bit_accuracy(ground_truth, prediction, expected_length=expected_bits)
            stage_results[stage].append(result)
            item[f"{stage}_decoded_bits"] = prediction
            item[f"{stage}_bit_errors"] = result["error_indices"]
            item[f"{stage}_num_errors"] = result["num_errors"]
            item[f"{stage}_bit_accuracy"] = result["accuracy"]
        audited.append(item)

    stages = {}
    for stage, results in stage_results.items():
        total_bits = sum(result["num_bits"] for result in results)
        total_errors = sum(result["num_errors"] for result in results)
        stages[stage] = {
            "N": len(results),
            "macro_bit_accuracy": sum(result["accuracy"] for result in results) / len(results),
            "micro_bit_accuracy": 1.0 - total_errors / total_bits,
            "total_bits": total_bits,
            "total_errors": total_errors,
        }
    return {
        "N": len(rows),
        "num_bits_per_sample": expected_bits,
        "stages": stages,
        "macro_bit_accuracy_before": stages["watermarked"]["macro_bit_accuracy"],
        "macro_bit_accuracy_attacked": stages["attacked"]["macro_bit_accuracy"],
        "micro_bit_accuracy_before": stages["watermarked"]["micro_bit_accuracy"],
        "micro_bit_accuracy_attacked": stages["attacked"]["micro_bit_accuracy"],
    }, audited


def main() -> int:
    args = build_parser().parse_args()
    for output in (args.output_json, args.output_rows):
        if output is not None and output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    with args.records.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {args.records}")
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise ValueError(f"Score extraction contains {len(errors)} error rows; first={errors[0].get('error')}")
    if any((row.get("method") or "").upper() != args.method for row in rows):
        raise ValueError("Records contain a different method")

    if args.method in SEMANTIC_METHODS:
        metric, detailed = semantic_report(args.method, rows, args.target_fpr, args.legacy_threshold), []
    else:
        metric, detailed = gs_report(rows, args.expected_gs_bits)
    report = {
        "protocol_version": 2,
        "paper_comparable": args.method == "GS" or args.target_fpr == 0.01,
        "method": args.method,
        "dataset": consistent_value(rows, "dataset"),
        "records": str(args.records.resolve()),
        "provenance": provenance(rows, args.method),
        "metric": metric,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_rows is not None:
        args.output_rows.parent.mkdir(parents=True, exist_ok=True)
        args.output_rows.write_text(json.dumps(detailed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
