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

from raven.metrics import summarize_detection

SEMANTIC_METHODS = {"TR", "RID", "HSTR", "HSQR"}
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", required=True, choices=["GS", "TR", "GM", "T2S", "RID", "HSTR", "HSQR"]
    )
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
    metric = {
        "N": {stage: len(values) for stage, values in raw.items()},
        "target_fpr": target_fpr,
        "actual_empirical_fpr": summary.calibration.actual_fpr,
        "threshold": summary.calibration.threshold,
        "false_positive_count": summary.calibration.false_positives,
        "before_tpr": summary.watermarked_tpr,
        "attacked_tpr_at_original_clean_threshold": summary.attacked_tpr,
        "attacked_tpr_at_recalibrated_threshold": None,
        "before_roc_auc": summary.watermarked_auc,
        "attacked_roc_auc": summary.attacked_auc,
        "attack_success_rate_at_original_clean_threshold": 1.0 - summary.attacked_tpr,
    }

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
        "legacy_threshold": legacy_threshold,
        "legacy_fixed_threshold_before_detect_rate": legacy_rates["watermarked"],
        "legacy_fixed_threshold_attacked_detect_rate": legacy_rates["attacked"],
        "legacy_actual_clean_fpr": legacy_rates["clean"],
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



FOURIER_METHOD_CONFIG = {
    "RID": {
        "raw_metric": "rid_channel_min_l1",
        "canonical_metric": "rid_neg_channel_min_complex_l1",
        "score_definition": "rid_neg_channel_min_complex_l1",
    },
    "HSTR": {
        "raw_metric": "hstr_channel_min_l1",
        "canonical_metric": "hstr_score",
        "score_definition": "hstr_score=-min(channel_0_l1,channel_3_l1)",
    },
    "HSQR": {
        "raw_metric": "hsqr_l1_distance",
        "canonical_metric": "hsqr_score",
        "score_definition": "hsqr_negative_mean_complex_l1_distance",
    },
}


def fourier_report(method: str, rows: list[dict[str, str]], target_fpr: float, threshold_override: float | None) -> tuple[dict, list[dict]]:
    """RID/HSTR/HSQR detector family: raw L1 plus canonical higher-is-WM score."""
    config = FOURIER_METHOD_CONFIG[method]
    prefix = method.lower()
    raw = {
        stage: [finite_float(row, f"{stage}_raw_score") for row in rows]
        for stage in ("clean", "watermarked", "attacked")
    }
    canonical = {
        stage: [finite_float(row, f"{stage}_canonical_score") for row in rows]
        for stage in raw
    }
    for stage in raw:
        for index, row in enumerate(rows):
            method_raw = row.get(f"{stage}_{prefix}_raw_l1")
            method_score = row.get(f"{stage}_{prefix}_canonical_score")
            if method_raw not in (None, "") and float(method_raw) != raw[stage][index]:
                raise ValueError(f"run_id={row.get('run_id')}: {method} raw L1 column drift at {stage}")
            if method_score not in (None, "") and float(method_score) != canonical[stage][index]:
                raise ValueError(f"run_id={row.get('run_id')}: {method} canonical score column drift at {stage}")
            if canonical[stage][index] != -raw[stage][index]:
                raise ValueError(f"run_id={row.get('run_id')}: {method} canonical score is not -raw L1")
    summary = summarize_detection(
        canonical["clean"], canonical["watermarked"], canonical["attacked"], target_fpr
    )
    threshold = summary.calibration.threshold
    rates = {
        stage: sum(value >= threshold for value in values) / len(values)
        for stage, values in canonical.items()
    }
    legacy_threshold = None
    recorded_thresholds = {finite_float(row, "legacy_threshold") for row in rows}
    if threshold_override is not None:
        legacy_threshold = float(threshold_override)
    elif len(recorded_thresholds) == 1:
        legacy_threshold = next(iter(recorded_thresholds))
    legacy_rates = None
    if legacy_threshold is not None:
        legacy_rates = {
            stage: sum(legacy_detected(method, value, legacy_threshold) for value in values) / len(values)
            for stage, values in raw.items()
        }
    audited = []
    for index, row in enumerate(rows):
        item = {
            "run_id": row.get("run_id") or str(index),
            f"{prefix}_bundle_config_sha256": row.get(f"{prefix}_bundle_config_sha256", ""),
            f"{prefix}_selected_pattern_sha256": row.get(f"{prefix}_selected_pattern_sha256", ""),
            f"{prefix}_mask_sha256": row.get(f"{prefix}_mask_sha256", ""),
        }
        for stage in raw:
            item[f"{stage}_raw_l1"] = raw[stage][index]
            item[f"{stage}_canonical_score"] = canonical[stage][index]
        audited.append(item)
    metric = {
        "N": len(rows),
        "detector_metric": config["canonical_metric"],
        "raw_detector_metric": config["raw_metric"],
        "score_direction": "higher_is_watermarked",
        "raw_score_direction": "lower_is_watermarked",
        "score_definition": config["score_definition"],
        "threshold_type": "empirical_clean_1pct_fpr",
        "threshold_score_space": "canonical_score",
        "threshold_comparison_operator": ">=",
        "clean_calibrated_threshold": threshold,
        "clean_calibrated_actual_empirical_fpr": summary.calibration.actual_fpr,
        "clean_calibrated_false_positive_count": summary.calibration.false_positives,
        "target_fpr": target_fpr,
        "stages": {
            stage: {
                "N": len(values),
                "mean_raw_l1": sum(raw[stage]) / len(raw[stage]),
                "mean_canonical_score": sum(canonical[stage]) / len(canonical[stage]),
                "canonical_detection_rate_at_clean_calibrated_threshold": rates[stage],
                "raw_l1_distribution": distribution(raw[stage]),
                "canonical_score_distribution": distribution(canonical[stage]),
            }
            for stage, values in canonical.items()
        },
        "mean_canonical_score_before": sum(canonical["watermarked"]) / len(canonical["watermarked"]),
        "mean_canonical_score_attacked": sum(canonical["attacked"]) / len(canonical["attacked"]),
        "mean_raw_l1_before": sum(raw["watermarked"]) / len(raw["watermarked"]),
        "mean_raw_l1_attacked": sum(raw["attacked"]) / len(raw["attacked"]),
        "before_detection_rate_at_clean_calibrated_threshold": rates["watermarked"],
        "attacked_detection_rate_at_clean_calibrated_threshold": rates["attacked"],
        "clean_detection_rate_at_clean_calibrated_threshold": rates["clean"],
        "attack_success_rate_at_clean_calibrated_threshold": 1.0 - rates["attacked"],
        "attack_success_definition": (
            "1 - attacked detection rate at the clean-calibrated empirical "
            f"threshold on {config['canonical_metric']}"
        ),
        "before_roc_auc": summary.watermarked_auc,
        "attacked_roc_auc": summary.attacked_auc,
        "statistically_valid_for_target_fpr": len(rows) >= math.ceil(1.0 / target_fpr),
        "legacy_threshold_raw_l1": legacy_threshold,
        "legacy_threshold_comparison_operator": "raw_l1_detected_when_-raw_l1 > threshold",
        "legacy_fixed_threshold_rates": legacy_rates,
    }
    return metric, audited

def gs_report(
    rows: list[dict[str, str]], expected_bits: int, target_fpr: float = 0.01
) -> tuple[dict, list[dict]]:
    raw = {
        stage: [finite_float(row, f"{stage}_raw_score") for row in rows]
        for stage in ("clean", "watermarked", "attacked")
    }
    summary = summarize_detection(
        raw["clean"], raw["watermarked"], raw["attacked"], target_fpr
    )
    legacy_thresholds = {finite_float(row, "legacy_threshold") for row in rows}
    if len(legacy_thresholds) != 1:
        raise ValueError(f"Expected one GS legacy threshold, got {sorted(legacy_thresholds)}")
    legacy_threshold = next(iter(legacy_thresholds))
    official_onebit = {finite_float(row, "gs_official_tau_onebit") for row in rows}
    official_bits = {finite_float(row, "gs_official_tau_bits") for row in rows}
    if len(official_onebit) != 1 or len(official_bits) != 1:
        raise ValueError("Mixed official GS threshold provenance")
    tau_onebit = next(iter(official_onebit))
    tau_bits = next(iter(official_bits))

    audited = []
    for index, row in enumerate(rows):
        item = {
            "run_id": row.get("run_id") or str(index),
            "gs_secret_index": row.get("gs_secret_index", ""),
            "gs_secret_bundle_sha256": row.get("gs_secret_bundle_sha256", ""),
        }
        for stage in raw:
            item[f"{stage}_bit_accuracy"] = finite_float(row, f"{stage}_raw_score")
            item[f"{stage}_decoded_bits_sha256"] = row.get(
                f"{stage}_decoded_bits_sha256", ""
            )
        audited.append(item)

    stages = {
        stage: {
            "N": len(values),
            "macro_bit_accuracy": sum(values) / len(values),
            "distribution": distribution(values),
        }
        for stage, values in raw.items()
    }
    legacy_rates = {
        stage: sum(value > legacy_threshold for value in values) / len(values)
        for stage, values in raw.items()
    }
    official_onebit_rates = {
        stage: sum(value >= tau_onebit for value in values) / len(values)
        for stage, values in raw.items()
    }
    official_traceability_rates = {
        stage: sum(value >= tau_bits for value in values) / len(values)
        for stage, values in raw.items()
    }
    return {
        "N": len(rows),
        "num_bits_per_sample": expected_bits,
        "stages": stages,
        "macro_bit_accuracy_before": stages["watermarked"]["macro_bit_accuracy"],
        "macro_bit_accuracy_attacked": stages["attacked"]["macro_bit_accuracy"],
        "target_fpr": target_fpr,
        "clean_calibrated_threshold_at_target_fpr": summary.calibration.threshold,
        "clean_calibrated_actual_empirical_fpr": summary.calibration.actual_fpr,
        "clean_calibrated_false_positive_count": summary.calibration.false_positives,
        "before_tpr_at_clean_calibrated_threshold": summary.watermarked_tpr,
        "attacked_tpr_at_clean_calibrated_threshold": summary.attacked_tpr,
        "before_roc_auc": summary.watermarked_auc,
        "attacked_roc_auc": summary.attacked_auc,
        "legacy_fixed_threshold": legacy_threshold,
        "legacy_fixed_threshold_comparison_operator": ">",
        "legacy_fixed_threshold_rates": legacy_rates,
        "official_tau_onebit": tau_onebit,
        "official_tau_bits": tau_bits,
        "official_threshold_comparison_operator": ">=",
        "official_onebit_rates": official_onebit_rates,
        "official_traceability_rates": official_traceability_rates,
        "statistically_valid_for_target_fpr": len(rows) >= math.ceil(1.0 / target_fpr),
    }, audited


def optional_float(row: dict[str, str], column: str) -> float | None:
    """Parse a column that the detector may legitimately have left unavailable."""
    value = row.get(column)
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"run_id={row.get('run_id')}: non-finite {column}: {value}")
    return parsed


def optional_mean(rows: list[dict[str, str]], column: str) -> float | None:
    values = [optional_float(row, column) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def gm_report(rows: list[dict[str, str]], target_fpr: float) -> tuple[dict, list[dict]]:
    """GaussMarker's own detector family.

    The primary score is ``gm_raw_bit_accuracy`` — the official spatial-domain
    ChaCha20-decrypt bit accuracy — because this cohort's bundle carries neither
    the GNR restorer nor the ring classifier, so GaussMarker's official ensemble
    score does not exist for it and ``gm_provider`` correctly declines to
    fabricate one. That absence is recorded explicitly rather than papered over.

    With no official threshold available, the threshold family used here is an
    empirical one calibrated from this run's own clean-negative cohort. It is
    named ``empirical_clean_1pct_fpr`` and reported with its measured FPR; it is
    NOT GaussMarker's official decision rule and must never be presented as one.
    The frequency-domain ring L1 is reported alongside as a secondary raw score.
    """
    raw = {
        stage: [finite_float(row, f"{stage}_gm_raw_bit_accuracy") for row in rows]
        for stage in ("clean", "watermarked", "attacked")
    }
    ring_l1 = {
        stage: [finite_float(row, f"{stage}_gm_raw_ring_l1") for row in rows]
        for stage in raw
    }
    summary = summarize_detection(
        raw["clean"], raw["watermarked"], raw["attacked"], target_fpr
    )
    threshold = summary.calibration.threshold
    detection_rates = {
        stage: sum(value >= threshold for value in values) / len(values)
        for stage, values in raw.items()
    }
    restored = {stage: optional_mean(rows, f"{stage}_gm_restored_bit_accuracy") for stage in raw}
    classifier = {
        stage: optional_mean(rows, f"{stage}_gm_classifier_probability") for stage in raw
    }
    gnr_used = {str(row.get("gm_gnr_used", "")).lower() for row in rows}
    classifier_used = {str(row.get("gm_classifier_used", "")).lower() for row in rows}
    if gnr_used != {"false"} or classifier_used != {"false"}:
        raise ValueError(
            "GM rows disagree about GNR/classifier availability; the ensemble "
            f"score family would apply: gnr={sorted(gnr_used)} "
            f"classifier={sorted(classifier_used)}"
        )
    audited = [
        {
            "run_id": row.get("run_id") or str(index),
            "gm_bundle_config_sha256": row.get("gm_bundle_config_sha256", ""),
            **{
                f"{stage}_{name}": finite_float(row, f"{stage}_gm_raw_{suffix}")
                for stage in raw
                for name, suffix in (("bit_accuracy", "bit_accuracy"), ("ring_l1", "ring_l1"))
            },
        }
        for index, row in enumerate(rows)
    ]
    return {
        "N": len(rows),
        "detector_metric": "gm_raw_bit_accuracy",
        "score_direction": "higher_is_watermarked",
        "score_definition": consistent_value(rows, "gm_score_definition"),
        "detector_report_label": consistent_value(rows, "gm_report_label"),
        "secondary_detector_metric": "gm_raw_ring_l1",
        "stages": {
            stage: {
                "N": len(values),
                "macro_bit_accuracy": sum(values) / len(values),
                "mean_ring_l1": sum(ring_l1[stage]) / len(ring_l1[stage]),
                "distribution": distribution(values),
                "ring_l1_distribution": distribution(ring_l1[stage]),
                "mean_restored_bit_accuracy": restored[stage],
                "mean_classifier_probability": classifier[stage],
            }
            for stage, values in raw.items()
        },
        "macro_bit_accuracy_before": sum(raw["watermarked"]) / len(raw["watermarked"]),
        "macro_bit_accuracy_attacked": sum(raw["attacked"]) / len(raw["attacked"]),
        "target_fpr": target_fpr,
        "threshold_type": "empirical_clean_1pct_fpr",
        "threshold_comparison_operator": ">=",
        "clean_calibrated_threshold": threshold,
        "clean_calibrated_actual_empirical_fpr": summary.calibration.actual_fpr,
        "clean_calibrated_false_positive_count": summary.calibration.false_positives,
        "before_detection_rate_at_clean_calibrated_threshold": detection_rates["watermarked"],
        "attacked_detection_rate_at_clean_calibrated_threshold": detection_rates["attacked"],
        "clean_detection_rate_at_clean_calibrated_threshold": detection_rates["clean"],
        "attack_success_rate_at_clean_calibrated_threshold": 1.0 - detection_rates["attacked"],
        "attack_success_definition": (
            "1 - attacked detection rate at the clean-calibrated empirical "
            "threshold on gm_raw_bit_accuracy"
        ),
        "before_roc_auc": summary.watermarked_auc,
        "attacked_roc_auc": summary.attacked_auc,
        "official_ensemble_threshold_available": False,
        "official_ensemble_threshold_unavailable_reason": (
            "the cohort's GM bundle carries no GNR restorer and no ring "
            "classifier, so GaussMarker's official ensemble score and its "
            "official decision threshold do not exist for this cohort"
        ),
        "statistically_valid_for_target_fpr": len(rows) >= math.ceil(1.0 / target_fpr),
    }, audited


def t2s_report(rows: list[dict[str, str]], target_fpr: float) -> tuple[dict, list[dict]]:
    """T2SMark's own detector family.

    The primary score is ``score_true_key`` (upstream ``norm1_w``), and T2S's own
    stored decision rule is ``paired_key_comparison``:
    ``score_true_key > score_control_key`` per image, where the control key is
    upstream's ``fake_key = 1 - master_key``. The threshold is therefore
    per-sample, not a cohort scalar, and no scalar is invented for it.

    That rule is a RAVEN deployment extension — upstream evaluates a cohort ROC
    and defines no per-image decision — and the provenance strings recorded by
    the detector say so. It is not TPR@1%FPR. A clean-calibrated empirical
    threshold on score_true_key is reported separately as a secondary family so
    the two are never conflated.
    """
    stages = ("clean", "watermarked", "attacked")
    true_key = {
        stage: [finite_float(row, f"{stage}_t2s_score_true_key") for row in rows]
        for stage in stages
    }
    control_key = {
        stage: [finite_float(row, f"{stage}_t2s_score_control_key") for row in rows]
        for stage in stages
    }
    paired_rates = {}
    for stage in stages:
        recorded = [
            str(row.get(f"{stage}_t2s_detection_success", "")).lower() for row in rows
        ]
        if set(recorded) - {"true", "false"}:
            raise ValueError(f"missing T2S detection decision for stage {stage}")
        computed = [
            first > second
            for first, second in zip(true_key[stage], control_key[stage])
        ]
        if [str(value).lower() for value in computed] != recorded:
            raise ValueError(
                f"stage {stage}: stored T2S decision disagrees with "
                "score_true_key > score_control_key"
            )
        paired_rates[stage] = sum(computed) / len(computed)
    summary = summarize_detection(
        true_key["clean"], true_key["watermarked"], true_key["attacked"], target_fpr
    )
    margins = {
        stage: [
            first - second for first, second in zip(true_key[stage], control_key[stage])
        ]
        for stage in stages
    }
    audited = [
        {
            "run_id": row.get("run_id") or str(index),
            "t2s_watermark_id": row.get("t2s_watermark_id", ""),
            "t2s_state_sha256": row.get("t2s_state_sha256", ""),
            **{
                f"{stage}_{name}": finite_float(row, f"{stage}_t2s_{name}")
                for stage in stages
                for name in ("score_true_key", "score_control_key", "score_margin")
            },
        }
        for index, row in enumerate(rows)
    ]
    return {
        "N": len(rows),
        "detector_metric": "t2s_score_true_key",
        "score_direction": "higher_is_watermarked",
        "threshold_type": "paired_key_comparison_control_key",
        "threshold_comparison_operator": ">",
        # Deliberately null: the comparand is each sample's own control-key
        # score, so there is no cohort-wide scalar threshold to report.
        "threshold": None,
        "decision_rule": consistent_value(rows, "t2s_decision_rule"),
        "decision_rule_provenance": (
            "RAVEN paired_key_comparison deployment extension; upstream T2SMark "
            "evaluates a cohort ROC and defines no per-image decision rule"
        ),
        "not_claimed": "this is not TPR at a calibrated 1% FPR",
        "stages": {
            stage: {
                "N": len(true_key[stage]),
                "mean_score_true_key": sum(true_key[stage]) / len(true_key[stage]),
                "mean_score_control_key": sum(control_key[stage]) / len(control_key[stage]),
                "mean_score_margin": sum(margins[stage]) / len(margins[stage]),
                "paired_key_comparison_detection_rate": paired_rates[stage],
                "mean_key_accuracy": optional_mean(rows, f"{stage}_t2s_key_accuracy"),
                "mean_message_accuracy": optional_mean(rows, f"{stage}_t2s_message_accuracy"),
                "distribution": distribution(true_key[stage]),
            }
            for stage in stages
        },
        "mean_score_true_key_before": sum(true_key["watermarked"]) / len(true_key["watermarked"]),
        "mean_score_true_key_attacked": sum(true_key["attacked"]) / len(true_key["attacked"]),
        "before_detection_rate_at_paired_key_comparison": paired_rates["watermarked"],
        "attacked_detection_rate_at_paired_key_comparison": paired_rates["attacked"],
        "empirical_clean_fpr_at_paired_key_comparison": paired_rates["clean"],
        "attack_success_rate_at_paired_key_comparison": 1.0 - paired_rates["attacked"],
        "attack_success_definition": (
            "1 - attacked detection rate under the stored T2S "
            "paired_key_comparison rule (score_true_key > score_control_key)"
        ),
        "before_roc_auc": summary.watermarked_auc,
        "attacked_roc_auc": summary.attacked_auc,
        # Secondary, clearly separated from the stored per-image rule above.
        "target_fpr": target_fpr,
        "secondary_threshold_type": "empirical_clean_1pct_fpr",
        "secondary_clean_calibrated_threshold": summary.calibration.threshold,
        "secondary_clean_calibrated_actual_empirical_fpr": summary.calibration.actual_fpr,
        "secondary_before_tpr": summary.watermarked_tpr,
        "secondary_attacked_tpr": summary.attacked_tpr,
        "statistically_valid_for_target_fpr": len(rows) >= math.ceil(1.0 / target_fpr),
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

    if args.method == "TR":
        metric, detailed = semantic_report(args.method, rows, args.target_fpr, args.legacy_threshold), []
    elif args.method in {"RID", "HSTR", "HSQR"}:
        metric, detailed = fourier_report(args.method, rows, args.target_fpr, args.legacy_threshold)
    elif args.method == "GM":
        metric, detailed = gm_report(rows, args.target_fpr)
    elif args.method == "T2S":
        metric, detailed = t2s_report(rows, args.target_fpr)
    else:
        metric, detailed = gs_report(rows, args.expected_gs_bits, args.target_fpr)
    report = {
        "protocol_version": 2,
        "paper_comparable": (
            args.target_fpr == 0.01
            and (
                args.method not in {"GS", "GM", "T2S"}
                or len(rows) >= math.ceil(1.0 / args.target_fpr)
            )
        ),
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
