"""Focused, CPU-only checks for RAVEN's unified RID paper-table protocol."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "raven_repro"))

from eval_bench_wm.utils.wm import runner_common  # noqa: E402
from raven.detectors.fourier_detector import aggregate  # noqa: E402
from raven.evaluation.metrics import unified_detection_report  # noqa: E402
from raven.evaluation.scoring import canonical_score, raw_score  # noqa: E402


SCORE_DEFINITION = "rid_neg_channel_min_complex_l1"


def _row(cohort: str, score: float) -> dict:
    return {
        "status": "scored",
        "evaluation_cohort": cohort,
        # The raw RingID detector is an L1 distance (lower is watermarked).
        # The evaluator must only negate it at the canonical score boundary.
        "raw_score": -score,
        "canonical_score": score,
        "score_definition": SCORE_DEFINITION,
    }


def test_raw_rid_score_convention_is_unchanged_at_evaluation_boundary() -> None:
    raw_l1 = 12.3456789
    result = {"l1_dist": [raw_l1]}
    assert raw_score("RID", result) == raw_l1
    assert canonical_score("RID", raw_l1, result) == -raw_l1


def test_unified_rid_protocol_uses_shared_strict_roc_and_freezes_threshold() -> None:
    clean = [0.40, 0.30, 0.20, 0.10]
    watermarked = [0.60, 0.50, 0.45]
    attacked = [0.55, 0.44, 0.20]

    expected = runner_common.official_roc(
        watermarked, clean, 0.01, score_definition=SCORE_DEFINITION,
    )
    report = unified_detection_report(
        clean, watermarked, attacked, score_definition=SCORE_DEFINITION,
    )

    assert report["clean_scores"] == clean
    assert report["watermarked_scores"] == watermarked
    assert report["fpr_rule"] == "strict_less_than"
    assert report["calibrated_threshold_before"] == expected["threshold"]
    assert report["actual_fpr_before"] == expected["empirical_fpr"]
    assert report["tpr_before"] == expected["empirical_tpr"]
    assert report["actual_fpr_before"] < report["fpr_target"]
    assert report["fixed_threshold_after"] == report["calibrated_threshold_before"]
    assert report["tpr_after"] == pytest.approx(1.0 / 3.0)
    assert report["attack_success_rate"] == pytest.approx(2.0 / 3.0)


def test_strict_fpr_excludes_an_operating_point_at_exactly_one_percent() -> None:
    # At threshold 0.9, exactly one of 100 clean negatives is positive.  A
    # <= 1% policy could then advance to threshold 0.8 and get TPR=1; the
    # unified benchmark's strict policy must retain only the FPR=0 ROC point.
    clean = [0.9] + [0.0] * 99
    report = unified_detection_report(
        clean, [0.8], [0.8], score_definition=SCORE_DEFINITION,
    )

    assert math.isinf(report["calibrated_threshold_before"])
    assert report["actual_fpr_before"] == 0.0
    assert report["tpr_before"] == 0.0
    assert report["fpr_rule"] == "strict_less_than"


def test_rid_aggregate_serializes_clean_negative_unified_report() -> None:
    clean = [0.40, 0.30, 0.20, 0.10]
    watermarked = [0.60, 0.50, 0.45]
    attacked = [0.55, 0.44, 0.20]
    rows = (
        [_row("original_clean", score) for score in clean]
        + [_row("original_watermarked", score) for score in watermarked]
        + [_row("attacked_watermarked", score) for score in attacked]
    )

    result = aggregate(rows, method="RID")
    report = result["unified_evaluation"]

    assert report["evaluation_protocol"] == "unified_clean_negative_tpr_at_fpr_1pct"
    assert result["evaluation_protocol"] == report["evaluation_protocol"]
    assert result["threshold_policy"] == report["threshold_policy"]
    assert report["threshold_policy"] == "calibrate_before_attack_then_freeze"
    assert report["clean_scores"] == clean
    assert report["watermarked_scores"] == watermarked
    assert report["attacked_watermarked_scores"] == attacked
    assert report["fixed_threshold_after"] == report["calibrated_threshold_before"]
    assert result["detection_summary"]["attacked_watermarked_tpr"] == report["tpr_after"]
