import math

import pytest

from raven.metrics import (
    bit_accuracy,
    calibrate_threshold,
    canonical_watermark_score,
    detection_rate,
    summarize_detection,
)


def test_canonical_score_directions():
    assert canonical_watermark_score("TR", 0.02) == -0.02
    assert canonical_watermark_score("RID", 12.0) == -12.0
    assert canonical_watermark_score("GS", 0.75) == 0.75


def test_threshold_respects_false_positive_budget_and_ties():
    calibration = calibrate_threshold([10, 9, 9, 8, 7], target_fpr=0.4)
    assert calibration.max_false_positives == 2
    assert calibration.false_positives == 1
    assert calibration.actual_fpr == 0.2
    assert detection_rate([10, 9], calibration.threshold) == 0.5


def test_one_percent_calibration_for_one_thousand_unique_scores():
    calibration = calibrate_threshold(list(range(1000)), target_fpr=0.01)
    assert calibration.false_positives == 10
    assert calibration.actual_fpr == pytest.approx(0.01)


def test_detection_summary_uses_clean_negatives():
    summary = summarize_detection(
        clean_scores=list(range(100)),
        watermarked_scores=[200, 201],
        attacked_scores=[200, -1],
        target_fpr=0.01,
    )
    assert summary.calibration.false_positives == 1
    assert summary.watermarked_tpr == 1.0
    assert summary.attacked_tpr == 0.5


def test_bit_accuracy_reports_error_indices():
    result = bit_accuracy("0011", "0111", expected_length=4)
    assert result == {"num_bits": 4, "num_errors": 1, "accuracy": 0.75, "error_indices": [1]}


def test_empty_clean_scores_are_rejected():
    with pytest.raises(ValueError):
        calibrate_threshold([], target_fpr=0.01)
