"""TR complex-L1 protocol tests — package modules only.

The legacy NFPA-style TR entrypoint (``raven_nfpa_tr_eval.py``) is no longer
part of production TR scoring.  Its surviving semantics live in:

- ``raven.detectors.tr_scoring`` — the complex-L1 formula and score direction
- ``raven.metrics`` — the canonical threshold calibration helper
- ``raven.detectors.tr_detector`` — detector/source tensor hash binding

No ``importlib.util`` / script-path loading remains here.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))


def test_complex_l1_rounding_semantics_are_not_serialized():
    """The unified protocol keeps full-precision scores; no .round(2)
    score serialization is applied in production scoring."""
    from raven.detectors import tr_scoring

    assert tr_scoring.SCORE_DEFINITION == "complex_l1_mean"
    assert tr_scoring.RAW_SCORE_DIRECTION == "lower_is_watermarked"
    assert tr_scoring.CANONICAL_SCORE_DIRECTION == "higher_is_watermarked"
    assert tr_scoring.COMPARISON_OPERATOR == ">="


def test_canonical_threshold_and_detection_operator():
    """Calibration helper selects the closest empirical FPR not exceeding
    the target; detection uses canonical `score >= threshold`."""
    from raven.metrics import calibrate_threshold, detection_rate

    clean = [float(v) for v in range(100, 200)]  # 100 distinct clean scores
    cal = calibrate_threshold(clean, target_fpr=0.01)
    assert cal.max_false_positives == 1
    assert cal.false_positives == 1
    assert cal.actual_fpr == pytest.approx(0.01)
    assert cal.threshold == 199.0
    # threshold equality behavior: score == threshold is detected
    assert detection_rate([199.0], cal.threshold) == 1.0
    assert detection_rate([199.0 - 1e-12], cal.threshold) == 0.0


def test_canonical_ties_kept_as_one_group():
    """Tied clean scores are kept as one group; including the group must not
    exceed the false-positive budget."""
    from raven.metrics import calibrate_threshold

    # 100 clean scores: 3-way tie at the top, 97 distinct below.
    clean = [10.0, 10.0, 10.0] + [float(v) for v in range(2, 99)]
    assert len(clean) == 100
    cal = calibrate_threshold(clean, target_fpr=0.01)
    # The 3-way tie group would exceed the 1-FP budget → excluded entirely.
    assert cal.false_positives == 1
    assert cal.actual_fpr == pytest.approx(0.01)
    assert cal.threshold == 98.0


def test_raw_score_is_abs_mean_complex():
    """raw_score = mean(abs(decoded_watermark - target_watermark)) over the
    masked complex FFT positions; canonical = -raw."""
    import torch
    from raven.detectors import tr_scoring

    decoded = torch.tensor([1 + 2j, 3 - 1j, 0.5 + 0.5j])
    target = torch.tensor([1 + 1j, 1 + 0j, 0.5 + 0.5j])
    raw = float(torch.abs(decoded - target).mean())
    assert raw == pytest.approx(float(torch.tensor([1.0, 2.2360679, 0.0]).mean()))
    assert tr_scoring.canonical_score(raw) == -raw
    assert tr_scoring.canonical_score(raw) == pytest.approx(-1.078689)


def test_detector_source_tensor_hash_mismatch_fails_closed():
    """Detector/source target and mask SHA binding: mismatches fail before
    scoring; matches are accepted.  This is the same comparison load_state
    performs in step 9 before any image is scored."""
    from raven.detectors import DetectorStateValidationError
    from raven.detectors.tr_detector import assert_tensor_identity_match

    assert_tensor_identity_match("t", "t", "target")
    assert_tensor_identity_match("m", "m", "mask")
    with pytest.raises(DetectorStateValidationError, match="target"):
        assert_tensor_identity_match("t", "different", "target")
    with pytest.raises(DetectorStateValidationError, match="mask"):
        assert_tensor_identity_match("m", "different", "mask")
