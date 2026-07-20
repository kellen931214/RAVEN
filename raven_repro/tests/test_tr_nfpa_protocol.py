import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "raven_nfpa_tr_eval.py"
    spec = importlib.util.spec_from_file_location("formal_nfpa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nfpa = load_module()


def test_nfpa_round2_matches_numpy_golden_values():
    assert nfpa.nfpa_round2(1.234) == 1.23
    assert nfpa.nfpa_round2(1.235) == float(nfpa.np.asarray(1.235, dtype=nfpa.np.float64).round(2))


def test_nfpa_strict_threshold_and_rounded_ties_report_actual_fpr():
    assert nfpa.nfpa_detection_rate([0.99, 1.0, 1.01], 1.0) == pytest.approx(1 / 3)
    rows = []
    for index in range(100):
        base = 1.0 if index < 3 else 2.0 + index
        rows.append({
            "original_clean_l1_nfpa_rounded2": base,
            "watermarked_l1_nfpa_rounded2": 0.0,
            "attacked_clean_l1_nfpa_rounded2": base,
            "attacked_watermarked_l1_nfpa_rounded2": 1.0,
        })
    result = nfpa.aggregate_nfpa_protocol(rows, "nfpa_rounded2")
    assert result["before_threshold"] == 1.0
    assert result["before_actual_fpr"] == 0.0
    assert result["before_false_positives"] == 0
    assert result["original_clean_threshold"] == 1.0
    assert result["original_clean_actual_fpr"] == 0.0
    assert result["original_clean_fp_count"] == 0
    assert result["attacked_clean_recalibrated_threshold"] == 1.0
    assert result["attacked_clean_actual_fpr"] == 0.0
    assert result["attacked_clean_fp_count"] == 0
