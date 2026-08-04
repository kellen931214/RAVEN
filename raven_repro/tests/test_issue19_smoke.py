"""Issue #19 regression tests — enforce detector score contracts and metric
cohort completeness.

All tests directly call ``evaluate_detector(...)`` with mocked ``score_image``
and check:

- row status in detector_records.jsonl
- aggregate result (scored_count, failed_count, cohort_counts)
- stage status
- metric_availability
- missing_scoring_cohorts / missing_metric_cohorts

Run:  pytest -q raven_repro/tests/test_issue19_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_record(run_id="1", role="watermarked", method="TR", **kw):
    return {
        "run_id": run_id,
        "role": role,
        "method": method,
        "input_path": kw.get("input_path", f"/tmp/in_{run_id}.png"),
        "output_path": f"/tmp/out/{role}/{run_id}/output.png",
        "prompt": kw.get("prompt", ""),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -24.0,
        "debug_info_path": "",
        "debug_info_retained": False,
        "source_metadata": kw.get("source_metadata", {}),
    }


def _write_fake_run(tmp_path, method="TR", records=None):
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = tmp_path / "run"
    out.mkdir()
    cfg = {"method": method, "dataset": "test"}
    write_config(out, cfg)
    if records is None:
        records = [_make_record("1", "watermarked", method=method)]
    for r in records:
        role = r.get("role", "watermarked")
        rid = r["run_id"]
        write_record(out, role, rid, r)
        img = out / "samples" / role / rid / "output.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")
    rebuild_records_jsonl(out)
    return out


def _read_detector_rows(output_dir):
    from raven.experiment_io import detector_records_path
    path = detector_records_path(output_dir)
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


TR_META = {"w_seed": "99", "w_channel": "3", "w_radius": "10",
           "w_pattern": "ring", "w_mask_shape": "circle",
           "w_measurement": "l1_complex", "w_injection": "complex"}

T2S_META = {"t2s_state_path": "/tmp/fake.pt", "t2s_state_sha256": "abc",
            "t2s_provider_config_sha256": "def", "t2s_protocol_mode": "official"}


# ---------------------------------------------------------------------------
# Score contract validation — threshold-based methods
# ---------------------------------------------------------------------------
class TestScoreContractThreshold:
    """Threshold-based method score validation (TR, GS, GM, RID, HSTR, HSQR)."""

    @staticmethod
    def _patch_tr(monkeypatch, fake_score_fn):
        import raven.detectors.tr_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_none_result_fails_scoring(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING, ROW_STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: None)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2  # orig_wm + att_wm entries
            assert result["available"] is False

            rows = _read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_FAILED_SCORING for r in rows)
            assert all("None" in r.get("error", "") for r in rows)

    def test_empty_dict_fails_scoring(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING, ROW_STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

            rows = _read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_FAILED_SCORING for r in rows)
            assert all("raw_score" in r.get("error", "") for r in rows)

    def test_missing_canonical_score_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {"raw_score": 0.001})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

    def test_missing_raw_score_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch,
                       lambda *a, **kw: {"canonical_score": 10.0})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

    def test_nan_score_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": float("nan"), "canonical_score": float("nan"),
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

    def test_inf_score_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": float("inf"), "canonical_score": float("inf"),
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

    def test_non_numeric_score_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": "not_a_number", "canonical_score": "also_bad",
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["scored_count"] == 0

    def test_valid_score_passes(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import ROW_STATUS_SCORED

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")

            assert result["scored_count"] == 2
            assert result["failed_count"] == 0

            rows = _read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
            assert all(r.get("raw_score") == 0.001 for r in rows)
            assert all(r.get("canonical_score") == 10.0 for r in rows)


# ---------------------------------------------------------------------------
# T2S score contract validation
# ---------------------------------------------------------------------------
class TestScoreContractT2S:
    """T2S-specific score validation."""

    @staticmethod
    def _patch_t2s(monkeypatch, fake_score_fn):
        import raven.detectors.t2s_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_valid_t2s_passes(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.85, "canonical_score": 0.85,
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_score_margin": 0.45,
            "t2s_detection_success": True,
            "t2s_key_accuracy": 1.0,
            "t2s_bit_accuracy": 0.98,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")

            assert result["scored_count"] == 2
            assert result["failed_count"] == 0

    def test_nan_true_key_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "raw_score": float("nan"), "canonical_score": float("nan"),
            "t2s_score_true_key": float("nan"),
            "t2s_score_control_key": 0.40,
            "t2s_score_margin": float("nan"),
            "t2s_detection_success": False,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0

    def test_inf_control_key_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": float("inf"),
            "t2s_detection_success": True,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["scored_count"] == 0

    def test_missing_detection_success_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["scored_count"] == 0

    def test_accuracy_out_of_range_fails(self, monkeypatch):
        """Accuracy fields must be in [0, 1]."""
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": True,
            "t2s_key_accuracy": 2.5,  # > 1
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2

    def test_negative_bit_accuracy_fails(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": True,
            "t2s_bit_accuracy": -0.5,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["scored_count"] == 0

    def test_t2s_without_clean_cohort_ok(self, monkeypatch):
        """T2S does NOT need original_clean cohort."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        rec_wm = _make_record("1", "watermarked", method="T2S",
                               source_metadata=T2S_META)
        rec_wm2 = _make_record("2", "watermarked", method="T2S",
                                source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": True,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2], out,
                                       "T2S", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            ma = result["metric_availability"]
            assert ma["primary_report_available"] is True
            assert ma["threshold_report_available"] is False
            # T2S has no threshold-based report
            assert ma["primary_report"] == "paired_key_detection_report"

    # ---- Focused: detection_success type validation ----
    def test_detection_success_string_false_fails(self, monkeypatch):
        """t2s_detection_success='false' (string) → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": "false",  # string, not bool
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0

    def test_detection_success_int_1_fails(self, monkeypatch):
        """t2s_detection_success=1 (int) → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": 1,  # int, not bool
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0

    def test_detection_success_none_fails(self, monkeypatch):
        """t2s_detection_success=None → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": None,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0

    def test_margin_nan_with_valid_keys_fails(self, monkeypatch):
        """t2s_score_margin=NaN but true/control keys valid → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": True,
            "t2s_score_margin": float("nan"),
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0

    def test_margin_inf_with_valid_keys_fails(self, monkeypatch):
        """t2s_score_margin=inf but true/control keys valid → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="T2S",
                           source_metadata=T2S_META)
        self._patch_t2s(monkeypatch, lambda *a, **kw: {
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_detection_success": True,
            "t2s_score_margin": float("inf"),
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec])
            result = evaluate_detector([rec], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0


# ---------------------------------------------------------------------------
# Metric cohort completeness — threshold-based methods
# ---------------------------------------------------------------------------
class TestMetricCohortCompleteness:
    """Cohort-based metric availability for threshold detectors."""

    @staticmethod
    def _patch_tr(monkeypatch, fake_score_fn):
        import raven.detectors.tr_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_missing_original_clean_blocks_threshold_report(self, monkeypatch):
        """No original_clean → threshold_report unavailable, stage completed_with_errors."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED_WITH_ERRORS

        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                                source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2], out,
                                       "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED_WITH_ERRORS
            assert result["scored_count"] == 4  # wm×2 each → 4 entries
            assert result["available"] is True  # any_report available
            ma = result["metric_availability"]
            assert ma["threshold_report_available"] is False
            assert ma["threshold_report"] is None
            assert ma["any_report_available"] is True
            assert "original_clean" in result["missing_metric_cohorts"]

    def test_complete_three_cohort_threshold_ok(self, monkeypatch):
        """clean + watermarked + attacked → completed, threshold_report available."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                                source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                       out, "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["failed_count"] == 0
            ma = result["metric_availability"]
            assert ma["primary_report_available"] is True
            assert ma["threshold_report_available"] is True
            assert ma["recalibrated_report_available"] is True
            assert "original_clean" in ma["scored_cohorts"]
            assert "original_watermarked" in ma["scored_cohorts"]
            assert "attacked_watermarked" in ma["scored_cohorts"]
            assert result["missing_metric_cohorts"] == []

    def test_no_attacked_clean_does_not_block_original_report(self, monkeypatch):
        """attacked_clean missing state, but 3 primary cohorts succeed → completed.

        Uses evaluation_entry to make ONLY attacked_clean raise
        DetectorMissingStateError while original_clean, original_watermarked,
        attacked_watermarked all score successfully.  Missing state in the
        optional cohort is a soft failure — it does not downgrade the primary
        report.  Hard failures (scoring_error, etc.) in optional cohorts
        would still propagate.
        """
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_COMPLETED, ROW_STATUS_FAILED_MISSING_STATE,
            DetectorMissingStateError,
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)

        def fake_score(provider_info, image_path, *,
                       record=None, evaluation_entry=None, steps=50):
            if (evaluation_entry is not None
                    and evaluation_entry.get("evaluation_cohort") == "attacked_clean"):
                raise DetectorMissingStateError(
                    "optional detector state missing for attacked_clean")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        self._patch_tr(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")

            # Stage must be completed — primary cohorts all OK,
            # optional failure is soft (missing_required_state only)
            assert result["status"] == STATUS_COMPLETED, (
                f"expected completed, got {result['status']}"
            )
            # Primary counts: 3 primary entries (orig_clean + orig_wm + att_wm)
            assert result["primary_scored_count"] == 3
            assert result["primary_failed_count"] == 0
            # Optional: 1 attacked_clean entry failed
            assert result["optional_requested_count"] == 1
            assert result["optional_scored_count"] == 0
            assert result["optional_failed_count"] == 1
            # optional_metrics_incomplete flag set
            assert result.get("optional_metrics_incomplete") is True

            ma = result["metric_availability"]
            assert ma["threshold_report_available"] is True
            assert ma["primary_report_available"] is True
            # Recalibrated cohorts ARE available (attacked_clean exists in image_index)
            # but NO rows scored → recalibrated_report_available = False
            assert ma["recalibrated_report_available"] is False

            # Check detector_records.jsonl
            rows = _read_detector_rows(out)
            statuses = {(r["evaluation_cohort"], r["status"]) for r in rows}
            assert ("original_clean", "scored") in statuses
            assert ("original_watermarked", "scored") in statuses
            assert ("attacked_watermarked", "scored") in statuses
            assert ("attacked_clean", ROW_STATUS_FAILED_MISSING_STATE) in statuses

    def test_cannot_convert_to_float_fails(self, monkeypatch):
        """String that can't convert to float → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": "hello", "canonical_score": "world",
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING


# ---------------------------------------------------------------------------
# Stage status semantics
# ---------------------------------------------------------------------------
class TestStageStatus:
    """Stage-level status based on score validity + metric completeness."""

    @staticmethod
    def _patch_tr(monkeypatch, fake_score_fn):
        import raven.detectors.tr_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_zero_scores_is_failed_scoring(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR", source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: None)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])
            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING

    def test_partial_failure_with_sufficient_data_is_completed_with_errors(self, monkeypatch):
        """Some scores fail (missing state) but enough remain for primary metrics.

        Uses DetectorMissingStateError so failures are ``missing_required_state``
        which softens to ``completed_with_errors`` when primary report is
        available (Issue #25 D.4).  Scoring errors (None return) would NOT
        soften and would produce ``failed_scoring`` instead.
        """
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_COMPLETED_WITH_ERRORS, DetectorMissingStateError,
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                                source_metadata=TR_META)

        call_count = [0]

        def partial_fail(*a, **kw):
            call_count[0] += 1
            if call_count[0] >= 5:  # Fail some entries → missing state
                raise DetectorMissingStateError("missing state for this row")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        self._patch_tr(monkeypatch, partial_fail)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                       out, "TR", device="cpu")

            assert result["failed_count"] > 0
            assert result["status"] == STATUS_COMPLETED_WITH_ERRORS
            assert result["available"] is True

    def test_all_required_cohorts_and_zero_failures_is_completed(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                                source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                       out, "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["failed_count"] == 0
            assert result["available"] is True


# ---------------------------------------------------------------------------
# Output structure — required fields present
# ---------------------------------------------------------------------------
class TestOutputStructure:
    """Verify all required fields exist in the aggregate result."""

    @staticmethod
    def _patch_tr(monkeypatch, fake_score_fn):
        import raven.detectors.tr_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_all_required_aggregate_fields(self, monkeypatch):
        from experiments.eval import evaluate_detector

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)
        self._patch_tr(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")

            # Required top-level fields
            for field in ("stage", "method", "status", "available",
                          "requested_count", "scored_count", "failed_count",
                          "cohort_counts", "metric_availability",
                          "missing_scoring_cohorts", "missing_metric_cohorts",
                          "primary_requested_count", "primary_scored_count",
                          "primary_failed_count",
                          "optional_requested_count", "optional_scored_count",
                          "optional_failed_count"):
                assert field in result, f"Missing top-level field: {field}"

            ma = result["metric_availability"]
            for field in ("scored_cohorts", "cohort_counts",
                          "primary_report_available", "any_report_available",
                          "threshold_report_available",
                          "recalibrated_cohorts_available",
                          "recalibrated_report_available"):
                assert field in ma, f"Missing metric_availability field: {field}"


# ---------------------------------------------------------------------------
# Recalibrated report availability — must reflect actual aggregate output
# ---------------------------------------------------------------------------
class TestRecalibratedReport:
    """recalibrated_report_available must only be True when aggregate
    actually contains a recalibrated result block."""

    @staticmethod
    def _patch_gs(monkeypatch, fake_score_fn):
        import raven.detectors.gs_detector as mod
        monkeypatch.setattr(mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_gs_cohorts_ok_but_no_recalibrated_report(self, monkeypatch):
        """GS adapter has no recalibrated block → recalibrated_report_available=False."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        GS_META = {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"}
        rec_clean = _make_record("1", "clean", method="GS",
                                  source_metadata=GS_META)
        rec_wm = _make_record("1", "watermarked", method="GS",
                               source_metadata=GS_META)

        self._patch_gs(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.8, "canonical_score": 0.8,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            ma = result["metric_availability"]
            # All 4 cohorts scored successfully
            assert ma["threshold_report_available"] is True
            # Recalibrated cohorts ARE present
            assert ma["recalibrated_cohorts_available"] is True
            # But GS adapter emits no recalibrated block
            assert ma["recalibrated_report_available"] is False
            assert "recalibrated_unavailable_reason" in ma

    def test_tr_with_recalibrated_data_shows_available(self, monkeypatch):
        """TR aggregate with tr_recalibrated → recalibrated_report_available=True
        only if recalibrated_metrics_available is True."""
        from experiments.eval import evaluate_detector

        rec_clean = _make_record("1", "clean", method="TR",
                                  source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                               source_metadata=TR_META)

        # TR aggregate sets recalibrated_metrics_available based on
        # attacked_clean cohort existing + summarize_detection succeeding.
        # With this mock (all 4 cohorts → valid scores), the TR aggregate
        # will compute detection_summary from clean+wm+att but
        # tr_recalibrated needs attacked_clean + clean both non-empty.
        # Our mock returns valid scores for everything, so recalibrated
        # should be computed.
        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(tr_mod, "score_image",
                            lambda *a, **kw: {"raw_score": 0.001,
                                              "canonical_score": 10.0})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")

            ma = result["metric_availability"]
            assert ma["recalibrated_cohorts_available"] is True
            # TR actually emitted tr_recalibrated with valid metrics
            assert ma["recalibrated_report_available"] is True
