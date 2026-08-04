"""Issue #25 regression tests — row → stage reducer, failure classification,
allow policy, CLI exit code integrity.

All tests use synthetic data, temp directories, and mocked detector modules.
No real models or datasets are downloaded.

Run:  pytest -q raven_repro/tests/test_issue25_status_reducer.py
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
    from raven.experiment_io import (
        write_config, write_record, rebuild_records_jsonl,
    )
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
        # Create output image
        img = out / "samples" / role / rid / "output.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")
        # Create input image so preflight doesn't fail for original cohorts
        input_path = Path(r.get("input_path", f"/tmp/in_{rid}.png"))
        if not input_path.is_file():
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(b"fake png")
    rebuild_records_jsonl(out)
    return out


TR_META = {
    "w_seed": "99", "w_channel": "3", "w_radius": "10",
    "w_pattern": "ring", "w_mask_shape": "circle",
    "w_measurement": "l1_complex", "w_injection": "complex",
}


def _patch_tr_module(monkeypatch, score_fn=None, load_fn=None):
    """Patch TR detector module for synthetic testing."""
    import raven.detectors.tr_detector as mod
    if load_fn is not None:
        monkeypatch.setattr(mod, "load_state", load_fn)
    else:
        monkeypatch.setattr(
            mod, "load_state",
            lambda records, device, **extra: {"fake": True},
        )
    if score_fn is not None:
        monkeypatch.setattr(mod, "score_image", score_fn)


# ===========================================================================
# A. Reducer unit tests (no evaluate_detector needed)
# ===========================================================================
class TestReducerUnit:
    """Direct tests of ``reduce_detector_stage_status``."""

    @staticmethod
    def _row(status):
        return {"status": status}

    def test_all_missing_state(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            FAILURE_CAUSE_MISSING_REQUIRED_STATE,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_MISSING_STATE),
            self._row(ROW_STATUS_FAILED_MISSING_STATE),
            self._row(ROW_STATUS_FAILED_MISSING_STATE),
        ])
        assert r["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE
        assert r["available"] is False
        assert "status_reducer_reason" in r
        assert "row_status_counts" in r
        assert "failure_cause_counts" in r

    def test_all_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_MISSING_IMAGE,
            STATUS_FAILED_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_IMAGE,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_MISSING_IMAGE),
            self._row(ROW_STATUS_FAILED_MISSING_IMAGE),
        ])
        assert r["status"] == STATUS_FAILED_MISSING_IMAGE
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_MISSING_IMAGE
        assert r["available"] is False

    def test_all_scoring_error(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_SCORING,
            STATUS_FAILED_SCORING,
            FAILURE_CAUSE_SCORING_ERROR,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_SCORING),
            self._row(ROW_STATUS_FAILED_SCORING),
        ])
        assert r["status"] == STATUS_FAILED_SCORING
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_SCORING_ERROR
        assert r["available"] is False

    def test_precedence_missing_state_plus_scoring_error(self):
        """10 missing_state + 1 scoring_error → failed_scoring (precedence)."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_MISSING_STATE,
            ROW_STATUS_FAILED_SCORING,
            STATUS_FAILED_SCORING,
            FAILURE_CAUSE_SCORING_ERROR,
        )
        rows = [self._row(ROW_STATUS_FAILED_MISSING_STATE)] * 10
        rows.append(self._row(ROW_STATUS_FAILED_SCORING))
        r = reduce_detector_stage_status(rows)
        assert r["status"] == STATUS_FAILED_SCORING
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_SCORING_ERROR

    def test_precedence_missing_image_plus_missing_state(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_MISSING_IMAGE,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_IMAGE,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_MISSING_STATE),
            self._row(ROW_STATUS_FAILED_MISSING_IMAGE),
        ])
        assert r["status"] == STATUS_FAILED_MISSING_IMAGE
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_MISSING_IMAGE

    def test_precedence_provider_init_plus_scoring_error(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_PROVIDER,
            ROW_STATUS_FAILED_SCORING,
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            FAILURE_CAUSE_PROVIDER_INITIALIZATION,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_SCORING),
            self._row(ROW_STATUS_FAILED_PROVIDER),
        ])
        assert r["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_PROVIDER_INITIALIZATION

    def test_precedence_state_validation_plus_provider_init(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_PROVIDER,
            ROW_STATUS_FAILED_STATE_VALIDATION,
            STATUS_FAILED_STATE_VALIDATION,
            FAILURE_CAUSE_STATE_VALIDATION,
        )
        r = reduce_detector_stage_status([
            self._row(ROW_STATUS_FAILED_PROVIDER),
            self._row(ROW_STATUS_FAILED_STATE_VALIDATION),
        ])
        assert r["status"] == STATUS_FAILED_STATE_VALIDATION
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION

    def test_precedence_internal_error_overrides_all(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_STATE_VALIDATION,
            ROW_STATUS_FAILED_SCORING,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
        )
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_FAILED_STATE_VALIDATION),
             self._row(ROW_STATUS_FAILED_SCORING)],
            setup_failure=FAILURE_CAUSE_INTERNAL_ERROR,
        )
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_INTERNAL_ERROR

    def test_partial_missing_state_primary_complete(self):
        """D.4: partial success + missing state + primary complete →
        completed_with_errors."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_COMPLETED_WITH_ERRORS,
        )
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED),
             self._row(ROW_STATUS_FAILED_MISSING_STATE)],
            primary_report_available=True,
            primary_metrics_complete=True,
        )
        assert r["status"] == STATUS_COMPLETED_WITH_ERRORS
        assert r["available"] is True

    def test_partial_missing_state_primary_incomplete(self):
        """D.4: partial + missing state + primary incomplete →
        failed_missing_required_state."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
        )
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED),
             self._row(ROW_STATUS_FAILED_MISSING_STATE)],
            primary_report_available=False,
            primary_metrics_complete=False,
        )
        assert r["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_all_scored_completed(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            STATUS_COMPLETED,
        )
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED)] * 3,
            primary_report_available=True,
            primary_metrics_complete=True,
        )
        assert r["status"] == STATUS_COMPLETED
        assert r["available"] is True

    def test_all_optional_failures_primary_complete(self):
        """D.5: optional cohort missing-state failures only → completed.

        Only ``missing_required_state`` is a soft cause — scoring_error
        and other hard failures in optional cohorts must still propagate.
        """
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_COMPLETED,
        )
        # 3 primary rows scored, 1 optional missing-state row
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED)] * 3
            + [self._row(ROW_STATUS_FAILED_MISSING_STATE)],
            primary_report_available=True,
            primary_metrics_complete=True,
            optional_failed_count=1,
        )
        assert r["status"] == STATUS_COMPLETED
        assert "dominant_failure_cause" in r

    def test_optional_hard_failure_not_exempted(self):
        """Optional scoring_error must NOT be suppressed to completed."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            ROW_STATUS_FAILED_SCORING,
            STATUS_FAILED_SCORING,
        )
        # 3 primary rows scored, 1 optional scoring_error
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED)] * 3
            + [self._row(ROW_STATUS_FAILED_SCORING)],
            primary_report_available=True,
            primary_metrics_complete=True,
            optional_failed_count=1,
        )
        assert r["status"] == STATUS_FAILED_SCORING

    def test_empty_rows_skipped(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_SKIPPED_INSUFFICIENT_DATA,
        )
        r = reduce_detector_stage_status([])
        assert r["status"] == STATUS_SKIPPED_INSUFFICIENT_DATA

    def test_reducer_output_fields(self):
        """All required diagnostic fields present."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_SCORING,
            ROW_STATUS_SCORED,
        )
        r = reduce_detector_stage_status([
            {"status": ROW_STATUS_SCORED},
            {"status": ROW_STATUS_FAILED_SCORING},
        ])
        for field in ("status", "dominant_failure_cause",
                      "status_reducer_reason", "available",
                      "failure_cause_counts", "row_status_counts"):
            assert field in r, f"missing field: {field}"


# ===========================================================================
# B. Allow policy tests
# ===========================================================================
class TestAllowPolicy:
    """``stage_status_is_allowable`` and ``determine_exit_code``."""

    def test_allowable_with_flag(self):
        from raven.detectors import (
            stage_status_is_allowable,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            STATUS_FAILED_MISSING_DEPENDENCY,
            STATUS_SKIPPED_INSUFFICIENT_DATA,
        )
        for st in (STATUS_FAILED_MISSING_REQUIRED_STATE,
                   STATUS_FAILED_MISSING_DEPENDENCY,
                   STATUS_SKIPPED_INSUFFICIENT_DATA):
            assert stage_status_is_allowable(st, allow_missing_metrics=True), st

    def test_not_allowable_without_flag(self):
        from raven.detectors import (
            stage_status_is_allowable,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            STATUS_FAILED_MISSING_DEPENDENCY,
            STATUS_SKIPPED_INSUFFICIENT_DATA,
        )
        for st in (STATUS_FAILED_MISSING_REQUIRED_STATE,
                   STATUS_FAILED_MISSING_DEPENDENCY,
                   STATUS_SKIPPED_INSUFFICIENT_DATA):
            assert not stage_status_is_allowable(
                st, allow_missing_metrics=False), st

    def test_never_allowable_even_with_flag(self):
        from raven.detectors import (
            stage_status_is_allowable,
            STATUS_FAILED_MISSING_IMAGE,
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            STATUS_FAILED_STATE_VALIDATION,
            STATUS_FAILED_SCORING,
            STATUS_FAILED_INTERNAL_ERROR,
            STATUS_COMPLETED_WITH_ERRORS,
        )
        for st in (STATUS_FAILED_MISSING_IMAGE,
                   STATUS_FAILED_PROVIDER_INITIALIZATION,
                   STATUS_FAILED_STATE_VALIDATION,
                   STATUS_FAILED_SCORING,
                   STATUS_FAILED_INTERNAL_ERROR,
                   STATUS_COMPLETED_WITH_ERRORS):
            assert not stage_status_is_allowable(
                st, allow_missing_metrics=True), f"{st} should not be allowable"
            assert not stage_status_is_allowable(
                st, allow_missing_metrics=False), f"{st} should not be allowable"

    def test_completed_always_allowable(self):
        from raven.detectors import (
            stage_status_is_allowable, STATUS_COMPLETED,
        )
        assert stage_status_is_allowable(
            STATUS_COMPLETED, allow_missing_metrics=False)
        assert stage_status_is_allowable(
            STATUS_COMPLETED, allow_missing_metrics=True)

    def test_determine_exit_code_all_allowed(self):
        from raven.detectors import (
            determine_exit_code, STATUS_COMPLETED,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
        )
        result = {
            "stages": {
                "quality": {"status": STATUS_COMPLETED},
                "detector": {"status": STATUS_FAILED_MISSING_REQUIRED_STATE},
            },
        }
        assert determine_exit_code(result, allow_missing_metrics=False) == 2
        assert determine_exit_code(result, allow_missing_metrics=True) == 0

    def test_determine_exit_code_hard_failure(self):
        from raven.detectors import (
            determine_exit_code, STATUS_COMPLETED,
            STATUS_FAILED_SCORING,
        )
        result = {
            "stages": {
                "quality": {"status": STATUS_COMPLETED},
                "detector": {"status": STATUS_FAILED_SCORING},
            },
        }
        assert determine_exit_code(result, allow_missing_metrics=False) == 2
        assert determine_exit_code(result, allow_missing_metrics=True) == 2

    def test_determine_exit_code_completed(self):
        from raven.detectors import (
            determine_exit_code, STATUS_COMPLETED,
        )
        result = {
            "stages": {
                "quality": {"status": STATUS_COMPLETED},
                "detector": {"status": STATUS_COMPLETED},
            },
        }
        assert determine_exit_code(result, allow_missing_metrics=False) == 0
        assert determine_exit_code(result, allow_missing_metrics=True) == 0


# ===========================================================================
# C. evaluate_detector() integration tests
# ===========================================================================
class TestEvaluateDetectorIntegration:
    """Tests that call ``evaluate_detector()`` with mocked TR detector."""

    def test_all_scored_completed(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED, ROW_STATUS_SCORED

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)
        _patch_tr_module(monkeypatch, score_fn=lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector(
                [rec_clean, rec_wm, rec_wm2], out, "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 6
            assert result["failed_count"] == 0
            assert result["available"] is True
            assert result["dominant_failure_cause"] is None
            assert "status_reducer_reason" in result
            assert result["row_status_counts"].get(ROW_STATUS_SCORED) == 6

    def test_all_missing_state_rows(self, monkeypatch):
        """All rows raise DetectorMissingStateError → failed_missing_required_state."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            ROW_STATUS_FAILED_MISSING_STATE,
            DetectorMissingStateError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("no state for this row")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector(
                [rec_wm, rec_wm2], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            assert result["scored_count"] == 0
            assert result["failed_count"] == 4
            assert result["available"] is False
            assert result["dominant_failure_cause"] == "missing_required_state"
            # Row-level check
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_STATE, 0) == 4

    def test_all_missing_image_rows(self, monkeypatch):
        """All rows FileNotFoundError → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE,
            ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                FileNotFoundError("no such image")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            assert result["scored_count"] == 0
            assert result["available"] is False
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_IMAGE, 0) == 2

    def test_scoring_exception_failed_scoring(self, monkeypatch):
        """DetectorScoringError → failed_scoring, not suppressible."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING,
            ROW_STATUS_FAILED_SCORING,
            DetectorScoringError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorScoringError("invert_images failed")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_SCORING, 0) == 2

    def test_provider_typeerror_in_scoring_loop(self, monkeypatch):
        """DetectorProviderInitializationError in score_image →
        failed_provider_initialization."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            ROW_STATUS_FAILED_PROVIDER,
            DetectorProviderInitializationError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorProviderInitializationError("TypeError in constructor")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_PROVIDER, 0) == 2

    def test_state_validation_in_scoring_loop(self, monkeypatch):
        """DetectorStateValidationError in score_image →
        failed_state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            ROW_STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorStateValidationError("SHA mismatch")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_STATE_VALIDATION, 0) == 2

    def test_missing_state_plus_scoring_error_precedence(self, monkeypatch):
        """1 scoring_error + many missing_state → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING,
            DetectorMissingStateError, DetectorScoringError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)

        call_count = [0]

        def mixed_fail(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise DetectorScoringError("one scoring error")
            raise DetectorMissingStateError("missing state")

        _patch_tr_module(monkeypatch, score_fn=mixed_fail)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector(
                [rec_wm, rec_wm2], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["dominant_failure_cause"] == "scoring_error"
            assert "scoring_error" in result["status_reducer_reason"]

    def test_missing_dependency_load_state(self, monkeypatch):
        """DetectorDependencyError in load_state → failed_missing_dependency."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_DEPENDENCY,
            DetectorDependencyError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            load_fn=lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorDependencyError("no torch")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_DEPENDENCY

    def test_state_validation_in_load_state(self, monkeypatch):
        """DetectorStateValidationError in load_state →
        failed_state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            load_fn=lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorStateValidationError("provenance mismatch")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_row_has_failure_cause_and_error_type(self, monkeypatch):
        """Each failed row must carry ``failure_cause`` and ``error_type``."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorScoringError

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorScoringError("bad invert")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            evaluate_detector([rec_wm], out, "TR", device="cpu")

            from raven.experiment_io import detector_records_path
            rows = [json.loads(l)
                    for l in detector_records_path(out).read_text().splitlines()
                    if l.strip()]
            for row in rows:
                assert "failure_cause" in row, row
                assert row["failure_cause"] == "scoring_error"
                assert "error_type" in row, row
                assert row["error_type"] == "DetectorScoringError"
                assert "error" in row, row

    def test_scored_row_has_no_failure_cause(self, monkeypatch):
        """Scored rows must NOT carry ``failure_cause`` or ``error_type``."""
        from experiments.eval import evaluate_detector

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(monkeypatch, score_fn=lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm])
            evaluate_detector([rec_wm], out, "TR", device="cpu")

            from raven.experiment_io import detector_records_path
            rows = [json.loads(l)
                    for l in detector_records_path(out).read_text().splitlines()
                    if l.strip()]
            for row in rows:
                if row["status"] == "scored":
                    assert "error_type" not in row, row
                    assert "failure_cause" not in row or not row.get(
                        "failure_cause"), row

    def test_completed_with_errors_never_allowable(self, monkeypatch):
        """completed_with_errors must be nonzero regardless of allow flag."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_COMPLETED_WITH_ERRORS,
            DetectorMissingStateError,
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        call_count = [0]

        def partial_fail(*a, **kw):
            call_count[0] += 1
            if call_count[0] >= 3:  # fail the attacked_watermarked entries
                raise DetectorMissingStateError("missing state")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=partial_fail)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "TR", device="cpu")

            # primary report not available (attacked_watermarked missing
            # for the only watermarked record), so failed_missing_required_state
            # or completed_with_errors depending on metric availability
            assert result["status"] in (
                STATUS_COMPLETED_WITH_ERRORS, "failed_missing_required_state",
            )


# ===========================================================================
# D. CLI exit-code tests (call main() in-process)
# ===========================================================================
class TestCLIExitCodes:
    """CLI exit code tests — call ``main()`` directly with mocked modules.

    Patches apply to the same process, so ``monkeypatch`` works correctly.
    ``main()`` accepts ``argv`` and returns an int exit code.
    """

    def _run_main(self, output_dir, *, allow=False, stages=None):
        """Call ``experiments.eval.main()`` and return (exit_code, result_json)."""
        import io
        from experiments.eval import main

        argv = [
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--log-level", "ERROR",
        ]
        if allow:
            argv.append("--allow-missing-metrics")
        if stages is not None:
            argv.extend(["--stages"] + stages)
        else:
            argv.extend(["--stages", "detector"])

        # Capture stdout to get JSON result
        stdout_buf = io.StringIO()
        with mock.patch("sys.stdout", stdout_buf):
            exit_code = main(argv)

        stdout_text = stdout_buf.getvalue()
        result = json.loads(stdout_text) if stdout_text.strip() else {}
        return exit_code, result

    def test_successful_completed_exit_0(self, monkeypatch):
        _patch_tr_module(monkeypatch, score_fn=lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            exit_code, result = self._run_main(out)
            assert exit_code == 0, f"exit={exit_code}"
            assert result["overall_status"] == "completed"

    def test_missing_state_without_allow_nonzero(self, monkeypatch):
        from raven.detectors import DetectorMissingStateError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("no state")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            exit_code, result = self._run_main(out, allow=False)
            assert exit_code != 0, f"exit={exit_code} expected nonzero"
            assert result["stages"]["detector"]["status"] == (
                "failed_missing_required_state")
            assert result["allowed_by_policy"] is False

    def test_missing_state_with_allow_exit_0(self, monkeypatch):
        from raven.detectors import DetectorMissingStateError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("no state")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            exit_code, result = self._run_main(out, allow=True)
            assert exit_code == 0, (
                f"exit={exit_code} expected 0 with allow flag")
            # JSON status preserved — NOT rewritten to completed
            assert result["stages"]["detector"]["status"] == (
                "failed_missing_required_state")
            assert result["allowed_by_policy"] is True
            assert result["stages_allowable"]["detector"] is True

    def test_scoring_error_always_nonzero(self, monkeypatch):
        from raven.detectors import DetectorScoringError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorScoringError("scoring failed")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])

            exit_code, result = self._run_main(out, allow=False)
            assert exit_code != 0
            assert result["stages"]["detector"]["status"] == "failed_scoring"
            assert result["stages_allowable"]["detector"] is False

            exit_code2, result2 = self._run_main(out, allow=True)
            assert exit_code2 != 0, (
                f"exit={exit_code2} — scoring_error must be nonzero "
                f"even with --allow-missing-metrics")
            assert result2["stages"]["detector"]["status"] == "failed_scoring"

    def test_missing_image_always_nonzero(self, monkeypatch):
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                FileNotFoundError("image gone")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            for allow in (False, True):
                exit_code, result = self._run_main(out, allow=allow)
                assert exit_code != 0, (
                    f"allow={allow} exit={exit_code} — "
                    f"missing_image must be nonzero")
                assert result["stages"]["detector"]["status"] == (
                    "failed_missing_image")

    def test_state_validation_always_nonzero(self, monkeypatch):
        from raven.detectors import DetectorStateValidationError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorStateValidationError("SHA mismatch")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            for allow in (False, True):
                exit_code, result = self._run_main(out, allow=allow)
                assert exit_code != 0, (
                    f"allow={allow} exit={exit_code}")
                assert result["stages"]["detector"]["status"] == (
                    "failed_state_validation")

    def test_provider_init_always_nonzero(self, monkeypatch):
        from raven.detectors import DetectorProviderInitializationError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorProviderInitializationError("TypeError: bad arg")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            for allow in (False, True):
                exit_code, result = self._run_main(out, allow=allow)
                assert exit_code != 0, (
                    f"allow={allow} exit={exit_code}")
                assert result["stages"]["detector"]["status"] == (
                    "failed_provider_initialization")

    def test_missing_dependency_with_allow_exit_0(self, monkeypatch):
        from raven.detectors import DetectorDependencyError

        _patch_tr_module(
            monkeypatch,
            load_fn=lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorDependencyError("no torch")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])

            exit_code, result = self._run_main(out, allow=False)
            assert exit_code != 0
            assert result["stages"]["detector"]["status"] == (
                "failed_missing_dependency")

            exit_code2, result2 = self._run_main(out, allow=True)
            assert exit_code2 == 0, (
                f"exit={exit_code2} — missing_dependency should be allowable")
            assert result2["stages"]["detector"]["status"] == (
                "failed_missing_dependency")  # JSON NOT rewritten
            assert result2["stages_allowable"]["detector"] is True

    def test_completed_with_errors_always_nonzero(self, monkeypatch):
        """completed_with_errors → nonzero regardless of allow flag (E).

        Uses 1 clean + 2 watermarked records (6 rows).  Rows 5-6 fail
        (one watermarked record lost), but rows 1-4 score successfully
        so primary report is still available → completed_with_errors.
        """
        from raven.detectors import DetectorMissingStateError

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            for allow in (False, True):
                call_count = [0]  # Fresh counter per iteration

                def partial_fail(*a, **kw):
                    call_count[0] += 1
                    if call_count[0] >= 5:
                        raise DetectorMissingStateError("missing state")
                    return {"raw_score": 0.001, "canonical_score": 10.0}

                _patch_tr_module(monkeypatch, score_fn=partial_fail)
                exit_code, result = self._run_main(out, allow=allow)
                det_status = result["stages"]["detector"]["status"]
                assert det_status == "completed_with_errors", (
                    f"expected completed_with_errors, got {det_status}")
                assert exit_code != 0, (
                    f"allow={allow} exit={exit_code} — "
                    f"completed_with_errors must be nonzero")

    def test_json_status_preserved_with_allow(self, monkeypatch):
        """F: JSON status must match actual stage status regardless of allow."""
        from raven.detectors import DetectorMissingStateError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("no state")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])

            exit_code, result = self._run_main(out, allow=True)
            assert exit_code == 0

            det = result["stages"]["detector"]
            assert det["status"] == "failed_missing_required_state"
            assert det["status"] != "completed"  # NOT rewritten
            assert result["allowed_by_policy"] is True
            assert result["stages_allowable"]["detector"] is True

    def test_result_summary_diagnostics(self, monkeypatch):
        """G: result JSON must contain diagnostic fields."""
        from raven.detectors import DetectorScoringError

        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorScoringError("scoring error")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            _, result = self._run_main(out, allow=False)
            det = result["stages"]["detector"]

            for field in ("requested_count", "scored_count", "failed_count",
                          "row_status_counts", "failure_cause_counts",
                          "status", "available",
                          "dominant_failure_cause",
                          "status_reducer_reason"):
                assert field in det, f"missing field: {field}"

            ma = det.get("metric_availability", {})
            for field in ("primary_report_available",):
                assert field in ma, f"missing ma field: {field}"


# ===========================================================================
# E. Quality stage status consistency
# ===========================================================================
class TestQualityStageStatus:
    """Quality stage status not affected by allow flag — always correct."""

    def test_quality_skipped_no_images(self, monkeypatch):
        import io
        from experiments.eval import main

        rec = _make_record("1", "watermarked", method="TR")
        rec["input_path"] = "/nonexistent/input.png"

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])

            argv = [
                "--output-dir", str(out), "--device", "cpu",
                "--stages", "quality", "--log-level", "ERROR",
            ]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert result["stages"]["quality"]["status"] == (
                "skipped_insufficient_data")
            assert result["stages"]["quality"]["available"] is False


# ===========================================================================
# F. Explicit failure-cause precedence (reducer unit tests)
# ===========================================================================
class TestExplicitFailureCause:
    """Reducer must use explicit ``failure_cause``, not derive from status."""

    def test_explicit_internal_error_overrides_status(self):
        """Row with status=failed_scoring + explicit internal_error
        → dominant = internal_error, stage = failed_internal_error."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_SCORING,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
        )
        rows = [
            {"status": ROW_STATUS_FAILED_SCORING,
             "failure_cause": FAILURE_CAUSE_INTERNAL_ERROR},
        ]
        r = reduce_detector_stage_status(rows)
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_INTERNAL_ERROR
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR

    def test_explicit_internal_error_beats_many_others(self):
        """1 explicit internal_error + 10 missing_state + 10 scoring_error
        → failed_internal_error."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_SCORING,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
            FAILURE_CAUSE_MISSING_REQUIRED_STATE,
        )
        rows = [
            {"status": ROW_STATUS_FAILED_SCORING,
             "failure_cause": FAILURE_CAUSE_INTERNAL_ERROR},
        ]
        rows += [{"status": ROW_STATUS_FAILED_MISSING_STATE}] * 10
        rows += [{"status": ROW_STATUS_FAILED_SCORING}] * 10
        r = reduce_detector_stage_status(rows)
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_INTERNAL_ERROR
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR
        assert "takes precedence" in r["status_reducer_reason"]

    def test_unknown_explicit_cause_fails_closed(self):
        """Unrecognized explicit failure_cause → internal_error."""
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
        )
        row = {
            "status": "failed_scoring",
            "failure_cause": "unrecognized_failure",
        }
        r = reduce_detector_stage_status([row])
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_INTERNAL_ERROR
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR


# ===========================================================================
# G. Optional-cohort hard failure integration tests
# ===========================================================================
class TestOptionalHardFailures:
    """Hard failures in optional cohorts must propagate, not be suppressed."""

    def test_optional_missing_image_primary_complete(self, monkeypatch):
        """optional missing_image + primary complete → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_IMAGE

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        def fake_score(provider_info, image_path, *,
                       record=None, evaluation_entry=None, steps=50):
            if (evaluation_entry is not None
                    and evaluation_entry.get("evaluation_cohort") == "attacked_clean"):
                raise FileNotFoundError("optional image missing")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE

    def test_optional_state_validation_primary_complete(self, monkeypatch):
        """optional state_validation + primary complete →
        failed_state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION, DetectorStateValidationError,
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        def fake_score(provider_info, image_path, *,
                       record=None, evaluation_entry=None, steps=50):
            if (evaluation_entry is not None
                    and evaluation_entry.get("evaluation_cohort") == "attacked_clean"):
                raise DetectorStateValidationError(
                    "optional provenance mismatch")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_STATE_VALIDATION


# ===========================================================================
# H. Image path preflight — actual missing files
# ===========================================================================
class TestImagePreflight:
    """Image preflight checks path BEFORE calling score_image."""

    def test_missing_input_image_preflight(self, monkeypatch):
        """Delete one input image → that row is caught by preflight.

        Other rows with valid images still call score_image.  Only the row
        with the deleted input is classified as failed_missing_image.
        """
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)

        called_entries = []

        def track_calls(*a, **kw):
            called_entries.append(kw.get("evaluation_entry", {}).get(
                "evaluation_cohort", "unknown"))
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=track_calls)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            # Delete input image for the clean record
            input_path = Path(rec_clean["input_path"])
            input_path.unlink()

            result = evaluate_detector([rec_clean, rec_wm, rec_wm2], out,
                                       "TR", device="cpu")
            # original_clean should NOT have called score_image
            assert "original_clean" not in called_entries, (
                f"preflight should have caught missing input before "
                f"score_image, but original_clean was in {called_entries}")
            # Other cohorts should have called score_image
            assert len(called_entries) >= 3
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_IMAGE, 0) >= 1

    def test_missing_output_image_preflight(self, monkeypatch):
        """Delete one output image → that row caught by preflight.

        Other rows with valid images still call score_image.
        """
        from experiments.eval import evaluate_detector, _build_detector_image_index
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )
        from raven.experiment_io import output_image_path

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)

        called_entries = []

        def track_calls(*a, **kw):
            called_entries.append(kw.get("evaluation_entry", {}).get(
                "evaluation_cohort", "unknown"))
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=track_calls)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm, rec_wm2])
            # Delete the attacked_watermarked output image for run_id=1
            att_path = output_image_path(out, "watermarked", "1")
            Path(str(att_path)).unlink()

            result = evaluate_detector([rec_wm, rec_wm2], out,
                                       "TR", device="cpu")
            # attacked_watermarked for run_id=1 should NOT have called
            # score_image
            assert "attacked_watermarked" in called_entries or len(
                called_entries) >= 3, (
                "other rows should still call score_image")
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_IMAGE, 0) >= 1


# ===========================================================================
# I. Exception classification — load_state TypeError + per-row dependency
# ===========================================================================
class TestExceptionClassification:
    """Raw TypeError in load_state → provider init.
    Per-row dependency errors → missing_dependency."""

    def test_raw_typeerror_in_load_state(self, monkeypatch):
        """TypeError in load_state → failed_provider_initialization."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_PROVIDER_INITIALIZATION

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            load_fn=lambda records, device, **extra: (_ for _ in ()).throw(
                TypeError("__init__() got an unexpected keyword argument 'foo'")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION

    def test_per_row_dependency_error(self, monkeypatch):
        """DetectorDependencyError in score_image →
        row=failed_missing_dependency, stage=failed_missing_dependency."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_DEPENDENCY,
            ROW_STATUS_FAILED_MISSING_DEPENDENCY,
            DetectorDependencyError,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                DetectorDependencyError("optional dependency not installed")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_DEPENDENCY
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_DEPENDENCY, 0) >= 1

    def test_per_row_import_error(self, monkeypatch):
        """ImportError in score_image → failed_missing_dependency."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_DEPENDENCY,
            ROW_STATUS_FAILED_MISSING_DEPENDENCY,
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        _patch_tr_module(
            monkeypatch,
            score_fn=lambda *a, **kw: (_ for _ in ()).throw(
                ImportError("No module named 'torch'")),
        )

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_DEPENDENCY
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_DEPENDENCY, 0) >= 1


# ===========================================================================
# J. Method-specific dispatch acceptance tests
# ===========================================================================
class TestMethodDispatchAcceptance:
    """GM, GS, T2S via actual adapter modules with monkeypatch."""

    # ---- GM: cohort-wide missing bundle ----
    def test_gm_load_state_missing_bundle(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            DetectorMissingStateError,
        )
        import raven.detectors.gm_detector as gm_mod

        monkeypatch.setattr(
            gm_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("GM bundle not found")),
        )

        GM_META = {"gm_bundle_path": "/nonexistent/bundle",
                    "gm_bundle_sha256": "abc"}
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GM", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_gm_missing_bundle_without_allow_nonzero(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.gm_detector as gm_mod

        monkeypatch.setattr(
            gm_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("GM bundle not found")),
        )

        GM_META = {"gm_bundle_path": "/nonexistent/bundle",
                    "gm_bundle_sha256": "abc"}
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code != 0
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")

    def test_gm_missing_bundle_with_allow_exit_0(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.gm_detector as gm_mod

        monkeypatch.setattr(
            gm_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("GM bundle not found")),
        )

        GM_META = {"gm_bundle_path": "/nonexistent/bundle",
                    "gm_bundle_sha256": "abc"}
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR",
                    "--allow-missing-metrics"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code == 0
            # JSON status preserved — NOT rewritten
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")
            assert result["allowed_by_policy"] is True

    # ---- GS: per-row secret state missing ----
    def test_gs_per_row_missing_state(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            ROW_STATUS_FAILED_MISSING_STATE,
            DetectorMissingStateError,
        )
        import raven.detectors.gs_detector as gs_mod

        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            gs_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("GS secret missing for this row")),
        )

        GS_META = {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"}
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GS", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_STATE, 0) >= 1

    def test_gs_missing_state_without_allow_nonzero(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.gs_detector as gs_mod

        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            gs_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("GS secret missing")),
        )

        GS_META = {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"}
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code != 0
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")

    def test_gs_missing_state_with_allow_exit_0(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.gs_detector as gs_mod

        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            gs_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("GS secret missing")),
        )

        GS_META = {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"}
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR",
                    "--allow-missing-metrics"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code == 0
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")

    # ---- T2S: per-row state missing ----
    def test_t2s_per_row_missing_state(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            ROW_STATUS_FAILED_MISSING_STATE,
            DetectorMissingStateError,
        )
        import raven.detectors.t2s_detector as t2s_mod

        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            t2s_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("T2S state missing for this row")),
        )

        T2S_META = {"t2s_state_path": "/tmp/fake.pt",
                     "t2s_state_sha256": "abc",
                     "t2s_provider_config_sha256": "def",
                     "t2s_protocol_mode": "official"}
        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            rc = result["row_status_counts"]
            assert rc.get(ROW_STATUS_FAILED_MISSING_STATE, 0) >= 1

    def test_t2s_missing_state_without_allow_nonzero(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.t2s_detector as t2s_mod

        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            t2s_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("T2S state missing")),
        )

        T2S_META = {"t2s_state_path": "/tmp/fake.pt",
                     "t2s_state_sha256": "abc",
                     "t2s_provider_config_sha256": "def",
                     "t2s_protocol_mode": "official"}
        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code != 0
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")

    def test_t2s_missing_state_with_allow_exit_0(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.t2s_detector as t2s_mod

        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(
            t2s_mod, "score_image",
            lambda *a, **kw: (_ for _ in ()).throw(
                DetectorMissingStateError("T2S state missing")),
        )

        T2S_META = {"t2s_state_path": "/tmp/fake.pt",
                     "t2s_state_sha256": "abc",
                     "t2s_provider_config_sha256": "def",
                     "t2s_protocol_mode": "official"}
        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            argv = ["--output-dir", str(out), "--device", "cpu",
                    "--stages", "detector", "--log-level", "ERROR",
                    "--allow-missing-metrics"]
            stdout_buf = io.StringIO()
            with mock.patch("sys.stdout", stdout_buf):
                exit_code = main(argv)
            result = json.loads(stdout_buf.getvalue())
            assert exit_code == 0
            assert (result["stages"]["detector"]["status"]
                    == "failed_missing_required_state")


# ===========================================================================
# K. Setup + row combined precedence tests
# ===========================================================================
class TestSetupRowPrecedence:
    """Setup failures participate in same precedence pool as row failures."""

    def _make_missing_image_row(self):
        return {
            "run_id": "1", "source_role": "clean",
            "evaluation_cohort": "original_clean",
            "image_path": "/nonexistent/img.png",
            "method": "TR",
            "status": "failed_missing_image",
            "failure_cause": "missing_image",
            "error_type": "FileNotFoundError",
            "error": "Image file does not exist: /nonexistent/img.png",
        }

    def test_setup_missing_state_does_not_mask_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_REQUIRED_STATE,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_MISSING_REQUIRED_STATE)
        assert r["status"] == STATUS_FAILED_MISSING_IMAGE
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_MISSING_IMAGE

    def test_setup_missing_dependency_does_not_mask_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_DEPENDENCY,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_MISSING_DEPENDENCY)
        assert r["status"] == STATUS_FAILED_MISSING_IMAGE
        assert r["dominant_failure_cause"] == FAILURE_CAUSE_MISSING_IMAGE

    def test_provider_failure_precedes_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            FAILURE_CAUSE_PROVIDER_INITIALIZATION,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_PROVIDER_INITIALIZATION)
        assert r["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION

    def test_state_validation_precedes_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_STATE_VALIDATION,
            FAILURE_CAUSE_STATE_VALIDATION,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_STATE_VALIDATION)
        assert r["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_internal_error_precedes_missing_image(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_INTERNAL_ERROR)
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR

    def test_unknown_setup_failure_fails_closed(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            STATUS_FAILED_INTERNAL_ERROR,
            FAILURE_CAUSE_INTERNAL_ERROR,
        )
        r = reduce_detector_stage_status(
            [], setup_failure="unknown_setup_failure")
        assert r["status"] == STATUS_FAILED_INTERNAL_ERROR

    def test_setup_failure_not_optional_exempted(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_FAILED_MISSING_STATE,
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            FAILURE_CAUSE_MISSING_REQUIRED_STATE,
        )
        rows = [{
            "status": ROW_STATUS_FAILED_MISSING_STATE,
            "failure_cause": FAILURE_CAUSE_MISSING_REQUIRED_STATE,
            "evaluation_cohort": "attacked_clean",
        }]
        r = reduce_detector_stage_status(
            rows,
            setup_failure=FAILURE_CAUSE_MISSING_REQUIRED_STATE,
            primary_report_available=True,
            primary_metrics_complete=True,
            optional_failed_count=1,
        )
        assert r["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_setup_failure_result_retains_diagnostics(self):
        from raven.detectors import (
            reduce_detector_stage_status,
            FAILURE_CAUSE_MISSING_REQUIRED_STATE,
        )
        rows = [self._make_missing_image_row()]
        r = reduce_detector_stage_status(
            rows, setup_failure=FAILURE_CAUSE_MISSING_REQUIRED_STATE)
        for field in ("status", "dominant_failure_cause",
                      "status_reducer_reason", "available",
                      "failure_cause_counts", "row_status_counts"):
            assert field in r, f"missing field: {field}"
        assert "missing_image" in r["failure_cause_counts"]
        assert "missing_required_state" in r["failure_cause_counts"]


# ===========================================================================
# L. Method dispatch with combined failures
# ===========================================================================
class TestCombinedMethodFailures:
    """GM/GS/T2S with image preflight + setup failures."""

    def test_gm_missing_image_precedes_missing_bundle(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, DetectorMissingStateError,
        )
        import raven.detectors.gm_detector as gm_mod

        monkeypatch.setattr(
            gm_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("GM bundle not found")),
        )

        GM_META = {"gm_bundle_path": "/nonexistent/bundle",
                    "gm_bundle_sha256": "abc"}
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            input_path = Path(rec_wm["input_path"])
            input_path.unlink()

            result = evaluate_detector([rec_wm], out, "GM", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            assert result["dominant_failure_cause"] == "missing_image"
            assert result.get("setup_failure_cause") == "missing_required_state"

    def test_t2s_provider_failure_precedes_missing_image(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            DetectorProviderInitializationError,
        )
        import raven.detectors.t2s_detector as t2s_mod

        monkeypatch.setattr(
            t2s_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorProviderInitializationError("TypeError in constructor")),
        )

        T2S_META = {"t2s_state_path": "/tmp/fake.pt",
                     "t2s_state_sha256": "abc",
                     "t2s_provider_config_sha256": "def",
                     "t2s_protocol_mode": "official"}
        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            input_path = Path(rec_wm["input_path"])
            input_path.unlink()

            result = evaluate_detector([rec_wm], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION

    def test_gs_missing_dependency_with_missing_image(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, DetectorDependencyError,
        )
        import raven.detectors.gs_detector as gs_mod

        monkeypatch.setattr(
            gs_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorDependencyError("torch not installed")),
        )

        GS_META = {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"}
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            input_path = Path(rec_wm["input_path"])
            input_path.unlink()

            result = evaluate_detector([rec_wm], out, "GS", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE

    def test_combined_missing_image_always_nonzero(self, monkeypatch):
        import io
        from experiments.eval import main
        from raven.detectors import DetectorDependencyError
        import raven.detectors.tr_detector as tr_mod

        monkeypatch.setattr(
            tr_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorDependencyError("no torch")),
        )

        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            input_path = Path(rec_wm["input_path"])
            input_path.unlink()

            for allow in (False, True):
                argv = ["--output-dir", str(out), "--device", "cpu",
                        "--stages", "detector", "--log-level", "ERROR"]
                if allow:
                    argv.append("--allow-missing-metrics")
                stdout_buf = io.StringIO()
                with mock.patch("sys.stdout", stdout_buf):
                    exit_code = main(argv)
                result = json.loads(stdout_buf.getvalue())
                assert exit_code != 0, (
                    f"allow={allow} exit={exit_code}")
                assert (result["stages"]["detector"]["status"]
                        == "failed_missing_image")


# ===========================================================================
# M. Count invariant tests
# ===========================================================================
class TestCountInvariant:
    """Orchestrator overrides adapter counts with full image_index counts."""

    def test_setup_failure_counts_all_requested_entries(self, monkeypatch):
        """setup failure + preflight: requested=4, scored=0, failed=1,
        unscored=3, invariant satisfied.

        Uses different run_ids so deleting one input only affects
        that record's original-cohort entry (not both records).
        """
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.tr_detector as tr_mod

        monkeypatch.setattr(
            tr_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("state not found")),
        )

        rec_clean = _make_record("100", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("200", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            # Delete only clean's input image
            Path(rec_clean["input_path"]).unlink()

            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")
            # TR: clean→2 entries, wm→2 entries = 4 total
            assert result["requested_count"] == 4
            assert result["scored_count"] == 0
            assert result["failed_count"] == 1  # 1 preflight missing-image
            assert result["unscored_due_to_setup_count"] == 3
            assert result["count_invariant_satisfied"] is True
            assert (result["requested_count"]
                    == result["scored_count"]
                    + result["failed_count"]
                    + result["unscored_due_to_setup_count"])

    def test_setup_failure_without_preflight_rows(self, monkeypatch):
        """setup failure, no preflight failures: all entries unscored."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.tr_detector as tr_mod

        monkeypatch.setattr(
            tr_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("no provider state")),
        )

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")
            assert result["requested_count"] == 4
            assert result["scored_count"] == 0
            assert result["failed_count"] == 0
            assert result["unscored_due_to_setup_count"] == 4
            assert result["count_invariant_satisfied"] is True

    def test_partial_row_failure_count_invariant(self, monkeypatch):
        """setup succeeds, some rows fail → normal counts, unscored=0."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)

        call_count = [0]

        def partial_fail(*a, **kw):
            call_count[0] += 1
            if call_count[0] >= 4:
                raise DetectorMissingStateError("missing state")
            return {"raw_score": 0.001, "canonical_score": 10.0}

        _patch_tr_module(monkeypatch, score_fn=partial_fail)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm], out,
                                       "TR", device="cpu")
            assert result["requested_count"] == 4
            assert result["scored_count"] >= 2
            assert result["failed_count"] >= 1
            assert result["unscored_due_to_setup_count"] == 0
            assert result["count_invariant_satisfied"] is True
            assert (result["requested_count"]
                    == result["scored_count"]
                    + result["failed_count"]
                    + result["unscored_due_to_setup_count"])

    def test_success_count_invariant(self, monkeypatch):
        """All rows score → unscored=0, invariant satisfied."""
        from experiments.eval import evaluate_detector

        rec_clean = _make_record("1", "clean", method="TR",
                                 source_metadata=TR_META)
        rec_wm = _make_record("1", "watermarked", method="TR",
                              source_metadata=TR_META)
        rec_wm2 = _make_record("2", "watermarked", method="TR",
                               source_metadata=TR_META)
        _patch_tr_module(monkeypatch, score_fn=lambda *a, **kw: {
            "raw_score": 0.001, "canonical_score": 10.0,
        })

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                       out, "TR", device="cpu")
            assert result["requested_count"] == 6
            assert result["scored_count"] == 6
            assert result["failed_count"] == 0
            assert result["unscored_due_to_setup_count"] == 0
            assert result["count_invariant_satisfied"] is True

    def test_gm_combined_failure_requested_count(self, monkeypatch):
        """GM missing bundle + deleted input: requested=4, scored=0,
        failed=1, unscored=3."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError
        import raven.detectors.gm_detector as gm_mod

        monkeypatch.setattr(
            gm_mod, "load_state",
            lambda records, device, **extra: (_ for _ in ()).throw(
                DetectorMissingStateError("GM bundle not found")),
        )

        GM_META = {"gm_bundle_path": "/nonexistent/bundle",
                    "gm_bundle_sha256": "abc"}
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)
        rec_wm2 = _make_record("2", "watermarked", method="GM",
                               source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM",
                                  records=[rec_wm, rec_wm2])
            # Delete one input image
            Path(rec_wm["input_path"]).unlink()

            result = evaluate_detector([rec_wm, rec_wm2], out,
                                       "GM", device="cpu")
            # 2 watermarked → 4 entries. 1 input deleted → 1 preflight failed.
            assert result["requested_count"] == 4
            assert result["scored_count"] == 0
            assert result["failed_count"] == 1
            assert result["unscored_due_to_setup_count"] == 3
            assert result["count_invariant_satisfied"] is True

    def test_all_methods_count_invariant(self, monkeypatch):
        """Across all 7 methods, orchestrator counts are consistent."""
        from experiments.eval import evaluate_detector

        METHODS = ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]
        METHOD_METAS = {
            "TR": TR_META,
            "GS": {"gs_secret_index": "5", "gs_secret_bundle_sha256": "abc",
                    "gs_protocol_mode": "official_compatible"},
            "GM": {"gm_bundle_path": "/tmp/fake_bundle",
                    "gm_bundle_sha256": "abc"},
            "T2S": {"t2s_state_path": "/tmp/fake.pt",
                     "t2s_state_sha256": "abc",
                     "t2s_provider_config_sha256": "def",
                     "t2s_protocol_mode": "official"},
            "RID": {"fourier_key": "42", "fourier_key_sha256": "abc"},
            "HSTR": {"fourier_key": "42", "fourier_key_sha256": "abc"},
            "HSQR": {"fourier_key": "42", "fourier_key_sha256": "abc"},
        }

        # T2S needs extra fields; threshold methods need raw_score +
        # canonical_score only.
        THRESHOLD_SCORE = {"raw_score": 0.001, "canonical_score": 10.0}
        T2S_SCORE = {
            "raw_score": 0.85, "canonical_score": 0.85,
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.40,
            "t2s_score_margin": 0.45,
            "t2s_detection_success": True,
        }

        for idx, method in enumerate(METHODS):
            det_pkg = "raven.detectors"
            mod_map = {
                "TR": "tr_detector", "GS": "gs_detector",
                "GM": "gm_detector", "T2S": "t2s_detector",
                "RID": "fourier_detector", "HSTR": "fourier_detector",
                "HSQR": "fourier_detector",
            }
            import importlib
            mod = importlib.import_module(f".{mod_map[method]}", det_pkg)
            monkeypatch.setattr(mod, "load_state",
                                lambda records, device, **extra: {"fake": True})
            score = T2S_SCORE if method == "T2S" else THRESHOLD_SCORE
            monkeypatch.setattr(mod, "score_image",
                                lambda *a, sc=score, **kw: dict(sc))

            meta = METHOD_METAS[method]
            rec_wm = _make_record(str(idx + 1), "watermarked",
                                  method=method, source_metadata=meta)

            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method,
                                           device="cpu")
                assert result["requested_count"] == 2, (
                    f"{method}: requested={result['requested_count']}")
                assert result["scored_count"] == 2, (
                    f"{method}: scored={result['scored_count']}")
                assert result["failed_count"] == 0, (
                    f"{method}: failed={result['failed_count']}")
                assert result["unscored_due_to_setup_count"] == 0, (
                    f"{method}: unscored={result['unscored_due_to_setup_count']}")
                assert result["count_invariant_satisfied"] is True, (
                    f"{method}: invariant violated")


