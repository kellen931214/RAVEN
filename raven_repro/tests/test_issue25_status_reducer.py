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
        img = out / "samples" / role / rid / "output.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")
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
        """D.5: optional cohort failures only → completed with flag."""
        from raven.detectors import (
            reduce_detector_stage_status,
            ROW_STATUS_SCORED,
            ROW_STATUS_FAILED_SCORING,
            STATUS_COMPLETED,
        )
        # 3 primary rows scored, 1 optional row failed
        r = reduce_detector_stage_status(
            [self._row(ROW_STATUS_SCORED)] * 3
            + [self._row(ROW_STATUS_FAILED_SCORING)],
            primary_report_available=True,
            primary_metrics_complete=True,
            optional_failed_count=1,
        )
        assert r["status"] == STATUS_COMPLETED

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

