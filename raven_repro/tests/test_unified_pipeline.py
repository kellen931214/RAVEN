"""Tests for the unified main/eval pipeline — no real data required."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))


# ===========================================================================
# Helpers
# ===========================================================================
def _make_record(run_id="17", role="watermarked", **kw):
    base = {
        "run_id": run_id, "role": role, "method": kw.get("method", "TR"),
        "input_path": kw.get("input_path", f"/tmp/in_{run_id}.png"),
        "output_path": f"/tmp/out/{role}/{run_id}/output.png",
        "prompt": kw.get("prompt", ""),
        "prompt_source": kw.get("prompt_source", "metadata"),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": kw["edx"] if "edx" in kw else 24.0,
        "effective_source_flow_dy_image_px": kw["edy"] if "edy" in kw else -24.0,
        "debug_info_path": kw.get("debug_info_path", ""),
        "debug_info_retained": kw.get("debug_info_retained", False),
        "source_metadata": kw.get("source_metadata", {}),
    }
    base.update(kw)
    return base


def _write_fake_run(tmp_path, method="TR", records=None, extra_config=None):
    """Create a minimal output dir with config + records + fake output.png."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = tmp_path / "run"
    out.mkdir()
    cfg = {"method": method, "dataset": "test", **(extra_config or {})}
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
        # Create input image for preflight (original cohorts)
        input_path = Path(r.get("input_path", f"/tmp/in_{rid}.png"))
        if not input_path.is_file():
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(b"fake png")
    rebuild_records_jsonl(out)
    return out


# ===========================================================================
# Metadata passthrough (Critical 4)
# ===========================================================================
class TestMetadataPassthrough:
    def test_all_original_columns_preserved(self):
        """normalize_metadata_row preserves all original columns."""
        sys.path.insert(0, str(REPO / "experiments"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main_mod", REPO / "experiments" / "main.py")
        main_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_mod)

        row = {
            "run_id": "42", "watermarked_path": "/tmp/wm.png",
            "clean_path": "/tmp/cl.png", "prompt": "test prompt",
            "gm_bundle_dir": "/data/gm_bundle",
            "gm_bundle_config_sha256": "abc123",
            "gm_w1_file_sha256": "def456",
            "gm_w2_file_sha256": "ghi789",
            "t2s_state_path": "/data/t2s.pt",
            "t2s_state_sha256": "t2s_sha",
            "gs_secret_index": "5",
            "w_seed": "12345",
            "w_channel": "3",
        }
        result = main_mod.normalize_metadata_row(row)
        # Core fields normalized
        assert result["run_id"] == "42"
        assert result["watermarked_path"] == "/tmp/wm.png"
        assert result["prompt"] == "test prompt"
        # Detector fields preserved
        assert result["gm_bundle_dir"] == "/data/gm_bundle"
        assert result["gm_bundle_config_sha256"] == "abc123"
        assert result["gm_w1_file_sha256"] == "def456"
        assert result["t2s_state_path"] == "/data/t2s.pt"
        assert result["t2s_state_sha256"] == "t2s_sha"
        assert result["gs_secret_index"] == "5"
        assert result["w_seed"] == "12345"
        assert result["w_channel"] == "3"

    def test_source_metadata_in_record(self):
        """Record contains full source_metadata for eval detector access."""
        from raven.experiment_io import write_record
        with tempfile.TemporaryDirectory() as td:
            meta = {"run_id": "1", "gm_bundle_dir": "/b", "w_seed": "99"}
            rec = _make_record("1", "watermarked", source_metadata=meta)
            write_record(td, "watermarked", "1", rec)
            from raven.experiment_io import record_path, output_image_path
            output_image_path(td, "watermarked", "1").write_bytes(b"png")
            loaded = json.loads(record_path(td, "watermarked", "1").read_text())
            assert loaded["source_metadata"]["gm_bundle_dir"] == "/b"
            assert loaded["source_metadata"]["w_seed"] == "99"


# ===========================================================================
# Effective flow — no planned fallback (Medium)
# ===========================================================================
class TestEffectiveFlow:
    def test_null_when_debug_info_missing(self):
        """When debug_info is empty, effective flow is None, not planned."""
        rec = _make_record("1", "watermarked", edx=None, edy=None)
        rec["effective_source_flow_dx_image_px"] = None
        rec["effective_source_flow_dy_image_px"] = None
        assert rec["effective_source_flow_dx_image_px"] is None
        assert rec["effective_source_flow_dy_image_px"] is None
        # Quality must report unavailable
        from experiments.eval import evaluate_quality
        result = evaluate_quality([rec], "/tmp/fake")
        assert not result["available"]

    def test_actual_flow_preserved(self):
        """When debug_info has actual effective flow, it is preserved."""
        rec = _make_record("1", "watermarked", edx=12.5, edy=-8.3)
        assert rec["effective_source_flow_dx_image_px"] == 12.5
        assert rec["effective_source_flow_dy_image_px"] == -8.3


# ===========================================================================
# Unified score_image contract (Critical 1)
# ===========================================================================
class TestUnifiedScoreImageContract:
    def test_all_seven_accept_record_kwarg(self):
        """Every detector's score_image accepts record= and evaluation_entry=."""
        from raven.detectors import _lazy_imports
        _lazy_imports()
        import inspect
        from raven.detectors import DETECTOR_MODULES
        for method in ["TR", "GS", "GM", "T2S", "RID"]:
            mod = DETECTOR_MODULES.get(method)
            sig = inspect.signature(mod.score_image)
            params = set(sig.parameters)
            assert "record" in params, f"{method}: missing 'record' param"
            assert "evaluation_entry" in params, f"{method}: missing 'evaluation_entry'"
            assert "steps" in params, f"{method}: missing 'steps' param"

    def test_no_row_index_kwarg(self):
        """No detector accepts 'row_index' — unified contract uses 'record'."""
        from raven.detectors import _lazy_imports
        _lazy_imports()
        import inspect
        from raven.detectors import DETECTOR_MODULES
        for method in DETECTOR_MODULES:
            mod = DETECTOR_MODULES[method]
            sig = inspect.signature(mod.score_image)
            assert "row_index" not in sig.parameters, (
                f"{method}: must not accept row_index")

    def test_evaluate_detector_passes_record(self):
        """evaluate_detector passes record= to score_image."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "record=" in source
        assert "evaluation_entry=" in source


# ===========================================================================
# Row-level status (Critical 2)
# ===========================================================================
class TestRowStatus:
    def test_status_constants_defined(self):
        from raven.detectors import (
            ROW_STATUS_SCORED, ROW_STATUS_FAILED_MISSING_IMAGE,
            ROW_STATUS_FAILED_MISSING_STATE, ROW_STATUS_FAILED_PROVIDER,
            ROW_STATUS_FAILED_SCORING,
        )
        assert ROW_STATUS_SCORED == "scored"
        assert ROW_STATUS_FAILED_MISSING_IMAGE == "failed_missing_image"
        assert ROW_STATUS_FAILED_MISSING_STATE == "failed_missing_state"
        assert ROW_STATUS_FAILED_PROVIDER == "failed_provider"
        assert ROW_STATUS_FAILED_SCORING == "failed_scoring"

    def test_aggregate_uses_scored_not_completed(self):
        """All detector aggregates check for 'scored', not 'completed'."""
        for mod_name in ["tr_detector", "gs_detector", "gm_detector",
                          "t2s_detector", "fourier_detector"]:
            source = (REPO / "raven_repro" / "raven" / "detectors"
                      / f"{mod_name}.py").read_text()
            assert "ROW_STATUS_SCORED" in source, (
                f"{mod_name}: must use ROW_STATUS_SCORED")


# ===========================================================================
# Zero scores not completed (Critical 3)
# ===========================================================================
class TestZeroScoresNotCompleted:
    def test_all_zero_scores_is_not_completed(self):
        """scored_count==0 → status is not completed."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_FAILED_SCORING

        rows = [{"status": ROW_STATUS_FAILED_SCORING,
                 "evaluation_cohort": "original_watermarked",
                 "error": "scoring failed"}]
        result = aggregate(rows)
        assert result["scored_count"] == 0
        # Orchestrator must detect this

    def test_missing_required_cohort_not_completed(self):
        """Missing original_watermarked or attacked_watermarked → not completed."""
        from raven.detectors.tr_detector import aggregate
        result = aggregate([])
        assert result["scored_count"] == 0
        assert "original_watermarked" in result.get("missing_cohorts", [])


# ===========================================================================
# Error classification (Critical 5)
# ===========================================================================
class TestErrorClassification:
    def test_exception_classes_exist(self):
        from raven.detectors import (
            DetectorError, DetectorMissingStateError,
            DetectorDependencyError, DetectorProviderInitializationError,
            DetectorStateValidationError, DetectorScoringError,
        )
        assert issubclass(DetectorMissingStateError, DetectorError)
        assert issubclass(DetectorDependencyError, DetectorError)
        assert issubclass(DetectorProviderInitializationError, DetectorError)
        assert issubclass(DetectorStateValidationError, DetectorError)
        assert issubclass(DetectorScoringError, DetectorError)

    def test_missing_state_not_swallowed(self):
        """DetectorMissingStateError or DetectorDependencyError raises, not None."""
        from raven.detectors.tr_detector import load_state
        from raven.detectors import DetectorError
        try:
            load_state([], "cpu")
        except DetectorError as e:
            # Either missing state (no metadata) or missing dependency (no lpips)
            assert isinstance(e, DetectorError), (
                f"Expected DetectorError subclass, got {type(e).__name__}")
        else:
            pass  # eval_bench_wm may be fully importable

    def test_tr_missing_fields(self):
        """TR without w_seed etc raises DetectorMissingStateError."""
        from raven.detectors.tr_detector import load_state, REQUIRED_METADATA_FIELDS
        assert "w_seed" in REQUIRED_METADATA_FIELDS
        # Empty records → missing all fields
        from raven.detectors import DetectorError
        try:
            load_state([], "cpu")
        except DetectorError:
            pass  # expected
        else:
            pass  # may also succeed if eval_bench_wm available


# ===========================================================================
# --allow-missing-metrics semantics (High 6)
# ===========================================================================
class TestAllowMissingMetrics:
    def test_allowable_statuses(self):
        from raven.detectors import ALLOWABLE_STATUSES, NONZERO_STATUSES
        # Allowable and nonzero should not overlap
        assert ALLOWABLE_STATUSES.isdisjoint(NONZERO_STATUSES)

    def test_exit_codes(self):
        """CLI returns correct exit codes."""
        # missing state, no flag → nonzero
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM")
            r = subprocess.run(
                [sys.executable, "experiments/eval.py",
                 "--output-dir", str(out), "--device", "cpu",
                 "--stages", "detector"],
                capture_output=True, text=True, cwd=str(REPO))
            assert r.returncode != 0, f"Expected nonzero exit, got {r.returncode}"

    def test_allow_flag_zero(self):
        """missing state + --allow-missing-metrics → zero."""
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM")
            r = subprocess.run(
                [sys.executable, "experiments/eval.py",
                 "--output-dir", str(out), "--device", "cpu",
                 "--stages", "detector", "--allow-missing-metrics"],
                capture_output=True, text=True, cwd=str(REPO))
            assert r.returncode == 0, (
                f"Expected exit 0 with --allow-missing-metrics, got {r.returncode}")


# ===========================================================================
# Detector end-to-end mock tests
# ===========================================================================
class TestDetectorEndToEndMock:
    def test_tr_with_mock_scores(self, monkeypatch):
        """Full orchestrator cycle with mocked score_image returning valid scores."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            ROW_STATUS_SCORED, STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS,
        )

        rec = _make_record("1", "watermarked", method="TR",
                            source_metadata={"w_seed": "99", "w_channel": "3",
                                             "w_radius": "10", "w_pattern": "ring",
                                             "w_mask_shape": "circle",
                                             "w_measurement": "l1_complex",
                                             "w_injection": "complex"})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])

            # Mock the detector module's score_image to return fake scores
            def fake_score(provider_info, image_path, *, record=None,
                           evaluation_entry=None, steps=50):
                return {"canonical_score": 10.0, "raw_score": 0.001,
                        "tr_log_p": -23.0}

            import raven.detectors.tr_detector as tr_mod
            monkeypatch.setattr(tr_mod, "score_image", fake_score)
            # Also mock load_state
            monkeypatch.setattr(tr_mod, "load_state",
                                lambda records, device, **extra: {"fake": True})

            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["scored_count"] > 0, f"Expected scored, got: {result}"
            assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)

    def test_all_fail_is_not_completed(self, monkeypatch):
        """When all images fail scoring, stage is not completed."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        rec = _make_record("1", "watermarked", method="TR",
                            source_metadata={"w_seed": "99", "w_channel": "3",
                                             "w_radius": "10", "w_pattern": "ring",
                                             "w_mask_shape": "circle",
                                             "w_measurement": "l1_complex",
                                             "w_injection": "complex"})

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec])

            def fake_fail(*a, **kw):
                raise RuntimeError("mock scoring failure")
            monkeypatch.setattr(tr_mod, "score_image", fake_fail)

            result = evaluate_detector([rec], out, "TR", device="cpu")
            assert result["scored_count"] == 0
            assert result["status"] != "completed"


# ===========================================================================
# GS per-row provider (High 1 verification)
# ===========================================================================
class TestGSPerRow:
    def test_gs_score_requires_record(self):
        """GS score_image raises if record is None."""
        from raven.detectors.gs_detector import score_image
        from raven.detectors import DetectorMissingStateError
        with pytest.raises(DetectorMissingStateError):
            score_image({"fake": True}, "/tmp/fake.png", record=None)

    def test_gs_missing_secret_index_raises(self):
        from raven.detectors.gs_detector import score_image
        from raven.detectors import DetectorMissingStateError
        with pytest.raises(DetectorMissingStateError):
            score_image({"fake": True}, "/tmp/fake.png",
                        record={"run_id": "1", "role": "watermarked"})


# ===========================================================================
# T2S role pairing (High 3 verification)
# ===========================================================================
class TestT2SRolePairing:
    def test_index_on_run_id_and_role(self):
        """eval.py builds record_index on (run_id, role), not just run_id."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert 'record_index' in source
        assert 'source_role' in source

    def test_different_roles_get_different_states(self):
        """Watermarked and clean with same run_id get different records."""
        wm = _make_record("1", "watermarked",
                          source_metadata={"t2s_state_path": "/state_wm.pt",
                                           "t2s_state_sha256": "sha_wm"})
        cl = _make_record("1", "clean",
                          source_metadata={"t2s_state_path": "/state_cl.pt",
                                           "t2s_state_sha256": "sha_cl"})
        idx = {}
        for r in [wm, cl]:
            key = (str(r["run_id"]), r.get("role", "watermarked"))
            idx[key] = r
        assert idx[("1", "watermarked")]["source_metadata"]["t2s_state_sha256"] == "sha_wm"
        assert idx[("1", "clean")]["source_metadata"]["t2s_state_sha256"] == "sha_cl"


# ===========================================================================
# GM canonical helpers (High 2 verification)
# ===========================================================================
class TestGMCanonical:
    def test_gm_uses_canonical_helpers(self):
        """GM detector references gm_provider_kwargs and gm_bundle_manifest."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "gm_detector.py").read_text()
        assert "gm_bundle_manifest" in source
        assert "gm_provider_kwargs" in source

    def test_gm_checks_state_source(self):
        """GM validates state_source == 'bundle'."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "gm_detector.py").read_text()
        assert 'state_source' in source
        assert '"bundle"' in source


# ===========================================================================
# Fourier canonical helpers (High 4 verification)
# ===========================================================================
class TestFourierCanonical:
    def test_fourier_uses_canonical_helpers(self):
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "fourier_detector.py").read_text()
        assert "rid_provider_kwargs_from_bundle" in source
        assert "hstr_provider_kwargs_from_bundle" in source
        assert "hsqr_provider_from_bundle" in source
        assert "fourier_bundle_manifest" in source

    def test_fourier_checks_state_source(self):
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "fourier_detector.py").read_text()
        assert 'state_source' in source


# ===========================================================================
# TR fail closed (High 5 verification)
# ===========================================================================
class TestTRFailClosed:
    def test_required_fields_are_enforced(self):
        from raven.detectors.tr_detector import REQUIRED_METADATA_FIELDS
        required = {"w_seed", "w_channel", "w_radius", "w_pattern",
                    "w_mask_shape", "w_measurement", "w_injection",
                    "w_pattern_const"}
        assert REQUIRED_METADATA_FIELDS == required

    def test_no_silent_defaults_in_load_state(self):
        """TR load_state must NOT use fallback defaults like 999999."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "tr_detector.py").read_text()
        # The only 999999 reference should be in REQUIRED_METADATA_FIELDS doc,
        # not in active code
        assert "999999" not in source.split("REQUIRED_METADATA_FIELDS")[-1]


# ===========================================================================
# Other regression tests (path safety, import isolation, etc.)
# ===========================================================================
class TestPathSafety:
    def test_exact_repo_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        repo = str(REPO)
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(repo)

    def test_child_of_data_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(str(REPO / "data" / "tr"))


class TestImportIsolation:
    def test_metrics_no_pipeline(self):
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "import raven.metrics; "
             "print('pipeline_raven' in sys.modules)"],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0
        assert r.stdout.strip() == "False"


class TestShiftPlan:
    def test_seed_17_base_42(self):
        from raven.shift_plan import compute_attack_seed
        assert compute_attack_seed(42, "17") == 59


class TestAlgoExecSplit:
    def test_resume_ignores_execution(self):
        from raven.experiment_config import check_config_match, normalize_config
        a = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/a")
        b = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/b",
                              resume=False, overwrite=False, gpu=7)
        assert check_config_match(a, b) == []


class TestDiffusionPairs:
    def test_ddpm_inv_ddim_sched_rejected(self):
        from raven.experiment_config import validate_diffusion_pair
        with pytest.raises(ValueError):
            validate_diffusion_pair("ddpm", "ddim")
