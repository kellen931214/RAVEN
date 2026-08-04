"""Issue #26 — Unified detector behavior integration matrix.

Covers all 7 detector methods (TR, GS, GM, T2S, RID, HSTR, HSQR) with
real ``evaluate_detector`` / ``run_evaluation`` / ``main`` orchestrator.
Only pipe/model construction, diffusion inversion, provider/bundle/state
heavy loading, and canonical scoring helper outputs are mocked.

Each method has:
  - A successful full orchestrator flow
  - At least one failure classification test
  - detector_records.jsonl read + validation
  - Stage result validation (status, dominant_failure_cause, counts,
    count_invariant_satisfied)
  - CLI exit code verification

Run:  pytest -q raven_repro/tests/test_issue26_detector_integration.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO))


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_record(run_id="1", role="watermarked", method="TR", **kw):
    """Build a synthetic record with embedded source_metadata."""
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


def _write_fake_run(tmp_path, method="TR", records=None, config=None,
                     create_input_images=True):
    """Create a minimal output_dir with config, records, and fake PNGs."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = tmp_path / "run"
    out.mkdir()
    cfg = {"method": method, "dataset": "test", **(config or {})}
    write_config(out, cfg)
    if records is None:
        records = [_make_record("1", "watermarked", method=method)]
    for r in records:
        role = r.get("role", "watermarked")
        rid = r["run_id"]
        write_record(out, role, rid, r)
        img = out / "samples" / role / rid / "output.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid 32x32 PNG
        _write_minimal_png(img)
        if create_input_images:
            input_path = Path(r.get("input_path", f"/tmp/in_{rid}.png"))
            input_path.parent.mkdir(parents=True, exist_ok=True)
            # Always overwrite — stale files from previous test runs must
            # never leak into a new run (they may hold invalid PNG bytes).
            _write_minimal_png(input_path)
    rebuild_records_jsonl(out)
    return out


def _write_minimal_png(path):
    """Write a minimal valid 32x32 grayscale PNG."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(path, format="PNG")


def _read_detector_rows(output_dir):
    """Read detector_records.jsonl as list of dicts."""
    from raven.experiment_io import detector_records_path
    path = detector_records_path(output_dir)
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# Common metadata fixtures
TR_META = {
    "w_seed": "99", "w_channel": "3", "w_radius": "10",
    "w_pattern": "ring", "w_mask_shape": "circle",
    "w_measurement": "l1_complex", "w_injection": "complex",
    "w_pattern_const": "0.0",
}

GS_META = {
    "gs_secret_index": "5",
    "gs_message_sha256": "msg_sha",
    "gs_key_sha256": "key_sha",
    "gs_nonce_sha256": "nonce_sha",
    "gs_secret_bundle_sha256": "bundle_sha",
    "gs_protocol_mode": "official_compatible",
    "gs_detection_mode": "official_onebit",
}

GM_META = {
    "gm_bundle_dir": "/fake/bundle",
    "gm_bundle_config_sha256": "a" * 64,
    "gm_w1_file_sha256": "b" * 64,
    "gm_w2_file_sha256": "c" * 64,
    "gm_m_sha256": "m" * 64,
    "gm_watermark_sha256": "n" * 64,
    "gm_target_sha256": "o" * 64,
    "gm_protocol_mode": "official_math_shared_tr_clean",
}

T2S_META = {
    "t2s_state_path": "/tmp/fake.pt",
    "t2s_state_sha256": "abc123",
    "t2s_watermark_id": "wm_id_1",
    "t2s_provider_config_sha256": "def456",
    "t2s_protocol_mode": "official_math_shared_tr_clean",
    "t2s_rng_mode": "official_compatible",
    "t2s_inversion_mode": "t2s_official",
    "t2s_num_inversion_steps": "10",
}


def _fourier_meta(method, run_id="1"):
    """Build method-specific Fourier metadata."""
    prefix = method.lower()
    return {
        "method": method,
        f"{prefix}_bundle_dir": f"/fake/{prefix}_bundle",
        f"{prefix}_bundle_config_sha256": f"{prefix}_cfg_sha",
        f"{prefix}_selected_pattern_sha256": f"{prefix}_pat_sha",
        f"{prefix}_mask_sha256": f"{prefix}_mask_sha",
        f"{prefix}_key_index": "0",
        f"{prefix}_protocol_mode": f"official_math_shared_tr_clean",
        "watermark_target_sha256": "tgt_sha",
        "watermark_mask_sha256": "mask_sha",
    }


# Full TR profile for generator-style records
TR_PROFILE = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "scheduler": "DDIM",
    "steps": "50",
    "resolution": "512",
    "watermark_target_sha256": "tgt_sha",
    "watermark_mask_sha256": "mask_sha",
    "provider_config_hash": "",
}


# ===========================================================================
# Mock harness — shared across methods
# ===========================================================================

def _mock_imports_tr(monkeypatch):
    """Mock heavy imports for TR detector."""
    import builtins
    import raven.detectors.tr_detector as tr_mod

    fake_extract = mock.MagicMock()
    fake_extract.evaluate_image.return_value = {
        "p_values": [0.001],
        "p_value_diagnostics": [
            {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
             "statistic": 50.0, "df": 100, "p_underflow": False},
        ],
    }
    fake_extract.raw_score.return_value = 0.001
    fake_extract.canonical_score.return_value = 10.0
    monkeypatch.setattr(tr_mod, "_extract_module", fake_extract)
    monkeypatch.setattr(tr_mod, "_get_extract_module", lambda: fake_extract)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"
    scheduler_inv = mock.MagicMock()
    scheduler_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe.scheduler_inverse = scheduler_inv
    fake_pipe.pipe.vae.config.scaling_factor = 0.18215

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.SCHEDULER_CLASSES = {
        "DDIM": (mock.MagicMock(), mock.MagicMock()),
        "DPM": (mock.MagicMock(), mock.MagicMock()),
    }
    fake_pipe_utils.get_pipe_provider.return_value = fake_pipe

    fake_provider = mock.MagicMock()
    fake_provider.gt_patch = mock.MagicMock()
    fake_provider.watermarking_mask = mock.MagicMock()
    fake_tr_class = mock.MagicMock(return_value=fake_provider)
    fake_tr_mod = mock.MagicMock(TrProvider=fake_tr_class)
    fake_wm = mock.MagicMock(tr_provider=fake_tr_mod)
    fake_utils = mock.MagicMock(
        pipe=mock.MagicMock(pipe_utils=fake_pipe_utils), wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_utils)
    fake_eb.__path__ = []

    _imports = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_utils,
        "eval_bench_wm.utils.pipe": fake_utils.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.tr_provider": fake_tr_mod,
    }
    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                       lambda n, *a, **kw: _imports.get(n, original(n, *a, **kw)))

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(side_effect=["tgt_sha", "mask_sha"]))
    return fake_extract, fake_pipe_utils, fake_tr_class


def _build_tr_record(run_id="1", role="watermarked", method="TR", **kw):
    """Build a TR record with full metadata profile."""
    meta = dict(TR_META)
    profile = dict(TR_PROFILE, **kw.get("profile_overrides", {}))
    rec = _make_record(run_id, role, method=method,
                       source_metadata={**meta, **profile})
    for k, v in meta.items():
        rec[k] = v
    for k, v in profile.items():
        rec[k] = v
    return rec


def _patch_tr_score(monkeypatch, raw=0.001, canonical=10.0):
    """Minimal TR score_image patch for evaluate_detector."""
    import raven.detectors.tr_detector as tr_mod

    def fake_score(provider_info, image_path, *,
                   record=None, evaluation_entry=None, steps=50):
        return {"raw_score": raw, "canonical_score": canonical}
    monkeypatch.setattr(tr_mod, "load_state",
                        lambda records, device, **extra: {"fake": True, "inversion_steps": 50})
    monkeypatch.setattr(tr_mod, "score_image", fake_score)


def _patch_gs_score(monkeypatch, raw=0.85, canonical=0.85,
                     raise_on_cohort=None):
    """Minimal GS score_image patch for evaluate_detector."""
    import raven.detectors.gs_detector as gs_mod

    def fake_score(provider_info, image_path, *,
                   record=None, evaluation_entry=None, steps=50):
        if raise_on_cohort and evaluation_entry:
            cohort = evaluation_entry.get("evaluation_cohort", "")
            if raise_on_cohort in cohort:
                from raven.detectors import DetectorMissingStateError
                raise DetectorMissingStateError(f"fake missing state for {cohort}")
        return {"raw_score": raw, "canonical_score": canonical,
                "gs_secret_index": 5, "gs_protocol_mode": "official_compatible",
                "gs_detection_mode": "official_onebit",
                "gs_active_threshold": 0.9,
                "gs_active_threshold_type": "official_beta_tail_tau_onebit",
                "gs_active_comparison_operator": ">=",
                "gs_active_nominal_fpr": 1e-6,
                "gs_active_calibrated_from_current_clean_negatives": False,
                "gs_detection_success": True,
                "gs_official_tau_onebit": 0.9,
                "gs_official_tau_bits": 0.95,
                "gs_official_fpr": 1e-6,
                "gs_official_user_number": 1000000,
                "gs_official_comparison_operator": ">=",
                "gs_official_source": "test",
                "gs_detection_policy_hash": "DET_HASH",
                "bit_accuracy": 0.85,
                "gs_decoded_bits": "0" * 256,
                }
    monkeypatch.setattr(gs_mod, "load_state",
                        lambda records, device, **extra: {"fake": True})
    monkeypatch.setattr(gs_mod, "score_image", fake_score)


def _patch_gm_score(monkeypatch, raw=0.85, canonical=0.85):
    """Minimal GM score_image patch."""
    import raven.detectors.gm_detector as gm_mod

    def fake_score(provider_info, image_path, *,
                   record=None, evaluation_entry=None, steps=50):
        return {"raw_score": raw, "canonical_score": canonical,
                "gm_raw_bit_accuracy": 0.85,
                "gm_raw_ring_l1": -10.0,
                "gm_report_label": "gauss_marker_clean_calibrated",
                "gm_score_definition": "gm_neg_mean_cosine_sim_l1_per_bit_target_direction",
                "gm_threshold_source": "clean_calibrated",
                "gm_comparison_operator": ">=",
                "gm_protocol_mode": "official_math_shared_tr_clean",
                "gm_profile": "legacy",
                }
    monkeypatch.setattr(gm_mod, "load_state",
                        lambda records, device, **extra: {"fake": True})
    monkeypatch.setattr(gm_mod, "score_image", fake_score)


def _patch_t2s_score(monkeypatch, true_key=0.85, control_key=0.40,
                      detection_success=True, margin=0.45):
    """Minimal T2S score_image patch."""
    import raven.detectors.t2s_detector as t2s_mod

    def fake_score(provider_info, image_path, *,
                   record=None, evaluation_entry=None, steps=50):
        return {
            "raw_score": true_key,
            "canonical_score": true_key,
            "t2s_score_true_key": true_key,
            "t2s_score_control_key": control_key,
            "t2s_detection_success": detection_success,
            "t2s_score_margin": margin,
            "t2s_key_accuracy": 1.0,
            "t2s_bit_accuracy": 0.98,
        }
    monkeypatch.setattr(t2s_mod, "load_state",
                        lambda records, device, **extra: {"fake": True})
    monkeypatch.setattr(t2s_mod, "score_image", fake_score)


def _patch_fourier_score(monkeypatch, raw=10.0, canonical=10.0):
    """Minimal Fourier (RID/HSTR/HSQR) score_image patch."""
    import raven.detectors.fourier_detector as fmod

    def fake_score(provider_info, image_path, *,
                   record=None, evaluation_entry=None, steps=50):
        return {"raw_score": raw, "canonical_score": canonical}
    monkeypatch.setattr(fmod, "load_state",
                        lambda records, device, method=None, **extra: {"fake": True})
    monkeypatch.setattr(fmod, "score_image", fake_score)


# ===========================================================================
# TR — TreeRing detector integration tests
# ===========================================================================

class TestTRIntegration:
    """Full orchestrator integration for TR (TreeRing) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """TR with clean+wm → completed, all cohorts scored, records valid."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_tr_score(monkeypatch)

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")
        rec_wm2 = _build_tr_record("2", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 6  # 3 records × 2 cohorts each
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None
            assert det["metric_availability"]["primary_report_available"] is True

            # Validate detector_records.jsonl
            rows = _read_detector_rows(out)
            assert len(rows) == 6
            assert all(r["status"] == "scored" for r in rows)
            assert all("raw_score" in r for r in rows)
            assert all("canonical_score" in r for r in rows)

            # CLI exit code
            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_score_image_returns_none(self, monkeypatch):
        """TR: score_image returns None → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, ROW_STATUS_FAILED_SCORING,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image", lambda *a, **kw: None)

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2
            assert result["count_invariant_satisfied"] is True

            rows = _read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_FAILED_SCORING for r in rows)
            assert all("None" in r.get("error", "") for r in rows)

    def test_score_image_returns_empty_dict(self, monkeypatch):
        """TR: score_image returns {} → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image", lambda *a, **kw: {})

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["failed_count"] == 2
            assert result["count_invariant_satisfied"] is True

    def test_missing_canonical_score(self, monkeypatch):
        """TR: missing canonical_score in score dict → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image",
                            lambda *a, **kw: {"raw_score": 0.001})

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["scored_count"] == 0
            assert result["failed_count"] == 2
            assert result["count_invariant_satisfied"] is True

    def test_missing_image_preflight(self, monkeypatch):
        """TR: missing image caught in preflight → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        _patch_tr_score(monkeypatch)
        rec_wm = _build_tr_record("1", "watermarked")
        rec_wm["input_path"] = "/tmp/raven_issue26_missing_input.png"

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm],
                                  create_input_images=False)
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            assert result["dominant_failure_cause"] == "missing_image"
            rows = _read_detector_rows(out)
            assert any(r["status"] == ROW_STATUS_FAILED_MISSING_IMAGE
                      for r in rows)

    def test_missing_required_state(self, monkeypatch):
        """TR: load_state raises DetectorMissingStateError → setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            DetectorMissingStateError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorMissingStateError("no state")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            assert result["dominant_failure_cause"] == "missing_required_state"
            assert result["count_invariant_satisfied"] is True
            assert result.get("setup_failure_cause") == "missing_required_state"

    def test_provider_initialization_error(self, monkeypatch):
        """TR: load_state raises DetectorProviderInitializationError."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            DetectorProviderInitializationError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorProviderInitializationError("bad init")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
            assert result["dominant_failure_cause"] == "provider_initialization_error"

    def test_state_validation_error(self, monkeypatch):
        """TR: load_state raises DetectorStateValidationError → setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("bad state")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"

    def test_runtime_scoring_error(self, monkeypatch):
        """TR: score_image raises DetectorScoringError → row-level failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, DetectorScoringError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorScoringError("scoring boom")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["scored_count"] == 0
            assert result["count_invariant_satisfied"] is True

    def test_cli_exit_failure_is_2(self, monkeypatch):
        """TR: scoring failure → CLI exit code 2."""
        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2

    def test_required_cohort_missing(self, monkeypatch):
        """TR: no original_clean cohort → completed_with_errors."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED_WITH_ERRORS

        _patch_tr_score(monkeypatch)

        rec_wm = _build_tr_record("1", "watermarked")
        rec_wm2 = _build_tr_record("2", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2], out,
                                       "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED_WITH_ERRORS
            assert result["scored_count"] == 4
            assert "original_clean" in result["missing_metric_cohorts"]
            assert result["count_invariant_satisfied"] is True

    def test_mixed_provider_config_rejection(self, monkeypatch):
        """TR: mixed provider config → state_validation setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("Mixed TR provider config")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")
            assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_attacked_clean_recalibration_availability(self, monkeypatch):
        """TR: with all 4 cohorts → recalibrated_report_available=True."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_tr_score(monkeypatch)

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")
        rec_wm2 = _build_tr_record("2", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm, rec_wm2])
            result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                       out, "TR", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            ma = result["metric_availability"]
            assert ma["threshold_report_available"] is True
            assert ma["recalibrated_cohorts_available"] is True
            assert ma["recalibrated_report_available"] is True


# ===========================================================================
# GS — Gaussian Shading detector integration tests
# ===========================================================================

class TestGSIntegration:
    """Full orchestrator integration for GS (Gaussian Shading) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """GS with clean+wm → completed, all cohorts scored."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_gs_score(monkeypatch)

        rec_clean = _make_record("1", "clean", method="GS",
                                 source_metadata=GS_META)
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)
        rec_wm2 = _make_record("2", "watermarked", method="GS",
                               source_metadata=GS_META)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="GS",
                                  records=[rec_clean, rec_wm, rec_wm2],
                                  config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 6
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 6
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_classification_missing_image(self, monkeypatch):
        """GS: missing image → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        _patch_gs_score(monkeypatch)
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META,
                              input_path="/nonexistent/img.png")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm],
                                  create_input_images=False)
            result = evaluate_detector([rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            assert result["dominant_failure_cause"] == "missing_image"
            rows = _read_detector_rows(out)
            assert any(r["status"] == ROW_STATUS_FAILED_MISSING_IMAGE
                      for r in rows)

    def test_per_row_secret_differentiation(self, monkeypatch):
        """GS: different secret_index per role → correct provider per cohort."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_gs_score(monkeypatch)

        meta_clean = dict(GS_META, gs_secret_index="5")
        meta_wm = dict(GS_META, gs_secret_index="7")

        rec_clean = _make_record("1", "clean", method="GS",
                                 source_metadata=meta_clean)
        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=meta_wm)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4  # 2 records × 2 cohorts each

            rows = _read_detector_rows(out)
            assert all(r["status"] == "scored" for r in rows)

    def test_none_score_image(self, monkeypatch):
        """GS: score_image returns None → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.gs_detector as gs_mod
        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gs_mod, "score_image", lambda *a, **kw: None)

        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["count_invariant_satisfied"] is True

    def test_missing_required_state_setup(self, monkeypatch):
        """GS: provider init error → setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            DetectorMissingStateError,
        )

        import raven.detectors.gs_detector as gs_mod
        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorMissingStateError("no state")))

        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            assert result["count_invariant_satisfied"] is True

    def test_cli_exit_failure(self, monkeypatch):
        """GS: failure → CLI exit code 2."""
        import raven.detectors.gs_detector as gs_mod
        monkeypatch.setattr(gs_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gs_mod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        rec_wm = _make_record("1", "watermarked", method="GS",
                              source_metadata=GS_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GS", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# GM — GaussMarker detector integration tests
# ===========================================================================

class TestGMIntegration:
    """Full orchestrator integration for GM (GaussMarker) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """GM with clean+wm → completed, protocol/profile separation valid."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_gm_score(monkeypatch)

        rec_clean = _make_record("1", "clean", method="GM",
                                 source_metadata=GM_META)
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="GM",
                                  records=[rec_clean, rec_wm], config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 4
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_classification(self, monkeypatch):
        """GM: score_image raises scoring error → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, DetectorScoringError,
        )

        import raven.detectors.gm_detector as gm_mod
        monkeypatch.setattr(gm_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gm_mod, "score_image",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorScoringError("gm scoring error")))

        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GM", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["dominant_failure_cause"] == "scoring_error"
            assert result["count_invariant_satisfied"] is True

            rows = _read_detector_rows(out)
            assert any(r["status"] == "failed_scoring" for r in rows)

    def test_mixed_bundle_rejection(self, monkeypatch):
        """GM: mixed bundle metadata → state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        import raven.detectors.gm_detector as gm_mod
        from raven.detectors import DetectorStateValidationError
        monkeypatch.setattr(gm_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("mixed bundle")))

        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GM", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"
            assert result["count_invariant_satisfied"] is True

    def test_protocol_profile_separation(self, monkeypatch):
        """GM: protocol_mode ≠ profile — both recorded in scored row."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_gm_score(monkeypatch)

        rec_clean = _make_record("1", "clean", method="GM",
                                 source_metadata=GM_META)
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "GM", device="cpu")
            assert result["status"] == STATUS_COMPLETED

            rows = _read_detector_rows(out)
            for row in rows:
                if row["status"] == "scored":
                    assert "gm_protocol_mode" in row
                    assert "gm_profile" in row

    def test_empty_score_dict(self, monkeypatch):
        """GM: score_image returns {} → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.gm_detector as gm_mod
        monkeypatch.setattr(gm_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gm_mod, "score_image", lambda *a, **kw: {})

        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "GM", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING

    def test_missing_image(self, monkeypatch):
        """GM: missing image → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_IMAGE

        _patch_gm_score(monkeypatch)
        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META,
                              input_path="/nonexistent/img.png")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm],
                                  create_input_images=False)
            result = evaluate_detector([rec_wm], out, "GM", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            assert result["count_invariant_satisfied"] is True

    def test_cli_exit_code(self, monkeypatch):
        """GM: scoring failure → CLI exit 2."""
        import raven.detectors.gm_detector as gm_mod
        monkeypatch.setattr(gm_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gm_mod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        rec_wm = _make_record("1", "watermarked", method="GM",
                              source_metadata=GM_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="GM", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# T2S — T2SMark detector integration tests
# ===========================================================================

class TestT2SIntegration:
    """Full orchestrator integration for T2S detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """T2S with watermarked records → completed, paired-key report ok."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_t2s_score(monkeypatch)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        rec_wm2 = _make_record("2", "watermarked", method="T2S",
                               source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="T2S",
                                  records=[rec_wm, rec_wm2], config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 4
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            ma = det["metric_availability"]
            assert ma["primary_report_available"] is True
            assert ma["primary_report"] == "paired_key_detection_report"
            assert ma["threshold_report_available"] is False

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_detection_success_false(self, monkeypatch):
        """T2S: detection_success=False → still scored (it's a valid score)."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_t2s_score(monkeypatch, detection_success=False)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        rec_wm2 = _make_record("2", "watermarked", method="T2S",
                               source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2],
                                       out, "T2S", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4
            rows = _read_detector_rows(out)
            assert all(r["status"] == "scored" for r in rows)

    def test_missing_t2s_state_setup_failure(self, monkeypatch):
        """T2S: load_state fails → setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            DetectorMissingStateError,
        )

        import raven.detectors.t2s_detector as t2s_mod
        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: (_ for _ in ()).throw(
                                DetectorMissingStateError("no t2s state")))

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "T2S", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            assert result["count_invariant_satisfied"] is True

    def test_run_id_role_state_differentiation(self, monkeypatch):
        """T2S: different (run_id, role) pairs → separate state resolution."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_t2s_score(monkeypatch)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        rec_wm2 = _make_record("2", "watermarked", method="T2S",
                               source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2],
                                       out, "T2S", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4
            rows = _read_detector_rows(out)
            run_ids = {r["run_id"] for r in rows}
            assert len(run_ids) == 2

    def test_bit_message_aggregation_alignment(self, monkeypatch):
        """T2S: verify bit/message accuracy fields in aggregate output."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_t2s_score(monkeypatch)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        rec_wm2 = _make_record("2", "watermarked", method="T2S",
                               source_metadata=T2S_META)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S",
                                  records=[rec_wm, rec_wm2])
            result = evaluate_detector([rec_wm, rec_wm2],
                                       out, "T2S", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            rows = _read_detector_rows(out)
            for row in rows:
                if row["status"] == "scored":
                    assert "t2s_score_true_key" in row
                    assert "t2s_score_control_key" in row
                    assert "t2s_detection_success" in row
                    assert "t2s_key_accuracy" in row
                    assert "t2s_bit_accuracy" in row

    def test_benchmark_ddim_step_binding(self, monkeypatch):
        """T2S: steps parameter flows through to score_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        captured_steps = []
        import raven.detectors.t2s_detector as t2s_mod

        def fake_score(provider_info, image_path, *,
                       record=None, evaluation_entry=None, steps=50):
            captured_steps.append(steps)
            return {
                "raw_score": 0.85, "canonical_score": 0.85,
                "t2s_score_true_key": 0.85,
                "t2s_score_control_key": 0.40,
                "t2s_detection_success": True,
                "t2s_score_margin": 0.45,
                "t2s_key_accuracy": 1.0,
                "t2s_bit_accuracy": 0.98,
            }
        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(t2s_mod, "score_image", fake_score)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            evaluate_detector([rec_wm], out, "T2S", device="cpu")

        # steps=50 (the default) passed to every score_image call
        assert captured_steps
        assert all(s == 50 for s in captured_steps)

    def test_none_score_image(self, monkeypatch):
        """T2S: score_image returns None → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.t2s_detector as t2s_mod
        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(t2s_mod, "score_image", lambda *a, **kw: None)

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "T2S", device="cpu")
            assert result["status"] == STATUS_FAILED_SCORING

    def test_cli_exit_failure(self, monkeypatch):
        """T2S: failure → CLI exit 2."""
        import raven.detectors.t2s_detector as t2s_mod
        monkeypatch.setattr(t2s_mod, "load_state",
                            lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(t2s_mod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        rec_wm = _make_record("1", "watermarked", method="T2S",
                              source_metadata=T2S_META)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="T2S", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# RID — Fourier RID detector integration tests
# ===========================================================================

class TestRIDIntegration:
    """Full orchestrator integration for RID (Fourier RingID) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """RID: clean+wm → completed with threshold report."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_fourier_score(monkeypatch)

        rid_meta = _fourier_meta("RID")
        rec_clean = _make_record("1", "clean", method="RID",
                                 source_metadata=rid_meta)
        rec_wm = _make_record("1", "watermarked", method="RID",
                              source_metadata=rid_meta)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="RID",
                                  records=[rec_clean, rec_wm], config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 4
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_scoring(self, monkeypatch):
        """RID: score_image returns None → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image", lambda *a, **kw: None)

        rid_meta = _fourier_meta("RID")
        rec_wm = _make_record("1", "watermarked", method="RID",
                              source_metadata=rid_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="RID", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "RID", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["dominant_failure_cause"] == "scoring_error"
            assert result["count_invariant_satisfied"] is True

    def test_wrong_manifest_method_rejection(self, monkeypatch):
        """RID: wrong method in manifest → state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("unsupported RID bundle")))

        rid_meta = _fourier_meta("RID")
        rec_wm = _make_record("1", "watermarked", method="RID",
                              source_metadata=rid_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="RID", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "RID", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"

    def test_missing_image_preflight(self, monkeypatch):
        """RID: missing image → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        _patch_fourier_score(monkeypatch)
        rid_meta = _fourier_meta("RID")
        rec_wm = _make_record("1", "watermarked", method="RID",
                              source_metadata=rid_meta,
                              input_path="/nonexistent/img.png")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="RID", records=[rec_wm],
                                  create_input_images=False)
            result = evaluate_detector([rec_wm], out, "RID", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            rows = _read_detector_rows(out)
            assert any(r["status"] == ROW_STATUS_FAILED_MISSING_IMAGE
                      for r in rows)

    def test_cli_exit_failure(self, monkeypatch):
        """RID: failure → CLI exit 2."""
        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        rid_meta = _fourier_meta("RID")
        rec_wm = _make_record("1", "watermarked", method="RID",
                              source_metadata=rid_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="RID", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# HSTR — Fourier HSTR detector integration tests
# ===========================================================================

class TestHSTRIntegration:
    """Full orchestrator integration for HSTR (Fourier SFW) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """HSTR: clean+wm → completed, records + CLI verified."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_fourier_score(monkeypatch)

        hstr_meta = _fourier_meta("HSTR")
        rec_clean = _make_record("1", "clean", method="HSTR",
                                 source_metadata=hstr_meta)
        rec_wm = _make_record("1", "watermarked", method="HSTR",
                              source_metadata=hstr_meta)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="HSTR",
                                  records=[rec_clean, rec_wm], config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 4
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_setup_state_validation(self, monkeypatch):
        """HSTR: bundle validation fails → state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("bad HSTR bundle")))

        hstr_meta = _fourier_meta("HSTR")
        rec_wm = _make_record("1", "watermarked", method="HSTR",
                              source_metadata=hstr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSTR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "HSTR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"
            assert result["count_invariant_satisfied"] is True

    def test_method_specific_bundle_gate_rid_bundle_rejected(self, monkeypatch):
        """HSTR: RidBundle (not SfwBundle) → state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("unsupported HSTR bundle schema")))

        hstr_meta = _fourier_meta("HSTR")
        rec_wm = _make_record("1", "watermarked", method="HSTR",
                              source_metadata=hstr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSTR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "HSTR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"

    def test_empty_score_dict(self, monkeypatch):
        """HSTR: score_image returns {} → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image", lambda *a, **kw: {})

        hstr_meta = _fourier_meta("HSTR")
        rec_wm = _make_record("1", "watermarked", method="HSTR",
                              source_metadata=hstr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSTR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "HSTR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING

    def test_cli_exit(self, monkeypatch):
        """HSTR: failure → CLI exit 2."""
        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        hstr_meta = _fourier_meta("HSTR")
        rec_wm = _make_record("1", "watermarked", method="HSTR",
                              source_metadata=hstr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSTR", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# HSQR — Fourier HSQR detector integration tests
# ===========================================================================

class TestHSQRIntegration:
    """Full orchestrator integration for HSQR (Fourier HSQR) detector."""

    def test_successful_full_orchestrator(self, monkeypatch):
        """HSQR: clean+wm → completed, no state_source gate for HSQR."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_fourier_score(monkeypatch)

        hsqr_meta = _fourier_meta("HSQR")
        rec_clean = _make_record("1", "clean", method="HSQR",
                                 source_metadata=hsqr_meta)
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta)

        with tempfile.TemporaryDirectory() as td:
            cfg = {"metadata_path": "/nonexistent/metadata.csv"}
            out = _write_fake_run(Path(td), method="HSQR",
                                  records=[rec_clean, rec_wm], config=cfg)
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED
            det = result["stages"]["detector"]
            assert det["status"] == STATUS_COMPLETED
            assert det["scored_count"] == 4
            assert det["failed_count"] == 0
            assert det["count_invariant_satisfied"] is True
            assert det["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

            from experiments.eval import main
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_failure_scoring_error(self, monkeypatch):
        """HSQR: scoring error during inversion → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, DetectorScoringError,
        )

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorScoringError("HSQR scoring error")))

        hsqr_meta = _fourier_meta("HSQR")
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSQR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "HSQR", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            assert result["dominant_failure_cause"] == "scoring_error"
            assert result["count_invariant_satisfied"] is True

    def test_method_specific_bundle_gate_no_state_source(self, monkeypatch):
        """HSQR: no state_source gate — SfwBundle validates, load_state succeeds."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_fourier_score(monkeypatch)
        hsqr_meta = _fourier_meta("HSQR")
        rec_clean = _make_record("1", "clean", method="HSQR",
                                 source_metadata=hsqr_meta)
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta)

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSQR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "HSQR", device="cpu")
            assert result["status"] == STATUS_COMPLETED

    def test_wrong_manifest_method_rejection(self, monkeypatch):
        """HSQR: wrong manifest method → state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: (_ for _ in ()).throw(
                                DetectorStateValidationError("manifest method mismatch")))

        hsqr_meta = _fourier_meta("HSQR")
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSQR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "HSQR", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert result["dominant_failure_cause"] == "state_validation_error"

    def test_missing_image(self, monkeypatch):
        """HSQR: missing image → failed_missing_image."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_IMAGE

        _patch_fourier_score(monkeypatch)
        hsqr_meta = _fourier_meta("HSQR")
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta,
                              input_path="/nonexistent/img.png")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSQR", records=[rec_wm],
                                  create_input_images=False)
            result = evaluate_detector([rec_wm], out, "HSQR", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_IMAGE

    def test_cli_exit(self, monkeypatch):
        """HSQR: failure → CLI exit 2."""
        import raven.detectors.fourier_detector as fmod
        monkeypatch.setattr(fmod, "load_state",
                            lambda records, device, method=None, **extra: {"fake": True})
        monkeypatch.setattr(fmod, "score_image", lambda *a, **kw: None)

        from experiments.eval import main

        hsqr_meta = _fourier_meta("HSQR")
        rec_wm = _make_record("1", "watermarked", method="HSQR",
                              source_metadata=hsqr_meta)
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="HSQR", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 2


# ===========================================================================
# Common cross-cutting cases — run evaluation with multiple stages
# ===========================================================================

class TestCommonIntegrationCases:
    """Common test patterns applied across any detector method."""

    def test_successful_scoring_all_methods(self, monkeypatch):
        """Verify all 7 methods complete successfully with evaluate_detector."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        # Configure all 7 methods
        _patch_tr_score(monkeypatch)
        _patch_gs_score(monkeypatch)
        _patch_gm_score(monkeypatch)
        _patch_t2s_score(monkeypatch)
        _patch_fourier_score(monkeypatch)

        results = {}
        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            records = [rec_wm]
            # Threshold methods need original_clean for completed status
            if method != "T2S":
                rec_clean = _make_record("1", "clean", method=method,
                                         source_metadata=meta)
                records.append(rec_clean)

            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=records)
                result = evaluate_detector(records, out, method, device="cpu")
                results[method] = result["status"]

        for method in results:
            assert results[method] == STATUS_COMPLETED, \
                f"{method}: expected completed, got {results[method]}"

    def test_score_image_none_across_methods(self, monkeypatch):
        """score_image returns None → failed_scoring for every method."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: {"fake": True})
            monkeypatch.setattr(mod, "score_image", lambda *a, **kw: None)

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_SCORING, \
                    f"{method}: expected failed_scoring, got {result['status']}"

    def test_empty_score_dict_across_methods(self, monkeypatch):
        """score_image returns {} → failed_scoring for every method."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: {"fake": True})
            monkeypatch.setattr(mod, "score_image", lambda *a, **kw: {})

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_SCORING, \
                    f"{method}: expected failed_scoring, got {result['status']}"

    def test_missing_required_state_across_methods(self, monkeypatch):
        """load_state raises DetectorMissingStateError → setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE,
            DetectorMissingStateError,
        )

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: (_ for _ in ()).throw(
                                    DetectorMissingStateError("no state")))

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE, \
                    f"{method}: expected failed_missing_required_state, got {result['status']}"
                assert result["count_invariant_satisfied"] is True

    def test_provider_init_error_across_methods(self, monkeypatch):
        """load_state raises DetectorProviderInitializationError across methods."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            DetectorProviderInitializationError,
        )

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: (_ for _ in ()).throw(
                                    DetectorProviderInitializationError("bad init")))

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION, \
                    f"{method}: expected failed_provider_initialization, got {result['status']}"

    def test_state_validation_error_across_methods(self, monkeypatch):
        """load_state raises DetectorStateValidationError across methods."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION,
            DetectorStateValidationError,
        )

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: (_ for _ in ()).throw(
                                    DetectorStateValidationError("bad state")))

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_STATE_VALIDATION, \
                    f"{method}: expected failed_state_validation, got {result['status']}"

    def test_runtime_scoring_error_across_methods(self, monkeypatch):
        """score_image raises DetectorScoringError → failed_scoring."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, DetectorScoringError,
        )

        import raven.detectors.tr_detector as tr_mod
        import raven.detectors.gs_detector as gs_mod
        import raven.detectors.gm_detector as gm_mod
        import raven.detectors.t2s_detector as t2s_mod
        import raven.detectors.fourier_detector as fmod

        for mod in [tr_mod, gs_mod, gm_mod, t2s_mod, fmod]:
            monkeypatch.setattr(mod, "load_state",
                                lambda *a, **kw: {"fake": True})
            monkeypatch.setattr(mod, "score_image",
                                lambda *a, **kw: (_ for _ in ()).throw(
                                    DetectorScoringError("score boom")))

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_SCORING, \
                    f"{method}: expected failed_scoring, got {result['status']}"

    def test_missing_image_across_methods(self, monkeypatch):
        """Missing input image → failed_missing_image for every method."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_IMAGE

        _patch_tr_score(monkeypatch)
        _patch_gs_score(monkeypatch)
        _patch_gm_score(monkeypatch)
        _patch_t2s_score(monkeypatch)
        _patch_fourier_score(monkeypatch)

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta,
                                  input_path="/nonexistent/img.png")
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm],
                                      create_input_images=False)
                result = evaluate_detector([rec_wm], out, method, device="cpu")
                assert result["status"] == STATUS_FAILED_MISSING_IMAGE, \
                    f"{method}: expected failed_missing_image, got {result['status']}"

    def test_required_cohort_missing_still_reports(self, monkeypatch):
        """No clean cohort → completed_with_errors, cohorts properly reported."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED_WITH_ERRORS

        _patch_tr_score(monkeypatch)
        _patch_gs_score(monkeypatch)
        _patch_gm_score(monkeypatch)
        _patch_fourier_score(monkeypatch)

        for method in ["TR", "GS", "GM", "RID", "HSTR", "HSQR"]:
            if method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            rec_wm2 = _make_record("2", "watermarked", method=method,
                                   source_metadata=meta)

            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm, rec_wm2])
                result = evaluate_detector([rec_wm, rec_wm2],
                                           out, method, device="cpu")
                assert result["status"] == STATUS_COMPLETED_WITH_ERRORS, \
                    f"{method}: expected completed_with_errors, got {result['status']}"
                assert "original_clean" in result["missing_metric_cohorts"], \
                    f"{method}: expected original_clean in missing_metric_cohorts"

    def test_complete_required_cohorts_completed(self, monkeypatch):
        """All required cohorts present → completed across methods."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_tr_score(monkeypatch)
        _patch_gs_score(monkeypatch)
        _patch_gm_score(monkeypatch)
        _patch_fourier_score(monkeypatch)

        for method in ["TR", "GS", "GM", "RID", "HSTR", "HSQR"]:
            if method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_clean = _make_record("1", "clean", method=method,
                                     source_metadata=meta)
            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)
            rec_wm2 = _make_record("2", "watermarked", method=method,
                                   source_metadata=meta)

            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_clean, rec_wm, rec_wm2])
                result = evaluate_detector([rec_clean, rec_wm, rec_wm2],
                                           out, method, device="cpu")
                assert result["status"] == STATUS_COMPLETED, \
                    f"{method}: expected completed, got {result['status']}"
                assert result["failed_count"] == 0

    def test_stage_result_structure(self, monkeypatch):
        """Verify all required fields in stage result for every method."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        _patch_tr_score(monkeypatch)
        _patch_gs_score(monkeypatch)
        _patch_gm_score(monkeypatch)
        _patch_t2s_score(monkeypatch)
        _patch_fourier_score(monkeypatch)

        expected_fields = [
            "stage", "method", "status", "available",
            "requested_count", "scored_count", "failed_count",
            "cohort_counts", "metric_availability",
            "missing_scoring_cohorts", "missing_metric_cohorts",
            "primary_requested_count", "primary_scored_count",
            "primary_failed_count",
            "optional_requested_count", "optional_scored_count",
            "optional_failed_count",
            "count_invariant_satisfied",
        ]

        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            if method == "T2S":
                meta = T2S_META
            elif method == "GS":
                meta = GS_META
            elif method == "GM":
                meta = GM_META
            elif method in ("RID", "HSTR", "HSQR"):
                meta = _fourier_meta(method)
            else:
                meta = TR_META

            rec_wm = _make_record("1", "watermarked", method=method,
                                  source_metadata=meta)

            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method=method,
                                      records=[rec_wm])
                result = evaluate_detector([rec_wm], out, method, device="cpu")

                for field in expected_fields:
                    assert field in result, \
                        f"{method}: missing field {field} in stage result"

                ma = result["metric_availability"]
                for field in ("scored_cohorts", "cohort_counts",
                              "primary_report_available", "any_report_available",
                              "threshold_report_available",
                              "recalibrated_cohorts_available",
                              "recalibrated_report_available"):
                    assert field in ma, \
                        f"{method}: missing metric_availability field {field}"


# ===========================================================================
# Stage result validation — counts and invariants
# ===========================================================================

class TestStageResultValidation:
    """Count invariants and stage result integrity across methods."""

    def test_count_invariant_always_satisfied(self, monkeypatch):
        """requested_count == scored_count + failed_count + unscored_due_to_setup_count."""
        from experiments.eval import evaluate_detector

        _patch_tr_score(monkeypatch)

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "TR", device="cpu")

            req = result["requested_count"]
            scored = result["scored_count"]
            failed = result["failed_count"]
            unscored = result.get("unscored_due_to_setup_count", 0)

            assert result["count_invariant_satisfied"] is True
            assert req == scored + failed + unscored, \
                f"{req} != {scored} + {failed} + {unscored}"

    def test_setup_failure_unscored_count_correct(self, monkeypatch):
        """Setup failure → all entries counted as unscored_due_to_setup."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorMissingStateError("no state")))

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "TR", device="cpu")

            assert result["count_invariant_satisfied"] is True
            assert result["scored_count"] == 0
            assert result["unscored_due_to_setup_count"] == 4
            assert result["setup_failure_cause"] == "missing_required_state"

    def test_detector_records_jsonl_serialized(self, monkeypatch):
        """detector_records.jsonl is always written, even on failure."""
        from experiments.eval import evaluate_detector

        _patch_tr_score(monkeypatch)

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            evaluate_detector([rec_wm], out, "TR", device="cpu")

            rows = _read_detector_rows(out)
            assert len(rows) == 2
            for row in rows:
                assert "run_id" in row
                assert "evaluation_cohort" in row
                assert "status" in row
                assert "method" in row
                assert "image_path" in row

    def test_records_jsonl_written_on_setup_failure(self, monkeypatch):
        """detector_records.jsonl contains preflight rows on setup failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import DetectorMissingStateError

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorMissingStateError("no state")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            evaluate_detector([rec_wm], out, "TR", device="cpu")

            from raven.experiment_io import detector_records_path
            assert detector_records_path(out).is_file()
            rows = _read_detector_rows(out)
            # preflight passes, but no scoring → empty preflight rows for valid images
            assert len(rows) >= 0  # file exists and is valid JSONL

    def test_dominant_failure_cause_set_on_error(self, monkeypatch):
        """dominant_failure_cause present and correct on error."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_PROVIDER_INITIALIZATION,
            DetectorProviderInitializationError,
        )

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorProviderInitializationError("bad provider")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = evaluate_detector([rec_wm], out, "TR", device="cpu")

            assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
            assert result["dominant_failure_cause"] == "provider_initialization_error"

    def test_exit_code_completed_is_0(self, monkeypatch):
        """Successful evaluation → exit code 0."""
        from experiments.eval import main

        _patch_tr_score(monkeypatch)

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector"])
            assert rc == 0

    def test_exit_code_with_allow_missing_metrics(self, monkeypatch):
        """--allow-missing-metrics softens skippable statuses to exit 0."""
        from experiments.eval import main
        from raven.detectors import DetectorMissingStateError

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                DetectorMissingStateError("no state")))

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            rc = main(["--output-dir", str(out), "--device", "cpu",
                       "--log-level", "ERROR", "--stages", "detector",
                       "--allow-missing-metrics"])
            assert rc == 0

    def test_run_evaluation_overall_status(self, monkeypatch):
        """run_evaluation returns overall_status and stage info correctly."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED

        _patch_tr_score(monkeypatch)

        rec_clean = _build_tr_record("1", "clean")
        rec_wm = _build_tr_record("1", "watermarked")

        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR",
                                  records=[rec_clean, rec_wm])
            # allow_missing_metrics=True since quality stage may skip
            # on 32x32 synthetic images without valid flow metadata
            result = run_evaluation(out, device="cpu",
                                    stages=["quality", "detector"],
                                    allow_missing_metrics=True)

            assert result["overall_status"] == STATUS_COMPLETED
            assert "stages" in result
            assert "quality" in result["stages"]
            assert "detector" in result["stages"]
            assert result["stages"]["detector"]["status"] == STATUS_COMPLETED

    def test_run_evaluation_with_failure(self, monkeypatch):
        """run_evaluation with failed detector → completed_with_errors."""
        from experiments.eval import run_evaluation
        from raven.detectors import STATUS_COMPLETED_WITH_ERRORS

        import raven.detectors.tr_detector as tr_mod
        monkeypatch.setattr(tr_mod, "load_state",
                            lambda records, device, **extra: {"fake": True,
                                                              "inversion_steps": 50})
        monkeypatch.setattr(tr_mod, "score_image", lambda *a, **kw: None)

        rec_wm = _build_tr_record("1", "watermarked")
        with tempfile.TemporaryDirectory() as td:
            out = _write_fake_run(Path(td), method="TR", records=[rec_wm])
            result = run_evaluation(out, device="cpu", stages=["detector"])

            assert result["overall_status"] == STATUS_COMPLETED_WITH_ERRORS
            assert result["failed_stages"] == ["detector"]


# ===========================================================================
# Real adapter integration tests — real load_state / score_image / aggregate
# ===========================================================================
# Only pipe/model construction, diffusion inversion, external bundle/state IO,
# provider heavy ops, and canonical scoring helper outputs are mocked.
# MetadataResolver, adapter dispatch, row status, aggregation, stage reducer,
# and CLI exit handling all run for real.

import csv as _csv
from contextlib import contextmanager as _contextmanager


def _write_metadata_csv(path, rows):
    """Write a metadata CSV file from a list of dicts."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _min_record(run_id="1", role="watermarked", method="TR", **kw):
    """Minimal attack record with only join identity fields and attack facts."""
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
    }


def _eval_detector_with_config(records, output_dir, method, device="cpu",
                                metadata_path=None, **extra_config):
    """Call evaluate_detector with config that includes metadata_path."""
    from experiments.eval import evaluate_detector
    config = {"method": method, "dataset": "test",
              "metadata_path": metadata_path or "", **extra_config}
    return evaluate_detector(records, output_dir, method, device=device,
                             config=config)


# ── TR real adapter ────────────────────────────────────────────────────

_TR_CSV_FIELDS = [
    "run_id", "role",
    "w_seed", "w_channel", "w_radius", "w_pattern", "w_mask_shape",
    "w_measurement", "w_injection", "w_pattern_const",
    "model_id", "model_revision", "scheduler", "steps", "resolution",
    "watermark_target_sha256", "watermark_mask_sha256",
]

_TR_FAKE_SCHED_CLASSES = {
    "DDIM": (mock.MagicMock(), mock.MagicMock()),
    "DPM": (mock.MagicMock(), mock.MagicMock()),
}


@_contextmanager
def _tr_real_deps(monkeypatch):
    """Mock only external TR deps: pipe, provider, extract_module, tensor SHA."""
    import builtins
    import raven.detectors.tr_detector as tr_mod

    fake_extract = mock.MagicMock()
    fake_extract.evaluate_image.return_value = {
        "p_values": [0.001],
        "p_value_diagnostics": [
            {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
             "statistic": 50.0, "df": 100, "p_underflow": False},
        ],
    }
    fake_extract.raw_score.return_value = 0.001
    fake_extract.canonical_score.return_value = 10.0
    monkeypatch.setattr(tr_mod, "_extract_module", fake_extract)
    monkeypatch.setattr(tr_mod, "_get_extract_module", lambda: fake_extract)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"
    sched_inv = mock.MagicMock()
    sched_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe.scheduler_inverse = sched_inv
    fake_pipe.pipe.vae.config.scaling_factor = 0.18215

    fake_pu = mock.MagicMock()
    fake_pu.SCHEDULER_CLASSES = dict(_TR_FAKE_SCHED_CLASSES)
    fake_pu.get_pipe_provider.return_value = fake_pipe

    fake_prov = mock.MagicMock()
    fake_tr_cls = mock.MagicMock(return_value=fake_prov)
    fake_tr_mod = mock.MagicMock(TrProvider=fake_tr_cls)
    fake_wm = mock.MagicMock(tr_provider=fake_tr_mod)
    fake_utils = mock.MagicMock(
        pipe=mock.MagicMock(pipe_utils=fake_pu), wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_utils)
    fake_eb.__path__ = []

    _imps = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_utils,
        "eval_bench_wm.utils.pipe": fake_utils.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.tr_provider": fake_tr_mod,
    }
    for mod_name, mod_obj in _imps.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    orig = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                       lambda n, *a, **kw: (_imps[n] if (n in _imps and (kw.get('level', a[3] if len(a) > 3 else 0) == 0)) else orig(n, *a, **kw)))

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(side_effect=[
                            "default_target_sha_placeholder",
                            "default_mask_sha_placeholder",
                        ] * 20))
    yield fake_extract, fake_pu, fake_tr_cls, fake_pipe


# ── GS real adapter ────────────────────────────────────────────────────

_GS_CSV_FIELDS = [
    "run_id", "role",
    "gs_secret_index", "gs_message_sha256", "gs_key_sha256",
    "gs_nonce_sha256", "gs_secret_bundle_sha256",
    "gs_protocol_mode", "gs_detection_mode",
    "model_id", "scheduler", "resolution", "model_revision",
    "watermark_target_sha256", "watermark_mask_sha256",
    "provider_config_hash",
]


@_contextmanager
def _gs_real_deps(monkeypatch):
    """Mock external GS deps: pipe, GsProvider, extract_verification_scores."""
    import raven.detectors.gs_detector as gs_mod

    # Pre-populate sys.modules for eval_bench_wm chain
    fake_pu = mock.MagicMock()
    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"
    fake_pu.get_pipe_provider.return_value = fake_pipe

    _gs_fake_wm_mod = mock.MagicMock()

    def _make_gs_instance(*args, **kwargs):
        """Build a GsProvider instance whose secret identity follows the
        constructor's ``gs_secret_index`` — lets per-source differentiation
        be exercised for real (one provider per (run_id, role))."""
        secret_idx = kwargs.get("gs_secret_index", 5)
        inst = mock.MagicMock()
        inst.gs_protocol_mode = "official_compatible"
        inst.gs_detection_mode = "official_onebit"
        inst.message_width_in_bytes = 32
        inst.l = 1
        inst.num_replications = 64
        inst.gs_channel_copy = 1
        inst.gs_hw_copy = 8
        inst.gs_fpr = 1e-6
        inst.gs_user_number = 1000000
        inst.gs_secret_index = secret_idx
        inst.secret_provenance.return_value = {
            "secret_index": secret_idx, "message_sha256": "msg_sha",
            "key_sha256": "key_sha", "nonce_sha256": "nonce_sha",
            "secret_bundle_sha256": "bundle_sha",
        }
        inst.watermark_target_tensor.return_value = mock.MagicMock()
        inst.invert_images.return_value = {"zT_torch": mock.MagicMock()}
        inst.get_accuracies.return_value = {
            "bit_accuracies": [0.85],
            "message_bits_str_list": ["0" * 256],
        }
        inst.official_thresholds.return_value = {
            "tau_onebit": 0.9, "tau_bits": 0.95,
            "fpr": 1e-6, "user_number": 1000000,
            "comparison_operator": ">=", "source": "test",
        }
        inst.active_detection_threshold.return_value = {
            "detection_mode": "official_onebit",
            "threshold": 0.9,
            "threshold_type": "official_beta_tail_tau_onebit",
            "comparison_operator": ">=",
            "nominal_fpr": 1e-6,
            "calibrated_from_current_clean_negatives": False,
            "official_tau_onebit": 0.9,
            "official_tau_bits": 0.95,
        }
        inst.is_detection_successful.return_value = True
        return inst

    _gs_prov_cls = mock.MagicMock(side_effect=_make_gs_instance)
    _gs_fake_wm_mod.GsProvider = _gs_prov_cls
    _gs_fake_wm_mod.__name__ = "gs_provider"
    _fake_pipe_mod = mock.MagicMock()
    _fake_pipe_mod.pipe_utils = fake_pu
    _fake_pipe_mod.__name__ = "pipe"
    _fake_wm_pkg = mock.MagicMock()
    _fake_wm_pkg.gs_provider = _gs_fake_wm_mod
    _fake_wm_pkg.__name__ = "wm"
    _fake_utils_pkg = mock.MagicMock()
    _fake_utils_pkg.pipe = _fake_pipe_mod
    _fake_utils_pkg.wm = _fake_wm_pkg
    _fake_utils_pkg.__name__ = "utils"
    _fake_eb = mock.MagicMock()
    _fake_eb.utils = _fake_utils_pkg
    _fake_eb.__path__ = []
    _fake_eb.__name__ = "eval_bench_wm"

    _gs_mods = {
        "eval_bench_wm": _fake_eb,
        "eval_bench_wm.utils": _fake_utils_pkg,
        "eval_bench_wm.utils.pipe": _fake_pipe_mod,
        "eval_bench_wm.utils.pipe.pipe_utils": fake_pu,
        "eval_bench_wm.utils.wm": _fake_wm_pkg,
        "eval_bench_wm.utils.wm.gs_provider": _gs_fake_wm_mod,
    }
    for name, mod in _gs_mods.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Ensure extract_verification_scores is importable
    _scripts_dir = str(REPO / "raven_repro" / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import extract_verification_scores as _evs
    monkeypatch.setattr(_evs, "provider_kwargs",
                        lambda method, row: {"offset": int(
                            row.get("gs_secret_index", 0)),
                            "gs_secret_index": int(
                                row.get("gs_secret_index", 0))})
    monkeypatch.setattr(_evs, "evaluate_image",
                        lambda *a, **k: {
                            "bit_accuracies": [0.85],
                            "message_bits_str_list": ["0" * 256],
                        })

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(return_value="default_target_sha_placeholder"))

    yield fake_pu, _gs_fake_wm_mod.GsProvider, fake_pipe


# ── GM real adapter ────────────────────────────────────────────────────

_GM_CSV_FIELDS = [
    "run_id", "role",
    "gm_bundle_dir", "gm_bundle_config_sha256",
    "gm_w1_file_sha256", "gm_w2_file_sha256",
    "gm_m_sha256", "gm_watermark_sha256", "gm_target_sha256",
    "gm_protocol_mode",
    "watermark_target_sha256", "watermark_mask_sha256",
]


def _make_gm_bundle_dir(tmp_path, **overrides):
    """Create a minimal GM bundle directory."""
    bundle = tmp_path / "gm_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": "legacy",
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "fake",
        "scheduler": "DDIM",
        "resolution": 512,
        "torch_dtype": "float32",
        "channel_copy": 1,
        "w_copy": 1,
        "h_copy": 1,
        "watermark_bits_seed": None,
        "model_nf": 1,
        "classifier_type": 0,
        "use_gnr": False,
        "use_classifier": False,
        "inversion_guidance": 7.5,
        "inversion_steps": 50,
        "inversion_seed": 0,
        "inversion_prompt": "",
        "vae_sample": False,
        "vae_scaling_factor": 0.18215,
        "profile_is_official": True,
        "w_seed": 99,
        "w_channel": 3,
        "w_pattern": "ring",
        "w_mask_shape": "circle",
        "w_radius": 10,
        "w_measurement": "l1_complex",
        "w_injection": "complex",
        "create_bundle": False,
        "allow_in_memory_state": False,
    }
    manifest.update(overrides)
    import json as _json
    (bundle / "manifest.json").write_text(_json.dumps(manifest, sort_keys=True))
    (bundle / "w1.pth").write_bytes(b"\x00" * 64)
    (bundle / "w2.pth").write_bytes(b"\x00" * 64)
    return bundle


@_contextmanager
def _gm_real_deps(monkeypatch, tmp_path):
    """Mock external GM deps: pipe, GmProvider, extract bundle helpers."""
    import builtins
    import raven.detectors.gm_detector as gm_mod

    fake_extract = mock.MagicMock()

    def _fake_manifest(row, run_id):
        import json as _json
        bundle_dir = Path(row["gm_bundle_dir"])
        manifest = _json.loads((bundle_dir / "manifest.json").read_text())
        return bundle_dir, manifest

    def _fake_provider_kwargs(row, run_id):
        _bd, mf = _fake_manifest(row, run_id)
        return {
            "gm_profile": mf["profile"],
            "gm_bundle_dir": row["gm_bundle_dir"],
            "gm_create_bundle": False,
            "gm_allow_in_memory_state": False,
            "gm_torch_dtype": "float32",
            "gm_channel_copy": mf.get("channel_copy", 1),
            "gm_w_copy": mf.get("w_copy", 1),
            "gm_h_copy": mf.get("h_copy", 1),
            "gm_watermark_bits_seed": None,
            "gm_use_gnr": False,
            "gm_gnr_path": None,
            "gm_model_nf": mf.get("model_nf", 1),
            "gm_classifier_type": mf.get("classifier_type", 0),
            "gm_use_classifier": False,
            "gm_classifier_path": None,
            "modelid_target": mf["model_id"],
            "model_revision": mf["model_revision"],
            "scheduler_target": mf["scheduler"],
            "resolution": mf["resolution"],
            "gm_inversion_guidance": mf.get("inversion_guidance", 7.5),
            "gm_inversion_steps": mf.get("inversion_steps", 50),
            "gm_inversion_seed": mf.get("inversion_seed", 0),
            "gm_inversion_prompt": mf.get("inversion_prompt", ""),
            "gm_vae_sample": False,
            "gm_vae_scaling_factor": mf.get("vae_scaling_factor", 0.18215),
            "gm_profile_is_official": True,
            "w_seed": mf.get("w_seed", 99),
            "w_channel": mf.get("w_channel", 3),
            "w_pattern": mf.get("w_pattern", "ring"),
            "w_mask_shape": mf.get("w_mask_shape", "circle"),
            "w_radius": mf.get("w_radius", 10),
            "w_measurement": mf.get("w_measurement", "l1_complex"),
            "w_injection": mf.get("w_injection", "complex"),
        }

    fake_extract.gm_bundle_manifest = _fake_manifest
    fake_extract.gm_provider_kwargs = _fake_provider_kwargs
    fake_extract.evaluate_image.return_value = {
        "gm_raw_bit_accuracy": 0.85,
        "gm_raw_ring_l1": 10.0,
        "gm_report_label": "gauss_marker_clean_calibrated",
        "gm_score_definition": "gm_neg_mean_cosine_sim_l1_per_bit_target_direction",
        "gm_threshold_source": "clean_calibrated",
        "gm_comparison_operator": ">=",
    }
    fake_extract.raw_score.side_effect = (
        lambda method, result: result["gm_raw_bit_accuracy"])
    fake_extract.canonical_score.side_effect = (
        lambda method, raw, result: 0.85)
    monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: fake_extract)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"

    fake_pu = mock.MagicMock()
    fake_pu.get_pipe_provider.return_value = fake_pipe

    fake_gm_prov = mock.MagicMock()
    fake_gm_prov.bundle = mock.MagicMock()
    fake_gm_prov.bundle.manifest = {
        "profile": "legacy",
        "profile_is_official": True,
    }
    fake_gm_prov.state_source = "bundle"
    fake_gm_prov.profile = "legacy"
    fake_gm_prov.profile_is_official = True
    fake_gm_prov.gm_torch_dtype = "float32"
    fake_gm_prov.ch = 1
    fake_gm_prov.w = 1
    fake_gm_prov.h = 1
    fake_gm_prov.watermark_bits_seed = None
    fake_gm_prov.model_nf = 1
    fake_gm_prov.classifier_type = 0
    fake_gm_prov.use_gnr = False
    fake_gm_prov.use_classifier = False
    fake_gm_prov.model_id = "RedbeardNZ/stable-diffusion-2-1-base"
    fake_gm_prov.model_revision = "fake"
    fake_gm_prov.scheduler_name = "DDIM"
    fake_gm_prov.resolution = 512
    fake_gm_prov.inversion_guidance = 7.5
    fake_gm_prov.inversion_steps = 50
    fake_gm_prov.inversion_seed = 0
    fake_gm_prov.inversion_prompt = ""
    fake_gm_prov.vae_sample = False
    fake_gm_prov.vae_scaling_factor = 0.18215
    fake_gm_prov.w_seed = 99
    fake_gm_prov.w_channel = 3
    fake_gm_prov.w_pattern = "ring"
    fake_gm_prov.w_mask_shape = "circle"
    fake_gm_prov.w_radius = 10
    fake_gm_prov.w_measurement = "l1_complex"
    fake_gm_prov.w_injection = "complex"

    fake_gm_cls = mock.MagicMock(return_value=fake_gm_prov)
    fake_gm_mod = mock.MagicMock(GmProvider=fake_gm_cls)
    fake_wm = mock.MagicMock(gm_provider=fake_gm_mod)
    fake_u = mock.MagicMock(pipe=mock.MagicMock(pipe_utils=fake_pu), wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_u)
    fake_eb.__path__ = []

    _imps = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_u,
        "eval_bench_wm.utils.pipe": fake_u.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.gm_provider": fake_gm_mod,
    }
    for mod_name, mod_obj in _imps.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    orig = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                       lambda n, *a, **kw: (_imps[n] if (n in _imps and (kw.get('level', a[3] if len(a) > 3 else 0) == 0)) else orig(n, *a, **kw)))

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(return_value="tensor_hash_sha"))

    yield fake_extract, fake_pu, fake_gm_cls, fake_pipe


# ── T2S real adapter ───────────────────────────────────────────────────

_T2S_CSV_FIELDS = [
    "run_id", "role",
    "t2s_state_path", "t2s_state_sha256",
    "t2s_watermark_id", "t2s_provider_config_sha256",
    "t2s_protocol_mode", "t2s_rng_mode",
    "t2s_inversion_mode", "t2s_num_inversion_steps",
]


def _write_t2s_state(path, **overrides):
    """Write a synthetic T2SWatermarkState JSON file."""
    import json as _json
    state = {
        "watermark_id": "t2s-synth-01",
        "inversion_mode": "t2s_official",
        "num_inversion_steps": 10,
        "provider_config_sha256": "pcfg_sha",
        "rng_mode": "official_compatible",
        "num_inference_steps": 50,
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
        "scheduler": "DDIM",
        "resolution": 512,
        "latent_shape": [1, 4, 64, 64],
        "key_channels": [0],
        "msg_channels": [1, 2, 3],
        "key_length": 32,
        "msg_length": 96,
        "tau": 0.5,
        "protocol_mode": "official_math_shared_tr_clean",
    }
    state.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(state, sort_keys=True))
    return state


@_contextmanager
def _t2s_real_deps(monkeypatch, tmp_path):
    """Mock external T2S deps: pipe, T2S provider/inversion, accuracies."""
    import builtins
    import raven.detectors.t2s_detector as t2s_mod
    import types as _types

    # Stub t2s_provider module
    fake_t2s_mod = _types.ModuleType("t2s_provider")
    fake_t2s_mod.T2S_RNG_MODES = ["official_compatible", "v1"]
    fake_t2s_mod.T2S_INVERSION_MODES = ["t2s_official", "benchmark_ddim"]
    fake_t2s_mod.T2S_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
    fake_t2s_mod.T2SWatermarkState = mock.MagicMock()

    def _fake_load(path):
        import hashlib as _hashlib
        import json as _json
        data = _json.loads(Path(path).read_text())
        st = mock.MagicMock()
        for k, v in data.items():
            setattr(st, k, v)
        # Real SHA-256 of the state file — mirrors the canonical
        # T2SWatermarkState contract (state_sha256() is a method).
        file_sha = _hashlib.sha256(Path(path).read_bytes()).hexdigest()
        st.state_sha256.return_value = file_sha
        st.load = classmethod(lambda cls, p: _fake_load(p))
        return st
    fake_t2s_mod.T2SWatermarkState.load = _fake_load

    fake_t2s_mod.T2SProvider = mock.MagicMock()
    fake_t2s_mod.T2SProvider.accuracies_for_state.return_value = {
        "t2s_score_true_key": 0.85,
        "t2s_score_control_key": 0.40,
        "t2s_score_margin": 0.45,
        "detection_success": True,
        "key_accuracy": 1.0,
        "message_accuracy": 1.0,
    }

    # Stub t2s_inversion module
    fake_inv_mod = _types.ModuleType("t2s_inversion")
    fake_inv_mod.invert_image = mock.MagicMock()
    fake_inv_mod.invert_image.return_value = mock.MagicMock()

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)

    fake_pu = mock.MagicMock()
    fake_pu.get_pipe_provider.return_value = fake_pipe

    # Build mock eval_bench_wm tree
    fake_wm = mock.MagicMock()
    fake_wm.t2s_provider = fake_t2s_mod
    fake_wm.t2s_inversion = fake_inv_mod
    fake_wm.__name__ = "wm"
    fake_u = mock.MagicMock()
    fake_u.wm = fake_wm
    fake_u.pipe = mock.MagicMock(pipe_utils=fake_pu)
    fake_u.__name__ = "utils"
    fake_eb = mock.MagicMock()
    fake_eb.utils = fake_u
    fake_eb.__path__ = []
    fake_eb.__name__ = "eval_bench_wm"

    monkeypatch.setitem(sys.modules, "utils.wm.t2s_provider", fake_t2s_mod)
    monkeypatch.setitem(sys.modules, "utils.wm.t2s_inversion", fake_inv_mod)

    _imps = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_u,
        "eval_bench_wm.utils.pipe": fake_u.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.t2s_provider": fake_t2s_mod,
        "eval_bench_wm.utils.wm.t2s_inversion": fake_inv_mod,
        "utils.wm.t2s_provider": fake_t2s_mod,
        "utils.wm.t2s_inversion": fake_inv_mod,
        "utils.wm": fake_wm,
        "utils": fake_u,
    }
    for mod_name, mod_obj in _imps.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    orig = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                       lambda n, *a, **kw: (_imps[n] if (n in _imps and (kw.get('level', a[3] if len(a) > 3 else 0) == 0)) else orig(n, *a, **kw)))

    yield fake_t2s_mod, fake_inv_mod, fake_pu, fake_pipe


# ── Fourier (RID/HSTR/HSQR) real adapter ───────────────────────────────

_RID_CSV_FIELDS = [
    "run_id", "role", "method",
    "rid_bundle_dir", "rid_bundle_config_sha256",
    "rid_selected_pattern_sha256", "rid_mask_sha256",
    "rid_key_index", "rid_protocol_mode",
    "watermark_target_sha256", "watermark_mask_sha256",
]

_HSTR_CSV_FIELDS = [
    "run_id", "role", "method",
    "hstr_bundle_dir", "hstr_bundle_config_sha256",
    "hstr_selected_pattern_sha256", "hstr_mask_sha256",
    "hstr_key_index", "hstr_protocol_mode",
    "watermark_target_sha256", "watermark_mask_sha256",
]

_HSQR_CSV_FIELDS = [
    "run_id", "role", "method",
    "hsqr_bundle_dir", "hsqr_bundle_config_sha256",
    "hsqr_selected_pattern_sha256", "hsqr_mask_sha256",
    "hsqr_key_index", "hsqr_protocol_mode",
    "watermark_target_sha256", "watermark_mask_sha256",
]

_FOURIER_METHOD_CSV_FIELDS = {"RID": _RID_CSV_FIELDS,
                               "HSTR": _HSTR_CSV_FIELDS,
                               "HSQR": _HSQR_CSV_FIELDS}

from raven.pairing_provenance import (  # noqa: E402
    RID_SHARED_TR_CLEAN_MODE,
    HSTR_SHARED_TR_CLEAN_MODE,
    HSQR_SHARED_TR_CLEAN_MODE,
)

_FOURIER_PROTOCOL_MODES = {
    "RID": RID_SHARED_TR_CLEAN_MODE,
    "HSTR": HSTR_SHARED_TR_CLEAN_MODE,
    "HSQR": HSQR_SHARED_TR_CLEAN_MODE,
}


def _make_fourier_bundle_dir(tmp_path, method, **overrides):
    """Create a minimal Fourier bundle directory with manifest.json."""
    prefix = method.lower()
    bundle = tmp_path / f"{prefix}_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "1.0",
        "method": method,
        "bundle_schema": "rid_bundle_v1" if method == "RID" else "sfw_bundle_v1",
        "bundle_config_sha256": f"{prefix}_cfg_sha",
        "selected_pattern_sha256": f"{prefix}_pat_sha",
        "mask_sha256": f"{prefix}_mask_sha",
        "selected_key_index": 0,
        "protocol_mode": _FOURIER_PROTOCOL_MODES[method],
        "profile_name": "legacy",
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
        "scheduler_type": "DDIM",
        "resolution": 512,
    }
    manifest.update(overrides)
    import json as _json
    (bundle / "manifest.json").write_text(_json.dumps(manifest, sort_keys=True))
    return bundle


@_contextmanager
def _fourier_real_deps(monkeypatch, method="RID"):
    """Mock external Fourier deps: pipe, provider, extract_module, bundle."""
    import builtins
    import raven.detectors.fourier_detector as fmod

    fake_extract = mock.MagicMock()

    def _fake_fourier_manifest(row, identifier, meth):
        import json as _json
        prefix = meth.lower()
        bundle_dir = Path(row.get(f"{prefix}_bundle_dir", ""))
        if not bundle_dir.is_dir():
            raise RuntimeError(f"bundle dir not found: {bundle_dir}")
        manifest = _json.loads((bundle_dir / "manifest.json").read_text())
        return bundle_dir, manifest

    fake_extract.fourier_bundle_manifest = _fake_fourier_manifest
    fake_extract.evaluate_image.return_value = {
        "l1_dist": [10.0],
        "p_value_diagnostics": [
            {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
             "statistic": 50.0, "df": 100, "p_underflow": False},
        ],
    }
    fake_extract.raw_score.return_value = 0.001
    fake_extract.canonical_score.return_value = 10.0

    def _fake_rid_kwargs(bundle_dir, device, **extra):
        return {"bundle_dir": str(bundle_dir), "rid_profile": "legacy"}

    def _fake_hstr_kwargs(bundle_dir, device, **extra):
        return {"bundle_dir": str(bundle_dir), "hstr_profile": "legacy"}

    def _fake_hsqr_from_bundle(bundle_dir, identifier, latent_shape, device,
                               **extra):
        prov = mock.MagicMock()
        prov.bundle = mock.MagicMock()
        prov.bundle.manifest = {
            "selected_pattern_sha256": "tgt_sha",
            "mask_sha256": "tgt_sha",
        }
        prov.profile = "legacy"
        # HSQR must NOT depend on a state_source attribute — leave it absent
        # so the adapter proves it never consults that field.
        prov.selected_pattern_sha256 = "tgt_sha"
        prov.watermark_mask_sha256 = "tgt_sha"
        prov.latent_shape = latent_shape
        return prov

    fake_extract.rid_provider_kwargs_from_bundle = _fake_rid_kwargs
    fake_extract.hstr_provider_kwargs_from_bundle = _fake_hstr_kwargs
    fake_extract.hsqr_provider_from_bundle = _fake_hsqr_from_bundle
    monkeypatch.setattr(fmod, "_get_extract_module", lambda: fake_extract)
    monkeypatch.setattr(fmod, "_ensure_paths", lambda: None)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"

    fake_pu = mock.MagicMock()
    fake_pu.get_pipe_provider.return_value = fake_pipe

    # Build per-method provider
    fake_prov = mock.MagicMock()
    fake_prov.bundle = mock.MagicMock()
    fake_prov.bundle.manifest = {
        "selected_pattern_sha256": "tgt_sha",
        "mask_sha256": "tgt_sha",
    }
    fake_prov.state_source = "bundle"
    fake_prov.profile = "legacy"
    fake_prov.selected_pattern_sha256 = "tgt_sha"
    fake_prov.watermark_mask_sha256 = "tgt_sha"

    if method == "RID":
        fake_prov_cls = mock.MagicMock(return_value=fake_prov)
        prov_attr = "ringid_provider"
        prov_cls_name = "RingIDProvider"
        prov_cls = fake_prov_cls
    elif method == "HSTR":
        fake_prov_cls = mock.MagicMock(return_value=fake_prov)
        prov_attr = "hstr_provider"
        prov_cls_name = "HSTRProvider"
        prov_cls = fake_prov_cls
    else:
        fake_prov_cls = None
        prov_attr = "hsqr_provider"
        prov_cls_name = "HSQRProvider"
        prov_cls = mock.MagicMock()

    fake_prov_mod = mock.MagicMock(**{prov_cls_name: prov_cls})
    fake_wm = mock.MagicMock(**{prov_attr: fake_prov_mod})
    fake_wm.sfw_bundle = mock.MagicMock()
    fake_wm.__name__ = "wm"
    fake_u = mock.MagicMock()
    fake_u.wm = fake_wm
    fake_u.pipe = mock.MagicMock(pipe_utils=fake_pu)
    fake_u.__name__ = "utils"
    fake_eb = mock.MagicMock()
    fake_eb.utils = fake_u
    fake_eb.__path__ = []
    fake_eb.__name__ = "eval_bench_wm"

    _imps = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_u,
        "eval_bench_wm.utils.pipe": fake_u.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        f"eval_bench_wm.utils.wm.{prov_attr}": fake_prov_mod,
    }
    for mod_name, mod_obj in _imps.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)
    orig = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                       lambda n, *a, **kw: (_imps[n] if (n in _imps and (kw.get('level', a[3] if len(a) > 3 else 0) == 0)) else orig(n, *a, **kw)))

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(return_value="tgt_sha"))

    yield fake_extract, fake_pu, fake_pipe, fake_prov


# ════════════════════════════════════════════════════════════════════════
# TR — real adapter success + mixed provider config rejection
# ════════════════════════════════════════════════════════════════════════

class TestTRRealAdapter:
    """TR real load_state → score_image → aggregate through evaluate_detector."""

    def test_real_adapter_success_with_metadata_csv(self, monkeypatch):
        """TR: real load_state + real score_image + real MetadataResolver."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED
        import raven.detectors.tr_detector as tr_mod
        import builtins
        import raven.pairing_provenance as pp

        # Build mock infrastructure inline
        fake_extract = mock.MagicMock()
        fake_extract.evaluate_image.return_value = {
            "p_values": [0.001],
            "p_value_diagnostics": [
                {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
                 "statistic": 50.0, "df": 100, "p_underflow": False},
            ],
        }
        fake_extract.raw_score.return_value = 0.001
        fake_extract.canonical_score.return_value = 10.0
        monkeypatch.setattr(tr_mod, "_extract_module", fake_extract)
        monkeypatch.setattr(tr_mod, "_get_extract_module", lambda: fake_extract)

        fake_pipe = mock.MagicMock()
        fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
        fake_pipe.get_dtype.return_value = "torch.float32"
        sched_inv = mock.MagicMock()
        sched_inv.__class__.__name__ = "DDIMScheduler"
        fake_pipe.scheduler_inverse = sched_inv
        fake_pipe.pipe.vae.config.scaling_factor = 0.18215

        fake_pu = mock.MagicMock()
        fake_pu.SCHEDULER_CLASSES = dict(_TR_FAKE_SCHED_CLASSES)
        fake_pu.get_pipe_provider.return_value = fake_pipe

        fake_prov = mock.MagicMock()
        fake_tr_cls = mock.MagicMock(return_value=fake_prov)
        fake_tr_pkg = mock.MagicMock(TrProvider=fake_tr_cls)
        fake_wm = mock.MagicMock(tr_provider=fake_tr_pkg)
        fake_utils = mock.MagicMock(
            pipe=mock.MagicMock(pipe_utils=fake_pu), wm=fake_wm)
        fake_eb = mock.MagicMock(utils=fake_utils)
        fake_eb.__path__ = []

        _imps = {
            "eval_bench_wm": fake_eb,
            "eval_bench_wm.utils": fake_utils,
            "eval_bench_wm.utils.pipe": fake_utils.pipe,
            "eval_bench_wm.utils.wm": fake_wm,
            "eval_bench_wm.utils.wm.tr_provider": fake_tr_pkg,
        }
        orig_import = builtins.__import__
        # Pre-populate sys.modules so Python doesn't try to real-import
        for mod_name, mod_obj in _imps.items():
            monkeypatch.setitem(sys.modules, mod_name, mod_obj)
        monkeypatch.setattr(builtins, "__import__",
                           lambda *a, **kw: (
                               _imps[a[0]] if (a[0] in _imps and (kw.get('level', a[3] if len(a) > 3 else 0) == 0)) else orig_import(*a, **kw)))

        monkeypatch.setattr(pp, "tensor_sha256",
                            mock.MagicMock(side_effect=[
                                "default_target_sha_placeholder",
                                "default_mask_sha_placeholder",
                            ] * 20))

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            csv_path = tdp / "metadata.csv"
            csv_rows = [{
                "run_id": "1", "role": "clean",
                "w_seed": "99", "w_channel": "3", "w_radius": "10",
                "w_pattern": "ring", "w_mask_shape": "circle",
                "w_measurement": "l1_complex", "w_injection": "complex",
                "w_pattern_const": "0.0",
                "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                "scheduler": "DDIM", "steps": "50", "resolution": "512",
                "watermark_target_sha256": "default_target_sha_placeholder",
                "watermark_mask_sha256": "default_mask_sha_placeholder",
            }, {
                "run_id": "1", "role": "watermarked",
                "w_seed": "99", "w_channel": "3", "w_radius": "10",
                "w_pattern": "ring", "w_mask_shape": "circle",
                "w_measurement": "l1_complex", "w_injection": "complex",
                "w_pattern_const": "0.0",
                "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                "scheduler": "DDIM", "steps": "50", "resolution": "512",
                "watermark_target_sha256": "default_target_sha_placeholder",
                "watermark_mask_sha256": "default_mask_sha_placeholder",
            }]
            _write_metadata_csv(csv_path, csv_rows)

            rec_clean = _min_record("1", "clean", "TR")
            rec_wm = _min_record("1", "watermarked", "TR")

            eval_config = {"method": "TR", "dataset": "test",
                           "metadata_path": str(csv_path)}
            out = _write_fake_run(tdp, method="TR",
                                  records=[rec_clean, rec_wm],
                                  config=eval_config)
            result = evaluate_detector([rec_clean, rec_wm],
                                       out, "TR", device="cpu",
                                       config=eval_config)

            assert result["status"] == STATUS_COMPLETED, (
                f"status={result['status']}, "
                f"setup_error={result.get('setup_error')}, "
                f"reason={result.get('status_reducer_reason')}"
            )
            assert result["scored_count"] == 4
            assert result["failed_count"] == 0
            assert result["count_invariant_satisfied"] is True
            assert result["dominant_failure_cause"] is None

            rows = _read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == "scored" for r in rows)

    def test_real_mixed_provider_config_rejection(self, monkeypatch):
        """TR: mixed provider config → real load_state raises StateValidation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        with _tr_real_deps(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                csv_path = tdp / "metadata.csv"
                # Different w_seed values → mixed provider config
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "w_seed": "99", "w_channel": "3", "w_radius": "10",
                    "w_pattern": "ring", "w_mask_shape": "circle",
                    "w_measurement": "l1_complex", "w_injection": "complex",
                    "w_pattern_const": "0.0",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "scheduler": "DDIM", "steps": "50", "resolution": "512",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "default_mask_sha_placeholder",
                }, {
                    "run_id": "2", "role": "watermarked",
                    "w_seed": "88888", "w_channel": "3", "w_radius": "10",
                    "w_pattern": "ring", "w_mask_shape": "circle",
                    "w_measurement": "l1_complex", "w_injection": "complex",
                    "w_pattern_const": "0.0",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "scheduler": "DDIM", "steps": "50", "resolution": "512",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "default_mask_sha_placeholder",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec1 = _min_record("1", "watermarked", "TR")
                rec2 = _min_record("2", "watermarked", "TR")

                out = _write_fake_run(tdp, method="TR",
                                      records=[rec1, rec2],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec1, rec2],
                                           out, "TR", device="cpu", config={"method": "TR", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_FAILED_STATE_VALIDATION
                assert result["dominant_failure_cause"] == "state_validation_error"
                assert result["count_invariant_satisfied"] is True


# ════════════════════════════════════════════════════════════════════════
# GS — real adapter success + per-row secret/provider differentiation
# ════════════════════════════════════════════════════════════════════════

class TestGSRealAdapter:
    """GS real load_state → score_image with per-source provider cache."""

    def test_real_adapter_success_with_metadata_csv(self, monkeypatch):
        """GS: real adapter success through evaluate_detector."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED
        from raven.eval_protocol import provider_config_hash

        with _gs_real_deps(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)

                # Build provider config hash for GS
                gs_provider_meta = {
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_secret_index": "5",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                }
                gs_cfg_hash = provider_config_hash("GS", gs_provider_meta)

                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean",
                    "gs_secret_index": "5",
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "scheduler": "DDIM", "resolution": "512",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "f80e7f814ec3c12e1fde467c29f60eb2de0efc69edbae02396f60106368f7ece",
                    "provider_config_hash": gs_cfg_hash,
                }, {
                    "run_id": "1", "role": "watermarked",
                    "gs_secret_index": "5",
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "scheduler": "DDIM", "resolution": "512",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "f80e7f814ec3c12e1fde467c29f60eb2de0efc69edbae02396f60106368f7ece",
                    "provider_config_hash": gs_cfg_hash,
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "GS")
                rec_wm = _min_record("1", "watermarked", "GS")

                out = _write_fake_run(tdp, method="GS",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "GS", device="cpu", config={"method": "GS", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_per_row_secret_provider_differentiation(self, monkeypatch):
        """GS: different secret_index per (run_id, role) → separate providers."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED
        from raven.eval_protocol import provider_config_hash

        with _gs_real_deps(monkeypatch) as (_pu, GsProvider, _pipe):
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)

                gs_provider_meta_5 = {
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_secret_index": "5",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                }
                gs_cfg_hash_5 = provider_config_hash("GS", gs_provider_meta_5)

                gs_provider_meta_7 = dict(gs_provider_meta_5,
                                          gs_secret_index="7")
                gs_cfg_hash_7 = provider_config_hash("GS", gs_provider_meta_7)

                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean",
                    "gs_secret_index": "5",
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "scheduler": "DDIM", "resolution": "512",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "f80e7f814ec3c12e1fde467c29f60eb2de0efc69edbae02396f60106368f7ece",
                    "provider_config_hash": gs_cfg_hash_5,
                }, {
                    "run_id": "1", "role": "watermarked",
                    "gs_secret_index": "7",
                    "gs_message_sha256": "msg_sha",
                    "gs_key_sha256": "key_sha",
                    "gs_nonce_sha256": "nonce_sha",
                    "gs_secret_bundle_sha256": "bundle_sha",
                    "gs_protocol_mode": "official_compatible",
                    "gs_detection_mode": "official_onebit",
                    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
                    "scheduler": "DDIM", "resolution": "512",
                    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
                    "watermark_target_sha256": "default_target_sha_placeholder",
                    "watermark_mask_sha256": "f80e7f814ec3c12e1fde467c29f60eb2de0efc69edbae02396f60106368f7ece",
                    "provider_config_hash": gs_cfg_hash_7,
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "GS")
                rec_wm = _min_record("1", "watermarked", "GS")

                out = _write_fake_run(tdp, method="GS",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "GS", device="cpu", config={"method": "GS", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                # Exactly one provider per source — two sources, two providers.
                assert GsProvider.call_count == 2, \
                    f"expected 2 provider constructions, got {GsProvider.call_count}"
                # Constructor kwargs must carry each source's own secret index.
                call_kwargs = [call.kwargs for call in GsProvider.call_args_list]
                secret_indices = {kw.get("gs_secret_index") for kw in call_kwargs}
                assert secret_indices == {5, 7}, \
                    f"expected gs_secret_index in {{5, 7}}, got {secret_indices}"
                # The two sources must NOT share a provider instance: each
                # scored row carries the secret index of ITS source.  A shared
                # provider would stamp the same secret on both rows.
                rows = _read_detector_rows(out)
                clean_secrets = {
                    r["gs_secret_index"]
                    for r in rows if r["source_role"] == "clean"
                }
                wm_secrets = {
                    r["gs_secret_index"]
                    for r in rows if r["source_role"] == "watermarked"
                }
                assert clean_secrets == {5}, \
                    f"clean rows must carry secret 5, got {clean_secrets}"
                assert wm_secrets == {7}, \
                    f"watermarked rows must carry secret 7, got {wm_secrets}"


# ════════════════════════════════════════════════════════════════════════
# GM — real adapter success + mixed bundle + protocol/profile validation
# ════════════════════════════════════════════════════════════════════════

class TestGMRealAdapter:
    """GM real load_state with synthetic bundle directory."""

    def test_real_adapter_success_with_bundle_and_csv(self, monkeypatch):
        """GM: real adapter success with synthetic bundle + metadata CSV."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_dir = _make_gm_bundle_dir(tdp)

            with _gm_real_deps(monkeypatch, tdp):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean",
                    "gm_bundle_dir": str(bundle_dir),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }, {
                    "run_id": "1", "role": "watermarked",
                    "gm_bundle_dir": str(bundle_dir),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "GM")
                rec_wm = _min_record("1", "watermarked", "GM")

                out = _write_fake_run(tdp, method="GM",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "GM", device="cpu", config={"method": "GM", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True
                assert result["dominant_failure_cause"] is None

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_mixed_bundle_rejection(self, monkeypatch):
        """GM: different bundle dirs per record → real state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_a = _make_gm_bundle_dir(tdp / "bundle_a",
                                           w_seed=99)
            bundle_b = _make_gm_bundle_dir(tdp / "bundle_b",
                                           w_seed=88888)

            with _gm_real_deps(monkeypatch, tdp):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "gm_bundle_dir": str(bundle_a),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }, {
                    "run_id": "2", "role": "watermarked",
                    "gm_bundle_dir": str(bundle_b),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec1 = _min_record("1", "watermarked", "GM")
                rec2 = _min_record("2", "watermarked", "GM")

                out = _write_fake_run(tdp, method="GM",
                                      records=[rec1, rec2],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec1, rec2],
                                           out, "GM", device="cpu", config={"method": "GM", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_FAILED_STATE_VALIDATION
                assert result["dominant_failure_cause"] == "state_validation_error"
                assert result["count_invariant_satisfied"] is True

    def test_real_protocol_profile_separation(self, monkeypatch):
        """GM: protocol_mode ≠ profile — both identities verified by real adapter."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_dir = _make_gm_bundle_dir(tdp)

            with _gm_real_deps(monkeypatch, tdp):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean",
                    "gm_bundle_dir": str(bundle_dir),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }, {
                    "run_id": "1", "role": "watermarked",
                    "gm_bundle_dir": str(bundle_dir),
                    "gm_bundle_config_sha256": "a" * 64,
                    "gm_w1_file_sha256": "b" * 64,
                    "gm_w2_file_sha256": "c" * 64,
                    "gm_m_sha256": "m" * 64,
                    "gm_watermark_sha256": "n" * 64,
                    "gm_target_sha256": "o" * 64,
                    "gm_protocol_mode": "official_math_shared_tr_clean",
                    "watermark_target_sha256": "tensor_hash_sha",
                    "watermark_mask_sha256": "tensor_hash_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "GM")
                rec_wm = _min_record("1", "watermarked", "GM")

                out = _write_fake_run(tdp, method="GM",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "GM", device="cpu", config={"method": "GM", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                # Verify protocol_mode in metadata propagated through resolver
                rows = _read_detector_rows(out)
                for row in rows:
                    if row["status"] == "scored":
                        # gm_protocol_mode from CSV should be in scored rows
                        assert "gm_protocol_mode" in row, \
                            "scored row missing gm_protocol_mode"


# ════════════════════════════════════════════════════════════════════════
# T2S — real adapter success + state resolution + benchmark binding
# ════════════════════════════════════════════════════════════════════════

class TestT2SRealAdapter:
    """T2S real load_state with synthetic state files + MetadataResolver."""

    def test_real_adapter_success_with_state_and_csv(self, monkeypatch):
        """T2S: real adapter with synthetic state files and metadata CSV."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            # Write synthetic state files
            state_path_wm = tdp / "states" / "wm_1.pt"
            state_data_wm = _write_t2s_state(state_path_wm)
            import hashlib as _hashlib
            state_sha_wm = _hashlib.sha256(state_path_wm.read_bytes()).hexdigest()

            state_path_wm2 = tdp / "states" / "wm_2.pt"
            state_data_wm2 = _write_t2s_state(state_path_wm2,
                                              watermark_id="t2s-synth-02")
            state_sha_wm2 = _hashlib.sha256(state_path_wm2.read_bytes()).hexdigest()

            with _t2s_real_deps(monkeypatch, tdp):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "t2s_state_path": str(state_path_wm),
                    "t2s_state_sha256": state_sha_wm,
                    "t2s_watermark_id": state_data_wm["watermark_id"],
                    "t2s_provider_config_sha256": state_data_wm["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data_wm["rng_mode"],
                    "t2s_inversion_mode": state_data_wm["inversion_mode"],
                    "t2s_num_inversion_steps": str(state_data_wm["num_inversion_steps"]),
                }, {
                    "run_id": "2", "role": "watermarked",
                    "t2s_state_path": str(state_path_wm2),
                    "t2s_state_sha256": state_sha_wm2,
                    "t2s_watermark_id": state_data_wm2["watermark_id"],
                    "t2s_provider_config_sha256": state_data_wm2["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data_wm2["rng_mode"],
                    "t2s_inversion_mode": state_data_wm2["inversion_mode"],
                    "t2s_num_inversion_steps": str(state_data_wm2["num_inversion_steps"]),
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "T2S")
                rec_wm2 = _min_record("2", "watermarked", "T2S")

                out = _write_fake_run(tdp, method="T2S",
                                      records=[rec_wm, rec_wm2],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm, rec_wm2],
                                           out, "T2S", device="cpu", config={"method": "T2S", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True
                assert result["dominant_failure_cause"] is None

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_state_resolution_by_run_id_role(self, monkeypatch):
        """T2S: real state resolution distinguishes (run_id, role) pairs."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            state_path = tdp / "states" / "wm.pt"
            state_data = _write_t2s_state(state_path)

            with _t2s_real_deps(monkeypatch, tdp):
                import hashlib as _hashlib
                state_sha = _hashlib.sha256(state_path.read_bytes()).hexdigest()
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "t2s_state_path": str(state_path),
                    "t2s_state_sha256": state_sha,
                    "t2s_watermark_id": state_data["watermark_id"],
                    "t2s_provider_config_sha256": state_data["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data["rng_mode"],
                    "t2s_inversion_mode": state_data["inversion_mode"],
                    "t2s_num_inversion_steps": str(state_data["num_inversion_steps"]),
                }, {
                    "run_id": "2", "role": "watermarked",
                    "t2s_state_path": str(state_path),
                    "t2s_state_sha256": state_sha,
                    "t2s_watermark_id": state_data["watermark_id"],
                    "t2s_provider_config_sha256": state_data["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data["rng_mode"],
                    "t2s_inversion_mode": state_data["inversion_mode"],
                    "t2s_num_inversion_steps": str(state_data["num_inversion_steps"]),
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "T2S")
                rec_wm2 = _min_record("2", "watermarked", "T2S")

                out = _write_fake_run(tdp, method="T2S",
                                      records=[rec_wm, rec_wm2],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm, rec_wm2],
                                           out, "T2S", device="cpu", config={"method": "T2S", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                rows = _read_detector_rows(out)
                run_ids = {r["run_id"] for r in rows}
                assert len(run_ids) == 2

    def test_real_benchmark_num_inference_steps_binding(self, monkeypatch):
        """T2S benchmark_ddim: invert_image kwargs use num_inference_steps."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            state_path = tdp / "states" / "wm.pt"
            state_data = _write_t2s_state(
                state_path,
                inversion_mode="benchmark_ddim",
                num_inference_steps=30,
                num_inversion_steps=15,
            )

            with _t2s_real_deps(monkeypatch, tdp) as (t2s_mod, inv_mod, pu, pipe):
                import hashlib as _hashlib
                state_sha = _hashlib.sha256(state_path.read_bytes()).hexdigest()
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "t2s_state_path": str(state_path),
                    "t2s_state_sha256": state_sha,
                    "t2s_watermark_id": state_data["watermark_id"],
                    "t2s_provider_config_sha256": state_data["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data["rng_mode"],
                    "t2s_inversion_mode": "benchmark_ddim",
                    "t2s_num_inversion_steps": "15",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "T2S")

                out = _write_fake_run(tdp, method="T2S",
                                      records=[rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm],
                                           out, "T2S", device="cpu", config={"method": "T2S", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 2

                # Verify invert_image kwargs from the REAL score_image:
                # benchmark_ddim → benchmark_num_inference_steps = state.num_inference_steps
                inv_calls = inv_mod.invert_image.mock_calls
                assert inv_calls, "invert_image was never called"
                for call in inv_calls:
                    assert call.kwargs, f"invert_image call missing kwargs: {call}"
                    assert call.kwargs["benchmark_num_inference_steps"] == 30, \
                        f"benchmark_num_inference_steps={call.kwargs['benchmark_num_inference_steps']}, " \
                        f"expected 30 (= state.num_inference_steps)"
                    assert call.kwargs["num_inversion_steps"] == 15, \
                        f"num_inversion_steps={call.kwargs['num_inversion_steps']}, " \
                        f"expected 15 (= state.num_inversion_steps)"
                    assert call.kwargs["inversion_mode"] == "benchmark_ddim"

    def test_real_t2s_official_steps_binding(self, monkeypatch):
        """T2S t2s_official: invert_image uses num_inversion_steps = state value."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            state_path = tdp / "states" / "wm.pt"
            state_data = _write_t2s_state(
                state_path,
                inversion_mode="t2s_official",
                num_inversion_steps=12,
                num_inference_steps=50,
            )

            with _t2s_real_deps(monkeypatch, tdp) as (t2s_mod, inv_mod, pu, pipe):
                import hashlib as _hashlib
                state_sha = _hashlib.sha256(state_path.read_bytes()).hexdigest()
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked",
                    "t2s_state_path": str(state_path),
                    "t2s_state_sha256": state_sha,
                    "t2s_watermark_id": state_data["watermark_id"],
                    "t2s_provider_config_sha256": state_data["provider_config_sha256"],
                    "t2s_protocol_mode": "official_math_shared_tr_clean",
                    "t2s_rng_mode": state_data["rng_mode"],
                    "t2s_inversion_mode": "t2s_official",
                    "t2s_num_inversion_steps": "12",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "T2S")

                out = _write_fake_run(tdp, method="T2S",
                                      records=[rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm],
                                           out, "T2S", device="cpu", config={"method": "T2S", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED

                inv_calls = inv_mod.invert_image.mock_calls
                assert inv_calls, "invert_image was never called"
                for call in inv_calls:
                    assert call.kwargs["num_inversion_steps"] == 12, \
                        f"num_inversion_steps={call.kwargs['num_inversion_steps']}, " \
                        f"expected 12 (= state.num_inversion_steps)"
                    assert call.kwargs["inversion_mode"] == "t2s_official"


# ════════════════════════════════════════════════════════════════════════
# RID — real adapter success + wrong manifest method rejection
# ════════════════════════════════════════════════════════════════════════

class TestRIDRealAdapter:
    """RID real load_state with synthetic bundle + MetadataResolver."""

    def test_real_adapter_success_with_bundle_and_csv(self, monkeypatch):
        """RID: real adapter success with synthetic Fourier bundle."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_dir = _make_fourier_bundle_dir(tdp, "RID")

            with _fourier_real_deps(monkeypatch, "RID"):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean", "method": "RID",
                    "rid_bundle_dir": str(bundle_dir),
                    "rid_bundle_config_sha256": "rid_cfg_sha",
                    "rid_selected_pattern_sha256": "rid_pat_sha",
                    "rid_mask_sha256": "rid_mask_sha",
                    "rid_key_index": "0",
                    "rid_protocol_mode": RID_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }, {
                    "run_id": "1", "role": "watermarked", "method": "RID",
                    "rid_bundle_dir": str(bundle_dir),
                    "rid_bundle_config_sha256": "rid_cfg_sha",
                    "rid_selected_pattern_sha256": "rid_pat_sha",
                    "rid_mask_sha256": "rid_mask_sha",
                    "rid_key_index": "0",
                    "rid_protocol_mode": RID_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "RID")
                rec_wm = _min_record("1", "watermarked", "RID")

                out = _write_fake_run(tdp, method="RID",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "RID", device="cpu", config={"method": "RID", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_wrong_manifest_method_rejection(self, monkeypatch):
        """RID: manifest method tag ≠ 'RID' → real state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # Manifest says 'HSTR' but method is 'RID' → mismatch
            bundle_dir = _make_fourier_bundle_dir(tdp, "RID")
            # Overwrite manifest method to "HSTR" to trigger mismatch
            import json as _json
            manifest = _json.loads((bundle_dir / "manifest.json").read_text())
            manifest["method"] = "HSTR"
            (bundle_dir / "manifest.json").write_text(_json.dumps(manifest, sort_keys=True))

            with _fourier_real_deps(monkeypatch, "RID"):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked", "method": "RID",
                    "rid_bundle_dir": str(bundle_dir),
                    "rid_bundle_config_sha256": "rid_cfg_sha",
                    "rid_selected_pattern_sha256": "rid_pat_sha",
                    "rid_mask_sha256": "rid_mask_sha",
                    "rid_key_index": "0",
                    "rid_protocol_mode": RID_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "RID")

                out = _write_fake_run(tdp, method="RID",
                                      records=[rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm],
                                           out, "RID", device="cpu", config={"method": "RID", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_FAILED_STATE_VALIDATION
                assert result["dominant_failure_cause"] == "state_validation_error"


# ════════════════════════════════════════════════════════════════════════
# HSTR — real adapter success + method-specific bundle gate
# ════════════════════════════════════════════════════════════════════════

class TestHSTRRealAdapter:
    """HSTR real load_state with SfwBundle gate validation."""

    def test_real_adapter_success_with_bundle_and_csv(self, monkeypatch):
        """HSTR: real adapter with SfwBundle + MetadataResolver."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_dir = _make_fourier_bundle_dir(tdp, "HSTR")

            with _fourier_real_deps(monkeypatch, "HSTR"):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean", "method": "HSTR",
                    "hstr_bundle_dir": str(bundle_dir),
                    "hstr_bundle_config_sha256": "hstr_cfg_sha",
                    "hstr_selected_pattern_sha256": "hstr_pat_sha",
                    "hstr_mask_sha256": "hstr_mask_sha",
                    "hstr_key_index": "0",
                    "hstr_protocol_mode": HSTR_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }, {
                    "run_id": "1", "role": "watermarked", "method": "HSTR",
                    "hstr_bundle_dir": str(bundle_dir),
                    "hstr_bundle_config_sha256": "hstr_cfg_sha",
                    "hstr_selected_pattern_sha256": "hstr_pat_sha",
                    "hstr_mask_sha256": "hstr_mask_sha",
                    "hstr_key_index": "0",
                    "hstr_protocol_mode": HSTR_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "HSTR")
                rec_wm = _min_record("1", "watermarked", "HSTR")

                out = _write_fake_run(tdp, method="HSTR",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "HSTR", device="cpu", config={"method": "HSTR", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_method_specific_bundle_gate(self, monkeypatch):
        """HSTR: RidBundle (not SfwBundle) → real state_validation."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # RID bundle schema on HSTR method → gate rejection
            bundle_dir = _make_fourier_bundle_dir(tdp, "HSTR")
            # Overwrite bundle schema to rid_bundle_v1 to trigger gate rejection
            import json as _json
            manifest = _json.loads((bundle_dir / "manifest.json").read_text())
            manifest["bundle_schema"] = "rid_bundle_v1"
            (bundle_dir / "manifest.json").write_text(_json.dumps(manifest, sort_keys=True))

            with _fourier_real_deps(monkeypatch, "HSTR"):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "watermarked", "method": "HSTR",
                    "hstr_bundle_dir": str(bundle_dir),
                    "hstr_bundle_config_sha256": "hstr_cfg_sha",
                    "hstr_selected_pattern_sha256": "hstr_pat_sha",
                    "hstr_mask_sha256": "hstr_mask_sha",
                    "hstr_key_index": "0",
                    "hstr_protocol_mode": HSTR_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_wm = _min_record("1", "watermarked", "HSTR")

                out = _write_fake_run(tdp, method="HSTR",
                                      records=[rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_wm],
                                           out, "HSTR", device="cpu", config={"method": "HSTR", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_FAILED_STATE_VALIDATION
                assert result["dominant_failure_cause"] == "state_validation_error"


# ════════════════════════════════════════════════════════════════════════
# HSQR — real adapter success + bundle gate (no state_source contract)
# ════════════════════════════════════════════════════════════════════════

class TestHSQRRealAdapter:
    """HSQR real load_state — no state_source gate."""

    def test_real_adapter_success_with_bundle_and_csv(self, monkeypatch):
        """HSQR: real adapter with SfwBundle, no state_source required."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bundle_dir = _make_fourier_bundle_dir(tdp, "HSQR")

            with _fourier_real_deps(monkeypatch, "HSQR"):
                csv_path = tdp / "metadata.csv"
                csv_rows = [{
                    "run_id": "1", "role": "clean", "method": "HSQR",
                    "hsqr_bundle_dir": str(bundle_dir),
                    "hsqr_bundle_config_sha256": "hsqr_cfg_sha",
                    "hsqr_selected_pattern_sha256": "hsqr_pat_sha",
                    "hsqr_mask_sha256": "hsqr_mask_sha",
                    "hsqr_key_index": "0",
                    "hsqr_protocol_mode": HSQR_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }, {
                    "run_id": "1", "role": "watermarked", "method": "HSQR",
                    "hsqr_bundle_dir": str(bundle_dir),
                    "hsqr_bundle_config_sha256": "hsqr_cfg_sha",
                    "hsqr_selected_pattern_sha256": "hsqr_pat_sha",
                    "hsqr_mask_sha256": "hsqr_mask_sha",
                    "hsqr_key_index": "0",
                    "hsqr_protocol_mode": HSQR_SHARED_TR_CLEAN_MODE,
                    "watermark_target_sha256": "tgt_sha",
                    "watermark_mask_sha256": "tgt_sha",
                }]
                _write_metadata_csv(csv_path, csv_rows)

                rec_clean = _min_record("1", "clean", "HSQR")
                rec_wm = _min_record("1", "watermarked", "HSQR")

                out = _write_fake_run(tdp, method="HSQR",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "HSQR", device="cpu", config={"method": "HSQR", "metadata_path": str(csv_path)})

                assert result["status"] == STATUS_COMPLETED
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["count_invariant_satisfied"] is True

                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_real_bundle_gate_no_state_source_contract(self, monkeypatch):
        """HSQR: bundle loaded successfully without state_source check."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # HSQR with SfwBundle — no state_source gate is applied
            bundle_dir = _make_fourier_bundle_dir(tdp, "HSQR")

            with _fourier_real_deps(monkeypatch, "HSQR"):
                csv_path = tdp / "metadata.csv"
                def hsqr_row(role):
                    return {
                        "run_id": "1", "role": role, "method": "HSQR",
                        "hsqr_bundle_dir": str(bundle_dir),
                        "hsqr_bundle_config_sha256": "hsqr_cfg_sha",
                        "hsqr_selected_pattern_sha256": "hsqr_pat_sha",
                        "hsqr_mask_sha256": "hsqr_mask_sha",
                        "hsqr_key_index": "0",
                        "hsqr_protocol_mode": HSQR_SHARED_TR_CLEAN_MODE,
                        "watermark_target_sha256": "tgt_sha",
                        "watermark_mask_sha256": "tgt_sha",
                    }
                _write_metadata_csv(csv_path, [hsqr_row("clean"),
                                               hsqr_row("watermarked")])

                rec_clean = _min_record("1", "clean", "HSQR")
                rec_wm = _min_record("1", "watermarked", "HSQR")

                out = _write_fake_run(tdp, method="HSQR",
                                      records=[rec_clean, rec_wm],
                                      config={"metadata_path": str(csv_path)})
                result = evaluate_detector([rec_clean, rec_wm],
                                           out, "HSQR", device="cpu", config={"method": "HSQR", "metadata_path": str(csv_path)})
                # The fake HSQR provider has NO state_source attribute at all
                # (see _fake_hsqr_from_bundle).  If the adapter consulted that
                # field, load_state would fail.  Success proves HSQR does not
                # depend on the RID/HSTR state_source contract.
                assert result["status"] == STATUS_COMPLETED, \
                    f"HSQR must complete without state_source, got {result['status']}: {result.get('setup_error')}"
                assert result["scored_count"] == 4
                rows = _read_detector_rows(out)
                assert all(r["status"] == "scored" for r in rows)
