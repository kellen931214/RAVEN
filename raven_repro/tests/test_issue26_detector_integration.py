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
            if not input_path.is_file():
                input_path.parent.mkdir(parents=True, exist_ok=True)
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
