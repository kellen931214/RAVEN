"""Issue #22 complete regression tests — fail-closed TR cohort configuration.

Covers:
- w_pattern_const in REQUIRED_METADATA_FIELDS, passed to provider
- Strict type validation (missing vs invalid separation)
- Profile identity fail-closed (partial field → missing state)
- Pipe built from verified profile, not hard-coded
- Provider-derived target/mask SHA verification
- Canonical scoring boundary (raw_score/canonical_score failures)
- Missing image → FileNotFoundError (never DetectorMissingStateError)
- Aggregate required cohorts (original_clean mandatory for primary)
- Integration: real evaluate_detector with mocked pipe/provider/scoring
- Verified provenance in provider_info and scored rows

Run:  pytest -q raven_repro/tests/test_issue22_tr_detector.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO))


# ===========================================================================
# Synthetic metadata
# ===========================================================================
TR_META_COMPLETE: dict[str, str] = {
    "w_seed": "99",
    "w_channel": "3",
    "w_radius": "10",
    "w_pattern": "ring",
    "w_mask_shape": "circle",
    "w_measurement": "l1_complex",
    "w_injection": "complex",
    "w_pattern_const": "0.0",
}

TR_PROFILE: dict[str, str] = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "scheduler": "DDIM",
    "inverse_scheduler": "DDIMScheduler",
    "steps": "50",
    "resolution": "512",
    "detector_dtype": "torch.float32",
    "vae_id": "checkpoint-default",
    "vae_scaling_factor": "0.18215",
    "provider_config_hash": "e5d3d80eed2103aa130c80280bf0d1387531eddc2b14c0eaea51ea3fcfb54df1",
    "watermark_target_sha256": "default_target_sha_placeholder",
    "watermark_mask_sha256": "default_mask_sha_placeholder",
}


def _make_record(run_id="1", role="watermarked", method="TR",
                 provider_meta=None, profile=None, **kw):
    """Build a synthetic record with TR fields at top level (mimics
    MetadataResolver.enrich_record).  Auto-computes provider_config_hash
    when missing."""
    if provider_meta is None:
        provider_meta = dict(TR_META_COMPLETE)
    if profile is None:
        profile = dict(TR_PROFILE)

    # Always recompute provider_config_hash from the actual provider_meta
    # so it matches what load_state will compute from the record.
    # Skip silently when provider_meta contains deliberately invalid values
    # (negative tests).
    try:
        from raven.eval_protocol import provider_config_hash
        profile["provider_config_hash"] = provider_config_hash(
            "TR", provider_meta)
    except (ValueError, TypeError):
        pass

    record = {
        "run_id": run_id, "role": role, "method": method,
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
        "source_metadata": {**provider_meta, **profile},
        **kw,
    }
    for key in provider_meta:
        record[key] = provider_meta[key]
    for key in profile:
        record[key] = profile[key]
    return record


# ===========================================================================
# Mock harness for load_state unit tests
# ===========================================================================
@contextmanager
def _mock_load_state_deps(monkeypatch):
    """Replace heavy imports so load_state can run CPU-only.  Also patches
    raven.pairing_provenance.tensor_sha256 so downstream callers can set
    the return values."""
    import raven.detectors.tr_detector as tr_mod
    import builtins

    fake_extract = mock.MagicMock()
    monkeypatch.setattr(tr_mod, "_extract_module", fake_extract)
    monkeypatch.setattr(tr_mod, "_get_extract_module", lambda: fake_extract)

    fake_pipe_obj = mock.MagicMock()
    fake_pipe_obj.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe_obj.get_dtype.return_value = "torch.float32"
    scheduler_inv = mock.MagicMock()
    scheduler_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe_obj.scheduler_inverse = scheduler_inv
    fake_pipe_obj.pipe.vae.config.scaling_factor = 0.18215

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.get_pipe_provider.return_value = fake_pipe_obj

    fake_tr_provider_class = mock.MagicMock()
    fake_tr_module = mock.MagicMock(TrProvider=fake_tr_provider_class)
    fake_wm = mock.MagicMock(tr_provider=fake_tr_module)
    fake_utils = mock.MagicMock(
        pipe=mock.MagicMock(pipe_utils=fake_pipe_utils), wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_utils)
    fake_eb.__path__ = []

    _import_modules = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_utils,
        "eval_bench_wm.utils.pipe": fake_utils.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.tr_provider": fake_tr_module,
    }

    original_import = builtins.__import__

    def _mock_import(name, globals=None, locals=None, fromlist=(),
                     level=0):
        if name in _import_modules:
            return _import_modules[name]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    yield fake_pipe_obj, fake_tr_provider_class


def _patch_tensor_sha256(monkeypatch, *sha_values):
    """Patch raven.pairing_provenance.tensor_sha256 to return the given values
    in order.  Defaults match TR_PROFILE placeholders."""
    import raven.pairing_provenance as pp
    if not sha_values:
        sha_values = (TR_PROFILE["watermark_target_sha256"],
                      TR_PROFILE["watermark_mask_sha256"])
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(side_effect=list(sha_values)))


# ===========================================================================
# 1 — w_pattern_const contract
# ===========================================================================
class TestWPatternConst:
    def test_field_in_required_set(self):
        from raven.detectors.tr_detector import REQUIRED_METADATA_FIELDS
        assert "w_pattern_const" in REQUIRED_METADATA_FIELDS

    def test_missing_raises_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE)
        del meta["w_pattern_const"]
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError,
                               match="w_pattern_const"):
                load_state(records, "cpu")

    def test_value_passed_to_provider(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        meta = dict(TR_META_COMPLETE, w_pattern_const="0.75")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            call_kwargs = tr_cls.call_args.kwargs
            assert call_kwargs["w_pattern_const"] == 0.75

    def test_mixed_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", provider_meta=dict(TR_META_COMPLETE,
                                                  w_pattern_const="0.0")),
            _make_record("2", provider_meta=dict(TR_META_COMPLETE,
                                                  w_pattern_const="0.75")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_nan_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_pattern_const=str(float("nan")))
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_pattern_const"):
                load_state(records, "cpu")

    def test_inf_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_pattern_const=str(float("inf")))
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_pattern_const"):
                load_state(records, "cpu")


# ===========================================================================
# 2 — Strict type / value validation
# ===========================================================================
class TestStrictTypeValidation:
    def test_w_seed_not_integer_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_seed="abc")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="w_seed"):
                load_state(records, "cpu")

    def test_w_channel_negative_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_channel="-1")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="w_channel"):
                load_state(records, "cpu")

    def test_w_radius_zero_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_radius="0")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="w_radius"):
                load_state(records, "cpu")

    def test_w_radius_negative_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_radius="-5")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="w_radius"):
                load_state(records, "cpu")

    def test_w_pattern_unknown_value_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_pattern="garbage")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="w_pattern"):
                load_state(records, "cpu")

    def test_w_mask_shape_unknown_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_mask_shape="triangle")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_mask_shape"):
                load_state(records, "cpu")

    def test_w_injection_unknown_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_injection="real_only")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_injection"):
                load_state(records, "cpu")

    def test_w_pattern_const_non_numeric_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_pattern_const="abc")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_pattern_const"):
                load_state(records, "cpu")


# ===========================================================================
# 3 — Profile identity fail-closed
# ===========================================================================
class TestProfileIdentity:
    def test_missing_model_id_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile = dict(TR_PROFILE)
        del profile["model_id"]
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="model_id"):
                load_state(records, "cpu")

    def test_partial_field_one_record_missing(self, monkeypatch):
        """Row A has model_id, row B does not → missing state."""
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile_a = dict(TR_PROFILE)
        profile_b = dict(TR_PROFILE)
        del profile_b["model_id"]

        records = [
            _make_record("1", profile=profile_a),
            _make_record("2", profile=profile_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="model_id"):
                load_state(records, "cpu")

    def test_mixed_profile_field_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", profile=dict(TR_PROFILE, resolution="512")),
            _make_record("2", profile=dict(TR_PROFILE, resolution="256")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="resolution"):
                load_state(records, "cpu")

    def test_all_profile_fields_present_and_uniform_ok(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_make_record("1"), _make_record("2")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
        assert "verified_profile" in result
        vp = result["verified_profile"]
        assert vp["model_id"] == TR_PROFILE["model_id"]
        assert vp["scheduler"] == "DDIM"


# ===========================================================================
# 4 — Pipe built from verified profile
# ===========================================================================
class TestPipeProfile:
    def test_pipe_dtype_mismatch_detected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, detector_dtype="torch.float16")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="detector_dtype"):
                load_state(records, "cpu")

    def test_pipe_inverse_scheduler_mismatch_detected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, inverse_scheduler="DDPMScheduler")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="inverse scheduler"):
                load_state(records, "cpu")


# ===========================================================================
# 5 — Provider-derived target / mask verification
# ===========================================================================
class TestTargetMaskVerification:
    def test_source_target_must_match_provider_derived(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="source_target_sha",
                       watermark_mask_sha256="source_mask_sha")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch,
                                 "detector_target_sha", "source_mask_sha")
            with pytest.raises(DetectorStateValidationError,
                               match="target"):
                load_state(records, "cpu")

    def test_source_mask_must_match_provider_derived(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="target_sha",
                       watermark_mask_sha256="source_mask_sha")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch,
                                 "target_sha", "detector_mask_sha")
            with pytest.raises(DetectorStateValidationError,
                               match="mask"):
                load_state(records, "cpu")

    def test_matching_target_and_mask_ok(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="same_target",
                       watermark_mask_sha256="same_mask")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch, "same_target", "same_mask")
            result = load_state(records, "cpu")
            assert result["source_watermark_target_sha256"] == "same_target"
            assert result["detector_watermark_target_sha256"] == "same_target"
            assert result["source_watermark_mask_sha256"] == "same_mask"
            assert result["detector_watermark_mask_sha256"] == "same_mask"

    def test_target_missing_in_profile_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile = dict(TR_PROFILE)
        del profile["watermark_target_sha256"]
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError,
                               match="watermark_target_sha256"):
                load_state(records, "cpu")

    def test_mask_missing_in_profile_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile = dict(TR_PROFILE)
        del profile["watermark_mask_sha256"]
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError,
                               match="watermark_mask_sha256"):
                load_state(records, "cpu")

    def test_all_wrong_target_sha_is_validation_error(self, monkeypatch):
        """All rows have same — but wrong — target SHA → state validation."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="wrong_target",
                       watermark_mask_sha256="correct_mask")
        records = [_make_record("1", profile=profile),
                   _make_record("2", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch, "real_target", "correct_mask")
            with pytest.raises(DetectorStateValidationError,
                               match="target"):
                load_state(records, "cpu")


# ===========================================================================
# 6 — Scoring boundary taxonomy
# ===========================================================================
class TestScoringBoundary:
    def _make_provider_info(self):
        fake_mod = mock.MagicMock()
        fake_mod.evaluate_image.return_value = {
            "p_values": [0.001],
            "p_value_diagnostics": [{"log_p": -20.0, "sigma": 1.0,
                                     "lambda": 100.0, "statistic": 50.0,
                                     "df": 100, "p_underflow": False}],
        }
        fake_mod.raw_score.return_value = 0.001
        fake_mod.canonical_score.return_value = 10.0

        return {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "extract_module": fake_mod,
            "detector_provider_config_hash": "hash_abc",
            "source_watermark_target_sha256": "target_abc",
            "detector_watermark_target_sha256": "target_abc",
            "source_watermark_mask_sha256": "mask_abc",
            "detector_watermark_mask_sha256": "mask_abc",
            "verified_profile": {
                "model_id": "test/model", "model_revision": "rev",
                "scheduler": "DDIM", "inverse_scheduler": "DDIMScheduler",
                "steps": 50, "resolution": 512,
                "detector_dtype": "torch.float32",
            },
        }

    def _make_fake_image(self, tmp_path):
        from PIL import Image
        img = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img)
        return str(img)

    def test_missing_image_raises_file_not_found(self):
        from raven.detectors.tr_detector import score_image

        with pytest.raises(FileNotFoundError):
            score_image({"fake": True}, "/nonexistent/path.png")

    def test_raw_score_failure_is_scoring_error(self, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        info = self._make_provider_info()
        info["extract_module"].raw_score.side_effect = ValueError("bad raw")

        with pytest.raises(DetectorScoringError, match="bad raw"):
            score_image(info, self._make_fake_image(tmp_path))

    def test_canonical_score_failure_is_scoring_error(self, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        info = self._make_provider_info()
        info["extract_module"].canonical_score.side_effect = (
            RuntimeError("bad canonical"))

        with pytest.raises(DetectorScoringError, match="bad canonical"):
            score_image(info, self._make_fake_image(tmp_path))

    def test_nan_raw_score_is_scoring_error(self, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        info = self._make_provider_info()
        info["extract_module"].raw_score.return_value = float("nan")

        with pytest.raises(DetectorScoringError, match="raw_score"):
            score_image(info, self._make_fake_image(tmp_path))

    def test_non_numeric_raw_score_is_scoring_error(self, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        info = self._make_provider_info()
        info["extract_module"].raw_score.return_value = "not_a_number"

        with pytest.raises(DetectorScoringError, match="not numeric"):
            score_image(info, self._make_fake_image(tmp_path))

    def test_missing_diagnostics_is_scoring_error(self, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        info = self._make_provider_info()
        info["extract_module"].evaluate_image.return_value = {
            "p_values": [0.001],
        }

        with pytest.raises(DetectorScoringError,
                           match="p_value_diagnostics"):
            score_image(info, self._make_fake_image(tmp_path))

    def test_no_image_io_in_score_image(self):
        """score_image must not call Image.open — canonical helper does it."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "tr_detector.py").read_text()
        assert "Image.open" not in source
        assert "ImageOps" not in source


# ===========================================================================
# 7 — Aggregate required cohorts
# ===========================================================================
class TestAggregateCohorts:
    def test_original_clean_is_required_for_primary(self):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
        ]
        result = aggregate(rows)
        assert "original_clean" in result["missing_cohorts"]
        assert "detection_summary" not in result

    def test_attacked_clean_missing_does_not_block_primary(self):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 4.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 11.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 6.0},
        ]
        result = aggregate(rows)
        assert "detection_summary" in result
        assert result["missing_cohorts"] == []
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False

    def test_four_cohorts_recalibrated_available(self):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 4.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 11.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 6.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 3.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 2.0},
        ]
        result = aggregate(rows)
        assert "detection_summary" in result
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is True
        assert result["tr_recalibrated"]["attacked_clean_count"] == 2


# ===========================================================================
# 8 — Integration harness (standalone functions, not class methods)
# ===========================================================================
def _write_fake_run(tmp_path, method="TR", records=None, config=None):
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
        img.write_bytes(b"fake png")
        input_path = Path(r.get("input_path", f"/tmp/in_{rid}.png"))
        if not input_path.is_file():
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(b"fake png")
    rebuild_records_jsonl(out)
    return out


def _read_detector_rows(output_dir):
    from raven.experiment_io import detector_records_path
    path = detector_records_path(output_dir)
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@contextmanager
def _patch_integration(monkeypatch,
                       target_sha=TR_PROFILE["watermark_target_sha256"],
                       mask_sha=TR_PROFILE["watermark_mask_sha256"],
                       raw_score_val=0.001,
                       canonical_score_val=10.0):
    """Mock only pipe, provider construction, canonical scoring, and
    tensor hashing.  Everything else — load_state, score_image,
    evaluate_detector, aggregate, stage reducer — runs for real."""
    import raven.detectors.tr_detector as tr_mod
    import builtins

    # Mock extract module (canonical scoring)
    fake_extract = mock.MagicMock()
    try:
        log_p = math.log(float(raw_score_val)) if float(raw_score_val) > 0 else -690.0
    except (ValueError, TypeError):
        log_p = -690.0
    fake_extract.evaluate_image.return_value = {
        "p_values": [raw_score_val],
        "p_value_diagnostics": [
            {"log_p": log_p,
             "sigma": 1.0, "lambda": 100.0, "statistic": 50.0,
             "df": 100, "p_underflow": False},
        ],
    }
    fake_extract.raw_score.return_value = raw_score_val
    fake_extract.canonical_score.return_value = canonical_score_val
    monkeypatch.setattr(tr_mod, "_extract_module", fake_extract)
    monkeypatch.setattr(tr_mod, "_get_extract_module", lambda: fake_extract)

    # Mock eval_bench_wm imports
    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"
    scheduler_inv = mock.MagicMock()
    scheduler_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe.scheduler_inverse = scheduler_inv
    fake_pipe.pipe.vae.config.scaling_factor = 0.18215

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.get_pipe_provider.return_value = fake_pipe

    fake_provider = mock.MagicMock()
    fake_provider.gt_patch = mock.MagicMock()
    fake_provider.watermarking_mask = mock.MagicMock()

    fake_tr_provider_class = mock.MagicMock(return_value=fake_provider)
    fake_tr_module = mock.MagicMock(TrProvider=fake_tr_provider_class)
    fake_wm = mock.MagicMock(tr_provider=fake_tr_module)
    fake_utils = mock.MagicMock(
        pipe=mock.MagicMock(pipe_utils=fake_pipe_utils), wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_utils)
    fake_eb.__path__ = []

    _import_modules = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_utils,
        "eval_bench_wm.utils.pipe": fake_utils.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.tr_provider": fake_tr_module,
    }

    original_import = builtins.__import__
    def _mock_import(name, globals=None, locals=None, fromlist=(),
                     level=0):
        if name in _import_modules:
            return _import_modules[name]
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", _mock_import)

    # Mock tensor_sha256
    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(side_effect=[target_sha, mask_sha]))

    yield fake_pipe, fake_provider, fake_tr_provider_class


# ===========================================================================
# 9 — Integration tests with real evaluate_detector
# ===========================================================================
class TestIntegrationSuccess:
    """Successful uniform cohort — real evaluate_detector path."""

    def test_three_cohort_primary_report(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")

        rec_clean = _make_record("1", "clean", profile=profile)
        rec_wm = _make_record("1", "watermarked", profile=profile)
        rec_wm2 = _make_record("2", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, target_sha="ok_target",
                                mask_sha="ok_mask"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm, rec_wm2])
                result = evaluate_detector(
                    [rec_clean, rec_wm, rec_wm2], out, "TR", device="cpu")

        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] > 0
        assert result["failed_count"] == 0
        ma = result["metric_availability"]
        assert ma["primary_report_available"] is True
        assert ma["threshold_report_available"] is True
        # clean records auto-populate attacked_clean entries in orchestrator,
        # so recalibrated is available when all entries score successfully
        assert ma["recalibrated_report_available"] is True

        rows = _read_detector_rows(out)
        assert all(r["status"] == "scored" for r in rows)

    def test_four_cohort_recalibration(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")

        rec_clean = _make_record("1", "clean", profile=profile)
        rec_wm = _make_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, target_sha="ok_target",
                                mask_sha="ok_mask"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm])
                result = evaluate_detector(
                    [rec_clean, rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_COMPLETED
        ma = result["metric_availability"]
        assert ma["recalibrated_report_available"] is True

    def test_provider_constructed_once(self, monkeypatch):
        """Exactly one TrProvider constructed for a uniform cohort."""
        from experiments.eval import evaluate_detector

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")

        rec_clean = _make_record("1", "clean", profile=profile)
        rec_wm = _make_record("1", "watermarked", profile=profile)
        rec_wm2 = _make_record("2", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, target_sha="ok_target",
                                mask_sha="ok_mask") as (_pipe, _prov, tr_cls):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm, rec_wm2])
                evaluate_detector(
                    [rec_clean, rec_wm, rec_wm2], out, "TR", device="cpu")
            assert tr_cls.call_count == 1, (
                f"Expected 1 provider, got {tr_cls.call_count}")


class TestIntegrationFailures:
    """Validation and scoring failure taxonomy through real orchestrator."""

    def test_missing_profile_field_setup_failure(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_REQUIRED_STATE

        profile = dict(TR_PROFILE)
        del profile["model_id"]
        rec_wm = _make_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_wm])
                result = evaluate_detector(
                    [rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_missing_target_sha_setup_failure(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_REQUIRED_STATE

        profile = dict(TR_PROFILE)
        del profile["watermark_target_sha256"]
        rec_wm = _make_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_wm])
                result = evaluate_detector(
                    [rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_target_sha_mismatch_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="wrong_target",
                       watermark_mask_sha256="ok_mask")
        rec_wm = _make_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, target_sha="real_target",
                                mask_sha="ok_mask"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_wm])
                result = evaluate_detector(
                    [rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_mixed_provider_config_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")

        rec_a = _make_record("1", "watermarked",
                             provider_meta=dict(TR_META_COMPLETE,
                                                w_seed="99"),
                             profile=profile)
        rec_b = _make_record("2", "watermarked",
                             provider_meta=dict(TR_META_COMPLETE,
                                                w_seed="88888"),
                             profile=profile)

        with _patch_integration(monkeypatch, target_sha="ok_target",
                                mask_sha="ok_mask"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_a, rec_b])
                result = evaluate_detector(
                    [rec_a, rec_b], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_canonical_helper_failure_scoring_error(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")

        rec_clean = _make_record("1", "clean", profile=profile)
        rec_wm = _make_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, target_sha="ok_target",
                                mask_sha="ok_mask",
                                raw_score_val="bad_string",
                                canonical_score_val="also_bad"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm])
                result = evaluate_detector(
                    [rec_clean, rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        assert result["scored_count"] == 0


# ===========================================================================
# 10 — Complete contract and provenance verification
# ===========================================================================
class TestCompleteContract:
    def test_required_fields_match_canonical(self):
        from raven.detectors.tr_detector import REQUIRED_METADATA_FIELDS
        from raven.eval_protocol import TR_PROVIDER_FIELDS
        canonical = set(TR_PROVIDER_FIELDS)
        assert REQUIRED_METADATA_FIELDS == canonical, (
            f"adapter {sorted(REQUIRED_METADATA_FIELDS)} != "
            f"canonical {sorted(canonical)}"
        )

    def test_all_required_fields_in_provider_kwargs(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_make_record("1")]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            kwargs = tr_cls.call_args.kwargs
            for field in ("w_seed", "w_channel", "w_radius", "w_pattern",
                          "w_mask_shape", "w_measurement", "w_injection",
                          "w_pattern_const"):
                assert field in kwargs, f"{field} not passed to TrProvider"

    def test_verified_provenance_in_provider_info(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE,
                       watermark_target_sha256="ok_target",
                       watermark_mask_sha256="ok_mask")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch, "ok_target", "ok_mask")
            result = load_state(records, "cpu")

        for key in ("source_provider_config_hash",
                    "detector_provider_config_hash",
                    "source_watermark_target_sha256",
                    "detector_watermark_target_sha256",
                    "source_watermark_mask_sha256",
                    "detector_watermark_mask_sha256",
                    "verified_profile"):
            assert key in result, f"Missing {key} in provider_info"

    def test_scored_row_has_provenance_fields(self, tmp_path):
        from raven.detectors.tr_detector import score_image

        fake_mod = mock.MagicMock()
        fake_mod.evaluate_image.return_value = {
            "p_values": [0.001],
            "p_value_diagnostics": [
                {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
                 "statistic": 50.0, "df": 100, "p_underflow": False},
            ],
        }
        fake_mod.raw_score.return_value = 0.001
        fake_mod.canonical_score.return_value = 10.0

        info = {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "extract_module": fake_mod,
            "detector_provider_config_hash": "h",
            "source_watermark_target_sha256": "st",
            "detector_watermark_target_sha256": "dt",
            "source_watermark_mask_sha256": "sm",
            "detector_watermark_mask_sha256": "dm",
            "verified_profile": {
                "model_id": "m", "model_revision": "r",
                "scheduler": "DDIM", "inverse_scheduler": "DDIMScheduler",
                "steps": 50, "resolution": 512,
                "detector_dtype": "torch.float32",
            },
        }

        from PIL import Image
        img = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img)

        score = score_image(info, str(img))

        for field in ("tr_provider_config_hash", "tr_provider_config_verified",
                      "tr_source_watermark_target_sha256",
                      "tr_detector_watermark_target_sha256",
                      "tr_target_verified",
                      "tr_source_watermark_mask_sha256",
                      "tr_detector_watermark_mask_sha256",
                      "tr_mask_verified", "tr_model_id", "tr_scheduler"):
            assert field in score, f"Missing {field} in score dict"
