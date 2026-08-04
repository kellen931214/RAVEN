"""Issue #20 — canonical per-sample GS detection (comprehensive).

Covers the full acceptance contract:

1. Per-source provider cache (one provider per source, not per image)
2. Required metadata validation BEFORE provider_kwargs call
3. Formal provider configuration identity (require_uniform_provider_config),
   pipe config and detection mode kept OUT of the formal hash
4. Pipe from verified provider config, ``revision=`` kwarg, no fallback
5. Metadata index prevents record cross-use
6. Secret state failure structured classification
7. Canonical scoring helpers (evaluate_image, raw_score, canonical_score)
   in one fail-closed boundary
8. Required scoring outputs no default fallback, decoded bits 0/1 exact length
9. Official thresholds fail closed, [0,1] range, operator validated
10. Explicit verified provenance (source/detector pairs + flags)
11. Missing image → FileNotFoundError
12. Real evaluate_detector integration, shared-clean mode included

All tests use mocks only — no secret bundles, models, or datasets downloaded.

Run:  pytest -q raven_repro/tests/test_issue20_gs_detector.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO / "raven_repro" / "scripts"))
sys.path.insert(0, str(REPO))

# Pre-populate sys.modules with fake heavy-import modules so load_state
# does not trigger the broken lpips/pipe_utils import chain.
_fake_pipe_utils = mock.MagicMock()
_fake_pipe_utils.get_pipe_provider = mock.MagicMock(
    return_value=mock.MagicMock())
_fake_pipe_utils.__name__ = "pipe_utils"

_fake_gs_provider_mod = mock.MagicMock()
_fake_gs_provider_mod.GsProvider = mock.MagicMock()
_fake_gs_provider_mod.__name__ = "gs_provider"

for _mod_name in (
    "eval_bench_wm.utils.pipe.pipe_utils",
    "eval_bench_wm.utils.pipe",
    "eval_bench_wm.utils.wm.gs_provider",
    "eval_bench_wm.utils.wm",
    "eval_bench_wm.utils",
    "eval_bench_wm",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = mock.MagicMock()
        sys.modules[_mod_name].__name__ = _mod_name

sys.modules["eval_bench_wm.utils.pipe.pipe_utils"] = _fake_pipe_utils
sys.modules["eval_bench_wm.utils.wm.gs_provider"] = _fake_gs_provider_mod
# Bind submodule attributes on parent package mocks so
# `from eval_bench_wm.utils.pipe import pipe_utils` resolves to our fakes.
sys.modules["eval_bench_wm.utils.pipe"].pipe_utils = _fake_pipe_utils
sys.modules["eval_bench_wm.utils.wm"].gs_provider = _fake_gs_provider_mod

import extract_verification_scores  # noqa: E402 — land in sys.modules

from raven.detectors.gs_detector import (  # noqa: E402
    score_image,
    load_state,
    aggregate,
    REQUIRED_METADATA_FIELDS,
    describe_required_artifacts,
    _validate_required_gs_metadata,
    _build_metadata_index,
    _validate_pipe_config_uniformity,
    _validate_gs_provider_config,
    _construct_provider,
    _validate_scoring_result,
    _validate_thresholds,
    _validate_decoded_bits,
    _strict_nonneg_int,
)
from raven.detectors import (  # noqa: E402
    DetectorMissingStateError,
    DetectorStateValidationError,
    DetectorScoringError,
    DetectorProviderInitializationError,
    DetectorDependencyError,
    ROW_STATUS_SCORED,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_SCORING,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_SCORING_ERROR,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION,
)
from raven.pairing_provenance import GS_SHARED_TR_CLEAN_MODE  # noqa: E402
from raven.eval_protocol import (  # noqa: E402
    canonical_json_hash,
    require_uniform_provider_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MESSAGE_WIDTH_BYTES = 32
DECODED_BITS_256 = "0" * (MESSAGE_WIDTH_BYTES * 8)  # 256 chars


def _secret_provenance(secret_index=5, **overrides):
    msg = overrides.pop("message_sha256", f"msg_{secret_index:04d}_sha256")
    key = overrides.pop("key_sha256", f"key_{secret_index:04d}_sha256")
    nonce = overrides.pop("nonce_sha256", f"nonce_{secret_index:04d}_sha256")
    bundle = overrides.pop("secret_bundle_sha256",
                           f"bundle_{secret_index:04d}_sha256")
    result = {
        "secret_index": secret_index,
        "message_sha256": msg,
        "key_sha256": key,
        "nonce_sha256": nonce,
        "secret_bundle_sha256": bundle,
    }
    result.update(overrides)
    return result


def _target_tensor():
    return torch.zeros((1, 4, 8, 8), dtype=torch.uint8)


def _mock_provider_instance(secret_idx=5, protocol="official_compatible",
                            bit_acc=0.85, decoded=None,
                            detection_mode="official_onebit",
                            detection_success=True, **runtime_attrs):
    if decoded is None:
        decoded = DECODED_BITS_256
    inst = mock.MagicMock()
    inst.secret_provenance.return_value = _secret_provenance(secret_idx)
    inst.watermark_target_tensor.return_value = _target_tensor()
    inst.gs_protocol_mode = protocol
    inst.gs_detection_mode = detection_mode
    inst.message_width_in_bytes = MESSAGE_WIDTH_BYTES
    inst.l = 1
    inst.num_replications = 64
    inst.gs_channel_copy = 1
    inst.gs_hw_copy = 8
    inst.gs_fpr = 1e-6
    inst.gs_user_number = 1000000
    for k, v in runtime_attrs.items():
        setattr(inst, k, v)
    inst.invert_images.return_value = {"zT_torch": torch.zeros(1, 4, 64, 64)}
    inst.get_accuracies.return_value = {
        "bit_accuracies": [bit_acc],
        "message_bits_str_list": [decoded],
    }
    inst.official_thresholds.return_value = {
        "tau_onebit": 0.9,
        "tau_bits": 0.95,
        "fpr": 1e-6,
        "user_number": 1000000,
        "comparison_operator": ">=",
        "source": "test",
    }
    inst.active_detection_threshold.return_value = {
        "detection_mode": detection_mode,
        "threshold": 0.9,
        "threshold_type": "official_beta_tail_tau_onebit",
        "comparison_operator": ">=",
        "nominal_fpr": 1e-6,
        "calibrated_from_current_clean_negatives": False,
        "official_tau_onebit": 0.9,
        "official_tau_bits": 0.95,
    }
    inst.is_detection_successful.return_value = detection_success
    return inst


def _formal_hash(records):
    """Formal provider_config_hash via require_uniform_provider_config."""
    _, h = require_uniform_provider_config("GS", records)
    return h


def _resolved_metadata(run_id="1", role="watermarked", secret_index=5,
                       protocol="official_compatible", **kw):
    idx = secret_index
    rec = {
        "run_id": run_id,
        "role": role,
        "gs_secret_index": str(idx),
        "gs_message_sha256": f"msg_{idx:04d}_sha256",
        "gs_key_sha256": f"key_{idx:04d}_sha256",
        "gs_nonce_sha256": f"nonce_{idx:04d}_sha256",
        "gs_secret_bundle_sha256": f"bundle_{idx:04d}_sha256",
        "gs_protocol_mode": protocol,
        "watermark_target_sha256": "TGT_HASH",
        "watermark_mask_sha256": "MASK_SENTINEL",
        "provider_config_hash": "CFG_HASH",
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "unspecified",
        "scheduler": "DDIM",
        "resolution": "512",
        "gs_detection_mode": "official_onebit",
    }
    rec.update(kw)
    return rec


def _build_fake_png(tmp_path, name="test.png"):
    from PIL import Image as PILImage
    img_path = tmp_path / name
    img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path, format="PNG")
    return str(img_path)


def _eval_entry(run_id="1", source_role="watermarked", cohort="original_watermarked"):
    return {
        "run_id": run_id,
        "source_role": source_role,
        "evaluation_cohort": cohort,
        "image_path": "/tmp/x.png",
    }


def _mock_pipe():
    p = mock.MagicMock()
    p.get_latent_shape.return_value = (1, 4, 64, 64)
    p.get_dtype.return_value = torch.float32
    return p


def _configure_fake_modules(mock_pipe=None, gs_provider_cls=None):
    """Configure sys.modules mocks before load_state."""
    if mock_pipe is None:
        mock_pipe = _mock_pipe()
    _fake_pipe_utils.get_pipe_provider.return_value = mock_pipe
    if gs_provider_cls is not None:
        _fake_gs_provider_mod.GsProvider = gs_provider_cls


# ---------------------------------------------------------------------------
# 2. Required metadata preflight
# ---------------------------------------------------------------------------
class TestRequiredMetadataPreflight:
    """_validate_required_gs_metadata catches missing/bad fields."""

    def test_all_fields_present_passes(self):
        rec = _resolved_metadata()
        _validate_required_gs_metadata(rec)  # no exception

    def test_missing_run_id_raises(self):
        with pytest.raises(DetectorMissingStateError, match="run_id"):
            _validate_required_gs_metadata({})

    def test_missing_role_raises(self):
        with pytest.raises(DetectorMissingStateError, match="role"):
            _validate_required_gs_metadata({"run_id": "1"})

    def test_missing_gs_secret_index_raises(self):
        rec = _resolved_metadata()
        del rec["gs_secret_index"]
        with pytest.raises(DetectorMissingStateError,
                          match="gs_secret_index"):
            _validate_required_gs_metadata(rec)

    def test_empty_gs_secret_index_raises(self):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = ""
        with pytest.raises(DetectorMissingStateError,
                          match="gs_secret_index"):
            _validate_required_gs_metadata(rec)

    def test_none_secret_index_raises(self):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = None
        with pytest.raises(DetectorMissingStateError,
                          match="gs_secret_index"):
            _validate_required_gs_metadata(rec)

    def test_missing_provider_config_hash_raises(self):
        rec = _resolved_metadata()
        del rec["provider_config_hash"]
        with pytest.raises(DetectorMissingStateError,
                          match="provider_config_hash"):
            _validate_required_gs_metadata(rec)

    def test_missing_message_sha256_raises(self):
        rec = _resolved_metadata()
        rec["gs_message_sha256"] = ""
        with pytest.raises(DetectorMissingStateError,
                          match="gs_message_sha256"):
            _validate_required_gs_metadata(rec)

    # ---- strict secret index validation (section 6) ----
    @pytest.mark.parametrize("bad", [
        "-1", "1.5", "1e2", "abc", True, False, 1.5, float("inf"),
    ])
    def test_invalid_secret_index_values(self, bad):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = bad
        with pytest.raises((DetectorMissingStateError,
                            DetectorStateValidationError)):
            _validate_required_gs_metadata(rec)

    @pytest.mark.parametrize("good", [0, 5, "5", "0", 999999])
    def test_valid_secret_index_values(self, good):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = good
        _validate_required_gs_metadata(rec)

    def test_strict_nonneg_int_rejects_bool(self):
        with pytest.raises(ValueError, match="bool"):
            _strict_nonneg_int(True)
        with pytest.raises(ValueError, match="bool"):
            _strict_nonneg_int(False)

    def test_strict_nonneg_int_rejects_float(self):
        with pytest.raises(ValueError):
            _strict_nonneg_int(1.5)

    def test_strict_nonneg_int_rejects_scientific(self):
        with pytest.raises(ValueError):
            _strict_nonneg_int("1e2")

    def test_strict_nonneg_int_accepts_digits(self):
        assert _strict_nonneg_int("42") == 42
        assert _strict_nonneg_int(42) == 42


# ---------------------------------------------------------------------------
# 5. Metadata index
# ---------------------------------------------------------------------------
class TestMetadataIndex:
    def test_builds_unique_keys(self):
        recs = [
            _resolved_metadata("1", "clean"),
            _resolved_metadata("1", "watermarked"),
        ]
        idx = _build_metadata_index(recs)
        assert len(idx) == 2
        assert ("1", "clean") in idx
        assert ("1", "watermarked") in idx

    def test_duplicate_key_raises(self):
        recs = [
            _resolved_metadata("1", "watermarked"),
            _resolved_metadata("1", "watermarked"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="duplicate"):
            _build_metadata_index(recs)


# ---------------------------------------------------------------------------
# 3. Provider configuration identity — formal hash only
# ---------------------------------------------------------------------------
class TestProviderConfigIdentity:
    def test_uniform_config_passes(self):
        recs = [
            _resolved_metadata("1", "clean"),
            _resolved_metadata("1", "watermarked"),
        ]
        cfg, h, pipe_h, det_h = _validate_gs_provider_config(recs)
        assert "gs_protocol_mode" in cfg
        # pipe/detection fields NOT in formal config
        assert "model_id" not in cfg
        assert "gs_detection_mode" not in cfg

    def test_formal_hash_equals_require_uniform(self):
        """detector hash == require_uniform_provider_config hash."""
        recs = [_resolved_metadata("1", "clean"),
                _resolved_metadata("1", "watermarked")]
        _, h, _, _ = _validate_gs_provider_config(recs)
        formal = _formal_hash(recs)
        assert h == formal

    def test_formal_provider_hash_excludes_pipe_config(self):
        """Changing pipe config must NOT change the formal provider hash."""
        rec_a = _resolved_metadata("1", "watermarked",
                                   model_id="stabilityai/sd-2-1",
                                   scheduler="DPM")
        rec_b = _resolved_metadata("1", "watermarked")
        # Same embedding config, different pipe fields
        _, ha, _, _ = _validate_gs_provider_config([rec_a])
        _, hb, _, _ = _validate_gs_provider_config([rec_b])
        assert ha == hb

    def test_formal_provider_hash_excludes_detection_mode(self):
        """Changing gs_detection_mode must NOT change formal provider hash."""
        rec_a = _resolved_metadata("1", "watermarked",
                                   gs_detection_mode="official_onebit")
        rec_b = _resolved_metadata("1", "watermarked",
                                   gs_detection_mode="legacy_default")
        _, ha, _, _ = _validate_gs_provider_config([rec_a])
        _, hb, _, _ = _validate_gs_provider_config([rec_b])
        assert ha == hb

    def test_mixed_protocol_mode_raises(self):
        recs = [
            _resolved_metadata("1", "clean",
                              gs_protocol_mode="official_compatible"),
            _resolved_metadata("1", "watermarked",
                              gs_protocol_mode="legacy"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="not uniform"):
            _validate_gs_provider_config(recs)

    def test_mixed_detection_mode_raises(self):
        recs = [
            _resolved_metadata("1", "clean",
                              gs_detection_mode="official_onebit"),
            _resolved_metadata("1", "watermarked",
                              gs_detection_mode="legacy_default"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="detection mode"):
            _validate_gs_provider_config(recs)


# ---------------------------------------------------------------------------
# 5. Pipe config — fail closed, no fallback
# ---------------------------------------------------------------------------
class TestPipeConfig:
    @pytest.mark.parametrize("field", ["model_id", "model_revision",
                                       "scheduler", "resolution"])
    def test_missing_field_raises(self, field):
        rec = _resolved_metadata()
        rec[field] = ""
        with pytest.raises(DetectorMissingStateError,
                          match=f"missing required pipe config field: {field}"):
            _validate_pipe_config_uniformity([rec])

    def test_none_field_raises(self):
        rec = _resolved_metadata()
        rec["model_id"] = None
        with pytest.raises(DetectorMissingStateError,
                          match="model_id"):
            _validate_pipe_config_uniformity([rec])

    def test_invalid_resolution_raises(self):
        rec = _resolved_metadata()
        rec["resolution"] = "abc"
        with pytest.raises(DetectorStateValidationError,
                          match="resolution must be an integer"):
            _validate_pipe_config_uniformity([rec])

    def test_mixed_profile_raises(self):
        recs = [
            _resolved_metadata("1", "clean",
                              model_id="stabilityai/sd-2-1"),
            _resolved_metadata("1", "watermarked",
                              model_id="RedbeardNZ/sd-2-1-base"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="pipe config not uniform"):
            _validate_pipe_config_uniformity(recs)

    def test_no_fallback_defaults(self):
        """Absent pipe fields never fall back to hardcoded defaults."""
        rec = _resolved_metadata()
        for field in ("model_id", "model_revision", "scheduler", "resolution"):
            del rec[field]
        with pytest.raises(DetectorMissingStateError):
            _validate_pipe_config_uniformity([rec])

    def test_uniform_profile_passes(self):
        recs = [
            _resolved_metadata("1", "clean"),
            _resolved_metadata("1", "watermarked"),
        ]
        cfg = _validate_pipe_config_uniformity(recs)
        assert cfg["model_id"] == "RedbeardNZ/stable-diffusion-2-1-base"
        assert cfg["scheduler"] == "DDIM"
        assert cfg["resolution"] == 512


# ---------------------------------------------------------------------------
# 1. model_revision normalization
# ---------------------------------------------------------------------------
class TestModelRevisionNormalization:
    """Sentinel revisions normalize to None; never passed to pipe."""

    @pytest.mark.parametrize("value", [None, "", "none", "null",
                                       "unspecified", "UNSPECIFIED",
                                       "  unspecified  "])
    def test_sentinels_normalize_to_none(self, value):
        from raven.detectors.gs_detector import _normalize_model_revision
        assert _normalize_model_revision(value) is None

    @pytest.mark.parametrize("value", ["fp16", "9f4b8f2", "main",
                                       "some-tag"])
    def test_real_revisions_kept(self, value):
        from raven.detectors.gs_detector import _normalize_model_revision
        assert _normalize_model_revision(value) == value

    def test_unspecified_revision_is_not_passed_to_pipe(self, monkeypatch):
        captured = {}
        _fake_pipe_utils.get_pipe_provider.side_effect = (
            lambda **kw: captured.update(kw) or _mock_pipe())
        GsProvider = mock.MagicMock()
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        rec = _resolved_metadata("1", "watermarked",
                                 model_revision="unspecified")
        load_state([rec], "cpu")
        assert "revision" not in captured

    def test_none_revision_is_not_passed_to_pipe(self, monkeypatch):
        captured = {}
        _fake_pipe_utils.get_pipe_provider.side_effect = (
            lambda **kw: captured.update(kw) or _mock_pipe())
        GsProvider = mock.MagicMock()
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        rec = _resolved_metadata("1", "watermarked",
                                 model_revision="none")
        load_state([rec], "cpu")
        assert "revision" not in captured

    def test_empty_revision_is_not_passed_to_pipe(self, monkeypatch):
        captured = {}
        _fake_pipe_utils.get_pipe_provider.side_effect = (
            lambda **kw: captured.update(kw) or _mock_pipe())
        GsProvider = mock.MagicMock()
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        rec = _resolved_metadata("1", "watermarked",
                                 model_revision="null")
        load_state([rec], "cpu")
        assert "revision" not in captured

    def test_real_revision_is_passed_to_pipe(self, monkeypatch):
        captured = {}
        _fake_pipe_utils.get_pipe_provider.side_effect = (
            lambda **kw: captured.update(kw) or _mock_pipe())
        GsProvider = mock.MagicMock()
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        rec = _resolved_metadata("1", "watermarked",
                                 model_revision="9f4b8f2")
        load_state([rec], "cpu")
        assert captured["revision"] == "9f4b8f2"

    def test_revision_sentinels_have_same_pipe_hash(self):
        """unspecified / null / none produce identical pipe config hash."""
        rec_a = _resolved_metadata("1", "watermarked",
                                   model_revision="unspecified")
        rec_b = _resolved_metadata("1", "watermarked",
                                   model_revision="null")
        rec_c = _resolved_metadata("1", "watermarked",
                                   model_revision="none")
        cfg_a = _validate_pipe_config_uniformity([rec_a])
        cfg_b = _validate_pipe_config_uniformity([rec_b])
        cfg_c = _validate_pipe_config_uniformity([rec_c])
        assert cfg_a["model_revision"] is None
        assert cfg_a == cfg_b == cfg_c

    def test_pinned_revision_changes_pipe_hash(self):
        rec_a = _resolved_metadata("1", "watermarked",
                                   model_revision="9f4b8f2")
        rec_b = _resolved_metadata("1", "watermarked",
                                   model_revision="unspecified")
        cfg_a = _validate_pipe_config_uniformity([rec_a])
        cfg_b = _validate_pipe_config_uniformity([rec_b])
        assert cfg_a["model_revision"] == "9f4b8f2"
        assert cfg_a != cfg_b


# ---------------------------------------------------------------------------
# 4. Strict resolution validation
# ---------------------------------------------------------------------------
class TestStrictResolution:
    """Resolution must be a strictly positive integer."""

    @pytest.mark.parametrize("bad", [True, False, 512.5, 0, -1, "512.0",
                                     "abc", "1e2", None])
    def test_invalid_resolution_fails(self, bad):
        rec = _resolved_metadata()
        rec["resolution"] = bad
        if bad is None:
            with pytest.raises(DetectorMissingStateError):
                _validate_pipe_config_uniformity([rec])
        else:
            with pytest.raises(DetectorStateValidationError,
                              match="resolution"):
                _validate_pipe_config_uniformity([rec])

    @pytest.mark.parametrize("good", [512, "512", 768, "768", 1])
    def test_valid_resolution_passes(self, good):
        rec = _resolved_metadata()
        rec["resolution"] = good
        cfg = _validate_pipe_config_uniformity([rec])
        assert cfg["resolution"] == int(good)

    def test_strict_positive_int_rejects_bool(self):
        from raven.detectors.gs_detector import _strict_positive_int
        with pytest.raises(ValueError, match="bool"):
            _strict_positive_int(True, "resolution")
        with pytest.raises(ValueError, match="bool"):
            _strict_positive_int(False, "resolution")

    def test_strict_positive_int_rejects_float_and_nonpositive(self):
        from raven.detectors.gs_detector import _strict_positive_int
        for bad in (512.5, 0, -1, "512.0", "abc"):
            with pytest.raises(ValueError):
                _strict_positive_int(bad, "resolution")


# ---------------------------------------------------------------------------
# 5. role / source_role normalization
# ---------------------------------------------------------------------------
class TestRoleNormalization:
    """_resolved_source_role used consistently, no silent watermarked."""

    def test_source_role_only_clean(self):
        from raven.detectors.gs_detector import _resolved_source_role
        assert _resolved_source_role({"source_role": "clean"}) == "clean"

    def test_source_role_only_watermarked(self):
        from raven.detectors.gs_detector import _resolved_source_role
        assert _resolved_source_role({"source_role": "watermarked"}) == \
            "watermarked"

    def test_role_and_source_role_agree(self):
        from raven.detectors.gs_detector import _resolved_source_role
        assert _resolved_source_role(
            {"role": "clean", "source_role": "clean"}) == "clean"

    def test_role_source_role_conflict_raises(self):
        from raven.detectors.gs_detector import _resolved_source_role
        with pytest.raises(DetectorStateValidationError, match="conflict"):
            _resolved_source_role(
                {"role": "clean", "source_role": "watermarked"})

    def test_unknown_role_raises(self):
        from raven.detectors.gs_detector import _resolved_source_role
        with pytest.raises(DetectorStateValidationError, match="unknown"):
            _resolved_source_role({"role": "banana"})

    def test_missing_role_raises(self):
        from raven.detectors.gs_detector import _resolved_source_role
        with pytest.raises(DetectorMissingStateError, match="role"):
            _resolved_source_role({"run_id": "1"})

    def test_source_role_clean_record_indexed_as_clean(self):
        """Record with only source_role=clean must NOT index as watermarked."""
        rec = _resolved_metadata("1", "clean")
        del rec["role"]
        rec["source_role"] = "clean"
        idx = _build_metadata_index([rec])
        assert ("1", "clean") in idx
        assert ("1", "watermarked") not in idx

    def test_preflight_accepts_source_role_only(self):
        rec = _resolved_metadata("1", "watermarked")
        del rec["role"]
        rec["source_role"] = "watermarked"
        _validate_required_gs_metadata(rec)  # no exception


# ---------------------------------------------------------------------------
# 2. gs_detection_mode fail-closed validation
# ---------------------------------------------------------------------------
class TestDetectionModeValidation:
    """gs_detection_mode required, enum-validated, uniform."""

    def test_missing_detection_mode_raises(self):
        rec = _resolved_metadata()
        del rec["gs_detection_mode"]
        with pytest.raises(DetectorMissingStateError,
                          match="gs_detection_mode"):
            _validate_required_gs_metadata(rec)

    def test_empty_detection_mode_raises(self):
        rec = _resolved_metadata()
        rec["gs_detection_mode"] = ""
        with pytest.raises(DetectorMissingStateError,
                          match="gs_detection_mode"):
            _validate_required_gs_metadata(rec)

    @pytest.mark.parametrize("bad", ["foo", "official", "1", ">="])
    def test_invalid_detection_mode_raises(self, bad):
        rec = _resolved_metadata()
        rec["gs_detection_mode"] = bad
        with pytest.raises(DetectorStateValidationError,
                          match="unsupported gs_detection_mode"):
            _validate_required_gs_metadata(rec)

    @pytest.mark.parametrize("good", ["official_onebit",
                                      "official_traceability",
                                      "legacy_default"])
    def test_valid_detection_modes_pass(self, good):
        rec = _resolved_metadata()
        rec["gs_detection_mode"] = good
        _validate_required_gs_metadata(rec)  # no exception

    def test_mixed_detection_mode_hash_raises(self):
        recs = [
            _resolved_metadata("1", "clean",
                              gs_detection_mode="official_onebit"),
            _resolved_metadata("1", "watermarked",
                              gs_detection_mode="legacy_default"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="detection mode not uniform"):
            _validate_gs_provider_config(recs)

    def test_provider_runtime_mode_mismatch_raises(self, monkeypatch):
        """Provider runtime gs_detection_mode differs from metadata."""
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5, detection_mode="legacy_default")
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )
        canonical = {"gs_protocol_mode": "official_compatible",
                     "message_width_in_bytes": 32, "l": 1,
                     "num_replications": 64, "gs_channel_copy": 1,
                     "gs_hw_copy": 8, "gs_fpr": 1e-6,
                     "gs_user_number": 1000000}
        meta = _resolved_metadata("1", "watermarked",
                                  gs_detection_mode="official_onebit")
        with pytest.raises(DetectorStateValidationError,
                          match="gs_detection_mode"):
            _construct_provider(_mock_pipe(), GsProvider,
                               torch.device("cpu"), meta, canonical)

    def test_constructor_ignores_detection_mode_raises(self, monkeypatch):
        """Provider constructor ignores gs_detection_mode → mismatch."""
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5, detection_mode="official_onebit")
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        canonical = {"gs_protocol_mode": "official_compatible",
                     "message_width_in_bytes": 32, "l": 1,
                     "num_replications": 64, "gs_channel_copy": 1,
                     "gs_hw_copy": 8, "gs_fpr": 1e-6,
                     "gs_user_number": 1000000}
        meta = _resolved_metadata("1", "watermarked",
                                  gs_detection_mode="official_traceability")
        with pytest.raises(DetectorStateValidationError,
                          match="gs_detection_mode"):
            _construct_provider(_mock_pipe(), GsProvider,
                               torch.device("cpu"), meta, canonical)


# ---------------------------------------------------------------------------
# 2.5 Active detection policy
# ---------------------------------------------------------------------------
class TestActiveDetectionPolicy:
    """Provider-owned policy validated, no adapter math."""

    @staticmethod
    def _scored(monkeypatch, tmp_path, mode="official_onebit",
                threshold=0.9, tau_onebit=0.9, tau_bits=0.95,
                success=True, threshold_type="official_beta_tail_tau_onebit",
                operator=">="):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5, detection_mode=mode,
                                       detection_success=success)
        inst.active_detection_threshold.return_value = {
            "detection_mode": mode,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "comparison_operator": operator,
            "nominal_fpr": 1e-6,
            "calibrated_from_current_clean_negatives": False,
            "official_tau_onebit": tau_onebit,
            "official_tau_bits": tau_bits,
        }
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )
        meta = _resolved_metadata("1", "watermarked",
                                  gs_detection_mode=mode)
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        return score_image(prov_info, img, record=meta,
                          evaluation_entry=_eval_entry("1", "watermarked"))

    def test_official_onebit_uses_tau_onebit(self, monkeypatch, tmp_path):
        result = self._scored(monkeypatch, tmp_path, mode="official_onebit",
                              threshold=0.9)
        assert result["gs_detection_mode"] == "official_onebit"
        assert result["gs_active_threshold"] == 0.9
        assert result["gs_active_threshold_type"] == \
            "official_beta_tail_tau_onebit"
        assert result["gs_active_comparison_operator"] == ">="
        assert result["gs_active_nominal_fpr"] == 1e-6
        assert result["gs_active_calibrated_from_current_clean_negatives"] \
            is False
        assert result["gs_detection_success"] is True
        assert result["gs_official_tau_onebit"] == 0.9
        assert result["gs_detection_policy_hash"] == "DET_HASH"

    def test_official_traceability_uses_tau_bits(self, monkeypatch, tmp_path):
        result = self._scored(monkeypatch, tmp_path,
                              mode="official_traceability",
                              threshold=0.95,
                              threshold_type="official_beta_tail_tau_bits")
        assert result["gs_active_threshold"] == 0.95
        assert result["gs_active_threshold_type"] == \
            "official_beta_tail_tau_bits"

    def test_legacy_default_keeps_strict_gt(self, monkeypatch, tmp_path):
        result = self._scored(monkeypatch, tmp_path, mode="legacy_default",
                              threshold=0.9,
                              threshold_type="legacy_default_threshold",
                              operator=">")
        assert result["gs_active_comparison_operator"] == ">"

    def test_detection_success_is_bool(self, monkeypatch, tmp_path):
        result = self._scored(monkeypatch, tmp_path, success=True)
        assert isinstance(result["gs_detection_success"], bool)

    def test_nonbool_detection_success_raises(self, monkeypatch, tmp_path):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        inst.is_detection_successful.return_value = "yes"
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )
        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        with pytest.raises(DetectorScoringError, match="real bool"):
            score_image(prov_info, img, record=meta,
                       evaluation_entry=_eval_entry("1", "watermarked"))

    def test_onebit_must_use_tau_onebit(self, monkeypatch, tmp_path):
        """official_onebit with threshold != tau_onebit → scoring error."""
        with pytest.raises(DetectorScoringError, match="tau_onebit"):
            self._scored(monkeypatch, tmp_path, mode="official_onebit",
                         threshold=0.5, tau_onebit=0.9)

    def test_missing_policy_key_raises(self):
        from raven.detectors.gs_detector import _validate_active_policy
        with pytest.raises(DetectorScoringError, match="missing keys"):
            _validate_active_policy({"detection_mode": "official_onebit"},
                                    "official_onebit", "1")


# ---------------------------------------------------------------------------
# 1. Per-source provider cache
# ---------------------------------------------------------------------------
class TestProviderCache:
    """Provider constructed once per source, reused across cohorts."""

    @staticmethod
    def _mock_helpers(monkeypatch):
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {
                "offset": int(row.get("gs_secret_index", 0)),
                "gs_secret_index": int(row.get("gs_secret_index", 0)),
            },
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )

    def test_two_sources_two_providers(self, monkeypatch):
        GsProvider = mock.MagicMock()
        inst5 = _mock_provider_instance(5)
        inst7 = _mock_provider_instance(7)
        GsProvider.side_effect = [inst5, inst7]
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)

        meta_clean = _resolved_metadata("1", "clean", secret_index=5)
        meta_wm = _resolved_metadata("1", "watermarked", secret_index=7)

        prov_info = load_state([meta_clean, meta_wm], "cpu")
        assert GsProvider.call_count == 0  # not constructed in load_state

        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        r1 = score_image(
            prov_info, img,
            record=meta_clean,
            evaluation_entry=_eval_entry("1", "clean", "original_clean"),
        )
        assert GsProvider.call_count == 1

        r2 = score_image(
            prov_info, img,
            record=meta_clean,
            evaluation_entry=_eval_entry("1", "clean", "attacked_clean"),
        )
        assert GsProvider.call_count == 1  # cached

        r3 = score_image(
            prov_info, img,
            record=meta_wm,
            evaluation_entry=_eval_entry("1", "watermarked",
                                        "original_watermarked"),
        )
        assert GsProvider.call_count == 2

        r4 = score_image(
            prov_info, img,
            record=meta_wm,
            evaluation_entry=_eval_entry("1", "watermarked",
                                        "attacked_watermarked"),
        )
        assert GsProvider.call_count == 2  # cached

        assert r1["gs_secret_index"] == 5
        assert r3["gs_secret_index"] == 7

    def test_provider_cache_keys_correct(self, monkeypatch):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        assert ("1", "watermarked") in prov_info["metadata_index"]
        assert prov_info["provider_cache"] == {}

        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        score_image(prov_info, img, record=meta,
                   evaluation_entry=_eval_entry("1", "watermarked"))
        assert ("1", "watermarked") in prov_info["provider_cache"]


# ---------------------------------------------------------------------------
# 4. Provider runtime config validation
# ---------------------------------------------------------------------------
class TestProviderRuntimeConfig:
    """Constructor receives full canonical config; runtime fields verified."""

    def test_nondefault_provider_fields_passed_to_provider(self, monkeypatch):
        """Custom fields flow through provider_kwargs_full to constructor."""
        canonical = {
            "gs_protocol_mode": "official_compatible",
            "message_width_in_bytes": 32,
            "l": 1,
            "num_replications": 64,
            "gs_channel_copy": 1,
            "gs_hw_copy": 8,
            "gs_fpr": 1e-6,
            "gs_user_number": 1000000,
        }
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst

        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )

        meta = _resolved_metadata("1", "watermarked")
        _construct_provider(_mock_pipe(), GsProvider, torch.device("cpu"),
                           meta, canonical)

        # Constructor received the FULL canonical config merged with per-row
        _, kwargs = GsProvider.call_args
        for field in ("gs_protocol_mode", "message_width_in_bytes", "l",
                      "num_replications", "gs_channel_copy", "gs_hw_copy",
                      "gs_fpr", "gs_user_number", "gs_secret_index", "offset"):
            assert field in kwargs, f"missing {field} in provider kwargs"
        assert kwargs["gs_protocol_mode"] == "official_compatible"
        assert kwargs["message_width_in_bytes"] == 32
        assert kwargs["gs_secret_index"] == 5

    def test_runtime_field_mismatch_raises(self, monkeypatch):
        """Provider runtime field differs from canonical → state validation."""
        canonical = {
            "gs_protocol_mode": "official_compatible",
            "message_width_in_bytes": 32,
            "l": 1,
            "num_replications": 64,
            "gs_channel_copy": 1,
            "gs_hw_copy": 8,
            "gs_fpr": 1e-6,
            "gs_user_number": 1000000,
        }
        GsProvider = mock.MagicMock()
        # Provider reports wrong gs_hw_copy — constructor ignored kwarg
        inst = _mock_provider_instance(5, gs_hw_copy=4)
        GsProvider.return_value = inst

        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )

        meta = _resolved_metadata("1", "watermarked")
        with pytest.raises(DetectorStateValidationError,
                          match="gs_hw_copy mismatch"):
            _construct_provider(_mock_pipe(), GsProvider,
                               torch.device("cpu"), meta, canonical)


# ---------------------------------------------------------------------------
# 3. Shared-clean mode
# ---------------------------------------------------------------------------
class TestSharedCleanMode:
    """official_math_shared_tr_clean protocol passes through to provider."""

    def _shared_clean_meta(self, run_id="1", role="watermarked",
                           secret_index=5):
        rec = _resolved_metadata(
            run_id, role, secret_index=secret_index,
            protocol=GS_SHARED_TR_CLEAN_MODE,
            gs_protocol_mode=GS_SHARED_TR_CLEAN_MODE,
            gs_sampling_seed="",  # V2 has no sampling seed
            watermark_target_sha256="TGT_HASH",
            watermark_mask_sha256="MASK_SENTINEL",
        )
        rec["provider_config_hash"] = _formal_hash([rec])
        return rec

    @staticmethod
    def _mock_helpers(monkeypatch):
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )

    def test_shared_clean_mode_passed_to_provider(self, monkeypatch):
        """Provider receives gs_protocol_mode = official_math_shared_tr_clean."""
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(
            5, protocol=GS_SHARED_TR_CLEAN_MODE,
            gs_protocol_mode=GS_SHARED_TR_CLEAN_MODE)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {
                "offset": int(row.get("gs_secret_index", 0)),
                "gs_secret_index": int(row.get("gs_secret_index", 0)),
            },
        )

        meta = self._shared_clean_meta()
        prov_info = load_state([meta], "cpu")
        assert prov_info["canonical_config"]["gs_protocol_mode"] == \
            GS_SHARED_TR_CLEAN_MODE

        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        result = score_image(prov_info, img, record=meta,
                            evaluation_entry=_eval_entry(
                                "1", "watermarked"))
        assert result["gs_protocol_mode"] == GS_SHARED_TR_CLEAN_MODE

        # Constructor received the shared-clean mode
        _, kwargs = GsProvider.call_args
        assert kwargs["gs_protocol_mode"] == GS_SHARED_TR_CLEAN_MODE

    def test_shared_clean_mode_scores_successfully(self, monkeypatch):
        """Full shared-clean path returns a scored result."""
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(
            5, protocol=GS_SHARED_TR_CLEAN_MODE,
            gs_protocol_mode=GS_SHARED_TR_CLEAN_MODE)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {
                "offset": int(row.get("gs_secret_index", 0)),
                "gs_secret_index": int(row.get("gs_secret_index", 0)),
            },
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )

        meta = self._shared_clean_meta()
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        result = score_image(prov_info, img, record=meta,
                            evaluation_entry=_eval_entry(
                                "1", "watermarked"))
        assert result["status"] if "status" in result else True
        assert result["gs_protocol_mode"] == GS_SHARED_TR_CLEAN_MODE
        assert result["raw_score"] == 0.85
        assert result["canonical_score"] == 0.85


# ---------------------------------------------------------------------------
# 6. Secret state failure classification
# ---------------------------------------------------------------------------
class TestSecretStateFailureClassification:
    """IndexError → missing state, TypeError → provider init, etc."""

    def test_secret_index_out_of_range(self, monkeypatch):
        pipe = _mock_pipe()
        GsProvider = mock.MagicMock()
        GsProvider.side_effect = IndexError("list index out of range")

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 9999, "gs_secret_index": 9999},
        )

        meta = _resolved_metadata("1", "watermarked", secret_index=9999)
        with pytest.raises(DetectorMissingStateError,
                          match="out of range"):
            _construct_provider(pipe, GsProvider, torch.device("cpu"),
                               meta, {})

    def test_secret_provenance_index_error(self, monkeypatch):
        pipe = _mock_pipe()
        GsProvider = mock.MagicMock()
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
        inst.secret_provenance.side_effect = IndexError("out of range")
        GsProvider.return_value = inst

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )

        meta = _resolved_metadata("1", "watermarked")
        with pytest.raises(DetectorMissingStateError,
                          match="secret_provenance index"):
            _construct_provider(pipe, GsProvider, torch.device("cpu"),
                               meta, {})

    def test_constructor_type_error_is_provider_init(self, monkeypatch):
        pipe = _mock_pipe()
        GsProvider = mock.MagicMock()
        GsProvider.side_effect = TypeError("bad arg")

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )

        meta = _resolved_metadata("1", "watermarked")
        with pytest.raises(DetectorProviderInitializationError,
                          match="type error"):
            _construct_provider(pipe, GsProvider, torch.device("cpu"),
                               meta, {})


# ---------------------------------------------------------------------------
# 7. Canonical scoring helpers + 8. outputs validation
# ---------------------------------------------------------------------------
class TestCanonicalScoringHelpers:
    """evaluate_image, raw_score, canonical_score in one fail-closed boundary."""

    @staticmethod
    def _mock_helpers(monkeypatch):
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )

    def test_successful_scoring_path(self, monkeypatch, tmp_path):
        """evaluate_image called; raw/canonical from formal helpers."""
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)

        called = []
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch_mod, provider, pipe, path, steps: (
                called.append(("evaluate_image", path)),
                {"bit_accuracies": [0.85],
                 "message_bits_str_list": [DECODED_BITS_256]},
            )[1],
        )

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        result = score_image(prov_info, img, record=meta,
                            evaluation_entry=_eval_entry("1", "watermarked"))
        assert called[0] == ("evaluate_image", Path(img))
        assert result["raw_score"] == 0.85
        assert result["canonical_score"] == 0.85
        assert result["bit_accuracy"] == 0.85

    def test_raw_score_failure_is_scoring_error(self, monkeypatch, tmp_path):
        """raw_score helper raising → DetectorScoringError."""
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda *a, **k: {"bit_accuracies": [0.85],
                            "message_bits_str_list": [DECODED_BITS_256]},
        )
        monkeypatch.setattr(
            extract_verification_scores, "raw_score",
            lambda method, result: (_ for _ in ()).throw(
                RuntimeError("raw_score exploded")),
        )

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        with pytest.raises(DetectorScoringError,
                          match="raw_score exploded"):
            score_image(prov_info, img, record=meta,
                       evaluation_entry=_eval_entry("1", "watermarked"))

    def test_canonical_score_failure_is_scoring_error(self, monkeypatch,
                                                      tmp_path):
        """canonical_score helper raising → DetectorScoringError."""
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda *a, **k: {"bit_accuracies": [0.85],
                            "message_bits_str_list": [DECODED_BITS_256]},
        )
        monkeypatch.setattr(
            extract_verification_scores, "canonical_score",
            lambda method, raw, result: (_ for _ in ()).throw(
                RuntimeError("canonical exploded")),
        )

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        with pytest.raises(DetectorScoringError,
                          match="canonical exploded"):
            score_image(prov_info, img, record=meta,
                       evaluation_entry=_eval_entry("1", "watermarked"))

    def test_nonfinite_raw_score_is_scoring_error(self, monkeypatch, tmp_path):
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda *a, **k: {"bit_accuracies": [float("inf")],
                            "message_bits_str_list": [DECODED_BITS_256]},
        )

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        with pytest.raises(DetectorScoringError):
            score_image(prov_info, img, record=meta,
                       evaluation_entry=_eval_entry("1", "watermarked"))

    def test_nonfinite_canonical_score_is_scoring_error(self, monkeypatch,
                                                        tmp_path):
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda *a, **k: {"bit_accuracies": [0.85],
                            "message_bits_str_list": [DECODED_BITS_256]},
        )
        monkeypatch.setattr(
            extract_verification_scores, "canonical_score",
            lambda method, raw, result: float("nan"),
        )

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        with pytest.raises(DetectorScoringError):
            score_image(prov_info, img, record=meta,
                       evaluation_entry=_eval_entry("1", "watermarked"))


# ---------------------------------------------------------------------------
# 8. Scoring output validation — no default fallback
# ---------------------------------------------------------------------------
class TestScoringOutputValidation:
    """Missing/illegal scoring outputs → DetectorScoringError."""

    def test_missing_bit_accuracies(self):
        with pytest.raises(DetectorScoringError, match="missing or empty"):
            _validate_scoring_result({}, "1")

    def test_empty_bit_accuracies(self):
        with pytest.raises(DetectorScoringError, match="missing or empty"):
            _validate_scoring_result(
                {"bit_accuracies": [],
                 "message_bits_str_list": [DECODED_BITS_256]}, "1")

    def test_non_float_bit_accuracy(self):
        with pytest.raises(DetectorScoringError,
                          match="not convertible to float"):
            _validate_scoring_result(
                {"bit_accuracies": ["hello"],
                 "message_bits_str_list": [DECODED_BITS_256]}, "1")

    def test_nan_bit_accuracy(self):
        with pytest.raises(DetectorScoringError, match="non-finite"):
            _validate_scoring_result(
                {"bit_accuracies": [float("nan")],
                 "message_bits_str_list": [DECODED_BITS_256]}, "1")

    def test_out_of_range_bit_accuracy(self):
        with pytest.raises(DetectorScoringError, match="out of range"):
            _validate_scoring_result(
                {"bit_accuracies": [1.5],
                 "message_bits_str_list": [DECODED_BITS_256]}, "1")

    def test_missing_message_bits(self):
        with pytest.raises(DetectorScoringError, match="missing or empty"):
            _validate_scoring_result(
                {"bit_accuracies": [0.85],
                 "message_bits_str_list": []}, "1")

    def test_empty_decoded_string(self):
        with pytest.raises(DetectorScoringError, match="empty"):
            _validate_scoring_result(
                {"bit_accuracies": [0.85],
                 "message_bits_str_list": [""]}, "1")

    def test_valid_result_passes(self):
        _validate_scoring_result(
            {"bit_accuracies": [0.85],
             "message_bits_str_list": [DECODED_BITS_256]}, "1")


class TestDecodedBitsValidation:
    """Decoded bits must be binary string of exact length."""

    def test_non_binary_chars(self):
        with pytest.raises(DetectorScoringError,
                          match="non-binary"):
            _validate_decoded_bits("0101010x", "1", 32)

    def test_wrong_length(self):
        with pytest.raises(DetectorScoringError, match="length"):
            _validate_decoded_bits("1010", "1", 32)

    def test_empty_string(self):
        with pytest.raises(DetectorScoringError, match="empty"):
            _validate_decoded_bits("", "1", 32)

    def test_valid_256_bits(self):
        _validate_decoded_bits(DECODED_BITS_256, "1", 32)


# ---------------------------------------------------------------------------
# 9. Official thresholds fail closed
# ---------------------------------------------------------------------------
class TestThresholdValidation:
    """Missing/non-finite/out-of-range thresholds → DetectorScoringError."""

    def test_missing_tau_onebit(self):
        with pytest.raises(DetectorScoringError, match="missing tau_onebit"):
            _validate_thresholds({"tau_bits": 0.95}, "1")

    def test_nan_tau_bits(self):
        with pytest.raises(DetectorScoringError, match="non-finite"):
            _validate_thresholds(
                {"tau_onebit": 0.9, "tau_bits": float("nan")}, "1")

    def test_out_of_range_tau(self):
        with pytest.raises(DetectorScoringError, match="out of range"):
            _validate_thresholds(
                {"tau_onebit": 1.5, "tau_bits": 0.95}, "1")

    def test_non_numeric_threshold(self):
        with pytest.raises(DetectorScoringError,
                          match="not convertible"):
            _validate_thresholds(
                {"tau_onebit": "hello", "tau_bits": 0.95}, "1")

    def test_unsupported_operator(self):
        with pytest.raises(DetectorScoringError,
                          match="comparison operator"):
            _validate_thresholds(
                {"tau_onebit": 0.9, "tau_bits": 0.95,
                 "comparison_operator": "!="}, "1")

    def test_valid_thresholds_pass(self):
        rec = _validate_thresholds(
            {"tau_onebit": 0.9, "tau_bits": 0.95,
             "fpr": 1e-6, "user_number": 1000000,
             "comparison_operator": ">=", "source": "test"}, "1")
        assert rec["gs_official_tau_onebit"] == 0.9
        assert rec["gs_official_tau_bits"] == 0.95
        assert rec["gs_official_fpr"] == 1e-6
        assert rec["gs_official_user_number"] == 1000000
        assert rec["gs_official_comparison_operator"] == ">="
        assert rec["gs_official_source"] == "test"


# ---------------------------------------------------------------------------
# 10 + 11. Verified provenance + missing image
# ---------------------------------------------------------------------------
class TestVerifiedProvenance:
    """Score output carries source/detector pairs and verified flags."""

    @staticmethod
    def _mock_helpers(monkeypatch):
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_SENTINEL" if "mask" in str(p) else "CFG_HASH",
        )

    def test_scored_row_has_verified_fields(self, monkeypatch, tmp_path):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        img = _build_fake_png(tmp_path, "img.png")
        result = score_image(
            prov_info, img, record=meta,
            evaluation_entry=_eval_entry("1", "watermarked"),
        )

        assert result["source_watermark_target_sha256"] == "TGT_HASH"
        assert result["detector_watermark_target_sha256"] == "TGT_HASH"
        assert result["source_watermark_mask_sha256"] == "MASK_SENTINEL"
        assert result["detector_watermark_mask_sha256"] == "MASK_SENTINEL"
        assert result["source_provider_config_hash"] == "CFG_HASH"
        assert result["detector_provider_config_hash"] == "CFG_HASH"
        assert result["detector_pipe_config_hash"] == "PIPE_HASH"
        assert result["gs_detection_policy_hash"] == "DET_HASH"
        assert result["gs_secret_verified"] is True
        assert result["gs_target_verified"] is True
        assert result["gs_mask_verified"] is True
        assert result["provider_config_verified"] is True

    def test_missing_image_raises_file_not_found(self, monkeypatch):
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)
        self._mock_helpers(monkeypatch)

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        with pytest.raises(FileNotFoundError):
            score_image(
                prov_info, "/tmp/raven_issue20_definitely_missing_input.png",
                record=meta,
                evaluation_entry=_eval_entry("1", "watermarked"),
            )


# ---------------------------------------------------------------------------
# 4. Pipe constructor arguments
# ---------------------------------------------------------------------------
class TestPipeConstructorArgs:
    """Pipe gets verified config with ``revision=`` kwarg, no fallback."""

    def test_pipe_kwargs_use_revision_and_verified_config(self, monkeypatch):
        captured = {}

        def _fake_get_pipe(**kwargs):
            captured.update(kwargs)
            return _mock_pipe()

        _fake_pipe_utils.get_pipe_provider.side_effect = _fake_get_pipe
        GsProvider = mock.MagicMock()
        _configure_fake_modules(gs_provider_cls=GsProvider)
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible",
                       "message_width_in_bytes": 32, "l": 1,
                       "num_replications": 64, "gs_channel_copy": 1,
                       "gs_hw_copy": 8, "gs_fpr": 1e-6,
                       "gs_user_number": 1000000},
                      "CFG_HASH", "PIPE_HASH", "DET_HASH"),
        )

        rec = _resolved_metadata(
            "1", "watermarked",
            model_id="stabilityai/sd-2-1",
            model_revision="fp16",
            scheduler="DPM",
            resolution="768",
        )
        prov_info = load_state([rec], "cpu")
        assert captured["pretrained_model_name_or_path"] == "stabilityai/sd-2-1"
        assert captured["revision"] == "fp16"
        assert captured["schedulers_name"] == "DPM"
        assert captured["resolution"] == 768
        assert "model_revision" not in captured


# ---------------------------------------------------------------------------
# 12. Real evaluate_detector integration tests
# ---------------------------------------------------------------------------
class TestEvaluateDetectorIntegration:
    """evaluate_detector with real load_state + score_image, only heavy
    resources mocked."""

    @staticmethod
    def _make_record(run_id="1", role="watermarked", method="GS", **kw):
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

    def _write_fake_run(self, tmp_path, method="GS", records=None,
                        skip_input=False):
        from raven.experiment_io import (
            write_config, write_record, rebuild_records_jsonl,
        )
        out = tmp_path / "run"
        out.mkdir()
        cfg = {"method": method, "dataset": "test"}
        write_config(out, cfg)
        if records is None:
            records = [self._make_record("1", "watermarked", method=method)]
        for r in records:
            role = r.get("role", "watermarked")
            rid = r["run_id"]
            write_record(out, role, rid, r)
            img = out / "samples" / role / rid / "output.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"fake png")
            if not skip_input:
                input_path = Path(r.get("input_path",
                                        f"/tmp/in_{rid}.png"))
                if not input_path.is_file():
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    input_path.write_bytes(b"fake png")
        rebuild_records_jsonl(out)
        return out

    def _read_detector_rows(self, output_dir):
        from raven.experiment_io import detector_records_path
        path = detector_records_path(output_dir)
        if not path.is_file():
            return []
        return [json.loads(l)
                for l in path.read_text().splitlines() if l.strip()]

    def _setup_mocks(self, monkeypatch, protocol="official_compatible"):
        """Mock only the heavy-resource boundaries."""
        _configure_fake_modules(mock_pipe=_mock_pipe())

        # Mock GsProvider class
        self._gs_factory = mock.MagicMock()
        self._gs_instances = {}

        def _make_inst(*args, **kwargs):
            idx = kwargs.get("gs_secret_index", 0)
            if idx not in self._gs_instances:
                self._gs_instances[idx] = _mock_provider_instance(
                    idx, protocol=protocol, gs_protocol_mode=protocol)
            return self._gs_instances[idx]

        self._gs_factory.side_effect = _make_inst
        _fake_gs_provider_mod.GsProvider = self._gs_factory

        def _fake_eval(torch_mod, prov, pipe, path, steps):
            return {
                "bit_accuracies": [0.85],
                "message_bits_str_list": [DECODED_BITS_256],
            }

        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image", _fake_eval,
        )

        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )

        from raven.eval_protocol import canonical_json_hash
        self._mask_sentinel = canonical_json_hash(
            {"method": "GS", "mask": "not_applicable", "version": 1},
        )

    def _gs_meta(self, run_id="1", role="watermarked", secret_index=5,
                 protocol="official_compatible"):
        idx = secret_index
        rec = _resolved_metadata(
            run_id, role, secret_index=idx, protocol=protocol,
            gs_protocol_mode=protocol,
        )
        # Formal hash via require_uniform_provider_config
        rec["provider_config_hash"] = _formal_hash([rec])
        rec["watermark_target_sha256"] = "TGT_HASH"
        rec["watermark_mask_sha256"] = self._mask_sentinel
        return rec

    # ---- Successful per-source path (official_compatible) ----
    def test_two_sources_four_entries_two_providers(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED, ROW_STATUS_SCORED

        self._setup_mocks(monkeypatch)

        rec_clean = self._make_record(
            "1", "clean", method="GS",
            source_metadata=self._gs_meta("1", "clean", 5))
        rec_wm = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 7))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4
            assert result["failed_count"] == 0
            assert self._gs_factory.call_count == 2

            rows = self._read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
            for row in rows:
                assert row.get("gs_secret_verified") is True
                assert row.get("gs_target_verified") is True
                assert row.get("provider_config_verified") is True

    # ---- Shared-clean mode through real adapter ----
    def test_shared_clean_two_sources_four_entries(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED, ROW_STATUS_SCORED

        self._setup_mocks(monkeypatch, protocol=GS_SHARED_TR_CLEAN_MODE)

        rec_clean = self._make_record(
            "1", "clean", method="GS",
            source_metadata=self._gs_meta("1", "clean", 5,
                                          protocol=GS_SHARED_TR_CLEAN_MODE))
        rec_wm = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 7,
                                          protocol=GS_SHARED_TR_CLEAN_MODE))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4
            assert self._gs_factory.call_count == 2

            rows = self._read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
            for row in rows:
                assert row["gs_protocol_mode"] == GS_SHARED_TR_CLEAN_MODE
                assert row["provider_config_verified"] is True

    # ---- Missing secret index → failed_missing_required_state ----
    def test_missing_secret_index(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_REQUIRED_STATE

        self._setup_mocks(monkeypatch)

        bad_meta = dict(self._gs_meta("1", "watermarked", 5))
        del bad_meta["gs_secret_index"]

        rec = self._make_record(
            "1", "watermarked", method="GS", source_metadata=bad_meta)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            assert self._gs_factory.call_count == 0

            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE
                      for r in rows)

    # ---- Provider config hash mismatch → failed_state_validation ----
    def test_provider_config_hash_mismatch(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        self._setup_mocks(monkeypatch)

        bad_meta = dict(self._gs_meta("1", "watermarked", 5))
        bad_meta["gs_protocol_mode"] = "legacy"

        rec = self._make_record(
            "1", "watermarked", method="GS", source_metadata=bad_meta)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert self._gs_factory.call_count == 0

    # ---- Secret index out of range → failed_missing_required_state ----
    def test_secret_index_out_of_range(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_REQUIRED_STATE

        self._setup_mocks(monkeypatch)
        self._gs_factory.side_effect = IndexError("list index out of range")

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 9999))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE
                      for r in rows)

    # ---- Scoring output missing → failed_scoring ----
    def test_missing_bit_accuracies_is_scoring_error(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        self._setup_mocks(monkeypatch)

        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch, prov, pipe, path, steps: {
                "message_bits_str_list": [DECODED_BITS_256]},
        )

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 5))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_SCORING_ERROR
                      for r in rows)

    def test_missing_decoded_bits_is_scoring_error(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        self._setup_mocks(monkeypatch)

        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch, prov, pipe, path, steps: {
                "bit_accuracies": [0.85],
                "message_bits_str_list": [],
            },
        )

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 5))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING

    def test_nan_bit_accuracy_is_scoring_error(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        self._setup_mocks(monkeypatch)

        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch, prov, pipe, path, steps: {
                "bit_accuracies": [float("nan")],
                "message_bits_str_list": [DECODED_BITS_256],
            },
        )

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 5))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING

    # ---- Provenance mismatch → failed_state_validation ----
    def test_secret_sha_mismatch_is_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        self._setup_mocks(monkeypatch)

        bad_meta = dict(self._gs_meta("1", "watermarked", 5))
        bad_meta["gs_message_sha256"] = "wrong_hash"

        rec = self._make_record(
            "1", "watermarked", method="GS", source_metadata=bad_meta)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION
                      for r in rows)

    def test_target_mismatch_is_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        self._setup_mocks(monkeypatch)
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "WRONG_TARGET",
        )

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=self._gs_meta("1", "watermarked", 5))

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_mask_mismatch_is_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        self._setup_mocks(monkeypatch)

        bad_meta = dict(self._gs_meta("1", "watermarked", 5))
        bad_meta["watermark_mask_sha256"] = "wrong_mask"

        rec = self._make_record(
            "1", "watermarked", method="GS", source_metadata=bad_meta)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    # ---- Missing image → FileNotFoundError → failed_missing_image ----
    def test_missing_image_is_file_not_found(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
        )

        # Full adapter mock — unit FileNotFoundError test lives elsewhere
        import raven.detectors.gs_detector as gs_mod
        monkeypatch.setattr(gs_mod, "load_state",
                           lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(gs_mod, "score_image",
                           lambda *a, **kw: {"raw_score": 0.85,
                                             "canonical_score": 0.85})

        self._mask_sentinel = canonical_json_hash(
            {"method": "GS", "mask": "not_applicable", "version": 1})
        meta = self._gs_meta("1", "watermarked", 5)

        rec = self._make_record(
            "1", "watermarked", method="GS",
            source_metadata=meta,
            input_path="/tmp/raven_issue20_definitely_missing_input.png")

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec], skip_input=True)
            result = evaluate_detector([rec], out, "GS", device="cpu")
            assert result["status"] == STATUS_FAILED_MISSING_IMAGE
            rows = self._read_detector_rows(out)
            statuses = {(r["evaluation_cohort"], r["status"]) for r in rows}
            assert ("original_watermarked",
                    ROW_STATUS_FAILED_MISSING_IMAGE) in statuses

    # ---- Non-integer secret index → state_validation (preflight) ----
    def test_invalid_secret_index_is_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        self._setup_mocks(monkeypatch)

        bad_meta = dict(self._gs_meta("1", "watermarked", 5))
        bad_meta["gs_secret_index"] = "1.5"

        rec = self._make_record(
            "1", "watermarked", method="GS", source_metadata=bad_meta)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec])
            result = evaluate_detector(
                [rec], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            assert self._gs_factory.call_count == 0


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------
class TestAggregate:
    def test_scored_rows(self):
        rows = [
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_clean", "canonical_score": 0.45},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_clean", "canonical_score": 0.47},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.92},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.78},
            {"status": ROW_STATUS_FAILED_MISSING_STATE,
             "evaluation_cohort": "attacked_clean",
             "failure_cause": FAILURE_CAUSE_MISSING_REQUIRED_STATE},
        ]
        result = aggregate(rows)
        assert result["method"] == "GS"
        assert result["scored_count"] == 4
        assert result["failed_count"] == 1
        assert "detection_summary" in result

    def test_all_provenance_fields_present(self):
        required = set(REQUIRED_METADATA_FIELDS)
        assert "gs_secret_index" in required
        assert "provider_config_hash" in required
        assert "watermark_target_sha256" in required
        assert "watermark_mask_sha256" in required
        assert "gs_detection_mode" in required

    # ---- 3. official vs empirical summary separation ----
    def _policy_row(self, cohort, success, mode="official_onebit",
                    threshold=0.9):
        return {
            "status": ROW_STATUS_SCORED,
            "evaluation_cohort": cohort,
            "canonical_score": 0.85,
            "gs_detection_mode": mode,
            "gs_active_threshold": threshold,
            "gs_active_threshold_type": "official_beta_tail_tau_onebit",
            "gs_active_comparison_operator": ">=",
            "gs_active_nominal_fpr": 1e-6,
            "gs_active_calibrated_from_current_clean_negatives": False,
            "gs_detection_success": success,
        }

    def test_official_summary_uses_provider_decisions(self):
        rows = [
            self._policy_row("original_clean", False),
            self._policy_row("original_clean", False),
            self._policy_row("original_watermarked", True),
            self._policy_row("original_watermarked", True),
            self._policy_row("attacked_watermarked", True),
            self._policy_row("attacked_watermarked", False),
        ]
        result = aggregate(rows)
        official = result["gs_official_detection_summary"]
        assert official["detection_mode"] == "official_onebit"
        assert official["threshold"] == 0.9
        assert official["threshold_type"] == "official_beta_tail_tau_onebit"
        assert official["comparison_operator"] == ">="
        assert official["nominal_fpr"] == 1e-6
        assert official["calibrated_from_current_clean_negatives"] is False
        assert official["original_clean_positive_rate"] == 0.0
        assert official["original_watermarked_detection_rate"] == 1.0
        assert official["attacked_watermarked_detection_rate"] == 0.5
        assert official["attack_success"] == 0.5

    def test_official_summary_fails_closed_on_mixed_policy(self):
        rows = [
            self._policy_row("original_watermarked", True,
                             mode="official_onebit"),
            self._policy_row("attacked_watermarked", True,
                             mode="official_traceability"),
        ]
        result = aggregate(rows)
        official = result["gs_official_detection_summary"]
        assert "error" in official
        assert official["distinct_policies"] == 2

    def test_empirical_summary_separate_from_official(self):
        """clean_calibrated_1pct_fpr_summary is distinct, clearly labeled."""
        rows = [
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_clean", "canonical_score": 0.4},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_clean", "canonical_score": 0.5},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.9},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.95},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.7},
            {"status": ROW_STATUS_SCORED,
             "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.8},
        ]
        result = aggregate(rows)
        empirical = result["clean_calibrated_1pct_fpr_summary"]
        assert empirical["target_fpr"] == 0.01
        assert empirical["threshold_source"] == "current_original_clean_cohort"
        assert empirical["calibrated_from_current_clean_negatives"] is True
        # detection_summary is the deprecated empirical alias — must NOT look
        # like an official GS policy.
        alias = result["detection_summary"]
        assert alias["target_fpr"] == 0.01
        assert "detection_mode" not in alias
        assert "gs_official_detection_summary" not in result
