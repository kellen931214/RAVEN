"""Issue #20 — canonical per-sample GS detection (comprehensive).

Tests every acceptance criterion:

1. Per-source provider cache (one provider per source sample, not per image)
2. Required metadata validation BEFORE provider_kwargs call
3. Provider configuration identity (uniform config, hash match)
4. Pipe from verified provider config (not hardcoded)
5. Metadata index prevents record cross-use
6. Secret state failure structured classification
7. Canonical scoring helpers (evaluate_image, raw_score, canonical_score)
8. Required scoring outputs no default fallback
9. Official thresholds fail closed
10. Explicit verified provenance (source/detector pairs + flags)
11. Missing image → FileNotFoundError (not DetectorMissingStateError)
12. Real evaluate_detector integration tests

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
                            bit_acc=0.85, decoded="1010"):
    inst = mock.MagicMock()
    inst.secret_provenance.return_value = _secret_provenance(secret_idx)
    inst.watermark_target_tensor.return_value = _target_tensor()
    inst.gs_protocol_mode = protocol
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
    return inst


def _resolved_metadata(run_id="1", role="watermarked", secret_index=5, **kw):
    idx = secret_index
    rec = {
        "run_id": run_id,
        "role": role,
        "gs_secret_index": str(idx),
        "gs_message_sha256": f"msg_{idx:04d}_sha256",
        "gs_key_sha256": f"key_{idx:04d}_sha256",
        "gs_nonce_sha256": f"nonce_{idx:04d}_sha256",
        "gs_secret_bundle_sha256": f"bundle_{idx:04d}_sha256",
        "gs_protocol_mode": "official_compatible",
        "watermark_target_sha256": "TGT_HASH",
        "watermark_mask_sha256": "MASK_SENTINEL",
        "provider_config_hash": "CFG_HASH",
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "",
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


def _canonical_hash_for(rec):
    from raven.eval_protocol import canonical_json_hash
    cfg = {
        "gs_protocol_mode": rec.get("gs_protocol_mode", "official_compatible"),
        "message_width_in_bytes": 32,
        "l": 1,
        "num_replications": 64,
        "gs_channel_copy": 1,
        "gs_hw_copy": 8,
        "gs_fpr": 1e-6,
        "gs_user_number": 1000000,
        "model_id": rec.get("model_id", "RedbeardNZ/stable-diffusion-2-1-base"),
        "model_revision": rec.get("model_revision") or None,
        "scheduler": rec.get("scheduler", "DDIM"),
        "resolution": int(rec.get("resolution", 512)),
        "gs_detection_mode": rec.get("gs_detection_mode", "official_onebit"),
    }
    return canonical_json_hash(cfg)


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

    def test_empty_run_id_raises(self):
        with pytest.raises(DetectorMissingStateError, match="run_id"):
            _validate_required_gs_metadata({"run_id": "  ", "role": "wm"})

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

    def test_non_integer_secret_index_raises(self):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = "abc"
        with pytest.raises(DetectorStateValidationError,
                          match="non-negative integer"):
            _validate_required_gs_metadata(rec)

    def test_negative_secret_index_raises(self):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = "-1"
        with pytest.raises(DetectorStateValidationError,
                          match="non-negative integer"):
            _validate_required_gs_metadata(rec)

    def test_float_secret_index_raises(self):
        rec = _resolved_metadata()
        rec["gs_secret_index"] = "1.5"
        with pytest.raises(DetectorStateValidationError,
                          match="non-negative integer"):
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
# 3. Provider configuration identity
# ---------------------------------------------------------------------------
class TestProviderConfigIdentity:
    def test_uniform_config_passes(self):
        recs = [
            _resolved_metadata("1", "clean"),
            _resolved_metadata("1", "watermarked"),
        ]
        cfg, h = _validate_gs_provider_config(recs)
        assert "gs_protocol_mode" in cfg
        assert "model_id" in cfg

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

    def test_mixed_pipe_config_raises(self):
        recs = [
            _resolved_metadata("1", "clean",
                              model_id="stabilityai/sd-2-1"),
            _resolved_metadata("1", "watermarked",
                              model_id="RedbeardNZ/sd-2-1-base"),
        ]
        with pytest.raises(DetectorStateValidationError,
                          match="pipe config"):
            _validate_pipe_config_uniformity(recs)


# ---------------------------------------------------------------------------
# 1. Per-source provider cache
# ---------------------------------------------------------------------------
def _configure_fake_modules(mock_pipe=None, gs_provider_cls=None):
    """Configure sys.modules mocks before load_state."""
    if mock_pipe is None:
        mock_pipe = _mock_pipe()
    _fake_pipe_utils.get_pipe_provider.return_value = mock_pipe
    if gs_provider_cls is not None:
        _fake_gs_provider_mod.GsProvider = gs_provider_cls


class TestProviderCache:
    """Provider constructed once per source, reused across cohorts."""

    def test_two_sources_two_providers(self, monkeypatch):
        GsProvider = mock.MagicMock()
        inst5 = _mock_provider_instance(5)
        inst7 = _mock_provider_instance(7)
        GsProvider.side_effect = [inst5, inst7]
        _configure_fake_modules(gs_provider_cls=GsProvider)

        meta_clean = _resolved_metadata("1", "clean", secret_index=5)
        meta_wm = _resolved_metadata("1", "watermarked", secret_index=7)

        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible"}, "CFG_HASH"),
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

        prov_info = load_state([meta_clean, meta_wm], "cpu")
        assert GsProvider.call_count == 0  # not constructed in load_state

        # First call for clean source builds a provider
        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        r1 = score_image(
            prov_info, img,
            record=meta_clean,
            evaluation_entry=_eval_entry("1", "clean", "original_clean"),
        )
        assert GsProvider.call_count == 1

        # Second call — same source, different cohort — reuses cached provider
        r2 = score_image(
            prov_info, img,
            record=meta_clean,
            evaluation_entry=_eval_entry("1", "clean", "attacked_clean"),
        )
        assert GsProvider.call_count == 1  # still 1

        # Watermarked source → new provider
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
        assert GsProvider.call_count == 2  # still 2

        assert r1["gs_secret_index"] == 5
        assert r3["gs_secret_index"] == 7

    def test_provider_cache_keys_correct(self, monkeypatch):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)

        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible"}, "CFG_HASH"),
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

        meta = _resolved_metadata("1", "watermarked")
        prov_info = load_state([meta], "cpu")
        assert ("1", "watermarked") in prov_info["metadata_index"]
        assert prov_info["provider_cache"] == {}

        img = _build_fake_png(Path(tempfile.mkdtemp()), "img.png")
        score_image(prov_info, img, record=meta,
                   evaluation_entry=_eval_entry("1", "watermarked"))
        assert ("1", "watermarked") in prov_info["provider_cache"]


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
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
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
        inst.secret_provenance.side_effect = IndexError("out of range")
        GsProvider.return_value = inst

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
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
# 7. Canonical scoring helpers used
# ---------------------------------------------------------------------------
class TestCanonicalScoringHelpers:
    """evaluate_image, raw_score, canonical_score are called."""

    def test_evaluate_image_called(self, monkeypatch, tmp_path):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)

        fake_result = {
            "bit_accuracies": [0.85],
            "message_bits_str_list": ["1010"],
        }

        called = []
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch, provider, pipe, path, steps: (
                called.append(("evaluate_image", path)), fake_result)[1],
        )
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible"}, "CFG_HASH"),
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
        result = score_image(prov_info, img, record=meta,
                            evaluation_entry=_eval_entry("1", "watermarked"))

        assert called[0] == ("evaluate_image", Path(img))
        assert result["raw_score"] == 0.85
        assert result["canonical_score"] == 0.85


# ---------------------------------------------------------------------------
# 8. Required scoring outputs — no default fallback
# ---------------------------------------------------------------------------
class TestScoringOutputValidation:
    """Missing/illegal scoring outputs → DetectorScoringError."""

    def test_missing_bit_accuracies(self):
        with pytest.raises(DetectorScoringError, match="missing or empty"):
            _validate_scoring_result({}, "1")

    def test_empty_bit_accuracies(self):
        with pytest.raises(DetectorScoringError, match="missing or empty"):
            _validate_scoring_result(
                {"bit_accuracies": [], "message_bits_str_list": ["1"]}, "1")

    def test_non_float_bit_accuracy(self):
        with pytest.raises(DetectorScoringError,
                          match="not convertible to float"):
            _validate_scoring_result(
                {"bit_accuracies": ["hello"],
                 "message_bits_str_list": ["1010"]}, "1")

    def test_nan_bit_accuracy(self):
        with pytest.raises(DetectorScoringError, match="non-finite"):
            _validate_scoring_result(
                {"bit_accuracies": [float("nan")],
                 "message_bits_str_list": ["1010"]}, "1")

    def test_out_of_range_bit_accuracy(self):
        with pytest.raises(DetectorScoringError, match="out of range"):
            _validate_scoring_result(
                {"bit_accuracies": [1.5],
                 "message_bits_str_list": ["1010"]}, "1")

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
             "message_bits_str_list": ["1010"]}, "1")


# ---------------------------------------------------------------------------
# 9. Official thresholds fail closed
# ---------------------------------------------------------------------------
class TestThresholdValidation:
    """Missing/non-finite thresholds → DetectorScoringError."""

    def test_missing_tau_onebit(self):
        with pytest.raises(DetectorScoringError, match="missing tau_onebit"):
            _validate_thresholds({"tau_bits": 0.95}, "1")

    def test_nan_tau_bits(self):
        with pytest.raises(DetectorScoringError, match="non-finite"):
            _validate_thresholds(
                {"tau_onebit": 0.9, "tau_bits": float("nan")}, "1")

    def test_non_numeric_threshold(self):
        with pytest.raises(DetectorScoringError,
                          match="not convertible"):
            _validate_thresholds(
                {"tau_onebit": "hello", "tau_bits": 0.95}, "1")

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

    def test_scored_row_has_verified_fields(self, monkeypatch, tmp_path):
        GsProvider = mock.MagicMock()
        inst = _mock_provider_instance(5)
        GsProvider.return_value = inst
        _configure_fake_modules(gs_provider_cls=GsProvider)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible"}, "CFG_HASH"),
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
        result = score_image(
            prov_info, img, record=meta,
            evaluation_entry=_eval_entry("1", "watermarked"),
        )

        # source/detector pairs
        assert result["source_watermark_target_sha256"] == "TGT_HASH"
        assert result["detector_watermark_target_sha256"] == "TGT_HASH"
        assert result["source_watermark_mask_sha256"] == "MASK_SENTINEL"
        assert result["detector_watermark_mask_sha256"] == "MASK_SENTINEL"
        assert result["source_provider_config_hash"] == "CFG_HASH"
        assert result["detector_provider_config_hash"] == "CFG_HASH"

        # verified flags
        assert result["gs_secret_verified"] is True
        assert result["gs_target_verified"] is True
        assert result["gs_mask_verified"] is True
        assert result["provider_config_verified"] is True

        # backwards-compatible merged fields
        assert result["watermark_target_sha256"] == "TGT_HASH"
        assert result["watermark_mask_sha256"] == "MASK_SENTINEL"

    def test_missing_image_raises_file_not_found(self, monkeypatch):
        GsProvider = mock.MagicMock()
        GsProvider.return_value = _mock_provider_instance(5)
        _configure_fake_modules(gs_provider_cls=GsProvider)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_pipe_config_uniformity",
            lambda r: {"model_id": "x", "model_revision": None,
                      "scheduler": "DDIM", "resolution": 512},
        )
        monkeypatch.setattr(
            "raven.detectors.gs_detector._validate_gs_provider_config",
            lambda r: ({"gs_protocol_mode": "official_compatible"}, "CFG_HASH"),
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
        with pytest.raises(FileNotFoundError):
            score_image(
                prov_info, "/nonexistent/path.png", record=meta,
                evaluation_entry=_eval_entry("1", "watermarked"),
            )


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

    def _write_fake_run(self, tmp_path, method="GS", records=None):
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
            input_path = Path(r.get("input_path", f"/tmp/in_{rid}.png"))
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

    def _setup_mocks(self, monkeypatch):
        """Mock only the heavy-resource boundaries."""
        # Configure fake pipe module
        mock_pipe = _mock_pipe()
        _configure_fake_modules(mock_pipe=mock_pipe)

        # Mock GsProvider class
        self._gs_factory = mock.MagicMock()
        self._gs_instances = {}

        def _make_inst(*args, **kwargs):
            idx = kwargs.get("gs_secret_index", 0)
            if idx not in self._gs_instances:
                self._gs_instances[idx] = _mock_provider_instance(idx)
            return self._gs_instances[idx]

        self._gs_factory.side_effect = _make_inst
        _fake_gs_provider_mod.GsProvider = self._gs_factory

        # Mock evaluate_image to return controlled results
        def _fake_eval(torch_mod, prov, pipe, path, steps):
            return {
                "bit_accuracies": [0.85],
                "message_bits_str_list": ["1010"],
            }

        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image", _fake_eval,
        )

        # Mock tensor_sha256
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TGT_HASH",
        )

        # Compute canonical mask sentinel
        from raven.eval_protocol import canonical_json_hash
        self._mask_sentinel = canonical_json_hash(
            {"method": "GS", "mask": "not_applicable", "version": 1},
        )

    def _gs_meta(self, run_id="1", role="watermarked", secret_index=5):
        idx = secret_index
        cfg_hash = _canonical_hash_for(_resolved_metadata(
            run_id, role, secret_index=idx))
        return {
            "run_id": run_id,
            "role": role,
            "gs_secret_index": str(idx),
            "gs_message_sha256": f"msg_{idx:04d}_sha256",
            "gs_key_sha256": f"key_{idx:04d}_sha256",
            "gs_nonce_sha256": f"nonce_{idx:04d}_sha256",
            "gs_secret_bundle_sha256": f"bundle_{idx:04d}_sha256",
            "gs_protocol_mode": "official_compatible",
            "watermark_target_sha256": "TGT_HASH",
            "watermark_mask_sha256": self._mask_sentinel,
            "provider_config_hash": cfg_hash,
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "scheduler": "DDIM",
            "resolution": "512",
            "gs_detection_mode": "official_onebit",
        }

    # ---- Successful per-source path ----
    def test_two_sources_four_entries_two_providers(self, monkeypatch):
        """2 sources × 2 cohorts = 4 entries, 2 provider constructions."""
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

            # 2 sources → 2 provider instances
            assert self._gs_factory.call_count == 2

            rows = self._read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
            for row in rows:
                assert row.get("gs_secret_verified") is True
                assert row.get("gs_target_verified") is True
                assert row.get("provider_config_verified") is True

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
            # Provider was never constructed
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

        # evaluate_image returns no bit_accuracies
        monkeypatch.setattr(
            extract_verification_scores, "evaluate_image",
            lambda torch, prov, pipe, path, steps: {"message_bits_str_list": ["1"]},
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
                "message_bits_str_list": ["1010"],
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
        """Preflight catches missing image → failed_missing_image stage."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_IMAGE, ROW_STATUS_FAILED_MISSING_IMAGE,
            FAILURE_CAUSE_MISSING_IMAGE,
        )
        from raven.eval_protocol import canonical_json_hash

        # Use full adapter mock — unit test for FileNotFoundError is in
        # TestVerifiedProvenance.
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
            out = Path(td) / "run"
            out.mkdir()
            from raven.experiment_io import write_config, write_record
            write_config(out, {"method": "GS", "dataset": "test"})
            role = "watermarked"
            rid = "1"
            write_record(out, role, rid, rec)
            img = out / "samples" / role / rid / "output.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"fake png")
            from raven.experiment_io import rebuild_records_jsonl
            rebuild_records_jsonl(out)

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
        bad_meta["gs_secret_index"] = "-1"

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
