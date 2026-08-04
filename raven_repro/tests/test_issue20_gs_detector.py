"""Issue #20 — canonical per-sample GS detection.

Tests that ``gs_detector.score_image`` uses the canonical
``provider_kwargs`` helper from ``extract_verification_scores.py``,
constructs one ``GsProvider`` per source sample, validates every
provenance field (secret, message, key, nonce, bundle, target, mask,
protocol) against the provider's own identity, and classifies failures
according to the structured taxonomy.

All tests use mocks only — no secret bundles, models, or datasets are
downloaded.

Run:  pytest -q raven_repro/tests/test_issue20_gs_detector.py
"""

from __future__ import annotations

import json
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

# Import extract_verification_scores early so it lands in sys.modules
# before gs_detector.score_image tries to import it dynamically.
import extract_verification_scores  # noqa: E402

from raven.detectors.gs_detector import (  # noqa: E402
    score_image,
    load_state,
    aggregate,
    REQUIRED_METADATA_FIELDS,
    describe_required_artifacts,
)
from raven.detectors import (  # noqa: E402
    DetectorMissingStateError,
    DetectorStateValidationError,
    DetectorScoringError,
    DetectorProviderInitializationError,
    ROW_STATUS_SCORED,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_SCORING,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_SCORING_ERROR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
GS_MASK_CANONICAL = (
    "b94f11d5b7e78fa6bc75f3e4693b49438d3243edd2d3b8825aadd3c7e3a7 weather"
)


def _secret_provenance(secret_index=5, **overrides):
    """Return a controlled secret_provenance dict."""
    msg = overrides.pop("message_sha256", f"msg_{secret_index:04d}_sha256")
    key = overrides.pop("key_sha256", f"key_{secret_index:04d}_sha256")
    nonce = overrides.pop("nonce_sha256", f"nonce_{secret_index:04d}_sha256")
    bundle = overrides.pop(
        "secret_bundle_sha256", f"bundle_{secret_index:04d}_sha256"
    )
    result = {
        "secret_index": secret_index,
        "message_sha256": msg,
        "key_sha256": key,
        "nonce_sha256": nonce,
        "secret_bundle_sha256": bundle,
    }
    result.update(overrides)
    return result


def _mock_target_tensor():
    """Return a tiny fake tensor for watermark_target_tensor."""
    return torch.zeros((1, 4, 8, 8), dtype=torch.uint8)


def _build_fake_png(tmp_path, name="test.png"):
    """Create a minimal valid PNG file."""
    from PIL import Image as PILImage

    img_path = tmp_path / name
    img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path, format="PNG")
    return str(img_path)


# ---------------------------------------------------------------------------
# Helpers — build mock provider_info
# ---------------------------------------------------------------------------
def _make_provider_info(mock_provider_class=None):
    """Build a mock provider_info dict with a controlled GsProvider class."""
    mock_pipe = mock.MagicMock()
    mock_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    mock_pipe.get_dtype.return_value = torch.float32

    if mock_provider_class is None:
        mock_provider_class = mock.MagicMock()

    return {
        "pipe": mock_pipe,
        "provider_class": mock_provider_class,
        "device_obj": torch.device("cpu"),
    }


def _configure_provider(mock_provider_class, secret_prov, target_tensor,
                        protocol_mode="official_compatible",
                        bit_accuracy=0.85, decoded_bits="1010"):
    """Configure a mock provider class to return controlled values."""
    mock_inst = mock.MagicMock()
    mock_inst.secret_provenance.return_value = secret_prov
    mock_inst.watermark_target_tensor.return_value = target_tensor
    mock_inst.gs_protocol_mode = protocol_mode
    mock_inst.invert_images.return_value = {"zT_torch": torch.zeros(1, 4, 64, 64)}
    mock_inst.get_accuracies.return_value = {
        "bit_accuracies": [bit_accuracy],
        "message_bits_str_list": [decoded_bits],
    }
    mock_inst.official_thresholds.return_value = {
        "tau_onebit": 0.9,
        "tau_bits": 0.95,
        "fpr": 1e-6,
        "user_number": 1000000,
        "comparison_operator": ">=",
        "source": "test",
    }
    mock_provider_class.return_value = mock_inst
    return mock_inst


def _resolved_record(**overrides):
    """Build a resolved-metadata record with all required GS fields."""
    rec = {
        "run_id": "1",
        "role": "watermarked",
        "gs_secret_index": "5",
        "gs_message_sha256": "msg_0005_sha256",
        "gs_key_sha256": "key_0005_sha256",
        "gs_nonce_sha256": "nonce_0005_sha256",
        "gs_secret_bundle_sha256": "bundle_0005_sha256",
        "gs_protocol_mode": "official_compatible",
        "watermark_target_sha256": "TARGET_A",
        "watermark_mask_sha256": "MASK_A",
    }
    rec.update(overrides)
    return rec


# ---------------------------------------------------------------------------
# Unit tests — score_image failure classification
# ---------------------------------------------------------------------------
class TestGsDetectorMissingState:
    """Missing required state → DetectorMissingStateError."""

    def test_missing_image_raises(self, tmp_path, monkeypatch):
        """Image does not exist → DetectorMissingStateError."""
        provider_info = _make_provider_info()
        record = _resolved_record()
        with pytest.raises(DetectorMissingStateError, match="Image not found"):
            score_image(provider_info, "/nonexistent/path.png",
                       record=record)

    def test_missing_record_raises(self, tmp_path, monkeypatch):
        """Record is None → DetectorMissingStateError."""
        provider_info = _make_provider_info()
        img = _build_fake_png(tmp_path)
        with pytest.raises(DetectorMissingStateError,
                          match="per-row record"):
            score_image(provider_info, img, record=None)

    def test_missing_secret_index_raises(self, tmp_path, monkeypatch):
        """No gs_secret_index in record → DetectorMissingStateError."""
        provider_info = _make_provider_info()
        img = _build_fake_png(tmp_path)
        record = {"run_id": "1", "role": "watermarked"}

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 0}
        )

        with pytest.raises(DetectorMissingStateError,
                          match="missing gs_secret_index"):
            score_image(provider_info, img, record=record)

    def test_missing_message_sha256_raises(self, tmp_path, monkeypatch):
        """Missing gs_message_sha256 → DetectorMissingStateError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(gs_message_sha256="")
        with pytest.raises(DetectorMissingStateError,
                          match="missing gs_message_sha256"):
            score_image(provider_info, img, record=record)

    def test_missing_target_sha256_raises(self, tmp_path, monkeypatch):
        """Missing watermark_target_sha256 → DetectorMissingStateError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(watermark_target_sha256="")
        with pytest.raises(DetectorMissingStateError,
                          match="missing watermark_target_sha256"):
            score_image(provider_info, img, record=record)

    def test_missing_mask_sha256_raises(self, tmp_path, monkeypatch):
        """Missing watermark_mask_sha256 → DetectorMissingStateError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(watermark_mask_sha256="")
        with pytest.raises(DetectorMissingStateError,
                          match="missing watermark_mask_sha256"):
            score_image(provider_info, img, record=record)

    def test_missing_protocol_mode_raises(self, tmp_path, monkeypatch):
        """Missing gs_protocol_mode → DetectorMissingStateError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(gs_protocol_mode="")
        with pytest.raises(DetectorMissingStateError,
                          match="missing gs_protocol_mode"):
            score_image(provider_info, img, record=record)


# ---------------------------------------------------------------------------
# Unit tests — state validation (provenance mismatches)
# ---------------------------------------------------------------------------
class TestGsDetectorStateValidation:
    """Provenance mismatches → DetectorStateValidationError."""

    def _setup(self, tmp_path, monkeypatch, **record_overrides):
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(**record_overrides)
        return provider_info, img, record

    def test_message_sha256_mismatch(self, tmp_path, monkeypatch):
        """gs_message_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            gs_message_sha256="wrong_message_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="gs_message_sha256 mismatch"):
            score_image(provider_info, img, record=record)

    def test_key_sha256_mismatch(self, tmp_path, monkeypatch):
        """gs_key_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            gs_key_sha256="wrong_key_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="gs_key_sha256 mismatch"):
            score_image(provider_info, img, record=record)

    def test_nonce_sha256_mismatch(self, tmp_path, monkeypatch):
        """gs_nonce_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            gs_nonce_sha256="wrong_nonce_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="gs_nonce_sha256 mismatch"):
            score_image(provider_info, img, record=record)

    def test_secret_bundle_sha256_mismatch(self, tmp_path, monkeypatch):
        """gs_secret_bundle_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            gs_secret_bundle_sha256="wrong_bundle_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="gs_secret_bundle_sha256 mismatch"):
            score_image(provider_info, img, record=record)

    def test_target_sha256_mismatch(self, tmp_path, monkeypatch):
        """watermark_target_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            watermark_target_sha256="wrong_target_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="target SHA mismatch"):
            score_image(provider_info, img, record=record)

    def test_mask_sha256_mismatch(self, tmp_path, monkeypatch):
        """watermark_mask_sha256 mismatch → DetectorStateValidationError."""
        provider_info, img, record = self._setup(
            tmp_path, monkeypatch,
            watermark_mask_sha256="wrong_mask_hash"
        )
        with pytest.raises(DetectorStateValidationError,
                          match="mask SHA mismatch"):
            score_image(provider_info, img, record=record)

    def test_protocol_mode_mismatch(self, tmp_path, monkeypatch):
        """gs_protocol_mode mismatch → DetectorStateValidationError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        # Provider has official_compatible, record has legacy
        _configure_provider(mock_cls, prov, target,
                           protocol_mode="official_compatible")

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record(gs_protocol_mode="legacy")
        with pytest.raises(DetectorStateValidationError,
                          match="protocol_mode mismatch"):
            score_image(provider_info, img, record=record)

    def test_secret_index_mismatch(self, tmp_path, monkeypatch):
        """Secret index mismatch in provider → DetectorStateValidationError."""
        mock_cls = mock.MagicMock()
        # Provider returns index 7, record says 5
        prov = _secret_provenance(7)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record()
        with pytest.raises(DetectorStateValidationError,
                          match="secret_index mismatch"):
            score_image(provider_info, img, record=record)


# ---------------------------------------------------------------------------
# Unit tests — per-sample provider behavior
# ---------------------------------------------------------------------------
class TestGsDetectorPerSample:
    """Two rows with different secret indices construct distinct providers."""

    def test_different_secret_indices_different_providers(self, tmp_path,
                                                         monkeypatch):
        """Row A (index=5) and Row B (index=7) → different secret_provenance."""
        calls = []

        def _capture_kwargs(method, row):
            idx = int(row.get("gs_secret_index", 0))
            calls.append({"method": method, "secret_index": idx})
            return {"offset": idx, "gs_secret_index": idx}

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs", _capture_kwargs
        )

        # Mock provider_class to return a unique instance per call
        factory = mock.MagicMock()

        def _make_provider(*args, **kwargs):
            inst = mock.MagicMock()
            idx = kwargs.get("gs_secret_index", 0)
            inst.secret_provenance.return_value = _secret_provenance(idx)
            inst.watermark_target_tensor.return_value = _mock_target_tensor()
            inst.gs_protocol_mode = "official_compatible"
            inst.invert_images.return_value = {
                "zT_torch": torch.zeros(1, 4, 64, 64)
            }
            inst.get_accuracies.return_value = {
                "bit_accuracies": [0.85],
                "message_bits_str_list": ["1010"],
            }
            inst.official_thresholds.return_value = {
                "tau_onebit": 0.9, "tau_bits": 0.95,
                "fpr": 1e-6, "user_number": 1000000,
                "comparison_operator": ">=", "source": "test",
            }
            return inst

        factory.side_effect = _make_provider
        provider_info = _make_provider_info(factory)

        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_HASH"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_HASH"
        )

        img_a = _build_fake_png(tmp_path, "a.png")
        img_b = _build_fake_png(tmp_path, "b.png")

        rec_a = _resolved_record(
            run_id="A", gs_secret_index="5",
            gs_message_sha256="msg_0005_sha256",
            gs_key_sha256="key_0005_sha256",
            gs_nonce_sha256="nonce_0005_sha256",
            gs_secret_bundle_sha256="bundle_0005_sha256",
            watermark_target_sha256="TARGET_HASH",
            watermark_mask_sha256="MASK_HASH",
        )
        rec_b = _resolved_record(
            run_id="B", gs_secret_index="7",
            gs_message_sha256="msg_0007_sha256",
            gs_key_sha256="key_0007_sha256",
            gs_nonce_sha256="nonce_0007_sha256",
            gs_secret_bundle_sha256="bundle_0007_sha256",
            watermark_target_sha256="TARGET_HASH",
            watermark_mask_sha256="MASK_HASH",
        )

        result_a = score_image(provider_info, img_a, record=rec_a)
        result_b = score_image(provider_info, img_b, record=rec_b)

        # Two provider instances constructed
        assert factory.call_count == 2

        # Each provider got the correct kwargs
        kwargs_list = [c[1] for c in factory.call_args_list]
        assert kwargs_list[0]["gs_secret_index"] == 5
        assert kwargs_list[1]["gs_secret_index"] == 7

        # Results carry the correct per-row indices
        assert result_a["gs_secret_index"] == 5
        assert result_b["gs_secret_index"] == 7

        # Canonical kwargs called with correct rows
        assert len(calls) == 2
        assert calls[0]["secret_index"] == 5
        assert calls[1]["secret_index"] == 7

    def test_canonical_kwargs_called_with_resolved_row(self, tmp_path,
                                                       monkeypatch):
        """provider_kwargs receives the resolved metadata row."""
        mock_kwargs = mock.MagicMock()
        mock_kwargs.return_value = {"offset": 5, "gs_secret_index": 5}
        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs", mock_kwargs
        )

        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target)

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record()
        score_image(provider_info, img, record=record)

        mock_kwargs.assert_called_once_with("GS", record)
        # Verify the record passed has all required fields
        passed_record = mock_kwargs.call_args[0][1]
        assert passed_record["gs_secret_index"] == "5"
        assert passed_record["gs_protocol_mode"] == "official_compatible"


# ---------------------------------------------------------------------------
# Unit tests — successful scoring path
# ---------------------------------------------------------------------------
class TestGsDetectorSuccess:
    """Valid provider → scored result with official fields."""

    def test_successful_score_returns_official_fields(self, tmp_path,
                                                      monkeypatch):
        """All provenance matches → scored dict with all canonical fields."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        _configure_provider(mock_cls, prov, target, bit_accuracy=0.92,
                           decoded_bits="1100")

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record()
        result = score_image(provider_info, img, record=record)

        # Required threshold-detector fields
        assert result["raw_score"] == 0.92
        assert result["canonical_score"] == 0.92

        # GS-specific canonical fields
        assert result["bit_accuracy"] == 0.92
        assert "decoded_bits_sha256" in result
        assert result["gs_secret_index"] == 5
        assert result["gs_secret_bundle_sha256"] == "bundle_0005_sha256"
        assert result["gs_message_sha256"] == "msg_0005_sha256"
        assert result["gs_key_sha256"] == "key_0005_sha256"
        assert result["gs_nonce_sha256"] == "nonce_0005_sha256"
        assert result["gs_protocol_mode"] == "official_compatible"
        assert result["watermark_target_sha256"] == "TARGET_A"
        assert result["watermark_mask_sha256"] == "MASK_A"

        # Official thresholds preserved
        assert result["gs_official_tau_onebit"] == 0.9
        assert result["gs_official_tau_bits"] == 0.95

        # Score direction
        assert result["score_direction"] == "higher_is_watermarked"

    def test_scoring_failure_raises(self, tmp_path, monkeypatch):
        """Runtime inversion failure → DetectorScoringError."""
        mock_cls = mock.MagicMock()
        prov = _secret_provenance(5)
        target = _mock_target_tensor()
        inst = _configure_provider(mock_cls, prov, target)
        inst.invert_images.side_effect = RuntimeError("inversion exploded")

        provider_info = _make_provider_info(mock_cls)
        img = _build_fake_png(tmp_path)

        monkeypatch.setattr(
            extract_verification_scores, "provider_kwargs",
            lambda method, row: {"offset": 5, "gs_secret_index": 5}
        )
        monkeypatch.setattr(
            "raven.pairing_provenance.tensor_sha256",
            lambda t: "TARGET_A"
        )
        monkeypatch.setattr(
            "raven.eval_protocol.canonical_json_hash",
            lambda p: "MASK_A"
        )

        record = _resolved_record()
        with pytest.raises(DetectorScoringError,
                          match="inversion exploded"):
            score_image(provider_info, img, record=record)


# ---------------------------------------------------------------------------
# Unit tests — REQUIRED_METADATA_FIELDS completeness
# ---------------------------------------------------------------------------
class TestRequiredMetadataFields:
    """REQUIRED_METADATA_FIELDS covers all provenance fields."""

    def test_all_provenance_fields_present(self):
        """Every field validated in score_image is declared as required."""
        required = set(REQUIRED_METADATA_FIELDS)
        expected = {
            "gs_secret_index",
            "gs_message_sha256",
            "gs_key_sha256",
            "gs_nonce_sha256",
            "gs_secret_bundle_sha256",
            "gs_protocol_mode",
            "watermark_target_sha256",
            "watermark_mask_sha256",
        }
        assert required == expected

    def test_describe_required_artifacts_covers_all(self):
        """describe_required_artifacts mentions all provenance categories."""
        artifacts = describe_required_artifacts()
        text = " ".join(artifacts).lower()
        for token in ("secret", "message", "key", "nonce", "bundle",
                      "protocol", "target", "mask", "pipe"):
            assert token in text, f"Missing artifact mention: {token}"


# ---------------------------------------------------------------------------
# Unit tests — aggregate
# ---------------------------------------------------------------------------
class TestGsDetectorAggregate:
    """Aggregate function with mocked rows."""

    def test_aggregate_scored_rows(self):
        """Aggregate counts scored rows and builds detection summary."""
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 0.45},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 0.47},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.92},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.94},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.78},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.81},
            {"status": ROW_STATUS_FAILED_MISSING_STATE,
             "evaluation_cohort": "attacked_clean",
             "failure_cause": FAILURE_CAUSE_MISSING_REQUIRED_STATE},
            {"status": ROW_STATUS_FAILED_SCORING,
             "evaluation_cohort": "attacked_clean",
             "failure_cause": FAILURE_CAUSE_SCORING_ERROR},
        ]
        result = aggregate(rows)
        assert result["method"] == "GS"
        assert result["scored_count"] == 6
        assert result["failed_count"] == 2
        assert result["score_direction"] == "higher_is_watermarked"
        assert "detection_summary" in result
        assert result["cohort_counts"]["original_clean"] == 2
        assert result["cohort_counts"]["original_watermarked"] == 2
        assert result["cohort_counts"]["attacked_watermarked"] == 2

    def test_aggregate_missing_cohorts(self):
        """Missing required cohorts are listed."""
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 0.45},
        ]
        result = aggregate(rows)
        assert "original_watermarked" in result["missing_cohorts"]
        assert "attacked_watermarked" in result["missing_cohorts"]


# ---------------------------------------------------------------------------
# Integration tests — evaluate_detector with mocked adapter
# ---------------------------------------------------------------------------
class TestEvaluateDetectorGS:
    """Full evaluate_detector path with mocked gs_detector adapter."""

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

    GS_META = {
        "gs_secret_index": "5",
        "gs_message_sha256": "msg_hash",
        "gs_key_sha256": "key_hash",
        "gs_nonce_sha256": "nonce_hash",
        "gs_secret_bundle_sha256": "bundle_hash",
        "gs_protocol_mode": "official_compatible",
        "watermark_target_sha256": "target_hash",
        "watermark_mask_sha256": "mask_hash",
    }

    @staticmethod
    def _patch_gs(monkeypatch, fake_score_fn):
        import raven.detectors.gs_detector as mod
        monkeypatch.setattr(mod, "load_state",
                           lambda records, device, **extra: {"fake": True})
        monkeypatch.setattr(mod, "score_image", fake_score_fn)

    def test_complete_scored_path_writes_detector_records(self, monkeypatch):
        """evaluate_detector writes detector_records.jsonl with scored rows."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED, ROW_STATUS_SCORED

        rec_clean = self._make_record(
            "1", "clean", method="GS", source_metadata=self.GS_META)
        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        self._patch_gs(monkeypatch, lambda *a, **kw: {
            "raw_score": 0.85, "canonical_score": 0.85,
        })

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["scored_count"] == 4  # clean×2 + wm×2
            assert result["failed_count"] == 0

            rows = self._read_detector_rows(out)
            assert len(rows) == 4
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)

    def test_missing_state_in_optional_cohort(self, monkeypatch):
        """Missing state in attacked_clean → soft failure, primary OK."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_COMPLETED, DetectorMissingStateError,
        )

        rec_clean = self._make_record(
            "1", "clean", method="GS", source_metadata=self.GS_META)
        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        def fake_score(provider_info, image_path, *,
                      record=None, evaluation_entry=None, steps=50):
            if (evaluation_entry is not None
                    and evaluation_entry.get("evaluation_cohort")
                    == "attacked_clean"):
                raise DetectorMissingStateError(
                    "optional state missing")
            return {"raw_score": 0.85, "canonical_score": 0.85}

        self._patch_gs(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            assert result["primary_scored_count"] == 3
            assert result["primary_failed_count"] == 0
            assert result["optional_failed_count"] == 1
            assert result.get("optional_metrics_incomplete") is True

            rows = self._read_detector_rows(out)
            statuses = {(r["evaluation_cohort"], r["status"]) for r in rows}
            assert ("original_clean", "scored") in statuses
            assert ("original_watermarked", "scored") in statuses
            assert ("attacked_watermarked", "scored") in statuses
            assert ("attacked_clean",
                    ROW_STATUS_FAILED_MISSING_STATE) in statuses

    def test_state_validation_failure_is_hard(self, monkeypatch):
        """DetectorStateValidationError in any cohort → hard failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_STATE_VALIDATION, DetectorStateValidationError,
        )

        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        def fake_score(*a, **kw):
            raise DetectorStateValidationError("target SHA mismatch")

        self._patch_gs(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_STATE_VALIDATION
            rows = self._read_detector_rows(out)
            assert all(r["status"] == ROW_STATUS_FAILED_STATE_VALIDATION
                      for r in rows)
            assert all(r["failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION
                      for r in rows)

    def test_scoring_failure_is_hard(self, monkeypatch):
        """DetectorScoringError → hard failure."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, DetectorScoringError,
        )

        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        def fake_score(*a, **kw):
            raise DetectorScoringError("inversion failed")

        self._patch_gs(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_SCORING
            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_SCORING_ERROR
                      for r in rows)

    def test_missing_state_eligible_for_allow_missing_metrics(self, monkeypatch):
        """DetectorMissingStateError → allowable under --allow-missing-metrics."""
        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_MISSING_REQUIRED_STATE, DetectorMissingStateError,
        )

        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        def fake_score(*a, **kw):
            raise DetectorMissingStateError("secret not available")

        self._patch_gs(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_wm])
            result = evaluate_detector(
                [rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
            rows = self._read_detector_rows(out)
            assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE
                      for r in rows)
            assert all(r["status"] == ROW_STATUS_FAILED_MISSING_STATE
                      for r in rows)

    def test_scored_rows_carry_gs_fields(self, monkeypatch):
        """Scored rows in detector_records.jsonl contain all GS provenance."""
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        rec_clean = self._make_record(
            "1", "clean", method="GS", source_metadata=self.GS_META)
        rec_wm = self._make_record(
            "1", "watermarked", method="GS", source_metadata=self.GS_META)

        def fake_score(*a, **kw):
            return {
                "raw_score": 0.85,
                "canonical_score": 0.85,
                "bit_accuracy": 0.85,
                "decoded_bits_sha256": "abcdef",
                "gs_secret_index": 5,
                "gs_secret_bundle_sha256": "bundle_hash",
                "gs_message_sha256": "msg_hash",
                "gs_key_sha256": "key_hash",
                "gs_nonce_sha256": "nonce_hash",
                "gs_protocol_mode": "official_compatible",
                "watermark_target_sha256": "target_hash",
                "watermark_mask_sha256": "mask_hash",
                "gs_official_tau_onebit": 0.9,
                "gs_official_tau_bits": 0.95,
                "score_direction": "higher_is_watermarked",
            }

        self._patch_gs(monkeypatch, fake_score)

        with tempfile.TemporaryDirectory() as td:
            out = self._write_fake_run(
                Path(td), method="GS", records=[rec_clean, rec_wm])
            result = evaluate_detector(
                [rec_clean, rec_wm], out, "GS", device="cpu")

            assert result["status"] == STATUS_COMPLETED
            rows = self._read_detector_rows(out)
            for row in rows:
                if row["status"] == ROW_STATUS_SCORED:
                    assert "gs_secret_index" in row
                    assert "gs_protocol_mode" in row
                    assert "gs_official_tau_onebit" in row
                    assert "gs_official_tau_bits" in row
                    assert "decoded_bits_sha256" in row
                    assert "watermark_target_sha256" in row
                    assert "watermark_mask_sha256" in row
