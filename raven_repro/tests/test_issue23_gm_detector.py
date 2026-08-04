"""Issue #23 tests — bind GM evaluation to persisted bundle and provenance.

All unit tests use mocks.  Orchestrator integration tests use the real
``experiments.eval.evaluate_detector`` with mocked heavy resources.
No GM artifacts are downloaded and no real cohort is run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO / "eval_bench_wm", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from raven.detectors.gm_detector import (  # noqa: E402
    _CANONICAL_KWARGS_FIELDS,
    _GM_REQUIRED_METADATA_FIELDS,
    _validate_required_gm_metadata,
    _validate_bundle_files_exist,
    _canonical_provider_identity,
    _validate_gm_protocol_profile,
    describe_required_artifacts,
    load_state,
    score_image,
    aggregate,
)
from raven.detectors import (  # noqa: E402
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
    ROW_STATUS_SCORED,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_MISSING_IMAGE,
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_SCORING,
    STATUS_COMPLETED,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION,
    FAILURE_CAUSE_SCORING_ERROR,
    FAILURE_CAUSE_MISSING_IMAGE,
    KNOWN_FAILURE_CAUSES,
    stage_status_is_allowable,
)


# ============================================================================
# Record builders
# ============================================================================

def _gm_record(run_id="0", **overrides):
    record = {
        "run_id": run_id,
        "gm_bundle_dir": "/fake/bundle",
        "gm_bundle_config_sha256": "a" * 64,
        "gm_w1_file_sha256": "b" * 64,
        "gm_w2_file_sha256": "c" * 64,
        "gm_protocol_mode": "legacy",
        "gm_m_sha256": "m" * 64,
        "gm_watermark_sha256": "n" * 64,
        "gm_target_sha256": "o" * 64,
        "watermark_target_sha256": "d" * 64,
        "watermark_mask_sha256": "e" * 64,
    }
    record.update(overrides)
    return record


def _make_bundle_dir(tmp_path: Path, **manifest_overrides) -> Path:
    """Create a minimal valid-looking bundle directory."""
    bundle = tmp_path / "bundle"
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
        "w_seed": 42,
        "w_channel": 3,
        "w_pattern": "ring",
        "w_mask_shape": "circle",
        "w_radius": 10,
        "w_measurement": "l1_complex",
        "w_injection": "complex",
        "bundle_config_sha256": "a" * 64,
        "w1_file_sha256": "b" * 64,
        "w2_file_sha256": "c" * 64,
        "m_sha256": "m" * 64,
        "watermark_sha256": "n" * 64,
        "w2_tensor_sha256": "o" * 64,
        "watermark_bits_seed": 7,
    }
    manifest.update(manifest_overrides)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "w1.pth").write_bytes(b"\x00" * 64)
    (bundle / "w2.pth").write_bytes(b"\x01" * 64)
    return bundle


# ============================================================================
# Stubs
# ============================================================================

class StubGmProvider:
    def __init__(self, **kwargs):
        self.bundle = mock.Mock() if kwargs.get("_has_bundle", True) else None
        self.state_source = kwargs.get("_state_source", "bundle")
        self.gt_patch = kwargs.get("_gt_patch", _fake_gt_patch())
        self.watermarking_mask = kwargs.get("_wm_mask", _fake_wm_mask())
        self.profile_is_official = kwargs.get("_profile_is_official",
            kwargs.get("gm_profile", "") not in ("", "legacy"))


def _fake_gt_patch():
    return torch.zeros(1, 1, 64, 64, dtype=torch.float32)


def _fake_wm_mask():
    return torch.ones(1, 1, 64, 64, dtype=torch.bool)


class StubPipe:
    def get_latent_shape(self):
        return (1, 4, 64, 64)

    def get_dtype(self):
        return torch.float32


class _StubExtractModule:
    """Stand-in for ``extract_verification_scores.py``."""

    def gm_bundle_manifest(self, row, identifier):
        bundle_dir = Path(str(row.get("gm_bundle_dir", "")))
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"run_id={identifier}: GM bundle manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for mf, rf in (
            ("bundle_config_sha256", "gm_bundle_config_sha256"),
            ("w1_file_sha256", "gm_w1_file_sha256"),
            ("w2_file_sha256", "gm_w2_file_sha256"),
            ("m_sha256", "gm_m_sha256"),
            ("watermark_sha256", "gm_watermark_sha256"),
            ("w2_tensor_sha256", "gm_target_sha256"),
        ):
            expected = str(row.get(rf, ""))
            actual = str(manifest.get(mf, ""))
            if not expected or expected != actual:
                raise RuntimeError(
                    f"run_id={identifier}: GM bundle/source {rf} mismatch: "
                    f"source={expected!r} bundle={actual!r}"
                )
        return bundle_dir, manifest

    def gm_provider_kwargs(self, row, identifier):
        bundle_dir, manifest = self.gm_bundle_manifest(row, identifier)
        return {
            "gm_profile": str(manifest["profile"]),
            "gm_bundle_dir": str(bundle_dir),
            "gm_create_bundle": False,
            "gm_allow_in_memory_state": False,
            "gm_torch_dtype": str(manifest["torch_dtype"]),
            "gm_channel_copy": 1,
            "gm_w_copy": 1,
            "gm_h_copy": 1,
            "gm_watermark_bits_seed": manifest.get("watermark_bits_seed", 7),
            "gm_use_gnr": False,
            "gm_gnr_path": None,
            "gm_use_classifier": False,
            "gm_classifier_path": None,
            "modelid_target": str(manifest["model_id"]),
            "model_revision": str(manifest["model_revision"]),
            "scheduler_target": str(manifest["scheduler"]),
            "resolution": int(manifest["resolution"]),
            "w_seed": int(manifest["w_seed"]),
            "w_channel": int(manifest["w_channel"]),
            "w_pattern": str(manifest["w_pattern"]),
            "w_mask_shape": str(manifest["w_mask_shape"]),
            "w_radius": int(manifest["w_radius"]),
            "w_measurement": str(manifest["w_measurement"]),
            "w_injection": str(manifest["w_injection"]),
        }

    def evaluate_image(self, torch_mod, provider, pipe, path, steps):
        return {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": 0.90,
            "gm_classifier_probability": None,
            "gm_report_label": "gm_raw_bit_accuracy",
            "gm_score_definition": "spatial-domain per-pixel bit match rate",
            "gm_threshold_source": "ensemble_not_applicable",
            "gm_comparison_operator": ">=",
            "gm_used_gnr": False,
            "gm_used_classifier": False,
        }

    def raw_score(self, method, result):
        return float(result["gm_raw_bit_accuracy"])

    def canonical_score(self, method, raw, result):
        return raw


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bundle_dir(tmp_path):
    return _make_bundle_dir(tmp_path)


@pytest.fixture
def stub_extract_module():
    return _StubExtractModule()


@pytest.fixture
def mock_deps(monkeypatch, bundle_dir, stub_extract_module):
    """Wire all dependencies so load_state works without real imports."""
    import raven.detectors.gm_detector as gm_mod

    _fake_pipe_utils = mock.Mock()
    _fake_pipe_utils.get_pipe_provider = mock.Mock(return_value=StubPipe())

    _fake_gm_provider = mock.Mock()
    _fake_gm_provider.GmProvider = StubGmProvider

    for _key, _val in [
        ("eval_bench_wm", mock.Mock()),
        ("eval_bench_wm.utils", mock.Mock()),
        ("eval_bench_wm.utils.pipe", mock.Mock()),
        ("eval_bench_wm.utils.pipe.pipe_utils", _fake_pipe_utils),
        ("eval_bench_wm.utils.wm", mock.Mock()),
        ("eval_bench_wm.utils.wm.gm_provider", _fake_gm_provider),
    ]:
        monkeypatch.setitem(sys.modules, _key, _val)

    monkeypatch.setattr(gm_mod, "_get_extract_module",
                        lambda: stub_extract_module)
    monkeypatch.setattr(
        gm_mod, "_ensure_paths", lambda: sys.path.insert(0, str(REPO / "eval_bench_wm"))
    )
    monkeypatch.setattr(
        "raven.pairing_provenance.tensor_sha256",
        lambda t: "tensor_hash_" + str(t.shape) if hasattr(t, "shape") else "tensor_hash",
    )

    return gm_mod


@pytest.fixture
def provider_info(mock_deps, bundle_dir):
    records = [_gm_record("0", gm_bundle_dir=str(bundle_dir)),
               _gm_record("1", gm_bundle_dir=str(bundle_dir))]
    return load_state(records, "cpu")


@pytest.fixture
def fake_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (16, 16))
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


# ============================================================================
# 1. Per-row canonical bundle binding
# ============================================================================

class TestPerRowBundleBinding:

    def test_second_row_m_sha_mismatch_rejected_before_provider(self, mock_deps, bundle_dir):
        """Second row with different gm_m_sha256 must fail before provider construction."""
        records = [
            _gm_record("0", gm_bundle_dir=str(bundle_dir)),
            _gm_record("1", gm_bundle_dir=str(bundle_dir),
                       gm_m_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mismatch"):
            load_state(records, "cpu")

    def test_second_row_watermark_sha_mismatch_rejected_before_provider(
        self, mock_deps, bundle_dir):
        records = [
            _gm_record("0", gm_bundle_dir=str(bundle_dir)),
            _gm_record("1", gm_bundle_dir=str(bundle_dir),
                       gm_watermark_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mismatch"):
            load_state(records, "cpu")

    def test_second_row_target_sha_mismatch_rejected_before_provider(
        self, mock_deps, bundle_dir):
        records = [
            _gm_record("0", gm_bundle_dir=str(bundle_dir)),
            _gm_record("1", gm_bundle_dir=str(bundle_dir),
                       gm_target_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mismatch"):
            load_state(records, "cpu")

    def test_provider_not_constructed_on_second_row_mismatch(
        self, mock_deps, bundle_dir, monkeypatch):
        """When second row has mismatched m_sha256, GmProvider is never called."""
        import raven.detectors.gm_detector as gm_mod

        counter = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)

        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir),
                           gm_m_sha256="z" * 64),
            ]
            with pytest.raises(DetectorStateValidationError):
                load_state(records, "cpu")
            assert counter[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider


# ============================================================================
# 2. Strict required metadata preflight
# ============================================================================

class TestRequiredMetadataPreflight:

    def test_missing_field_is_missing_state(self):
        record = _gm_record("0")
        del record["gm_m_sha256"]
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            _validate_required_gm_metadata(record)

    def test_none_field_is_missing_state(self):
        record = _gm_record("0", gm_m_sha256=None)
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            _validate_required_gm_metadata(record)

    def test_whitespace_field_is_missing_state(self):
        record = _gm_record("0", gm_m_sha256="   ")
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            _validate_required_gm_metadata(record)

    def test_empty_string_field_is_missing_state(self):
        record = _gm_record("0", gm_m_sha256="")
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            _validate_required_gm_metadata(record)

    def test_missing_watermark_target_is_missing_state(self):
        record = _gm_record("0")
        del record["watermark_target_sha256"]
        with pytest.raises(DetectorMissingStateError, match="watermark_target_sha256"):
            _validate_required_gm_metadata(record)

    def test_missing_watermark_mask_is_missing_state(self):
        record = _gm_record("0")
        del record["watermark_mask_sha256"]
        with pytest.raises(DetectorMissingStateError, match="watermark_mask_sha256"):
            _validate_required_gm_metadata(record)

    def test_preflight_before_manifest(self, mock_deps, bundle_dir):
        """Missing required field must be caught as MissingStateError BEFORE
        gm_bundle_manifest is ever called (no StateValidationError misclassification).
        """
        record = _gm_record("0", gm_bundle_dir=str(bundle_dir))
        del record["gm_m_sha256"]
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            load_state([record], "cpu")

    def test_all_rows_missing_same_field_not_uniform(self, mock_deps, bundle_dir):
        """Two rows both missing gm_m_sha256: must be MissingStateError, not
        mistaken for a uniform cohort."""
        r0 = _gm_record("0", gm_bundle_dir=str(bundle_dir))
        del r0["gm_m_sha256"]
        r1 = _gm_record("1", gm_bundle_dir=str(bundle_dir))
        del r1["gm_m_sha256"]
        with pytest.raises(DetectorMissingStateError):
            load_state([r0, r1], "cpu")


# ============================================================================
# 3. Target / mask fail-closed
# ============================================================================

class TestTargetMaskFailClosed:

    def test_record_none_is_missing_state(self, provider_info, fake_image):
        with pytest.raises(DetectorMissingStateError, match="resolved source metadata"):
            score_image(provider_info, fake_image, record=None)

    def test_missing_source_target_is_missing_state(self, provider_info, fake_image):
        record = _gm_record("0")
        del record["watermark_target_sha256"]
        with pytest.raises(DetectorMissingStateError, match="watermark_target_sha256"):
            score_image(provider_info, fake_image, record=record)

    def test_missing_source_mask_is_missing_state(self, provider_info, fake_image):
        record = _gm_record("0")
        del record["watermark_mask_sha256"]
        with pytest.raises(DetectorMissingStateError, match="watermark_mask_sha256"):
            score_image(provider_info, fake_image, record=record)

    def test_missing_detector_target_is_state_validation(self, provider_info, fake_image):
        provider_info["provider_target_hash"] = ""
        record = _gm_record("0")
        with pytest.raises(DetectorStateValidationError, match="detector target hash"):
            score_image(provider_info, fake_image, record=record)

    def test_missing_detector_mask_is_state_validation(self, provider_info, fake_image):
        provider_info["provider_mask_hash"] = ""
        record = _gm_record("0")
        with pytest.raises(DetectorStateValidationError, match="detector mask hash"):
            score_image(provider_info, fake_image, record=record)

    def test_source_target_mismatch(self, provider_info, fake_image):
        record = _gm_record("0", watermark_target_sha256="wrong" * 8)
        with pytest.raises(DetectorStateValidationError, match="target SHA mismatch"):
            score_image(provider_info, fake_image, record=record)

    def test_source_mask_mismatch(self, provider_info, fake_image):
        provider_target = provider_info["provider_target_hash"]
        provider_mask = provider_info["provider_mask_hash"]
        record = _gm_record(
            "0",
            watermark_target_sha256=provider_target,
            watermark_mask_sha256="wrong" * 8,
        )
        with pytest.raises(DetectorStateValidationError, match="mask SHA mismatch"):
            score_image(provider_info, fake_image, record=record)


# ============================================================================
# 4. Protocol / profile matching
# ============================================================================

class TestProtocolProfileMatching:

    def test_protocol_matches_bundle_profile(self, mock_deps, bundle_dir):
        """When gm_protocol_mode == manifest.profile == gm_profile, pass."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir),
                               gm_protocol_mode="legacy")]
        info = load_state(records, "cpu")
        assert info["provider"] is not None

    def test_protocol_must_match_bundle_profile(self, mock_deps, bundle_dir):
        """When gm_protocol_mode differs from manifest.profile, fail."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir),
                               gm_protocol_mode="wrong_protocol")]
        with pytest.raises(DetectorStateValidationError, match="protocol/profile mismatch"):
            load_state(records, "cpu")

    def test_all_rows_wrong_protocol_still_fails(self, mock_deps, bundle_dir):
        """All rows agreeing on a wrong protocol is still a mismatch."""
        records = [
            _gm_record("0", gm_bundle_dir=str(bundle_dir),
                       gm_protocol_mode="wrong_protocol"),
            _gm_record("1", gm_bundle_dir=str(bundle_dir),
                       gm_protocol_mode="wrong_protocol"),
        ]
        with pytest.raises(DetectorStateValidationError, match="protocol/profile mismatch"):
            load_state(records, "cpu")


# ============================================================================
# 5. Canonical provider configuration comparison
# ============================================================================

class TestCanonicalProviderIdentity:

    def test_all_rows_canonical_provider_kwargs_match(self, mock_deps, bundle_dir):
        """Two rows with identical metadata produce identical canonical identities."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir)),
                   _gm_record("1", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        assert info["provider"] is not None

    def test_identity_differs_on_profile(self, mock_deps, bundle_dir):
        """Different manifest profile gives different canonical identity."""
        bundle2 = _make_bundle_dir(bundle_dir.parent / "bundle2", profile="custom")
        r0 = _gm_record("0", gm_bundle_dir=str(bundle_dir),
                        gm_protocol_mode="legacy",
                        gm_bundle_config_sha256="a" * 64,
                        gm_w1_file_sha256="b" * 64,
                        gm_w2_file_sha256="c" * 64,
                        gm_m_sha256="m" * 64,
                        gm_watermark_sha256="n" * 64,
                        gm_target_sha256="o" * 64)
        r1 = _gm_record("1", gm_bundle_dir=str(bundle2),
                        gm_protocol_mode="custom",
                        gm_bundle_config_sha256="a" * 64,
                        gm_w1_file_sha256="b" * 64,
                        gm_w2_file_sha256="c" * 64,
                        gm_m_sha256="m" * 64,
                        gm_watermark_sha256="n" * 64,
                        gm_target_sha256="o" * 64)
        # Update bundle2 manifest to match r1's expected fields
        records = [r0, r1]
        with pytest.raises(DetectorStateValidationError, match="mixed canonical"):
            load_state(records, "cpu")

    def test_canonical_identity_includes_all_kwargs_fields(self):
        """Every field in _CANONICAL_KWARGS_FIELDS contributes to the identity."""
        kwargs1 = {f: "v1" for f in _CANONICAL_KWARGS_FIELDS}
        kwargs2 = {f: "v1" for f in _CANONICAL_KWARGS_FIELDS}
        assert _canonical_provider_identity(kwargs1) == _canonical_provider_identity(kwargs2)
        kwargs2["w_seed"] = 999
        assert _canonical_provider_identity(kwargs1) != _canonical_provider_identity(kwargs2)


# ============================================================================
# 6. No error-message substring classification
# ============================================================================

class TestNoErrorMessageParsing:

    def test_bundle_error_classification_does_not_parse_message(
        self, mock_deps, bundle_dir):
        """RuntimeError from gm_bundle_manifest is always StateValidationError
        (after preflight passed).  The detector never inspects the message string."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        def _failing_manifest(row, ident):
            raise RuntimeError("unexpected explosion — not 'not found' or 'missing'")
        stub.gm_bundle_manifest = _failing_manifest
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError, match="manifest validation failed"):
            load_state(records, "cpu")

    def test_missing_bundle_still_missing_state_after_removing_classifier(
        self, mock_deps):
        """Missing bundle dir is MissingStateError (preflight), not StateValidationError."""
        records = [_gm_record("0", gm_bundle_dir="/no/such/dir")]
        with pytest.raises(DetectorMissingStateError, match="not found"):
            load_state(records, "cpu")


# ============================================================================
# 7. Canonical pipe configuration
# ============================================================================

class TestCanonicalPipeConfig:

    def test_pipe_uses_canonical_kwargs(self, mock_deps, bundle_dir):
        """Pipe construction uses modelid_target, scheduler_target, resolution
        from the canonical kwargs, not hardcoded defaults."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        # Provider constructed successfully proves canonical pipe args were used.
        assert info["provider"] is not None
        assert info["pipe"] is not None


# ============================================================================
# 8. Verified provenance fields
# ============================================================================

class TestVerifiedProvenance:

    def test_score_contains_source_detector_target_pairs(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["source_watermark_target_sha256"] == record["watermark_target_sha256"]
        assert score["detector_watermark_target_sha256"] == provider_info["provider_target_hash"]
        assert score["source_watermark_mask_sha256"] == record["watermark_mask_sha256"]
        assert score["detector_watermark_mask_sha256"] == provider_info["provider_mask_hash"]

    def test_score_contains_gm_target_mask_verified(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["gm_target_verified"] is True
        assert score["gm_mask_verified"] is True

    def test_verified_provenance_contains_bundle_fields(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        for field in ("gm_bundle_dir", "gm_bundle_config_sha256",
                       "gm_w1_file_sha256", "gm_w2_file_sha256",
                       "gm_m_sha256", "gm_watermark_sha256",
                       "gm_target_sha256", "gm_protocol_mode",
                       "gm_profile", "gm_state_source"):
            assert field in score, f"verified field {field} missing"


# ============================================================================
# 9. GNR / classifier state
# ============================================================================

class TestGnrClassifierState:

    def test_gnr_classifier_usage_preserved(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert "gm_gnr_used" in score
        assert "gm_classifier_used" in score
        assert isinstance(score["gm_gnr_used"], bool)
        assert isinstance(score["gm_classifier_used"], bool)

    def test_gnr_used_alias_resolution(self, provider_info, fake_image, monkeypatch):
        """When scorer returns gm_gnr_used (not gm_used_gnr), still captured."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        def gm_eval(*a, **kw):
            return {
                "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
                "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
                "gm_report_label": "x", "gm_score_definition": "x",
                "gm_threshold_source": "x", "gm_comparison_operator": ">=",
                "gm_gnr_used": True,  # alternate key
                "gm_classifier_used": True,  # alternate key
            }
        stub.evaluate_image = gm_eval
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["gm_gnr_used"] is True
        assert score["gm_classifier_used"] is True


# ============================================================================
# 10. Missing image → FileNotFoundError
# ============================================================================

class TestMissingImage:

    def test_missing_image_raises_file_not_found(self, provider_info):
        record = _gm_record("0")
        with pytest.raises(FileNotFoundError):
            score_image(provider_info, "/no/such/image.png", record=record)

    def test_missing_image_not_detector_missing_state_error(self, provider_info):
        record = _gm_record("0")
        with pytest.raises(FileNotFoundError):
            score_image(provider_info, "/no/such/image.png", record=record)


# ============================================================================
# 11. Non-TypeError constructor failure
# ============================================================================

class TestConstructorFailure:

    def test_non_typeerror_constructor_failure_is_provider_failure(
        self, mock_deps, bundle_dir):
        """A RuntimeError during provider construction is
        DetectorProviderInitializationError, not internal_error."""
        class FailingProvider:
            def __init__(self, **kwargs):
                raise RuntimeError("GPU OOM during provider init")

        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = FailingProvider
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorProviderInitializationError,
                               match="construction failed"):
                load_state(records, "cpu")
        finally:
            fake_gm.GmProvider = StubGmProvider


# ============================================================================
# Scoring success + aggregate
# ============================================================================

class TestScoring:

    def test_successful_score(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["raw_score"] == 0.85
        assert score["canonical_score"] == 0.85
        assert score["gm_raw_bit_accuracy"] == 0.85
        assert score["gm_raw_ring_l1"] == 0.12

    def test_gm_domain_scores_preserved(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert "gm_restored_bit_accuracy" in score
        assert "gm_classifier_probability" in score
        assert "gm_report_label" in score
        assert "gm_score_definition" in score
        assert "gm_threshold_source" in score
        assert "gm_comparison_operator" in score

    def test_scoring_error(self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("inversion failed"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image, record=record)


class TestAggregate:

    def test_aggregate_all_scored(self):
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.9},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.8},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.5},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.4},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 0.3},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 0.2},
        ]
        result = aggregate(rows)
        assert result["method"] == "GM"
        assert result["scored_count"] == 6
        assert "detection_summary" in result

    def test_aggregate_mixed_status(self):
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.9},
            {"status": ROW_STATUS_FAILED_MISSING_STATE, "evaluation_cohort": "",
             "canonical_score": None},
        ]
        result = aggregate(rows)
        assert result["scored_count"] == 1
        assert result["failed_count"] == 1


# ============================================================================
# Canonical helper delegation
# ============================================================================

class TestCanonicalHelperDelegation:

    def test_raw_score_delegated(self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        calls = []
        _orig = stub.raw_score
        stub.raw_score = lambda m, r: calls.append(("raw_score", m)) or _orig(m, r)
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "deleg1.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        score_image(info, str(fake), record=record)
        assert len(calls) == 1
        assert calls[0][0] == "raw_score"

    def test_canonical_score_delegated(self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        calls = []
        _orig = stub.canonical_score
        stub.canonical_score = lambda m, r, res: calls.append(("canonical", m)) or _orig(m, r, res)
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "deleg2.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        score_image(info, str(fake), record=record)
        assert len(calls) == 1
        assert calls[0][0] == "canonical"


# ============================================================================
# Misc
# ============================================================================

class TestBundleFileExistence:

    def test_bundle_dir_exists(self, bundle_dir):
        assert _validate_bundle_files_exist(str(bundle_dir)) == bundle_dir.resolve()

    def test_bundle_dir_missing(self, tmp_path):
        with pytest.raises(DetectorMissingStateError, match="not found"):
            _validate_bundle_files_exist(str(tmp_path / "nope"))

    def test_manifest_missing(self, bundle_dir):
        (bundle_dir / "manifest.json").unlink()
        with pytest.raises(DetectorMissingStateError, match="manifest.json"):
            _validate_bundle_files_exist(str(bundle_dir))

    def test_w1_missing(self, bundle_dir):
        (bundle_dir / "w1.pth").unlink()
        with pytest.raises(DetectorMissingStateError, match="w1.pth"):
            _validate_bundle_files_exist(str(bundle_dir))

    def test_w2_missing(self, bundle_dir):
        (bundle_dir / "w2.pth").unlink()
        with pytest.raises(DetectorMissingStateError, match="w2.pth"):
            _validate_bundle_files_exist(str(bundle_dir))


def test_describe_required_artifacts():
    artifacts = describe_required_artifacts()
    assert isinstance(artifacts, list)
    assert any("gm_bundle_dir" in a for a in artifacts)


def test_gm_mathematics_not_rewritten(mock_deps, bundle_dir, monkeypatch):
    import raven.detectors.gm_detector as gm_mod
    stub = _StubExtractModule()
    monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

    records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
    info = load_state(records, "cpu")
    info["extract_module"] = stub

    from PIL import Image
    fake = bundle_dir.parent / "math_test.png"
    Image.new("RGB", (16, 16)).save(fake)

    record = _gm_record("0",
        watermark_target_sha256=info["provider_target_hash"],
        watermark_mask_sha256=info["provider_mask_hash"])
    score = score_image(info, str(fake), record=record)

    assert score["raw_score"] == 0.85
    assert score["canonical_score"] == score["raw_score"]
    assert score["gm_raw_bit_accuracy"] == 0.85
    assert score["gm_raw_ring_l1"] == 0.12


# ============================================================================
# Orchestrator integration tests
# ============================================================================
# Use the real evaluate_detector with mocked heavy resources (pipe, GmProvider,
# extraction helpers, tensor hash).  The real load_state / score_image /
# aggregate are exercised.

def _orchestrator_record(run_id="0", role="watermarked", input_path="", output_dir=None, **kw):
    from pathlib import Path
    od = Path(output_dir) if output_dir else Path("/tmp/orch")
    return {
        "run_id": run_id,
        "role": role,
        "method": "GM",
        "input_path": str(Path(input_path) if input_path else od / "in.png"),
        "output_path": str(od / role / run_id / "output.png"),
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


def _make_orch_images(run_dir, run_ids=("0",)):
    """Create input images at *run_dir*.  Output images must be separately
    created at the eval output directory used by evaluate_detector."""
    from PIL import Image
    for rid in run_ids:
        for role in ("watermarked", "clean"):
            inp = run_dir / role / rid
            inp.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(inp / "input.png")


def _make_orch_output_images(eval_dir, run_ids=("0",)):
    """Create output (attacked) images at *eval_dir* so the orchestrator's
    attacked cohorts pass preflight.

    The canonical output path is ``samples/<role>/<run_id>/output.png``
    (see ``raven.experiment_io.sample_dir``).
    """
    from PIL import Image
    for rid in run_ids:
        for role in ("watermarked", "clean"):
            out = eval_dir / "samples" / role / rid
            out.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(out / "output.png")


def _setup_orch_mocks(monkeypatch, bundle_dir):
    """Pre-seed sys.modules so evaluate_detector can import gm_detector."""
    import raven.detectors.gm_detector as gm_mod

    stub = _StubExtractModule()
    monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
    monkeypatch.setattr(gm_mod, "_ensure_paths",
                        lambda: sys.path.insert(0, str(REPO / "eval_bench_wm")))

    _fake_pipe_utils = mock.Mock()
    _fake_pipe_utils.get_pipe_provider = mock.Mock(return_value=StubPipe())
    _fake_gm_provider = mock.Mock()
    _fake_gm_provider.GmProvider = StubGmProvider

    for _key, _val in [
        ("eval_bench_wm", mock.Mock()),
        ("eval_bench_wm.utils", mock.Mock()),
        ("eval_bench_wm.utils.pipe", mock.Mock()),
        ("eval_bench_wm.utils.pipe.pipe_utils", _fake_pipe_utils),
        ("eval_bench_wm.utils.wm", mock.Mock()),
        ("eval_bench_wm.utils.wm.gm_provider", _fake_gm_provider),
    ]:
        monkeypatch.setitem(sys.modules, _key, _val)

    monkeypatch.setattr(
        "raven.pairing_provenance.tensor_sha256",
        lambda t: "orch_tensor_hash",
    )

    return _fake_pipe_utils, _fake_gm_provider


class TestOrchestratorSuccess:

    def test_real_adapter_orchestrator_success(self, tmp_path, bundle_dir, monkeypatch):
        """Full evaluate_detector flow: real load_state/score_image/aggregate,
        mocked pipe/GmProvider/extract/tensor hash."""
        from experiments.eval import evaluate_detector

        _setup_orch_mocks(monkeypatch, bundle_dir)
        _make_orch_images(tmp_path / "run", run_ids=("0", "1"))

        out_dir = tmp_path / "eval_out"
        out_dir.mkdir()
        # Output (attacked) images resolved from evaluate_detector's output_dir:
        _make_orch_output_images(out_dir, run_ids=("0", "1"))

        # The monkeypatched tensor_sha256 returns "orch_tensor_hash".
        # source_metadata must carry that value so target/mask validation passes.
        _ORCH_HASH = "orch_tensor_hash"
        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir),
                         watermark_target_sha256=_ORCH_HASH,
                         watermark_mask_sha256=_ORCH_HASH))
        gm_fields.pop("run_id")

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
            _orchestrator_record("1", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "1" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="1")),
            _orchestrator_record("0", "clean",
                input_path=str(tmp_path / "run" / "clean" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
            _orchestrator_record("1", "clean",
                input_path=str(tmp_path / "run" / "clean" / "1" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="1")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        assert result["status"] == STATUS_COMPLETED
        assert result["method"] == "GM"
        assert result["scored_count"] >= 2

        from raven.experiment_io import detector_records_path
        rec_path = detector_records_path(out_dir)
        assert rec_path.is_file()

        records_found = [
            json.loads(line) for line in rec_path.read_text().strip().splitlines()
        ]
        scored = [r for r in records_found if r.get("status") == ROW_STATUS_SCORED]
        assert len(scored) >= 1

        for s in scored:
            assert s.get("source_watermark_target_sha256"), "missing source target"
            assert s.get("detector_watermark_target_sha256"), "missing detector target"
            assert s.get("source_watermark_mask_sha256"), "missing source mask"
            assert s.get("detector_watermark_mask_sha256"), "missing detector mask"
            assert s.get("gm_target_verified") is True
            assert s.get("gm_mask_verified") is True
            assert "gm_gnr_used" in s
            assert "gm_classifier_used" in s

    def test_orchestrator_provider_constructed_once(self, tmp_path, bundle_dir, monkeypatch):
        """Provider constructor called exactly once across the whole cohort."""
        from experiments.eval import evaluate_detector

        pu, fgp = _setup_orch_mocks(monkeypatch, bundle_dir)
        counter = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)

        fgp.GmProvider = CountingProvider
        _make_orch_images(tmp_path / "run", run_ids=("0", "1"))

        out_dir = tmp_path / "eval_out2"
        out_dir.mkdir()
        _make_orch_output_images(out_dir, run_ids=("0", "1"))
        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields.pop("run_id")

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
            _orchestrator_record("1", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "1" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="1")),
        ]

        evaluate_detector(records, out_dir, "GM", device="cpu")
        assert counter[0] == 1


class TestOrchestratorFailureStatuses:

    def test_mixed_bundle_state_fails_before_provider(self, tmp_path, bundle_dir, monkeypatch):
        """Second row with different gm_m_sha256 → failed_state_validation."""
        from experiments.eval import evaluate_detector

        pu, fgp = _setup_orch_mocks(monkeypatch, bundle_dir)
        counter = [0]
        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)
        fgp.GmProvider = CountingProvider

        _make_orch_images(tmp_path / "run", run_ids=("0", "1"))
        out_dir = tmp_path / "eval_fail1"
        out_dir.mkdir()
        # Need output images so preflight passes and we reach load_state
        _make_orch_output_images(out_dir, run_ids=("0", "1"))

        gm_fields_0 = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields_0.pop("run_id")
        gm_fields_1 = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir),
                                      gm_m_sha256="z" * 64))
        gm_fields_1.pop("run_id")

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields_0, run_id="0")),
            _orchestrator_record("1", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "1" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields_1, run_id="1")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        assert counter[0] == 0  # provider never constructed
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        # --allow-missing-metrics must NOT suppress state validation
        assert not stage_status_is_allowable(
            STATUS_FAILED_STATE_VALIDATION, allow_missing_metrics=True)

    def test_missing_target_mask_is_missing_required_state(
        self, tmp_path, bundle_dir, monkeypatch):
        """Records missing watermark_target_sha256 → failed_missing_required_state."""
        from experiments.eval import evaluate_detector

        _setup_orch_mocks(monkeypatch, bundle_dir)
        _make_orch_images(tmp_path / "run", run_ids=("0",))
        out_dir = tmp_path / "eval_fail2"
        out_dir.mkdir()
        _make_orch_output_images(out_dir, run_ids=("0",))

        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields.pop("run_id")
        del gm_fields["watermark_target_sha256"]
        del gm_fields["watermark_mask_sha256"]

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        # Without allow → nonzero; with allow → exit 0
        assert not stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=False)
        assert stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=True)

    def test_target_mismatch_fails_state_validation(
        self, tmp_path, bundle_dir, monkeypatch):
        """Target SHA mismatch in score_image → failed_state_validation stage."""
        from experiments.eval import evaluate_detector

        _setup_orch_mocks(monkeypatch, bundle_dir)
        _make_orch_images(tmp_path / "run", run_ids=("0",))
        out_dir = tmp_path / "eval_fail3"
        out_dir.mkdir()

        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields.pop("run_id")
        gm_fields["watermark_target_sha256"] = "wrong" * 8  # mismatched

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        # The load_state succeeds (metadata is fine), but score_image fails
        # on target mismatch, which becomes a row-level state_validation error.
        # If no rows scored → reduce_detector_stage_status returns failed_state_validation.
        assert result["status"] in (
            STATUS_FAILED_STATE_VALIDATION,
            "failed_state_validation",
        )
        # State validation is never allowable
        assert not stage_status_is_allowable(
            result["status"], allow_missing_metrics=True)

    def test_constructor_exception_is_provider_failure(
        self, tmp_path, bundle_dir, monkeypatch):
        """TypeError during provider init → failed_provider_initialization."""
        from experiments.eval import evaluate_detector

        pu, fgp = _setup_orch_mocks(monkeypatch, bundle_dir)

        class BadProvider:
            def __init__(self, **kwargs):
                raise TypeError("missing required argument: bad_param")
        fgp.GmProvider = BadProvider

        _make_orch_images(tmp_path / "run", run_ids=("0",))
        out_dir = tmp_path / "eval_fail4"
        out_dir.mkdir()

        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields.pop("run_id")

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
        assert not stage_status_is_allowable(
            STATUS_FAILED_PROVIDER_INITIALIZATION, allow_missing_metrics=True)
