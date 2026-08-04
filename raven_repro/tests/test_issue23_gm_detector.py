"""Issue #23 tests — bind GM evaluation to persisted bundle and provenance.

All tests use mocks.  No GM artifacts are downloaded and no real cohort is run.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO / "eval_bench_wm", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from raven.detectors.gm_detector import (  # noqa: E402
    _COHORT_UNIFORM_FIELDS,
    _VERIFIED_PROVENANCE_FIELDS,
    _validate_cohort_uniform,
    _validate_bundle_files_exist,
    _classify_bundle_error,
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
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_SCORING,
)

REQUIRED_METADATA_FIELDS = frozenset({
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_protocol_mode",
})


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _gm_record(run_id="0", **overrides):
    record = {
        "run_id": run_id,
        "gm_bundle_dir": "/fake/bundle",
        "gm_bundle_config_sha256": "a" * 64,
        "gm_w1_file_sha256": "b" * 64,
        "gm_w2_file_sha256": "c" * 64,
        "gm_protocol_mode": "official_math_shared_tr_clean",
        "gm_m_sha256": "m" * 64,
        "gm_watermark_sha256": "n" * 64,
        "gm_target_sha256": "o" * 64,
        "watermark_target_sha256": "d" * 64,
        "watermark_mask_sha256": "e" * 64,
    }
    record.update(overrides)
    return record


def _make_bundle_dir(tmp_path: Path) -> Path:
    """Create a minimal valid-looking bundle directory."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
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
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "w1.pth").write_bytes(b"\x00" * 64)
    (bundle / "w2.pth").write_bytes(b"\x01" * 64)
    return bundle


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------

class StubGmProvider:
    """Minimal stub that satisfies the detector's post-construction checks."""

    def __init__(self, **kwargs):
        self.bundle = mock.Mock() if kwargs.get("_has_bundle", True) else None
        self.state_source = kwargs.get("_state_source", "bundle")
        self.gt_patch = kwargs.get("_gt_patch", _fake_gt_patch())
        self.watermarking_mask = kwargs.get("_wm_mask", _fake_wm_mask())
        self.profile_is_official = kwargs.get("_profile_is_official", False)


def _fake_gt_patch():
    """Return a tensor whose ``tensor_sha256`` is deterministic and non-empty."""
    return torch.zeros(1, 1, 64, 64, dtype=torch.float32)


def _fake_wm_mask():
    """Return a boolean mask tensor with a deterministic hash."""
    return torch.ones(1, 1, 64, 64, dtype=torch.bool)


class StubPipe:
    def get_latent_shape(self):
        return (1, 4, 64, 64)

    def get_dtype(self):
        return torch.float32


# ---------------------------------------------------------------------------
# Stub extraction module
# ---------------------------------------------------------------------------

class _StubExtractModule:
    """Stand-in for ``extract_verification_scores.py`` so no real import occurs."""

    def gm_bundle_manifest(self, row, identifier):
        bundle_dir = Path(str(row.get("gm_bundle_dir", "")))
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest not found: {manifest_path}")
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle_dir(tmp_path):
    return _make_bundle_dir(tmp_path)


@pytest.fixture
def stub_extract_module():
    return _StubExtractModule()


@pytest.fixture
def mock_deps(monkeypatch, bundle_dir, stub_extract_module):
    """Wire all dependencies so load_state works without real imports.

    Pre-seeds ``sys.modules`` with fake modules for ``eval_bench_wm.*``
    so the ``from ... import ...`` statements inside ``load_state`` never
    trigger a real import (which would fail on missing ``lpips`` etc.).
    """
    import raven.detectors.gm_detector as gm_mod

    # ── Pre-seed sys.modules so real eval_bench_wm is never imported ──
    # Use setitem directly (not inside a context manager) so patches
    # persist for the entire test function, not just this fixture body.
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

    # Stub the extraction module loader
    monkeypatch.setattr(gm_mod, "_get_extract_module",
                        lambda: stub_extract_module)
    monkeypatch.setattr(
        gm_mod, "_ensure_paths", lambda: sys.path.insert(0, str(REPO / "eval_bench_wm"))
    )

    # Stub tensor_sha256 for reproducible hashes
    monkeypatch.setattr(
        "raven.pairing_provenance.tensor_sha256",
        lambda t: "tensor_hash_" + str(t.shape) if hasattr(t, "shape") else "tensor_hash",
    )

    return gm_mod


# ---------------------------------------------------------------------------
# Cross-row uniformity validation
# ---------------------------------------------------------------------------

class TestCohortUniformity:
    """Mixed bundles/configs must fail before any provider is constructed."""

    def test_all_rows_same(self):
        records = [_gm_record("0"), _gm_record("1")]
        uniform = _validate_cohort_uniform(records)
        assert uniform["gm_bundle_dir"] == "/fake/bundle"
        assert uniform["gm_bundle_config_sha256"] == "a" * 64

    def test_mixed_bundle_dir_fails(self):
        records = [
            _gm_record("0", gm_bundle_dir="/fake/a"),
            _gm_record("1", gm_bundle_dir="/fake/b"),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            _validate_cohort_uniform(records)

    def test_mixed_bundle_config_sha_fails(self):
        records = [
            _gm_record("0", gm_bundle_config_sha256="a" * 64),
            _gm_record("1", gm_bundle_config_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            _validate_cohort_uniform(records)

    def test_mixed_w1_sha_fails(self):
        records = [
            _gm_record("0", gm_w1_file_sha256="b" * 64),
            _gm_record("1", gm_w1_file_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            _validate_cohort_uniform(records)

    def test_mixed_w2_sha_fails(self):
        records = [
            _gm_record("0", gm_w2_file_sha256="c" * 64),
            _gm_record("1", gm_w2_file_sha256="z" * 64),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            _validate_cohort_uniform(records)

    def test_mixed_protocol_fails(self):
        records = [
            _gm_record("0", gm_protocol_mode="protocol_a"),
            _gm_record("1", gm_protocol_mode="protocol_b"),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            _validate_cohort_uniform(records)


# ---------------------------------------------------------------------------
# Bundle file existence
# ---------------------------------------------------------------------------

class TestBundleFileExistence:

    def test_bundle_dir_exists(self, bundle_dir):
        resolved = _validate_bundle_files_exist(str(bundle_dir))
        assert resolved == bundle_dir.resolve()

    def test_bundle_dir_missing(self, tmp_path):
        missing = tmp_path / "no_such_bundle"
        with pytest.raises(DetectorMissingStateError, match="not found"):
            _validate_bundle_files_exist(str(missing))

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


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestClassifyBundleError:

    def test_not_found_is_missing_state(self):
        exc = RuntimeError("run_id=0: GM bundle manifest not found: /x/manifest.json")
        assert _classify_bundle_error(exc) is DetectorMissingStateError

    def test_missing_lowercase_is_missing_state(self):
        exc = RuntimeError("something is missing from the bundle")
        assert _classify_bundle_error(exc) is DetectorMissingStateError

    def test_sha_mismatch_is_state_validation(self):
        exc = RuntimeError("run_id=0: GM bundle/source gm_bundle_config_sha256 mismatch")
        assert _classify_bundle_error(exc) is DetectorStateValidationError

    def test_unknown_message_is_state_validation(self):
        exc = RuntimeError("something unexpected happened")
        assert _classify_bundle_error(exc) is DetectorStateValidationError


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------

class TestLoadState:

    def test_successful_load(self, mock_deps, bundle_dir):
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        assert info["provider"] is not None
        assert info["pipe"] is not None
        assert info["bundle_dir"] == str(bundle_dir.resolve())
        assert "verified_provenance" in info
        assert info["verified_provenance"]["gm_bundle_dir"] == str(bundle_dir)
        assert info["verified_provenance"]["gm_bundle_config_sha256"] == "a" * 64

    def test_empty_records_fails(self, mock_deps):
        with pytest.raises(DetectorMissingStateError, match="at least one record"):
            load_state([], "cpu")

    def test_mixed_cohort_fails(self, mock_deps, bundle_dir):
        records = [
            _gm_record("0", gm_bundle_dir=str(bundle_dir)),
            _gm_record("1", gm_bundle_dir="/other/bundle"),
        ]
        with pytest.raises(DetectorStateValidationError, match="mixed"):
            load_state(records, "cpu")

    def test_missing_bundle_dir(self, mock_deps):
        records = [_gm_record("0", gm_bundle_dir="/no/such/dir")]
        with pytest.raises(DetectorMissingStateError, match="not found"):
            load_state(records, "cpu")

    def test_missing_manifest(self, mock_deps, bundle_dir):
        (bundle_dir / "manifest.json").unlink()
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorMissingStateError, match="manifest.json"):
            load_state(records, "cpu")

    def test_missing_w1(self, mock_deps, bundle_dir):
        (bundle_dir / "w1.pth").unlink()
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorMissingStateError, match="w1.pth"):
            load_state(records, "cpu")

    def test_bundle_sha_mismatch(self, mock_deps, bundle_dir):
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir),
                               gm_bundle_config_sha256="z" * 64)]
        with pytest.raises(DetectorStateValidationError, match="mismatch"):
            load_state(records, "cpu")

    def test_constructor_typeerror(self, mock_deps, bundle_dir):
        """A TypeError in GmProvider.__init__ becomes DetectorProviderInitializationError."""
        class BadProvider:
            def __init__(self, **kwargs):
                raise TypeError("missing required argument: bad_param")

        # Replace the GmProvider in the pre-seeded sys.modules entry.
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = BadProvider

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorProviderInitializationError, match="construction failed"):
            load_state(records, "cpu")

        # Restore
        fake_gm.GmProvider = StubGmProvider

    def test_state_source_not_bundle(self, mock_deps, bundle_dir):
        """Provider with state_source != 'bundle' raises DetectorStateValidationError."""
        class NoBundleProvider(StubGmProvider):
            def __init__(self, **kwargs):
                super().__init__(_has_bundle=False, _state_source="memory", **kwargs)

        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = NoBundleProvider

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError, match="persisted bundle"):
            load_state(records, "cpu")

        fake_gm.GmProvider = StubGmProvider

    def test_dependency_error(self, monkeypatch, bundle_dir):
        """When eval_bench_wm is not importable at all."""
        import raven.detectors.gm_detector as gm_mod

        # Make _ensure_paths a no-op
        monkeypatch.setattr(gm_mod, "_ensure_paths", lambda: None)
        # But leave the import path broken
        monkeypatch.setattr(
            gm_mod, "_get_extract_module",
            lambda: (_ for _ in ()).throw(ImportError("no eval_bench_wm")),
        )
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorDependencyError):
            load_state(records, "cpu")


# ---------------------------------------------------------------------------
# score_image
# ---------------------------------------------------------------------------

class TestScoreImage:

    @pytest.fixture
    def provider_info(self, mock_deps, bundle_dir):
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        return load_state(records, "cpu")

    @pytest.fixture
    def fake_image(self, tmp_path):
        from PIL import Image
        img = Image.new("RGB", (16, 16))
        path = tmp_path / "test.png"
        img.save(path)
        return str(path)

    def test_successful_score(self, provider_info, fake_image):
        score = score_image(provider_info, fake_image)
        assert "raw_score" in score
        assert "canonical_score" in score
        assert score["raw_score"] == 0.85
        assert score["canonical_score"] == 0.85

    def test_gm_domain_scores_preserved(self, provider_info, fake_image):
        """GM-specific fields (bit accuracy, ring L1) must appear in output."""
        score = score_image(provider_info, fake_image)
        assert score["gm_raw_bit_accuracy"] == 0.85
        assert score["gm_raw_ring_l1"] == 0.12
        assert "gm_restored_bit_accuracy" in score
        assert "gm_classifier_probability" in score

    def test_self_description_fields_preserved(self, provider_info, fake_image):
        """Detector self-description fields come from the scorer, not the row."""
        score = score_image(provider_info, fake_image)
        assert score["gm_report_label"] == "gm_raw_bit_accuracy"
        assert "gm_score_definition" in score
        assert "gm_threshold_source" in score
        assert "gm_comparison_operator" in score

    def test_verified_provenance_in_score(self, provider_info, fake_image):
        """Only verified provenance is saved; no blind copies from input row."""
        score = score_image(provider_info, fake_image)
        for field in _VERIFIED_PROVENANCE_FIELDS:
            assert field in score, f"verified provenance field {field} missing"
        # The score does NOT blindly copy e.g. gm_m_sha256 from the input row
        assert score["gm_bundle_config_sha256"] == "a" * 64

    def test_target_hash_in_score(self, provider_info, fake_image):
        score = score_image(provider_info, fake_image)
        assert "watermark_target_sha256" in score
        assert "watermark_mask_sha256" in score

    def test_source_target_mismatch(self, provider_info, fake_image):
        """When the row's target SHA disagrees with the provider, fail closed."""
        record = _gm_record("0", watermark_target_sha256="wrong" * 8)
        with pytest.raises(DetectorStateValidationError, match="target SHA mismatch"):
            score_image(provider_info, fake_image, record=record)

    def test_source_mask_mismatch(self, provider_info, fake_image):
        """When the row's mask SHA disagrees with the provider, fail closed.

        The target must match so the check reaches the mask comparison.
        """
        provider_mask = provider_info["provider_mask_hash"]
        provider_target = provider_info["provider_target_hash"]
        record = _gm_record(
            "0",
            watermark_target_sha256=provider_target,
            watermark_mask_sha256="wrong" * 8,
        )
        with pytest.raises(DetectorStateValidationError, match="mask SHA mismatch"):
            score_image(provider_info, fake_image, record=record)

    def test_missing_image(self, provider_info):
        with pytest.raises(DetectorMissingStateError, match="not found"):
            score_image(provider_info, "/no/such/image.png")

    def test_scoring_error(self, provider_info, fake_image, monkeypatch):
        """When evaluate_image raises, surface as DetectorScoringError."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("inversion failed"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image)

    def test_score_without_record(self, provider_info, fake_image):
        """score_image works without a record (no target/mask validation)."""
        score = score_image(provider_info, fake_image, record=None)
        assert "raw_score" in score
        assert "canonical_score" in score

    def test_score_with_record_missing_target(self, provider_info, fake_image):
        """When record has no watermark_target_sha256, skip validation."""
        record = _gm_record("0")
        del record["watermark_target_sha256"]
        del record["watermark_mask_sha256"]
        score = score_image(provider_info, fake_image, record=record)
        assert "raw_score" in score


# ---------------------------------------------------------------------------
# Canonical helper delegation
# ---------------------------------------------------------------------------

class TestCanonicalHelperDelegation:

    def test_raw_score_delegated(self, mock_deps, bundle_dir, monkeypatch):
        """Prove gm_detector calls the canonical raw_score from extract module."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        calls = []

        def tracking_raw(method, result):
            calls.append(("raw_score", method, result))
            return 0.99

        stub.raw_score = tracking_raw
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "score_test.png"
        Image.new("RGB", (16, 16)).save(fake)

        score_image(info, str(fake))
        assert len(calls) == 1
        assert calls[0][0] == "raw_score"
        assert calls[0][1] == "GM"

    def test_canonical_score_delegated(self, mock_deps, bundle_dir, monkeypatch):
        """Prove gm_detector calls the canonical canonical_score from extract module."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        calls = []

        def tracking_canonical(method, raw, result):
            calls.append(("canonical_score", method, raw))
            return raw

        stub.canonical_score = tracking_canonical
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "score_test2.png"
        Image.new("RGB", (16, 16)).save(fake)

        score_image(info, str(fake))
        assert len(calls) == 1
        assert calls[0][0] == "canonical_score"
        assert calls[0][1] == "GM"

    def test_evaluate_image_delegated(self, mock_deps, bundle_dir, monkeypatch):
        """Prove gm_detector calls the canonical evaluate_image."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        _original_eval = stub.evaluate_image
        calls = []

        def tracking_eval(torch_mod, provider, pipe, path, steps):
            calls.append(("evaluate_image", str(path), steps))
            return _original_eval(torch_mod, provider, pipe, path, steps)

        stub.evaluate_image = tracking_eval
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "score_test3.png"
        Image.new("RGB", (16, 16)).save(fake)

        score_image(info, str(fake))
        assert len(calls) == 1
        assert calls[0][0] == "evaluate_image"


# ---------------------------------------------------------------------------
# gm_bundle_manifest / gm_provider_kwargs delegation
# ---------------------------------------------------------------------------

class TestBundleManifestDelegation:

    def test_gm_bundle_manifest_called(self, mock_deps, bundle_dir, monkeypatch):
        """load_state must delegate to gm_bundle_manifest from the extract module.

        Called twice: once directly in load_state and once inside gm_provider_kwargs.
        """
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        _original_manifest = stub.gm_bundle_manifest
        manifest_calls = []

        def tracking_manifest(row, ident):
            manifest_calls.append((str(row.get("run_id")), ident))
            return _original_manifest(row, ident)

        stub.gm_bundle_manifest = tracking_manifest
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        load_state(records, "cpu")
        assert len(manifest_calls) >= 1
        assert manifest_calls[0][0] == "0"

    def test_gm_provider_kwargs_called(self, mock_deps, bundle_dir, monkeypatch):
        """load_state must delegate to gm_provider_kwargs from the extract module."""
        import raven.detectors.gm_detector as gm_mod

        stub = _StubExtractModule()
        _original_kwargs = stub.gm_provider_kwargs
        kwargs_calls = []

        def tracking_kwargs(row, ident):
            kwargs_calls.append((str(row.get("run_id")), ident))
            return _original_kwargs(row, ident)

        stub.gm_provider_kwargs = tracking_kwargs
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        load_state(records, "cpu")
        assert len(kwargs_calls) == 1
        assert kwargs_calls[0][0] == "0"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

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
        assert result["failed_count"] == 0
        assert "detection_summary" in result
        assert result["detection_summary"]["target_fpr"] == 0.01

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

    def test_aggregate_no_clean_cohort(self):
        """Without clean cohort, no detection_summary."""
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 0.9},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 0.5},
        ]
        result = aggregate(rows)
        assert "detection_summary" not in result


# ---------------------------------------------------------------------------
# REQUIRED_METADATA_FIELDS
# ---------------------------------------------------------------------------

class TestRequiredMetadataFields:

    def test_fields_match_expectation(self):
        assert REQUIRED_METADATA_FIELDS == {
            "gm_bundle_dir",
            "gm_bundle_config_sha256",
            "gm_w1_file_sha256",
            "gm_w2_file_sha256",
            "gm_protocol_mode",
        }

    def test_cohort_uniform_fields_are_subset(self):
        for field in _COHORT_UNIFORM_FIELDS:
            assert field in REQUIRED_METADATA_FIELDS


# ---------------------------------------------------------------------------
# describe_required_artifacts
# ---------------------------------------------------------------------------

def test_describe_required_artifacts():
    artifacts = describe_required_artifacts()
    assert isinstance(artifacts, list)
    assert any("gm_bundle_dir" in a for a in artifacts)
    assert any("manifest.json" in a for a in artifacts)
    assert any("w1.pth" in a for a in artifacts)
    assert any("w2.pth" in a for a in artifacts)


# ---------------------------------------------------------------------------
# GaussMarker mathematics is NOT rewritten
# ---------------------------------------------------------------------------

def test_gm_mathematics_not_rewritten(mock_deps, bundle_dir, monkeypatch):
    """The extract module's evaluate_image, raw_score, and canonical_score
    are NOT reimplemented in the detector. Their output flows through verbatim."""
    import raven.detectors.gm_detector as gm_mod

    # Use the real stub which returns known values
    stub = _StubExtractModule()
    monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

    records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
    info = load_state(records, "cpu")
    info["extract_module"] = stub

    from PIL import Image
    fake = bundle_dir.parent / "math_test.png"
    Image.new("RGB", (16, 16)).save(fake)

    score = score_image(info, str(fake))

    # raw_score returns gm_raw_bit_accuracy (not reimplemented)
    assert score["raw_score"] == stub.evaluate_image(None, None, None, None, 0)["gm_raw_bit_accuracy"]
    # canonical_score returns raw (not reimplemented)
    assert score["canonical_score"] == score["raw_score"]
    # GM domain scores pass through verbatim
    assert score["gm_raw_bit_accuracy"] == 0.85
    assert score["gm_raw_ring_l1"] == 0.12
