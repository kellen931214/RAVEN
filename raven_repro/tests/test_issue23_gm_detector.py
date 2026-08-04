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

from raven.pairing_provenance import GM_SHARED_TR_CLEAN_MODE  # noqa: E402

from raven.detectors.gm_detector import (  # noqa: E402
    _CANONICAL_KWARGS_FIELDS,
    _GM_REQUIRED_METADATA_FIELDS,
    _validate_required_gm_metadata,
    _validate_bundle_files_exist,
    _canonical_provider_identity,
    _validate_gm_protocol_mode,
    _validate_gm_provider_profile,
    _validate_scorer_outputs,
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
    STATUS_COMPLETED,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    stage_status_is_allowable,
)


# ============================================================================
# Record builders — gm_protocol_mode ≠ gm_profile
# ============================================================================

def _gm_record(run_id="0", **overrides):
    record = {
        "run_id": run_id,
        "gm_bundle_dir": "/fake/bundle",
        "gm_bundle_config_sha256": "a" * 64,
        "gm_w1_file_sha256": "b" * 64,
        "gm_w2_file_sha256": "c" * 64,
        "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
        "gm_m_sha256": "m" * 64,
        "gm_watermark_sha256": "n" * 64,
        "gm_target_sha256": "o" * 64,
        "watermark_target_sha256": "d" * 64,
        "watermark_mask_sha256": "e" * 64,
    }
    record.update(overrides)
    return record


def _make_bundle_dir(tmp_path: Path, **manifest_overrides) -> Path:
    """Create a minimal valid-looking bundle directory.

    Profile is ``legacy`` (the GmProvider bundle config), NOT the
    evaluation protocol (``GM_SHARED_TR_CLEAN_MODE``).
    """
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
        self.profile = kwargs.get("gm_profile", "legacy")
        self.profile_is_official = kwargs.get("_profile_is_official", False)


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

    # Pre-seed sys.modules so ``from ... import pipe_utils`` finds
    # the correct mock.  The parent modules must expose the submodules
    # as attributes so the import statement resolves them correctly.
    _pipe_pkg = mock.Mock()
    _pipe_pkg.pipe_utils = _fake_pipe_utils
    _wm_pkg = mock.Mock()
    _wm_pkg.gm_provider = _fake_gm_provider

    for _key, _val in [
        ("eval_bench_wm", mock.Mock()),
        ("eval_bench_wm.utils", mock.Mock()),
        ("eval_bench_wm.utils.pipe", _pipe_pkg),
        ("eval_bench_wm.utils.pipe.pipe_utils", _fake_pipe_utils),
        ("eval_bench_wm.utils.wm", _wm_pkg),
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
# 1. Protocol / profile separation
# ============================================================================

class TestProtocolProfileSeparation:

    def test_real_shared_clean_protocol_and_legacy_profile_are_valid(
        self, mock_deps, bundle_dir):
        """gm_protocol_mode=GM_SHARED_TR_CLEAN_MODE with profile=legacy passes."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir),
                               gm_protocol_mode=GM_SHARED_TR_CLEAN_MODE)]
        info = load_state(records, "cpu")
        vp = info["verified_provenance"]
        assert vp["gm_protocol_mode"] == GM_SHARED_TR_CLEAN_MODE
        assert vp["gm_profile"] == "legacy"

    def test_wrong_gm_protocol_mode_rejected(self):
        """Any gm_protocol_mode other than GM_SHARED_TR_CLEAN_MODE is rejected."""
        record = _gm_record("0", gm_protocol_mode="wrong_protocol")
        with pytest.raises(DetectorStateValidationError,
                           match="protocol mode"):
            _validate_gm_protocol_mode(record)

    def test_correct_gm_protocol_mode_accepted(self):
        record = _gm_record("0", gm_protocol_mode=GM_SHARED_TR_CLEAN_MODE)
        _validate_gm_protocol_mode(record)  # no exception

    def test_wrong_manifest_profile_rejected(self):
        """manifest.profile != gm_provider_kwargs gm_profile → error."""
        manifest = {"profile": "legacy"}
        kwargs = {"gm_profile": "custom"}
        with pytest.raises(DetectorStateValidationError,
                           match="provider profile mismatch"):
            _validate_gm_provider_profile(manifest, kwargs)

    def test_provider_actual_profile_mismatch_rejected(self):
        """Provider's actual .profile differs from kwargs gm_profile → error."""
        manifest = {"profile": "legacy"}
        kwargs = {"gm_profile": "legacy"}

        class WrongProfileProvider:
            profile = "custom"
            bundle = None
        with pytest.raises(DetectorStateValidationError,
                           match="provider profile mismatch"):
            _validate_gm_provider_profile(
                manifest, kwargs, provider=WrongProfileProvider())

    def test_provider_actual_profile_matches(self):
        """Provider .profile == kwargs gm_profile → passes."""
        manifest = {"profile": "legacy"}
        kwargs = {"gm_profile": "legacy"}

        class RightProfileProvider:
            profile = "legacy"
            bundle = None
        _validate_gm_provider_profile(
            manifest, kwargs, provider=RightProfileProvider())

    def test_verified_provenance_separates_protocol_and_profile(
        self, provider_info):
        vp = provider_info["verified_provenance"]
        assert "gm_protocol_mode" in vp
        assert "gm_profile" in vp
        assert vp["gm_protocol_mode"] == GM_SHARED_TR_CLEAN_MODE
        assert vp["gm_profile"] == "legacy"
        assert vp["gm_protocol_mode"] != vp["gm_profile"]


# ============================================================================
# 2. Per-row bundle binding
# ============================================================================

class TestPerRowBundleBinding:

    def test_second_row_m_sha_mismatch_rejected_before_provider(
        self, mock_deps, bundle_dir):
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
        self, mock_deps, bundle_dir):
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
# 3. Required metadata preflight
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

    def test_preflight_before_manifest(self, mock_deps, bundle_dir):
        record = _gm_record("0", gm_bundle_dir=str(bundle_dir))
        del record["gm_m_sha256"]
        with pytest.raises(DetectorMissingStateError, match="gm_m_sha256"):
            load_state([record], "cpu")

    def test_all_rows_missing_same_field_not_uniform(self, mock_deps, bundle_dir):
        r0 = _gm_record("0", gm_bundle_dir=str(bundle_dir))
        del r0["gm_m_sha256"]
        r1 = _gm_record("1", gm_bundle_dir=str(bundle_dir))
        del r1["gm_m_sha256"]
        with pytest.raises(DetectorMissingStateError):
            load_state([r0, r1], "cpu")


# ============================================================================
# 4. Target / mask fail-closed
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

    def test_missing_detector_target_is_state_validation(
        self, provider_info, fake_image):
        provider_info["provider_target_hash"] = ""
        record = _gm_record("0")
        with pytest.raises(DetectorStateValidationError, match="detector target hash"):
            score_image(provider_info, fake_image, record=record)

    def test_missing_detector_mask_is_state_validation(
        self, provider_info, fake_image):
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
        record = _gm_record(
            "0", watermark_target_sha256=provider_target,
            watermark_mask_sha256="wrong" * 8)
        with pytest.raises(DetectorStateValidationError, match="mask SHA mismatch"):
            score_image(provider_info, fake_image, record=record)


# ============================================================================
# 5. Pipe uses complete canonical profile (including revision)
# ============================================================================

class TestCanonicalPipeConfig:

    def test_pipe_uses_model_id_revision_scheduler_resolution(
        self, mock_deps, bundle_dir):
        """Pipe construction receives model_id, revision, scheduler, resolution
        from canonical kwargs."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        assert info["provider"] is not None
        assert info["pipe"] is not None

        fake_utils = sys.modules.get("eval_bench_wm.utils.pipe.pipe_utils")
        # get_pipe_provider was called
        assert fake_utils.get_pipe_provider.called
        _, kw = fake_utils.get_pipe_provider.call_args
        assert kw["pretrained_model_name_or_path"] == "RedbeardNZ/stable-diffusion-2-1-base"
        assert kw["resolution"] == 512
        assert kw["schedulers_name"] == "DDIM"
        assert kw.get("revision") == "fake"


# ============================================================================
# 6. Scoring contract — no fabricated defaults
# ============================================================================

class TestScorerOutputValidation:

    def test_required_outputs_present_and_valid(self):
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "gm_raw_bit_accuracy",
            "gm_score_definition": "spatial-domain bit match rate",
            "gm_threshold_source": "ensemble_not_applicable",
            "gm_comparison_operator": ">=",
        }
        _validate_scorer_outputs(result)  # no exception

    def test_missing_required_output_raises(self):
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "gm_raw_bit_accuracy",
            "gm_score_definition": "spatial-domain bit match rate",
            "gm_threshold_source": "ensemble_not_applicable",
            # gm_comparison_operator missing
        }
        with pytest.raises(ValueError, match="gm_comparison_operator"):
            _validate_scorer_outputs(result)

    def test_none_required_output_raises(self):
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": None,
            "gm_score_definition": "x",
            "gm_threshold_source": "x",
            "gm_comparison_operator": ">=",
        }
        with pytest.raises(ValueError, match="gm_report_label"):
            _validate_scorer_outputs(result)

    def test_nonfinite_bit_accuracy_raises(self):
        result = {
            "gm_raw_bit_accuracy": float("nan"),
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x",
            "gm_score_definition": "x",
            "gm_threshold_source": "x",
            "gm_comparison_operator": ">=",
        }
        with pytest.raises(ValueError, match="non-finite"):
            _validate_scorer_outputs(result)

    def test_bit_accuracy_out_of_range_raises(self):
        result = {
            "gm_raw_bit_accuracy": 1.5,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x",
            "gm_score_definition": "x",
            "gm_threshold_source": "x",
            "gm_comparison_operator": ">=",
        }
        with pytest.raises(ValueError, match="out of"):
            _validate_scorer_outputs(result)

    def test_optional_none_accepted(self):
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x",
            "gm_score_definition": "x",
            "gm_threshold_source": "x",
            "gm_comparison_operator": ">=",
            "gm_restored_bit_accuracy": None,
            "gm_classifier_probability": None,
        }
        _validate_scorer_outputs(result)  # no exception

    def test_optional_nonfinite_raises(self):
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x",
            "gm_score_definition": "x",
            "gm_threshold_source": "x",
            "gm_comparison_operator": ">=",
            "gm_restored_bit_accuracy": float("inf"),
        }
        with pytest.raises(ValueError, match="non-finite"):
            _validate_scorer_outputs(result)


# ============================================================================
# 7. Scoring path — raw_score/canonical_score failures wrapped
# ============================================================================

class TestScoringPathFailures:

    def test_raw_score_failure_is_scoring_error(
        self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.raw_score = lambda m, r: (_ for _ in ()).throw(
            ValueError("raw score computation failed"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image, record=record)

    def test_canonical_score_failure_is_scoring_error(
        self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.canonical_score = lambda m, r, res: (
            _ for _ in ()).throw(ValueError("canonical score failed"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image, record=record)

    def test_nonfinite_raw_score_is_scoring_error(
        self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.raw_score = lambda m, r: float("nan")
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image, record=record)

    def test_nonfinite_canonical_score_is_scoring_error(
        self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.canonical_score = lambda m, r, res: float("-inf")
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(provider_info, fake_image, record=record)


# ============================================================================
# 8. Malformed bundle — structured classification
# ============================================================================

class TestMalformedBundle:

    def test_malformed_manifest_json_is_state_validation(
        self, mock_deps, bundle_dir, monkeypatch):
        """When manifest.json is not valid JSON, it's StateValidationError
        (not MissingStateError — the files exist on disk)."""
        import raven.detectors.gm_detector as gm_mod

        def _broken_manifest(row, ident):
            raise json.JSONDecodeError("bad json", "{", 0)
        stub = _StubExtractModule()
        stub.gm_bundle_manifest = _broken_manifest
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError,
                           match="manifest validation failed"):
            load_state(records, "cpu")

    def test_manifest_missing_required_key_is_state_validation(
        self, mock_deps, bundle_dir, monkeypatch):
        """When manifest lacks a required key, StateValidationError."""
        import raven.detectors.gm_detector as gm_mod

        def _missing_key(row, ident):
            raise KeyError("bundle_config_sha256")
        stub = _StubExtractModule()
        stub.gm_bundle_manifest = _missing_key
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError,
                           match="manifest validation failed"):
            load_state(records, "cpu")

    def test_invalid_manifest_value_type_is_state_validation(
        self, mock_deps, bundle_dir, monkeypatch):
        """When a manifest value has wrong type, StateValidationError."""
        import raven.detectors.gm_detector as gm_mod

        def _wrong_type(row, ident):
            raise TypeError("resolution must be int, got str")
        stub = _StubExtractModule()
        stub.gm_bundle_manifest = _wrong_type
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError,
                           match="manifest validation failed"):
            load_state(records, "cpu")

    def test_provider_kwargs_exception_is_state_validation(
        self, mock_deps, bundle_dir, monkeypatch):
        """Any exception from gm_provider_kwargs → StateValidationError."""
        import raven.detectors.gm_detector as gm_mod

        def _failing_kwargs(row, ident):
            raise ValueError("unexpected manifest field")
        stub = _StubExtractModule()
        stub.gm_provider_kwargs = _failing_kwargs
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorStateValidationError,
                           match="provider kwargs validation failed"):
            load_state(records, "cpu")


# ============================================================================
# 9. Missing image → FileNotFoundError
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
# 10. Constructor failure
# ============================================================================

class TestConstructorFailure:

    def test_non_typeerror_constructor_failure_is_provider_failure(
        self, mock_deps, bundle_dir):
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
# Scoring + verified provenance + aggregate
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
        assert score["gm_raw_bit_accuracy"] == 0.85
        assert score["gm_raw_ring_l1"] == 0.12
        assert "gm_restored_bit_accuracy" in score
        assert "gm_classifier_probability" in score
        assert score["gm_report_label"] == "gm_raw_bit_accuracy"
        assert "gm_score_definition" in score
        assert "gm_threshold_source" in score
        assert "gm_comparison_operator" in score

    def test_verified_provenance_contains_protocol_and_profile(
        self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["gm_protocol_mode"] == GM_SHARED_TR_CLEAN_MODE
        assert score["gm_profile"] == "legacy"
        assert score["gm_protocol_mode"] != score["gm_profile"]

    def test_source_detector_target_pairs_preserved(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["source_watermark_target_sha256"]
        assert score["detector_watermark_target_sha256"]
        assert score["source_watermark_mask_sha256"]
        assert score["detector_watermark_mask_sha256"]

    def test_gnr_classifier_usage_preserved(self, provider_info, fake_image):
        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert "gm_gnr_used" in score
        assert "gm_classifier_used" in score
        assert isinstance(score["gm_gnr_used"], bool)
        assert isinstance(score["gm_classifier_used"], bool)

    def test_scoring_error(self, provider_info, fake_image, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: (
            _ for _ in ()).throw(RuntimeError("inversion failed"))
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


# ============================================================================
# Bundle file existence
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
# Orchestrator integration tests (real evaluate_detector, mocked heavy resources)
# ============================================================================

def _orchestrator_record(run_id="0", role="watermarked", input_path="",
                          output_dir=None, **kw):
    from pathlib import Path
    od = Path(output_dir) if output_dir else Path("/tmp/orch")
    return {
        "run_id": run_id, "role": role, "method": "GM",
        "input_path": str(Path(input_path) if input_path else od / "in.png"),
        "output_path": str(od / role / run_id / "output.png"),
        "prompt": kw.get("prompt", ""),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -24.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": kw.get("source_metadata", {}),
    }


def _make_orch_images(run_dir, run_ids=("0",)):
    from PIL import Image
    for rid in run_ids:
        for role in ("watermarked", "clean"):
            inp = run_dir / role / rid
            inp.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(inp / "input.png")


def _make_orch_output_images(eval_dir, run_ids=("0",)):
    from PIL import Image
    for rid in run_ids:
        for role in ("watermarked", "clean"):
            out = eval_dir / "samples" / role / rid
            out.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(out / "output.png")


def _setup_orch_mocks(monkeypatch, bundle_dir):
    import raven.detectors.gm_detector as gm_mod

    stub = _StubExtractModule()
    monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)
    monkeypatch.setattr(gm_mod, "_ensure_paths",
                        lambda: sys.path.insert(0, str(REPO / "eval_bench_wm")))

    _fake_pipe_utils = mock.Mock()
    _fake_pipe_utils.get_pipe_provider = mock.Mock(return_value=StubPipe())
    _fake_gm_provider = mock.Mock()
    _fake_gm_provider.GmProvider = StubGmProvider

    _pipe_pkg = mock.Mock()
    _pipe_pkg.pipe_utils = _fake_pipe_utils
    _wm_pkg = mock.Mock()
    _wm_pkg.gm_provider = _fake_gm_provider

    for _key, _val in [
        ("eval_bench_wm", mock.Mock()),
        ("eval_bench_wm.utils", mock.Mock()),
        ("eval_bench_wm.utils.pipe", _pipe_pkg),
        ("eval_bench_wm.utils.pipe.pipe_utils", _fake_pipe_utils),
        ("eval_bench_wm.utils.wm", _wm_pkg),
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
        from experiments.eval import evaluate_detector

        _setup_orch_mocks(monkeypatch, bundle_dir)
        _make_orch_images(tmp_path / "run", run_ids=("0", "1"))

        out_dir = tmp_path / "eval_out"
        out_dir.mkdir()
        _make_orch_output_images(out_dir, run_ids=("0", "1"))

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
            assert s.get("source_watermark_target_sha256")
            assert s.get("detector_watermark_target_sha256")
            assert s.get("source_watermark_mask_sha256")
            assert s.get("detector_watermark_mask_sha256")
            assert s.get("gm_target_verified") is True
            assert s.get("gm_mask_verified") is True
            assert "gm_gnr_used" in s
            assert "gm_classifier_used" in s

    def test_orchestrator_provider_constructed_once(
        self, tmp_path, bundle_dir, monkeypatch):
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
        ]

        evaluate_detector(records, out_dir, "GM", device="cpu")
        assert counter[0] == 1


class TestOrchestratorFailureStatuses:

    def test_mixed_bundle_state_fails_before_provider(
        self, tmp_path, bundle_dir, monkeypatch):
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

        assert counter[0] == 0
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert not stage_status_is_allowable(
            STATUS_FAILED_STATE_VALIDATION, allow_missing_metrics=True)

    def test_missing_target_mask_is_missing_required_state(
        self, tmp_path, bundle_dir, monkeypatch):
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
        assert not stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=False)
        assert stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=True)

    def test_target_mismatch_fails_state_validation(
        self, tmp_path, bundle_dir, monkeypatch):
        from experiments.eval import evaluate_detector

        _setup_orch_mocks(monkeypatch, bundle_dir)
        _make_orch_images(tmp_path / "run", run_ids=("0",))
        out_dir = tmp_path / "eval_fail3"
        out_dir.mkdir()
        _make_orch_output_images(out_dir, run_ids=("0",))

        gm_fields = dict(_gm_record("0", gm_bundle_dir=str(bundle_dir)))
        gm_fields.pop("run_id")
        gm_fields["watermark_target_sha256"] = "wrong" * 8

        records = [
            _orchestrator_record("0", "watermarked",
                input_path=str(tmp_path / "run" / "watermarked" / "0" / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gm_fields, run_id="0")),
        ]

        result = evaluate_detector(records, out_dir, "GM", device="cpu")

        assert result["status"] in (
            STATUS_FAILED_STATE_VALIDATION, "failed_state_validation")
        assert not stage_status_is_allowable(
            result["status"], allow_missing_metrics=True)

    def test_constructor_exception_is_provider_failure(
        self, tmp_path, bundle_dir, monkeypatch):
        from experiments.eval import evaluate_detector

        pu, fgp = _setup_orch_mocks(monkeypatch, bundle_dir)

        class BadProvider:
            def __init__(self, **kwargs):
                raise TypeError("missing required argument: bad_param")
        fgp.GmProvider = BadProvider

        _make_orch_images(tmp_path / "run", run_ids=("0",))
        out_dir = tmp_path / "eval_fail4"
        out_dir.mkdir()
        _make_orch_output_images(out_dir, run_ids=("0",))

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


# ============================================================================
# GNR usage — Bug 1 & Bug 2 fixes
# ============================================================================

class TestGnrUsage:

    @pytest.fixture
    def gnr_provider_info(self, provider_info):
        """provider_info with _canonical_kwargs gm_use_gnr=False."""
        provider_info["_canonical_kwargs"] = {"gm_use_gnr": False,
                                                "gm_use_classifier": False}
        return provider_info

    def test_canonical_false_scorer_true_contradiction(
        self, gnr_provider_info, fake_image):
        """canonical gm_use_gnr=False + scorer gm_used_gnr=True → DetectorScoringError."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": True, "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="usage contradiction"):
            score_image(gnr_provider_info, fake_image, record=record)

    def test_scorer_string_false_not_accepted(
        self, gnr_provider_info, fake_image):
        """gm_used_gnr='false' (string) → DetectorScoringError, NOT True."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": "false", "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="must be bool"):
            score_image(gnr_provider_info, fake_image, record=record)

    def test_value_1_not_accepted_as_bool(
        self, gnr_provider_info, fake_image):
        """gm_used_gnr=1 (int) → DetectorScoringError, NOT True."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": 1, "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="must be bool"):
            score_image(gnr_provider_info, fake_image, record=record)

    def test_conflicting_aliases_gnr(
        self, gnr_provider_info, fake_image):
        """gm_used_gnr=True + gm_gnr_used=False → DetectorScoringError."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": True, "gm_gnr_used": False,
            "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="conflicting"):
            score_image(gnr_provider_info, fake_image, record=record)

    def test_scorer_missing_fallback_to_canonical(
        self, gnr_provider_info, fake_image):
        """Scorer reports no GNR flag → fallback to canonical gm_use_gnr=False."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            # No gm_used_gnr / gm_gnr_used at all
            "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        score = score_image(gnr_provider_info, fake_image, record=record)
        assert score["gm_gnr_used"] is False

    def test_scorer_false_canonical_false_passes(
        self, gnr_provider_info, fake_image):
        """Scorer gm_used_gnr=False + canonical False → passes."""
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": False,
        }
        gnr_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=gnr_provider_info["provider_target_hash"],
            watermark_mask_sha256=gnr_provider_info["provider_mask_hash"])
        score = score_image(gnr_provider_info, fake_image, record=record)
        assert score["gm_gnr_used"] is False


# ============================================================================
# Classifier usage — symmetric to GNR
# ============================================================================

class TestClassifierUsage:

    @pytest.fixture
    def clf_provider_info(self, provider_info):
        provider_info["_canonical_kwargs"] = {"gm_use_gnr": False,
                                                "gm_use_classifier": False}
        return provider_info

    def test_canonical_false_scorer_true_contradiction(
        self, clf_provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": True,
        }
        clf_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=clf_provider_info["provider_target_hash"],
            watermark_mask_sha256=clf_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="usage contradiction"):
            score_image(clf_provider_info, fake_image, record=record)

    def test_scorer_string_false_not_accepted(
        self, clf_provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": "false",
        }
        clf_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=clf_provider_info["provider_target_hash"],
            watermark_mask_sha256=clf_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="must be bool"):
            score_image(clf_provider_info, fake_image, record=record)

    def test_conflicting_aliases_classifier(
        self, clf_provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False,
            "gm_used_classifier": True, "gm_classifier_used": False,
        }
        clf_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=clf_provider_info["provider_target_hash"],
            watermark_mask_sha256=clf_provider_info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError,
                           match="conflicting"):
            score_image(clf_provider_info, fake_image, record=record)

    def test_scorer_missing_fallback_to_canonical(
        self, clf_provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False,
            # No classifier flags
        }
        clf_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=clf_provider_info["provider_target_hash"],
            watermark_mask_sha256=clf_provider_info["provider_mask_hash"])
        score = score_image(clf_provider_info, fake_image, record=record)
        assert score["gm_classifier_used"] is False

    def test_scorer_false_canonical_false_passes(
        self, clf_provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": False,
        }
        clf_provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=clf_provider_info["provider_target_hash"],
            watermark_mask_sha256=clf_provider_info["provider_mask_hash"])
        score = score_image(clf_provider_info, fake_image, record=record)
        assert score["gm_classifier_used"] is False


# ============================================================================
# Scorer output — bool rejection in numeric fields
# ============================================================================

class TestBoolRejectionInNumericFields:

    def test_bool_rejected_in_bit_accuracy(self):
        from raven.detectors.gm_detector import _validate_scorer_outputs
        result = {
            "gm_raw_bit_accuracy": True,
            "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
        }
        with pytest.raises(ValueError, match="wrong type"):
            _validate_scorer_outputs(result)

    def test_bool_rejected_in_ring_l1(self):
        from raven.detectors.gm_detector import _validate_scorer_outputs
        result = {
            "gm_raw_bit_accuracy": 0.85,
            "gm_raw_ring_l1": False,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
        }
        with pytest.raises(ValueError, match="wrong type"):
            _validate_scorer_outputs(result)

    def test_bool_rejected_in_optional_restored_accuracy(self):
        from raven.detectors.gm_detector import _validate_scorer_outputs
        result = {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_restored_bit_accuracy": True,
        }
        with pytest.raises(ValueError, match="wrong type"):
            _validate_scorer_outputs(result)

    def test_ring_l1_no_range_bound(self):
        """gm_raw_ring_l1 must be finite but has NO [0,1] bound."""
        from raven.detectors.gm_detector import _validate_scorer_outputs
        result = {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 5000.0,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
        }
        _validate_scorer_outputs(result)  # no exception


# ============================================================================
# Missing bundle artifacts
# ============================================================================

class TestMissingBundleArtifacts:

    def test_w1_missing_is_detector_missing_state(self, bundle_dir):
        (bundle_dir / "w1.pth").unlink()
        with pytest.raises(DetectorMissingStateError, match="w1.pth"):
            _validate_bundle_files_exist(str(bundle_dir))

    def test_w2_missing_is_detector_missing_state(self, bundle_dir):
        (bundle_dir / "w2.pth").unlink()
        with pytest.raises(DetectorMissingStateError, match="w2.pth"):
            _validate_bundle_files_exist(str(bundle_dir))

    def test_w1_missing_via_load_state(self, mock_deps, bundle_dir):
        (bundle_dir / "w1.pth").unlink()
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorMissingStateError, match="w1.pth"):
            load_state(records, "cpu")

    def test_w2_missing_via_load_state(self, mock_deps, bundle_dir):
        (bundle_dir / "w2.pth").unlink()
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        with pytest.raises(DetectorMissingStateError, match="w2.pth"):
            load_state(records, "cpu")


# ============================================================================
# Canonical provider identity — field-level tests
# ============================================================================

class TestCanonicalProviderIdentityFields:

    def _base_kwargs(self):
        return {f: f"val_{f}" if f not in ("gm_channel_copy", "gm_w_copy",
            "gm_h_copy", "w_seed", "w_channel", "w_radius", "resolution",
            "gm_watermark_bits_seed") else 0
            for f in _CANONICAL_KWARGS_FIELDS}

    def test_same_fields_same_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        assert _canonical_provider_identity(k1) == _canonical_provider_identity(k2)

    def test_gm_profile_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["gm_profile"] = "different"
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_gm_bundle_dir_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["gm_bundle_dir"] = "/other/path"
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_gm_use_gnr_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["gm_use_gnr"] = not k1["gm_use_gnr"]
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_gm_use_classifier_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["gm_use_classifier"] = not k1["gm_use_classifier"]
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_modelid_target_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["modelid_target"] = "other/model"
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_model_revision_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["model_revision"] = "other_rev"
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_scheduler_target_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["scheduler_target"] = "DPM"
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_resolution_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["resolution"] = 768
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_w_seed_changes_identity(self):
        k1 = self._base_kwargs()
        k2 = self._base_kwargs()
        k2["w_seed"] = 999
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)


class TestMixedCanonicalIdentityRejectedBeforeProvider:

    def test_mixed_identity_via_custom_extract(
        self, mock_deps, bundle_dir, monkeypatch):
        """Custom extract module returns different kwargs for row 1 → mixed identity."""
        import raven.detectors.gm_detector as gm_mod
        counter = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)

        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class RowVaryingExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                if str(row.get("run_id")) == "1":
                    kw = dict(kw, gm_use_gnr=not kw.get("gm_use_gnr", False))
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: RowVaryingExtract())
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            with pytest.raises(DetectorStateValidationError,
                               match="mixed canonical"):
                load_state(records, "cpu")
            assert counter[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider


# ============================================================================
# Canonical helper delegation — raw_score/canonical_score + error wrapping
# ============================================================================

class TestCanonicalHelperDelegationFull:

    def test_raw_score_called_once(self, mock_deps, bundle_dir, monkeypatch):
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
        fake = bundle_dir.parent / "deleg_raw.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        score_image(info, str(fake), record=record)
        assert len(calls) == 1
        assert calls[0] == ("raw_score", "GM")

    def test_canonical_score_called_once(self, mock_deps, bundle_dir, monkeypatch):
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
        fake = bundle_dir.parent / "deleg_canon.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        score_image(info, str(fake), record=record)
        assert len(calls) == 1
        assert calls[0] == ("canonical", "GM")

    def test_raw_score_error_wrapped_as_scoring_error(
        self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.raw_score = lambda m, r: (_ for _ in ()).throw(
            RuntimeError("raw fail"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "deleg_err1.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(info, str(fake), record=record)

    def test_canonical_score_error_wrapped_as_scoring_error(
        self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.canonical_score = lambda m, r, res: (
            _ for _ in ()).throw(RuntimeError("canonical fail"))
        monkeypatch.setattr(gm_mod, "_get_extract_module", lambda: stub)

        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        info["extract_module"] = stub

        from PIL import Image
        fake = bundle_dir.parent / "deleg_err2.png"
        Image.new("RGB", (16, 16)).save(fake)

        record = _gm_record("0",
            watermark_target_sha256=info["provider_target_hash"],
            watermark_mask_sha256=info["provider_mask_hash"])
        with pytest.raises(DetectorScoringError, match="scoring failed"):
            score_image(info, str(fake), record=record)


# ============================================================================
# Canonical identity — gm_create_bundle / gm_allow_in_memory_state
# ============================================================================

class TestStateControlFieldsInIdentity:

    def test_gm_create_bundle_changes_identity(self):
        k1 = {f: "val" for f in _CANONICAL_KWARGS_FIELDS}
        k2 = dict(k1)
        k2["gm_create_bundle"] = not k1.get("gm_create_bundle", False)
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_gm_allow_in_memory_state_changes_identity(self):
        k1 = {f: "val" for f in _CANONICAL_KWARGS_FIELDS}
        k2 = dict(k1)
        k2["gm_allow_in_memory_state"] = not k1.get("gm_allow_in_memory_state", False)
        assert _canonical_provider_identity(k1) != _canonical_provider_identity(k2)

    def test_create_bundle_mixed_across_rows_rejected(
        self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        counter = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class CreateBundleVarying(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                if str(row.get("run_id")) == "1":
                    kw = dict(kw, gm_create_bundle=True)
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: CreateBundleVarying())
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_create_bundle"):
                load_state(records, "cpu")
            assert counter[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_allow_in_memory_mixed_across_rows_rejected(
        self, mock_deps, bundle_dir, monkeypatch):
        import raven.detectors.gm_detector as gm_mod
        counter = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                counter[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MemoryStateVarying(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                if str(row.get("run_id")) == "1":
                    kw = dict(kw, gm_allow_in_memory_state=True)
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MemoryStateVarying())
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_allow_in_memory_state"):
                load_state(records, "cpu")
            assert counter[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider


# ============================================================================
# Canonical config — score-time defense-in-depth
#
# These tests validate that a tampered or fabricated provider_info dict
# is caught at score time.  The primary canonical validation happens
# during load_state (see TestLoadTimeCanonicalValidation below).
# ============================================================================

class TestCanonicalConfigStrictBool:

    def _make_stub_result(self, **overrides):
        result = {
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None, "gm_classifier_probability": None,
            "gm_report_label": "x", "gm_score_definition": "x",
            "gm_threshold_source": "x", "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": False,
        }
        result.update(overrides)
        return result

    def test_canonical_gnr_string_false_rejected(self, provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: self._make_stub_result()
        provider_info["_canonical_kwargs"] = {
            "gm_use_gnr": "false", "gm_use_classifier": False}
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorStateValidationError,
                           match="must be bool"):
            score_image(provider_info, fake_image, record=record)

    def test_canonical_classifier_int_rejected(self, provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: self._make_stub_result()
        provider_info["_canonical_kwargs"] = {
            "gm_use_gnr": False, "gm_use_classifier": 0}
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        with pytest.raises(DetectorStateValidationError,
                           match="must be bool"):
            score_image(provider_info, fake_image, record=record)

    def test_canonical_gnr_true_fallback(self, provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: self._make_stub_result(
            gm_used_gnr=None)
        provider_info["_canonical_kwargs"] = {
            "gm_use_gnr": True, "gm_use_classifier": False}
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["gm_gnr_used"] is True

    def test_canonical_classifier_true_fallback(self, provider_info, fake_image):
        import raven.detectors.gm_detector as gm_mod
        stub = _StubExtractModule()
        stub.evaluate_image = lambda *a, **kw: self._make_stub_result(
            gm_used_classifier=None)
        provider_info["_canonical_kwargs"] = {
            "gm_use_gnr": False, "gm_use_classifier": True}
        provider_info["extract_module"] = stub

        record = _gm_record("0",
            watermark_target_sha256=provider_info["provider_target_hash"],
            watermark_mask_sha256=provider_info["provider_mask_hash"])
        score = score_image(provider_info, fake_image, record=record)
        assert score["gm_classifier_used"] is True


# ============================================================================
# Load-time canonical validation — malformed kwargs rejected BEFORE provider
# ============================================================================

class TestLoadTimeCanonicalValidation:

    def test_load_state_rejects_non_bool_gnr_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """gm_provider_kwargs returns gm_use_gnr='false' (string) →
        DetectorStateValidationError before provider construction."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MalformedGnrExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                kw["gm_use_gnr"] = "false"
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MalformedGnrExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_use_gnr.*must be bool"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_load_state_rejects_non_bool_classifier_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """gm_provider_kwargs returns gm_use_classifier=0 (int) →
        DetectorStateValidationError before provider construction."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MalformedClfExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                kw["gm_use_classifier"] = 0
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MalformedClfExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_use_classifier.*must be bool"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_uniform_create_bundle_true_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """All rows have gm_create_bundle=True (uniform, not mixed) →
        DetectorStateValidationError, provider never constructed."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class CreateBundleExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                kw["gm_create_bundle"] = True
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: CreateBundleExtract())
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_create_bundle"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_uniform_allow_in_memory_true_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """All rows have gm_allow_in_memory_state=True (uniform) →
        DetectorStateValidationError, provider never constructed."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MemoryStateExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                kw["gm_allow_in_memory_state"] = True
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MemoryStateExtract())
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_allow_in_memory_state"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_canonical_four_bool_valid_and_false_loads(
        self, mock_deps, bundle_dir):
        """All four canonical bool fields are valid False → load_state succeeds,
        exactly one provider constructed."""
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider
        try:
            records = [
                _gm_record("0", gm_bundle_dir=str(bundle_dir)),
                _gm_record("1", gm_bundle_dir=str(bundle_dir)),
            ]
            info = load_state(records, "cpu")
            assert info["provider"] is not None
            assert provider_calls[0] == 1
            ck = info["_canonical_kwargs"]
            assert ck["gm_use_gnr"] is False
            assert ck["gm_use_classifier"] is False
            assert ck["gm_create_bundle"] is False
            assert ck["gm_allow_in_memory_state"] is False
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_missing_gm_use_gnr_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """gm_provider_kwargs omits gm_use_gnr → DetectorStateValidationError."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MissingGnrExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                del kw["gm_use_gnr"]
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MissingGnrExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_use_gnr.*missing"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_missing_gm_use_classifier_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """gm_provider_kwargs omits gm_use_classifier → DetectorStateValidationError."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MissingClfExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                del kw["gm_use_classifier"]
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MissingClfExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="gm_use_classifier.*missing"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_non_dict_provider_kwargs_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch):
        """gm_provider_kwargs returns a list → DetectorStateValidationError
        before provider construction."""
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class NonDictExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                return ["not", "a", "dict"]

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: NonDictExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="must be a dict"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider


# ============================================================================
# Canonical key presence — all _CANONICAL_KWARGS_FIELDS must be present
# ============================================================================

_MISSING_KEY_CASES = [
    "gm_torch_dtype",
    "gm_channel_copy",
    "gm_gnr_path",
    "model_revision",
    "w_pattern",
]


class TestCanonicalKeyPresence:

    @pytest.mark.parametrize("missing_key", _MISSING_KEY_CASES)
    def test_missing_canonical_key_rejected_before_provider(
        self, mock_deps, bundle_dir, monkeypatch, missing_key):
        import raven.detectors.gm_detector as gm_mod
        provider_calls = [0]

        class CountingProvider(StubGmProvider):
            def __init__(self, **kw):
                provider_calls[0] += 1
                super().__init__(**kw)
        fake_gm = sys.modules.get("eval_bench_wm.utils.wm.gm_provider")
        fake_gm.GmProvider = CountingProvider

        class MissingKeyExtract(_StubExtractModule):
            def gm_provider_kwargs(self, row, identifier):
                kw = super().gm_provider_kwargs(row, identifier)
                del kw[missing_key]
                return kw

        monkeypatch.setattr(gm_mod, "_get_extract_module",
                            lambda: MissingKeyExtract())
        try:
            records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
            with pytest.raises(DetectorStateValidationError,
                               match="missing required keys"):
                load_state(records, "cpu")
            assert provider_calls[0] == 0
        finally:
            fake_gm.GmProvider = StubGmProvider

    def test_explicit_none_allowed_for_nullable_fields(
        self, mock_deps, bundle_dir):
        """gm_gnr_path and gm_classifier_path may be explicitly None.
        gm_watermark_bits_seed is NOT nullable (comes from manifest)."""
        records = [_gm_record("0", gm_bundle_dir=str(bundle_dir))]
        info = load_state(records, "cpu")
        ck = info["_canonical_kwargs"]
        assert ck["gm_gnr_path"] is None
        assert ck["gm_classifier_path"] is None
        assert ck["gm_watermark_bits_seed"] is not None
        assert info["provider"] is not None
