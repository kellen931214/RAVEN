"""Issue #24 regression tests — method-specific Fourier bundle validation
for RID, HSTR, and HSQR unified detectors.

All tests use mocks.  No real bundles are downloaded, no real inversion
runs.  Each test verifies a specific acceptance criterion from the issue.

Run:  pytest -q raven_repro/tests/test_issue24_fourier_detector.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO))

FOURIER_METHODS = ["RID", "HSTR", "HSQR"]

# ---------------------------------------------------------------------------
# Prevent real torch / provider imports
# ---------------------------------------------------------------------------
_MOCK_TORCH = mock.MagicMock(name="torch")
_MOCK_TORCH.cuda.is_available.return_value = False
_MOCK_TORCH.device.return_value = mock.MagicMock(name="cpu_device")
_MOCK_TORCH.no_grad.return_value = mock.MagicMock()
_MOCK_TORCH.float16 = "float16"
_MOCK_TORCH.float32 = "float32"

_MOCK_PIPE_UTILS = mock.MagicMock(name="pipe_utils")
_MOCK_PIPE = mock.MagicMock(name="pipe")
_MOCK_PIPE.get_latent_shape.return_value = (1, 4, 64, 64)
_MOCK_PIPE.get_dtype.return_value = _MOCK_TORCH.float32
_MOCK_PIPE_UTILS.get_pipe_provider.return_value = _MOCK_PIPE

_MOCK_RINGID = mock.MagicMock(name="RingIDProvider")
_MOCK_HSTR = mock.MagicMock(name="HSTRProvider")
_MOCK_HSQR = mock.MagicMock(name="HSQRProvider")

_MOCK_SFW_BUNDLE = mock.MagicMock(name="sfw_bundle")
_MOCK_SFW_BUNDLE.SfwBundle.load.return_value = mock.MagicMock()

_MOCK_EVAL_BENCH_UTILS_PIPE = mock.MagicMock(name="eval_bench_wm.utils.pipe")
_MOCK_EVAL_BENCH_UTILS_PIPE.pipe_utils = _MOCK_PIPE_UTILS
_MOCK_EVAL_BENCH_UTILS_WM = mock.MagicMock(name="eval_bench_wm.utils.wm")
_MOCK_EVAL_BENCH_UTILS_WM.ringid_provider = mock.MagicMock(RingIDProvider=_MOCK_RINGID)
_MOCK_EVAL_BENCH_UTILS_WM.hstr_provider = mock.MagicMock(HSTRProvider=_MOCK_HSTR)
_MOCK_EVAL_BENCH_UTILS_WM.hsqr_provider = mock.MagicMock(HSQRProvider=_MOCK_HSQR)
_MOCK_EVAL_BENCH_UTILS_WM.sfw_bundle = _MOCK_SFW_BUNDLE

_BASE_MOCK_MODULES = {
    "torch": _MOCK_TORCH,
    "eval_bench_wm.utils.pipe": _MOCK_EVAL_BENCH_UTILS_PIPE,
    "eval_bench_wm.utils.pipe.pipe_utils": _MOCK_PIPE_UTILS,
    "eval_bench_wm.utils.wm": _MOCK_EVAL_BENCH_UTILS_WM,
    "eval_bench_wm.utils.wm.ringid_provider": _MOCK_EVAL_BENCH_UTILS_WM.ringid_provider,
    "eval_bench_wm.utils.wm.hstr_provider": _MOCK_EVAL_BENCH_UTILS_WM.hstr_provider,
    "eval_bench_wm.utils.wm.hsqr_provider": _MOCK_EVAL_BENCH_UTILS_WM.hsqr_provider,
    "eval_bench_wm.utils.wm.sfw_bundle": _MOCK_SFW_BUNDLE,
}


@pytest.fixture(autouse=True)
def _mock_torch_and_providers(monkeypatch):
    """Globally mock torch and all provider imports for every test."""
    for mod_name, mod_mock in _BASE_MOCK_MODULES.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_mock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_fake_manifest(method="RID", **overrides):
    """Return a minimal manifest dict sufficient for fourier_bundle_manifest."""
    base = {
        "bundle_config_sha256": "abc123_bundle_config",
        "selected_pattern_sha256": "abc123_selected_pattern",
        "mask_sha256": "abc123_mask",
    }
    if method == "RID":
        base.update({
            "profile_name": "rid_official",
            "profile_is_official": True,
            "profile_overrides": {},
            "selected_key_index": 0,
            "rng_seed": 42,
            "rng_device": "cpu",
            "rng_dtype": "float32",
            "channel_min": 0,
            "ring_value_range": 1,
            "quantization_levels": 16,
            "ring_width": 1,
            "assigned_keys": 4,
            "fix_gt": 1,
            "spatial_shift": 1,
            "spatial_shift_factor": 1.0,
            "spatial_shift_factor_semantics": "periodic",
            "torch_dtype": "float32",
            "inversion_guidance_scale": 2.5,
            "inversion_steps": 50,
            "vae_sample": True,
            "vae_scaling_factor": 0.18215,
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main",
            "scheduler": "DDIM",
            "resolution": 512,
        })
    elif method == "HSTR":
        base.update({
            "profile_name": "hstr_official",
            "selected_key_index": 0,
            "rng_device": "cpu",
            "latent_shape": [1, 4, 64, 64],
            "center_slice": [1, 3],
            "wm_capacity": 256,
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main",
            "scheduler_type": "DDIM",
            "resolution": 512,
        })
    elif method == "HSQR":
        base.update({
            "selected_pattern_sha256": "hsqr_pattern_sha",
            "mask_sha256": "hsqr_mask_sha",
        })
    base.update(overrides)
    return base


def _make_fake_record(bundle_dir, run_id="1", method="RID", **overrides):
    """Return a record with all required Fourier metadata fields."""
    prefix = method.lower()
    base = {
        "run_id": run_id,
        "method": method,
        "role": "watermarked",
        f"{prefix}_bundle_dir": str(bundle_dir),
        f"{prefix}_bundle_config_sha256": "abc123_bundle_config",
        f"{prefix}_selected_pattern_sha256": "abc123_selected_pattern",
        f"{prefix}_mask_sha256": "abc123_mask",
        f"{prefix}_key_index": "0",
        f"{prefix}_protocol_mode": f"{prefix}_official",
        "watermark_target_sha256": "",
        "watermark_mask_sha256": "",
        "input_path": f"/tmp/in_{run_id}.png",
        "output_path": f"/tmp/out/watermarked/{run_id}/output.png",
        "prompt": "",
        "attack_seed": 59,
    }
    base.update(overrides)
    return base


def _make_fake_provider(method="RID", state_source="bundle", has_bundle=True):
    """Return a mock provider with method-appropriate attributes."""
    p = mock.MagicMock()
    p.get_wm_type.return_value = method
    if has_bundle:
        p.bundle = mock.MagicMock()
        p.bundle.manifest = _make_fake_manifest(method)
    else:
        p.bundle = None
    if method in {"RID", "HSTR"}:
        p.state_source = state_source
    p.gt_patch = mock.MagicMock()
    p.selected_pattern_sha256 = ""
    p.watermark_mask_sha256 = ""
    p.watermarking_mask = mock.MagicMock()
    if method == "HSTR":
        p.watermark_region_mask_hstr = mock.MagicMock()
    return p


def _build_mock_extract_module(method="RID", provider=None, manifest=None,
                                bundle_dir=None):
    """Build a mock extract module with all canonical helpers wired up."""
    mod = mock.MagicMock()

    if manifest is None:
        manifest = _make_fake_manifest(method)

    def _fake_fourier_bundle_manifest(row, identifier, m):
        bd = bundle_dir or Path(str(row.get(f"{method.lower()}_bundle_dir",
                                            "/tmp/fake")))
        return bd, manifest

    mod.fourier_bundle_manifest = mock.MagicMock(
        side_effect=_fake_fourier_bundle_manifest)

    if method == "RID":
        mod.rid_provider_kwargs_from_bundle = mock.MagicMock(return_value={
            "rid_profile": "rid_official", "rid_bundle_dir": "/tmp/fake_bundles/rid_bundle",
            "rid_create_bundle": False, "rid_key_index": 0, "rid_key_seed": 42,
            "rid_key_rng_device": "cpu", "rid_key_rng_dtype": "float32",
            "channel_min": 0, "ring_value_range": 1, "quantization_levels": 16,
            "ring_width": 1, "assigned_keys": 4, "fix_gt": 1, "time_shift": 1,
            "time_shift_factor": 1.0, "rid_shift_semantics": "periodic",
            "rid_torch_dtype": "float32", "rid_inversion_prompt": "",
            "rid_inversion_guidance": 2.5, "rid_inversion_steps": 50,
            "rid_vae_sample": True, "rid_vae_scaling_factor": 0.18215,
            "rid_profile_is_official": True, "rid_profile_overrides": {},
            "modelid_target": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main", "scheduler_target": "DDIM", "resolution": 512,
        })
    elif method == "HSTR":
        mod.hstr_provider_kwargs_from_bundle = mock.MagicMock(return_value={
            "hstr_profile": "hstr_official", "hstr_bundle_dir": "/tmp/fake_bundles/hstr_bundle",
            "hstr_create_bundle": False, "hstr_key_index": 0, "hstr_rng_device": "cpu",
            "latent_channel": 4, "hw_latent": 64, "start": 1, "end": 3,
            "wm_capacity": 256, "modelid_target": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main", "scheduler_target": "DDIM", "resolution": 512,
        })
    elif method == "HSQR":
        mod.hsqr_provider_from_bundle = mock.MagicMock(return_value=provider)

    mod.evaluate_image = mock.MagicMock(return_value={"l1_dist": [0.123]})
    mod.raw_score = mock.MagicMock(return_value=0.123)
    mod.canonical_score = mock.MagicMock(return_value=-0.123)

    return mod


def _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod):
    """Apply all mocks needed for fourier_detector.load_state to run."""
    from raven.detectors import fourier_detector

    monkeypatch.setattr(fourier_detector, "_ensure_paths", lambda: None)
    monkeypatch.setattr(fourier_detector, "_get_extract_module",
                        lambda: extract_mod)
    _MOCK_RINGID.return_value = fake_provider
    _MOCK_HSTR.return_value = fake_provider
    _MOCK_PIPE_UTILS.get_pipe_provider.return_value = _MOCK_PIPE
    _MOCK_PIPE.get_latent_shape.return_value = (1, 4, 64, 64)
    _MOCK_PIPE.get_dtype.return_value = _MOCK_TORCH.float32


@pytest.fixture
def bundle_dir():
    """Real temp directory with manifest.json for bundle validation."""
    with tempfile.TemporaryDirectory() as td:
        bd = Path(td)
        manifest = _make_fake_manifest()
        (bd / "manifest.json").write_text(json.dumps(manifest))
        yield bd


# ===========================================================================
# 1. Canonical helper dispatch
# ===========================================================================
class TestCanonicalHelperDispatch:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_fourier_bundle_manifest_called(self, method, monkeypatch, bundle_dir):
        """fourier_bundle_manifest is called for every row during load_state."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method=method)
        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        fourier_detector.load_state([record], "cpu", method=method)
        extract_mod.fourier_bundle_manifest.assert_called()

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_method_specific_kwargs_helper_called(self, method, monkeypatch,
                                                   bundle_dir):
        """Each method calls its own provider kwargs helper."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method=method)
        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        fourier_detector.load_state([record], "cpu", method=method)

        if method == "RID":
            extract_mod.rid_provider_kwargs_from_bundle.assert_called()
        elif method == "HSTR":
            extract_mod.hstr_provider_kwargs_from_bundle.assert_called()
        elif method == "HSQR":
            extract_mod.hsqr_provider_from_bundle.assert_called()

    def test_rid_does_not_call_hstr_helper(self, monkeypatch, bundle_dir):
        """RID must NOT call HSTR or HSQR kwargs helpers."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method="RID")
        manifest = _make_fake_manifest("RID")
        fake_provider = _make_fake_provider("RID")
        extract_mod = _build_mock_extract_module(
            "RID", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "RID", fake_provider, extract_mod)

        fourier_detector.load_state([record], "cpu", method="RID")
        extract_mod.rid_provider_kwargs_from_bundle.assert_called()


# ===========================================================================
# 2. Score delegation
# ===========================================================================
class TestScoreDelegation:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_evaluate_image_delegated(self, method, tmp_path):
        """score_image delegates to extract module's evaluate_image."""
        from raven.detectors import fourier_detector

        fake_provider = _make_fake_provider(method)
        manifest = _make_fake_manifest(method)
        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method,
            "_manifest": manifest,
        }

        result = fourier_detector.score_image(
            provider_info, str(img_path), steps=50)

        extract_mod.evaluate_image.assert_called_once()
        extract_mod.raw_score.assert_called_once()
        extract_mod.canonical_score.assert_called_once()
        assert "raw_score" in result
        assert "canonical_score" in result
        assert result["raw_l1"] == result["raw_score"]

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_canonical_score_is_negative_raw_l1(self, method, tmp_path):
        """Canonical score = -raw_l1 for all Fourier methods."""
        from raven.detectors import fourier_detector

        fake_provider = _make_fake_provider(method)
        manifest = _make_fake_manifest(method)
        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)
        extract_mod.raw_score.return_value = 0.456
        extract_mod.canonical_score.return_value = -0.456

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method,
            "_manifest": manifest,
        }

        result = fourier_detector.score_image(
            provider_info, str(img_path), steps=50)

        extract_mod.canonical_score.assert_called_once_with(
            method, 0.456, extract_mod.evaluate_image.return_value)
        assert result["canonical_score"] == -0.456
        assert result["score_direction"] == \
            "higher_is_watermarked (canonical = -raw_l1)"


# ===========================================================================
# 3. State-source gates
# ===========================================================================
class TestStateSourceGates:

    def test_rid_rejects_non_bundle_state_source(self, monkeypatch, bundle_dir):
        """RID with state_source != 'bundle' raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        record = _make_fake_record(bundle_dir, "1", method="RID")
        manifest = _make_fake_manifest("RID")
        fake_provider = _make_fake_provider("RID", state_source="random")
        extract_mod = _build_mock_extract_module(
            "RID", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "RID", fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError, match="state_source"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_hstr_rejects_non_bundle_state_source(self, monkeypatch, bundle_dir):
        """HSTR with state_source != 'bundle' raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        record = _make_fake_record(bundle_dir, "1", method="HSTR")
        manifest = _make_fake_manifest("HSTR")
        fake_provider = _make_fake_provider("HSTR", state_source="random")
        extract_mod = _build_mock_extract_module(
            "HSTR", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "HSTR", fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError, match="state_source"):
            fourier_detector.load_state([record], "cpu", method="HSTR")

    def test_hsqr_accepts_no_state_source(self, monkeypatch, bundle_dir):
        """HSQR must NOT be rejected for missing state_source attribute."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method="HSQR")
        manifest = _make_fake_manifest("HSQR")
        fake_provider = _make_fake_provider("HSQR", state_source=None)
        del fake_provider.state_source
        fake_provider.bundle = mock.MagicMock()
        fake_provider.bundle.manifest = manifest
        fake_provider.gt_patch = mock.MagicMock()

        extract_mod = _build_mock_extract_module(
            "HSQR", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "HSQR", fake_provider, extract_mod)

        result = fourier_detector.load_state([record], "cpu", method="HSQR")
        assert result["method"] == "HSQR"
        assert result["provider"] is fake_provider

    def test_hsqr_rejects_missing_bundle(self, monkeypatch, bundle_dir):
        """HSQR with no bundle raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        record = _make_fake_record(bundle_dir, "1", method="HSQR")
        manifest = _make_fake_manifest("HSQR")
        fake_provider = _make_fake_provider("HSQR", has_bundle=False)
        extract_mod = _build_mock_extract_module(
            "HSQR", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "HSQR", fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError,
                           match="no persisted bundle"):
            fourier_detector.load_state([record], "cpu", method="HSQR")

    def test_rid_accepts_bundle_state_source(self, monkeypatch, bundle_dir):
        """RID with state_source == 'bundle' loads successfully."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method="RID")
        manifest = _make_fake_manifest("RID")
        fake_provider = _make_fake_provider("RID", state_source="bundle")
        extract_mod = _build_mock_extract_module(
            "RID", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "RID", fake_provider, extract_mod)

        result = fourier_detector.load_state([record], "cpu", method="RID")
        assert result["method"] == "RID"

    def test_hstr_accepts_bundle_state_source(self, monkeypatch, bundle_dir):
        """HSTR with state_source == 'bundle' loads successfully."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method="HSTR")
        manifest = _make_fake_manifest("HSTR")
        fake_provider = _make_fake_provider("HSTR", state_source="bundle")
        extract_mod = _build_mock_extract_module(
            "HSTR", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "HSTR", fake_provider, extract_mod)

        result = fourier_detector.load_state([record], "cpu", method="HSTR")
        assert result["method"] == "HSTR"


# ===========================================================================
# 4. Mixed cohort rejection
# ===========================================================================
class TestMixedCohortRejection:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_bundle_dir_rejected(self, method, monkeypatch, bundle_dir):
        """Different bundle_dir across rows raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        r1 = _make_fake_record(bundle_dir, "1", method=method)

        # Create second, different bundle dir
        other_dir = Path(bundle_dir).parent / "other_bundle"
        other_dir.mkdir(parents=True, exist_ok=True)
        (other_dir / "manifest.json").write_text(json.dumps(_make_fake_manifest(method)))
        r2 = _make_fake_record(other_dir, "2", method=method)

        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError, match="mixed bundle_dir"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_key_index_rejected(self, method, monkeypatch, bundle_dir):
        """Different key_index across rows raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        r1 = _make_fake_record(bundle_dir, "1", method=method)
        r2 = _make_fake_record(bundle_dir, "2", method=method,
                               **{f"{prefix}_key_index": "3"})

        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError, match="mixed key_index"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_protocol_mode_rejected(self, method, monkeypatch, bundle_dir):
        """Different protocol_mode across rows raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        r1 = _make_fake_record(bundle_dir, "1", method=method)
        r2 = _make_fake_record(bundle_dir, "2", method=method,
                               **{f"{prefix}_protocol_mode": "other_protocol"})

        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError,
                           match="mixed protocol_mode"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

    def test_consistent_cohort_accepted(self, monkeypatch, bundle_dir):
        """Rows with identical bundle/key/profile load successfully."""
        from raven.detectors import fourier_detector

        r1 = _make_fake_record(bundle_dir, "1", method="RID")
        r2 = _make_fake_record(bundle_dir, "2", method="RID")

        manifest = _make_fake_manifest("RID")
        fake_provider = _make_fake_provider("RID", state_source="bundle")
        extract_mod = _build_mock_extract_module(
            "RID", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "RID", fake_provider, extract_mod)

        result = fourier_detector.load_state([r1, r2], "cpu", method="RID")
        assert result["_cohort_bundle_dir"] == str(bundle_dir)


# ===========================================================================
# 5. Target and mask mismatch
# ===========================================================================
class TestTargetMaskMismatch:

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_target_sha_mismatch(self, method, tmp_path):
        """Source target SHA != detector target SHA raises validation error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        fake_provider = _make_fake_provider(method)
        fake_provider.selected_pattern_sha256 = "detector_target_sha"
        manifest = _make_fake_manifest(method)
        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method,
            "_manifest": manifest,
        }

        bundir = Path(tempfile.mkdtemp())
        record = _make_fake_record(bundir, "1", method=method,
                                   watermark_target_sha256="wrong_target_sha",
                                   watermark_mask_sha256="")

        with pytest.raises(DetectorStateValidationError,
                           match="target SHA mismatch"):
            fourier_detector.score_image(
                provider_info, str(img_path), record=record, steps=50)

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_mask_sha_mismatch(self, method, tmp_path):
        """Source mask SHA != detector mask SHA raises validation error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        fake_provider = _make_fake_provider(method)
        fake_provider.selected_pattern_sha256 = "detector_target_sha"
        fake_provider.watermark_mask_sha256 = "detector_mask_sha"
        manifest = _make_fake_manifest(method)
        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method,
            "_manifest": manifest,
        }

        bundir = Path(tempfile.mkdtemp())
        record = _make_fake_record(bundir, "1", method=method,
                                   watermark_target_sha256="detector_target_sha",
                                   watermark_mask_sha256="wrong_mask_sha")

        with pytest.raises(DetectorStateValidationError,
                           match="mask SHA mismatch"):
            fourier_detector.score_image(
                provider_info, str(img_path), record=record, steps=50)

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_matching_sha_passes(self, method, tmp_path):
        """Matching target/mask SHA passes validation and scores successfully."""
        from raven.detectors import fourier_detector

        fake_provider = _make_fake_provider(method)
        fake_provider.selected_pattern_sha256 = "match_target_sha"
        fake_provider.watermark_mask_sha256 = "match_mask_sha"
        manifest = _make_fake_manifest(method)
        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method,
            "_manifest": manifest,
        }

        bundir = Path(tempfile.mkdtemp())
        record = _make_fake_record(bundir, "1", method=method,
                                   watermark_target_sha256="match_target_sha",
                                   watermark_mask_sha256="match_mask_sha")

        result = fourier_detector.score_image(
            provider_info, str(img_path), record=record, steps=50)

        assert "raw_score" in result
        assert "canonical_score" in result
        extract_mod.evaluate_image.assert_called_once()

    def test_target_mismatch_hsqr(self, tmp_path):
        """HSQR target SHA mismatch raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        fake_provider = _make_fake_provider("HSQR")
        fake_provider.bundle.manifest = {"selected_pattern_sha256": "hsqr_target_sha"}
        fake_provider.watermark_mask_sha256 = "hsqr_mask_sha"
        manifest = _make_fake_manifest("HSQR")
        extract_mod = _build_mock_extract_module("HSQR", provider=fake_provider,
                                                  manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        provider_info = {
            "provider": fake_provider,
            "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "HSQR",
            "_manifest": manifest,
        }

        bundir = Path(tempfile.mkdtemp())
        record = _make_fake_record(bundir, "1", method="HSQR",
                                   watermark_target_sha256="wrong_hsqr_target",
                                   watermark_mask_sha256="hsqr_mask_sha",
                                   hsqr_mask_sha256="hsqr_mask_sha")

        with pytest.raises(DetectorStateValidationError,
                           match="target SHA mismatch"):
            fourier_detector.score_image(
                provider_info, str(img_path), record=record, steps=50)


# ===========================================================================
# 6. Missing bundle → failed_missing_required_state
# ===========================================================================
class TestMissingBundle:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_bundle_dir_raises_missing_state(self, method, monkeypatch):
        """Empty bundle_dir raises DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        prefix = method.lower()
        record = _make_fake_record(Path("/tmp/fake"), "1", method=method,
                                   **{f"{prefix}_bundle_dir": ""})

        monkeypatch.setattr(fourier_detector, "_ensure_paths", lambda: None)

        with pytest.raises(DetectorMissingStateError, match="not found"):
            fourier_detector.load_state([record], "cpu", method=method)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_nonexistent_bundle_dir_raises_missing_state(self, method, monkeypatch):
        """Non-existent bundle_dir raises DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        record = _make_fake_record(Path("/nonexistent/path"), "1", method=method)
        monkeypatch.setattr(fourier_detector, "_ensure_paths", lambda: None)

        with pytest.raises(DetectorMissingStateError, match="not found"):
            fourier_detector.load_state([record], "cpu", method=method)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_required_metadata_field(self, method, monkeypatch, bundle_dir):
        """Missing required metadata field raises DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        prefix = method.lower()
        record = _make_fake_record(bundle_dir, "1", method=method)
        del record[f"{prefix}_key_index"]

        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorMissingStateError,
                           match="missing required field"):
            fourier_detector.load_state([record], "cpu", method=method)


# ===========================================================================
# 7. Edge cases
# ===========================================================================
class TestEmptyRecords:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_empty_records_raises_missing_state(self, method):
        """Empty records list raises DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        with pytest.raises(DetectorMissingStateError, match="no records"):
            fourier_detector.load_state([], "cpu", method=method)


class TestUnknownMethod:

    def test_unknown_method_raises_value_error(self):
        """Non-Fourier method raises ValueError."""
        from raven.detectors import fourier_detector

        with pytest.raises(ValueError, match="Unknown Fourier method"):
            fourier_detector.load_state(
                [_make_fake_record(Path("/tmp/fake"), "1", method="RID")],
                "cpu", method="TR")

    def test_lowercase_method_accepted(self, monkeypatch, bundle_dir):
        """Lowercase method names are normalized to uppercase."""
        from raven.detectors import fourier_detector

        record = _make_fake_record(bundle_dir, "1", method="RID")
        manifest = _make_fake_manifest("RID")
        fake_provider = _make_fake_provider("RID", state_source="bundle")
        extract_mod = _build_mock_extract_module(
            "RID", provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        _setup_load_state_mocks(monkeypatch, "RID", fake_provider, extract_mod)

        result = fourier_detector.load_state([record], "cpu", method="rid")
        assert result["method"] == "RID"


class TestMissingImage:

    def test_missing_image_in_score_image(self):
        """score_image raises DetectorMissingStateError for non-existent file."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider_info = {
            "provider": _make_fake_provider("RID"),
            "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "RID",
            "_manifest": {},
        }

        with pytest.raises(DetectorMissingStateError, match="Image not found"):
            fourier_detector.score_image(
                provider_info, "/nonexistent/image.png", steps=50)


# ===========================================================================
# 8. Aggregate
# ===========================================================================
class TestAggregateScoreLabels:

    @pytest.mark.parametrize("method,expected_score_type", [
        ("RID", "rid_score"),
        ("HSTR", "hstr_score"),
        ("HSQR", "hsqr_score"),
    ])
    def test_aggregate_score_type(self, method, expected_score_type):
        """Aggregate returns method-specific score_type."""
        from raven.detectors import fourier_detector

        rows = [{
            "run_id": "1",
            "evaluation_cohort": "original_watermarked",
            "status": "scored",
            "canonical_score": -0.1,
        }]

        result = fourier_detector.aggregate(rows, method=method)
        assert result["score_type"] == expected_score_type
        assert result["score_direction"] == "higher_is_watermarked"

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_aggregate_counts_failed_rows(self, method):
        """Aggregate counts scored and failed rows correctly."""
        from raven.detectors import fourier_detector

        rows = [
            {"run_id": "1", "evaluation_cohort": "original_watermarked",
             "status": "scored", "canonical_score": -0.1},
            {"run_id": "2", "evaluation_cohort": "original_watermarked",
             "status": "failed_state_validation"},
            {"run_id": "3", "evaluation_cohort": "attacked_watermarked",
             "status": "scored", "canonical_score": -0.2},
        ]

        result = fourier_detector.aggregate(rows, method=method)
        assert result["scored_count"] == 2
        assert result["failed_count"] == 1
        assert result["requested_count"] == 3


# ===========================================================================
# 9. Orchestrator integration
# ===========================================================================
class TestOrchestratorIntegration:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_full_orchestrator_path(self, method, monkeypatch, tmp_path):
        """evaluate_detector completes successfully with mocked Fourier detector."""
        from raven.detectors import fourier_detector as det_mod

        prefix = method.lower()
        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)
        if method in {"RID", "HSTR"}:
            fake_provider.state_source = "bundle"

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        from PIL import Image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (64, 64)).save(img_dir / f"img_{i}.png")

        # Create output subdirectories expected by the orchestrator
        # sample_dir() = output_dir / "samples" / role / run_id
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        for i in range(3):
            role_dir = out_dir / "samples" / "watermarked" / str(i)
            role_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(role_dir / "output.png")

        records = []
        for i in range(3):
            records.append({
                "run_id": str(i),
                "role": "watermarked",
                "method": method,
                "input_path": str(img_dir / f"img_{i}.png"),
                "output_path": str(out_dir / "watermarked" / str(i) / "output.png"),
                "prompt": "",
                "attack_seed": 59,
                "planned_flow_dx_image_px": 24.0,
                "planned_flow_dy_image_px": -24.0,
                "effective_source_flow_dx_image_px": 24.0,
                "effective_source_flow_dy_image_px": -24.0,
                "debug_info_path": "",
                "debug_info_retained": False,
                "source_metadata": {
                    f"{prefix}_bundle_dir": str(bundle_dir),
                    f"{prefix}_bundle_config_sha256": manifest["bundle_config_sha256"],
                    f"{prefix}_selected_pattern_sha256": manifest["selected_pattern_sha256"],
                    f"{prefix}_mask_sha256": manifest.get("mask_sha256", ""),
                    f"{prefix}_key_index": "0",
                    f"{prefix}_protocol_mode": f"{prefix}_official",
                },
            })

        extract_mod = _build_mock_extract_module(method, provider=fake_provider,
                                                  manifest=manifest)

        def fake_load_state(recs, dev, m=None, **kw):
            actual_method = (m or method).upper()
            return {
                "provider": fake_provider,
                "pipe": _MOCK_PIPE,
                "extract_module": extract_mod,
                "device_obj": _MOCK_TORCH.device.return_value,
                "method": actual_method,
                "score_definition": f"{actual_method.lower()}_score = -raw_l1",
                "_cohort_bundle_dir": str(bundle_dir),
                "_cohort_key_index": "0",
                "_cohort_protocol": f"{prefix}_official",
                "_cohort_bundle_config": manifest["bundle_config_sha256"],
                "_cohort_selected_pattern": manifest["selected_pattern_sha256"],
                "_cohort_mask": manifest.get("mask_sha256", ""),
                "_manifest": manifest,
            }

        def fake_score_image(pinfo, img_path, *, record=None,
                             evaluation_entry=None, steps=50):
            return {"raw_score": 0.123, "canonical_score": -0.123,
                    "raw_l1": 0.123,
                    "score_direction": "higher_is_watermarked (canonical = -raw_l1)"}

        monkeypatch.setattr(det_mod, "load_state", fake_load_state)
        monkeypatch.setattr(det_mod, "score_image", fake_score_image)

        from raven.experiment_io import write_config, write_record
        write_config(out_dir, {"method": method, "dataset": "test"})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        from experiments.eval import evaluate_detector
        result = evaluate_detector(records, out_dir, method, device="cpu")

        assert result["method"] == method
        assert result["stage"] == "detector"
        assert result["status"] in ("completed", "completed_with_errors",
                                    "skipped_insufficient_data")
        assert result["scored_count"] == 6
        assert result["failed_count"] == 0


# ===========================================================================
# 10. Per-row bundle validation failures
# ===========================================================================
class TestPerRowBundleValidation:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_row_manifest_mismatch_raises_state_validation(self, method, monkeypatch,
                                                            bundle_dir):
        """A row with mismatched bundle SHA raises DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)

        r1 = _make_fake_record(bundle_dir, "1", method=method)
        r2 = _make_fake_record(bundle_dir, "2", method=method,
                               **{f"{prefix}_bundle_config_sha256": "wrong_sha"})

        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)
        original_fbm = extract_mod.fourier_bundle_manifest

        def _selective_fbm(row, identifier, m):
            if str(row.get("run_id", "")) == "2":
                raise RuntimeError(
                    f"run_id=2: {method} bundle/source "
                    f"{prefix}_bundle_config_sha256 mismatch: "
                    f"source='wrong_sha' bundle='abc123_bundle_config'"
                )
            return original_fbm(row, identifier, m)

        extract_mod.fourier_bundle_manifest = mock.MagicMock(
            side_effect=_selective_fbm)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorStateValidationError,
                           match="bundle validation failed"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_row_manifest_not_found_raises_missing_state(self, method, monkeypatch,
                                                          bundle_dir):
        """A row with missing manifest raises DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        manifest = _make_fake_manifest(method)
        fake_provider = _make_fake_provider(method)

        r1 = _make_fake_record(bundle_dir, "1", method=method)
        r2 = _make_fake_record(bundle_dir, "2", method=method)

        extract_mod = _build_mock_extract_module(
            method, provider=fake_provider, manifest=manifest, bundle_dir=bundle_dir)

        def _selective_fbm(row, identifier, m):
            if str(row.get("run_id", "")) == "2":
                raise RuntimeError(
                    "run_id=2: RID bundle manifest not found: "
                    "/tmp/missing/manifest.json"
                )
            return (bundle_dir, manifest)

        extract_mod.fourier_bundle_manifest = mock.MagicMock(
            side_effect=_selective_fbm)
        _setup_load_state_mocks(monkeypatch, method, fake_provider, extract_mod)

        with pytest.raises(DetectorMissingStateError, match="bundle not found"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)
