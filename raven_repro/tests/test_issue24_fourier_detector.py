"""Issue #24 regression tests — method-specific Fourier bundle validation
for RID, HSTR, and HSQR unified detectors.

All tests use mocks.  No real bundles downloaded, no real inversion.
Integration tests use the REAL adapter (load_state + score_image) with
only external construction mocked.

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
# Global mocks — prevent real torch / provider imports
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
    for mod_name, mod_mock in _BASE_MOCK_MODULES.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_mock)


# ===========================================================================
# Canonical test data builders
# ===========================================================================
def _make_manifest(method="RID", **overrides):
    """Canonical manifest matching the provider schema for each method."""
    base = {
        "bundle_config_sha256": "abc123_bundle",
        "selected_pattern_sha256": "abc123_pattern",
        "mask_sha256": "abc123_mask",
        "selected_key_index": 0,
    }
    if method in {"RID", "HSTR"}:
        base.update({
            "profile_name": f"{method.lower()}_official",
            "rng_device": "cpu",
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main",
            "resolution": 512,
        })
    if method == "RID":
        base.update({
            "rng_seed": 42, "rng_dtype": "float32",
            "channel_min": 0, "ring_value_range": 1,
            "quantization_levels": 16, "ring_width": 1,
            "assigned_keys": 4, "fix_gt": 1,
            "spatial_shift": 1, "spatial_shift_factor": 1.0,
            "spatial_shift_factor_semantics": "periodic",
            "torch_dtype": "float32",
            "inversion_guidance_scale": 2.5, "inversion_steps": 50,
            "vae_sample": True, "vae_scaling_factor": 0.18215,
            "scheduler": "DDIM",
            "profile_is_official": True, "profile_overrides": {},
            "schema": "rid_bundle_v1",
        })
    elif method == "HSTR":
        base.update({
            "latent_shape": [1, 4, 64, 64],
            "center_slice": [1, 3], "wm_capacity": 256,
            "scheduler_type": "DDIM",
            "schema": "sfw_bundle_v1",
            "methods": ["HSTR", "HSQR"],
        })
    elif method == "HSQR":
        base.update({
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "main",
            "resolution": 512,
            "scheduler_type": "DDIM",
            "schema": "sfw_bundle_v1",
            "methods": ["HSTR", "HSQR"],
        })
    base.update(overrides)
    return base


def _make_record(bundle_dir, run_id="1", method="RID", **overrides):
    """A record carrying all required Fourier metadata fields."""
    prefix = method.lower()
    base = {
        "run_id": run_id,
        "method": method,
        "role": "watermarked",
        f"{prefix}_bundle_dir": str(bundle_dir),
        f"{prefix}_bundle_config_sha256": "abc123_bundle",
        f"{prefix}_selected_pattern_sha256": "abc123_pattern",
        f"{prefix}_mask_sha256": "abc123_mask",
        f"{prefix}_key_index": "0",
        f"{prefix}_protocol_mode": f"{prefix}_official",
        "watermark_target_sha256": "provider_target_sha",
        "watermark_mask_sha256": "provider_mask_sha",
        "input_path": f"/tmp/in_{run_id}.png",
        "output_path": f"/tmp/out/watermarked/{run_id}/output.png",
        "prompt": "",
        "attack_seed": 59,
    }
    base.update(overrides)
    return base


def _make_provider(method="RID", state_source="bundle", has_bundle=True):
    """Mock provider with attributes needed for validation."""
    p = mock.MagicMock()
    p.get_wm_type.return_value = method
    if has_bundle:
        p.bundle = mock.MagicMock()
        p.bundle.manifest = _make_manifest(method)
    else:
        p.bundle = None
    if method in {"RID", "HSTR"}:
        p.state_source = state_source
    p.gt_patch = mock.MagicMock()
    p.selected_pattern_sha256 = "provider_target_sha"
    p.watermark_mask_sha256 = "provider_mask_sha"
    p.watermarking_mask = mock.MagicMock()
    if method == "HSTR":
        p.watermark_region_mask_hstr = mock.MagicMock()
    return p


def _build_extract_module(method="RID", provider=None, manifest=None,
                           bundle_dir=None):
    """Mock extract_verification_scores module with all canonical helpers."""
    mod = mock.MagicMock()
    if manifest is None:
        manifest = _make_manifest(method)

    def _fake_fbm(row, identifier, m):
        bd = bundle_dir or Path(str(row.get(f"{method.lower()}_bundle_dir", "/tmp")))
        return bd, manifest

    mod.fourier_bundle_manifest = mock.MagicMock(side_effect=_fake_fbm)

    mod.rid_provider_kwargs_from_bundle = mock.MagicMock(return_value={
        "rid_profile": "rid_official", "rid_bundle_dir": "/tmp/bundles/rid",
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

    mod.hstr_provider_kwargs_from_bundle = mock.MagicMock(return_value={
        "hstr_profile": "hstr_official", "hstr_bundle_dir": "/tmp/bundles/hstr",
        "hstr_create_bundle": False, "hstr_key_index": 0, "hstr_rng_device": "cpu",
        "latent_channel": 4, "hw_latent": 64, "start": 1, "end": 3,
        "wm_capacity": 256, "modelid_target": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "main", "scheduler_target": "DDIM", "resolution": 512,
    })

    mod.hsqr_provider_from_bundle = mock.MagicMock(return_value=provider)

    mod.evaluate_image = mock.MagicMock(return_value={"l1_dist": [0.123]})
    mod.raw_score = mock.MagicMock(return_value=0.123)
    mod.canonical_score = mock.MagicMock(return_value=-0.123)

    return mod


def _setup_adapter_mocks(monkeypatch, method, provider, extract_mod):
    """Wire up all mocks so the REAL fourier_detector.load_state runs."""
    from raven.detectors import fourier_detector

    monkeypatch.setattr(fourier_detector, "_ensure_paths", lambda: None)
    monkeypatch.setattr(fourier_detector, "_get_extract_module",
                        lambda: extract_mod)
    _MOCK_RINGID.return_value = provider
    _MOCK_HSTR.return_value = provider
    _MOCK_PIPE_UTILS.get_pipe_provider.return_value = _MOCK_PIPE
    _MOCK_PIPE.get_latent_shape.return_value = (1, 4, 64, 64)
    _MOCK_PIPE.get_dtype.return_value = _MOCK_TORCH.float32


@pytest.fixture
def bundle_dir():
    """Real temp dir with manifest.json."""
    with tempfile.TemporaryDirectory() as td:
        bd = Path(td)
        manifest = _make_manifest()
        (bd / "manifest.json").write_text(json.dumps(manifest))
        yield bd


# ===========================================================================
# 1. Target/mask must be required provenance (fail-closed)
# ===========================================================================
class TestTargetMaskFailClosed:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_source_target_is_missing_state(self, method, tmp_path,
                                                     bundle_dir):
        """Empty watermark_target_sha256 → DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(mock.MagicMock(), method, provider, extract_mod)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="",
                              watermark_mask_sha256="provider_mask_sha",
                              **{f"{method.lower()}_bundle_config_sha256": "abc123_bundle",
                                 f"{method.lower()}_selected_pattern_sha256": "abc123_pattern",
                                 f"{method.lower()}_mask_sha256": "abc123_mask"})

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }

        with pytest.raises(DetectorMissingStateError,
                           match="missing watermark_target_sha256"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_source_mask_is_missing_state(self, method, tmp_path,
                                                   bundle_dir):
        """Empty watermark_mask_sha256 → DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(mock.MagicMock(), method, provider, extract_mod)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="")

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }

        with pytest.raises(DetectorMissingStateError,
                           match="missing watermark_mask_sha256"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    def test_target_mismatch_raises_state_validation(self, tmp_path, bundle_dir):
        """Source target != detector target → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider("RID")
        provider.selected_pattern_sha256 = "real_detector_sha"
        manifest = _make_manifest("RID")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "RID", "_manifest": manifest,
        }

        record = _make_record(bundle_dir, "1", method="RID",
                              watermark_target_sha256="wrong_sha",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorStateValidationError,
                           match="target SHA mismatch"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    def test_mask_mismatch_raises_state_validation(self, tmp_path, bundle_dir):
        """Source mask != detector mask → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider("RID")
        provider.selected_pattern_sha256 = "real_detector_sha"
        provider.watermark_mask_sha256 = "real_mask_sha"
        manifest = _make_manifest("RID")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "RID", "_manifest": manifest,
        }

        record = _make_record(bundle_dir, "1", method="RID",
                              watermark_target_sha256="real_detector_sha",
                              watermark_mask_sha256="wrong_mask")

        with pytest.raises(DetectorStateValidationError,
                           match="mask SHA mismatch"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    def test_detector_cannot_derive_target(self, tmp_path, bundle_dir, monkeypatch):
        """Detector with no target identity → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider("RID")
        provider.selected_pattern_sha256 = ""
        provider.gt_patch = None
        manifest = _make_manifest("RID", selected_pattern_sha256="")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "RID", "_manifest": manifest,
        }

        record = _make_record(bundle_dir, "1", method="RID",
                              watermark_target_sha256="some_target",
                              watermark_mask_sha256="provider_mask_sha")

        # Mock tensor_sha256 to return empty — can't derive identity
        with mock.patch("raven.pairing_provenance.tensor_sha256",
                        return_value=""):
            with pytest.raises(DetectorStateValidationError,
                               match="could not derive target identity"):
                fourier_detector.score_image(pinfo, str(img_path), record=record)


# ===========================================================================
# 2. HSQR mask cannot fallback to source record
# ===========================================================================
class TestHSQRMaskNoSourceFallback:

    def test_hsqr_mask_cannot_fallback_to_source_record(self, tmp_path,
                                                         bundle_dir):
        """HSQR mask from canonical sources only — never from record."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        # Build a provider with NO way to derive mask identity.
        # Use spec=[] to prevent auto-creation of attributes.
        provider = mock.MagicMock(spec=["get_wm_type", "bundle"])
        provider.get_wm_type.return_value = "HSQR"
        provider.bundle = mock.MagicMock()
        provider.bundle.manifest = {"selected_pattern_sha256": "hsqr_target"}
        # No watermark_mask_sha256, no watermarking_mask → cannot derive mask

        manifest = _make_manifest("HSQR", mask_sha256="")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "HSQR", "_manifest": manifest,
        }

        record = _make_record(bundle_dir, "1", method="HSQR",
                              watermark_target_sha256="hsqr_target",
                              watermark_mask_sha256="hsqr_mask_sha")

        with pytest.raises(DetectorStateValidationError,
                           match="could not derive mask identity"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    def test_hsqr_mask_from_provider_attribute_passes(self, tmp_path, bundle_dir):
        """HSQR mask from provider.watermark_mask_sha256 passes validation."""
        from raven.detectors import fourier_detector

        provider = _make_provider("HSQR")
        provider.watermark_mask_sha256 = "canonical_mask_sha"
        provider.bundle.manifest = {"selected_pattern_sha256": "canonical_target_sha"}
        manifest = _make_manifest("HSQR")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest)
        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "HSQR", "_manifest": manifest,
        }

        record = _make_record(bundle_dir, "1", method="HSQR",
                              watermark_target_sha256="canonical_target_sha",
                              watermark_mask_sha256="canonical_mask_sha")

        result = fourier_detector.score_image(pinfo, str(img_path), record=record)
        assert "canonical_score" in result


# ===========================================================================
# 3. Key index must match bundle/provider
# ===========================================================================
class TestKeyIndexMustMatchBundle:

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_key_index_must_match_bundle(self, method, monkeypatch, bundle_dir):
        """Row key_index != manifest selected_key_index → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        manifest = _make_manifest(method, selected_key_index=0)
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method,
                              **{f"{prefix}_key_index": "7",
                                 f"{prefix}_bundle_config_sha256": "abc123_bundle",
                                 f"{prefix}_selected_pattern_sha256": "abc123_pattern",
                                 f"{prefix}_mask_sha256": "abc123_mask"})

        with pytest.raises(DetectorStateValidationError, match="key_index"):
            fourier_detector.load_state([record], "cpu", method=method)

    def test_hsqr_key_index_must_match_bundle(self, monkeypatch, bundle_dir):
        """HSQR row key_index != manifest selected_key_index → error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSQR", selected_key_index=0)
        provider = _make_provider("HSQR")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR",
                              hsqr_key_index="7",
                              hsqr_bundle_config_sha256="abc123_bundle",
                              hsqr_selected_pattern_sha256="abc123_pattern",
                              hsqr_mask_sha256="abc123_mask")

        with pytest.raises(DetectorStateValidationError, match="key_index"):
            fourier_detector.load_state([record], "cpu", method="HSQR")

    def test_all_rows_consistently_wrong_key_index(self, monkeypatch, bundle_dir):
        """All rows agree on wrong key → still fails as DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", selected_key_index=0)
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method="RID", rid_key_index="7")
        r2 = _make_record(bundle_dir, "2", method="RID", rid_key_index="7")

        with pytest.raises(DetectorStateValidationError, match="key_index"):
            fourier_detector.load_state([r1, r2], "cpu", method="RID")


# ===========================================================================
# 4. Protocol/profile must match bundle
# ===========================================================================
class TestProtocolMustMatchBundle:

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_protocol_must_match_bundle_profile(self, method, monkeypatch,
                                                 bundle_dir):
        """Row protocol_mode != manifest profile_name → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        manifest = _make_manifest(method, profile_name=f"{prefix}_canonical")
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method,
                              **{f"{prefix}_protocol_mode": f"{prefix}_wrong"})

        with pytest.raises(DetectorStateValidationError, match="protocol"):
            fourier_detector.load_state([record], "cpu", method=method)

    def test_hsqr_protocol_must_match_bundle_profile(self, monkeypatch, bundle_dir):
        """HSQR protocol_mode != manifest profile_name → error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSQR", profile_name="hsqr_canonical")
        provider = _make_provider("HSQR")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR",
                              hsqr_protocol_mode="hsqr_wrong")

        with pytest.raises(DetectorStateValidationError, match="protocol"):
            fourier_detector.load_state([record], "cpu", method="HSQR")

    def test_all_rows_consistently_wrong_protocol(self, monkeypatch, bundle_dir):
        """All rows agree on wrong protocol → still fails."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", profile_name="rid_canonical")
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method="RID",
                          rid_protocol_mode="wrong_protocol")
        r2 = _make_record(bundle_dir, "2", method="RID",
                          rid_protocol_mode="wrong_protocol")

        with pytest.raises(DetectorStateValidationError, match="protocol"):
            fourier_detector.load_state([r1, r2], "cpu", method="RID")


# ===========================================================================
# 5. Method tag verification
# ===========================================================================
class TestMethodTagVerification:

    @pytest.mark.parametrize("eval_method,record_method", [
        ("RID", "HSTR"), ("HSTR", "RID"), ("HSQR", "RID"),
    ])
    def test_record_method_tag_must_match(self, eval_method, record_method,
                                           monkeypatch, bundle_dir):
        """Record method != evaluation method → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(eval_method)
        provider = _make_provider(eval_method)
        extract_mod = _build_extract_module(eval_method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, eval_method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=record_method)

        with pytest.raises(DetectorStateValidationError,
                           match="method tag"):
            fourier_detector.load_state([record], "cpu", method=eval_method)

    def test_manifest_schema_mismatch_rid_vs_sfw(self, monkeypatch, bundle_dir):
        """RID method with sfw_bundle schema → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", schema="sfw_bundle_v1",
                                  methods=["HSTR", "HSQR"])
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorStateValidationError,
                           match="not a RID bundle"):
            fourier_detector.load_state([record], "cpu", method="RID")


# ===========================================================================
# 6. Required metadata preflight (before canonical helper)
# ===========================================================================
class TestMetadataPreflight:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_metadata_checked_before_manifest_helper(self, method,
                                                              monkeypatch,
                                                              bundle_dir):
        """Missing required field raises DetectorMissingStateError BEFORE helper."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        prefix = method.lower()
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        # Remove a required field
        record = _make_record(bundle_dir, "1", method=method)
        del record[f"{prefix}_key_index"]

        with pytest.raises(DetectorMissingStateError,
                           match="is None"):
            fourier_detector.load_state([record], "cpu", method=method)

        # Canonical helper must NOT have been called
        extract_mod.fourier_bundle_manifest.assert_not_called()

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_empty_required_field_is_missing_state(self, method, monkeypatch,
                                                    bundle_dir):
        """Empty string for required field → DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        prefix = method.lower()
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method,
                              **{f"{prefix}_protocol_mode": "   "})

        with pytest.raises(DetectorMissingStateError,
                           match="missing or empty"):
            fourier_detector.load_state([record], "cpu", method=method)

        extract_mod.fourier_bundle_manifest.assert_not_called()

        extract_mod.fourier_bundle_manifest.assert_not_called()


# ===========================================================================
# 7. Structured failure classification (no message substring)
# ===========================================================================
class TestStructuredFailureClassification:

    def test_bundle_failure_does_not_parse_message(self, monkeypatch, bundle_dir):
        """fourier_bundle_manifest RuntimeError → DetectorStateValidationError
        regardless of what the message contains."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID")
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)

        # The helper raises RuntimeError with "not found" in the message,
        # but since path preflight already passed, it's state validation.
        extract_mod.fourier_bundle_manifest = mock.MagicMock(
            side_effect=RuntimeError("not found: manifest.json is missing"))

        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)
        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorStateValidationError,
                           match="bundle validation failed"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_typeerror_is_provider_initialization(self, monkeypatch, bundle_dir):
        """TypeError in provider construction → DetectorProviderInitializationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorProviderInitializationError

        manifest = _make_manifest("RID")
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        extract_mod.rid_provider_kwargs_from_bundle.return_value = {
            "modelid_target": "test/model",
            "resolution": 512,
            "scheduler_target": "DDIM",
        }
        _MOCK_RINGID.side_effect = TypeError("bad argument")

        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)
        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorProviderInitializationError,
                           match="provider construction failed"):
            fourier_detector.load_state([record], "cpu", method="RID")

        _MOCK_RINGID.side_effect = None


# ===========================================================================
# 8. Pipe configuration from bundle
# ===========================================================================
class TestPipeConfigurationFromBundle:

    def test_pipe_config_comes_from_bundle_rid(self, monkeypatch, bundle_dir):
        """RID pipe uses model_id/scheduler from kwargs (manifest-derived)."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest("RID")
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")
        fourier_detector.load_state([record], "cpu", method="RID")

        call_kwargs = _MOCK_PIPE_UTILS.get_pipe_provider.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs["pretrained_model_name_or_path"] == \
            "RedbeardNZ/stable-diffusion-2-1-base"
        assert kwargs["resolution"] == 512
        assert kwargs["schedulers_name"] == "DDIM"

    def test_pipe_config_comes_from_bundle_hsqr(self, monkeypatch, bundle_dir):
        """HSQR pipe uses model_id/scheduler from manifest."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest("HSQR")
        provider = _make_provider("HSQR")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")
        fourier_detector.load_state([record], "cpu", method="HSQR")

        call_kwargs = _MOCK_PIPE_UTILS.get_pipe_provider.call_args
        _, kwargs = call_kwargs
        assert kwargs["schedulers_name"] == "DDIM"


# ===========================================================================
# 9. Method-specific score definitions
# ===========================================================================
class TestScoreDefinitions:

    @pytest.mark.parametrize("method,expected_def", [
        ("RID", "rid_neg_channel_min_complex_l1"),
        ("HSTR", "hstr_score=-min(channel_0_l1,channel_3_l1)"),
        ("HSQR", "hsqr_negative_mean_complex_l1_distance"),
    ])
    def test_score_definition_is_method_specific(self, method, expected_def,
                                                  tmp_path, bundle_dir):
        """score_image returns the exact canonical score definition label."""
        from raven.detectors import fourier_detector

        provider = _make_provider(method)
        manifest = _make_manifest(method)

        # Align provider identity with record provenance values
        if method == "HSQR":
            provider.bundle.manifest = {
                "selected_pattern_sha256": "provider_target_sha",
                "mask_sha256": "provider_mask_sha",
            }
            provider.watermark_mask_sha256 = "provider_mask_sha"
        else:
            provider.selected_pattern_sha256 = "provider_target_sha"
            provider.watermark_mask_sha256 = "provider_mask_sha"

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
            "score_definition": expected_def,
        }

        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="provider_mask_sha")

        result = fourier_detector.score_image(pinfo, str(img_path), record=record)
        assert result["score_definition"] == expected_def

    @pytest.mark.parametrize("method,expected_def", [
        ("RID", "rid_neg_channel_min_complex_l1"),
        ("HSTR", "hstr_score=-min(channel_0_l1,channel_3_l1)"),
        ("HSQR", "hsqr_negative_mean_complex_l1_distance"),
    ])
    def test_aggregate_score_definition(self, method, expected_def):
        """aggregate returns method-specific score_definition."""
        from raven.detectors import fourier_detector

        rows = [{"run_id": "1", "evaluation_cohort": "original_watermarked",
                 "status": "scored", "canonical_score": -0.1}]
        result = fourier_detector.aggregate(rows, method=method)
        assert result["score_definition"] == expected_def


# ===========================================================================
# 10. Missing image → FileNotFoundError (Issue #25 taxonomy)
# ===========================================================================
class TestMissingImageFileNotFound:

    def test_missing_image_raises_file_not_found(self):
        """Missing image → FileNotFoundError, not DetectorMissingStateError."""
        from raven.detectors import fourier_detector

        pinfo = {
            "provider": _make_provider("RID"),
            "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": "RID", "_manifest": {},
        }

        with pytest.raises(FileNotFoundError, match="Image not found"):
            fourier_detector.score_image(pinfo, "/nonexistent/img.png")


# ===========================================================================
# 11. Orchestrator integration — REAL adapter, mocked externals
# ===========================================================================
class _OrchestratorFixtures:
    """Shared helpers for orchestrator integration tests."""

    @staticmethod
    def _build_orchestrator_env(tmp_path, method, monkeypatch, *,
                                 bundle_overrides=None,
                                 record_overrides=None):
        """Create temp dirs, records, metadata CSV, and wire up REAL adapter.

        Uses an explicit metadata CSV file (via config.metadata_path) so
        the MetadataResolver path is exercised deterministically.

        Returns (out_dir, records, extract_mod, provider, manifest, metadata_csv).
        """
        from raven.detectors import fourier_detector

        prefix = method.lower()
        manifest = _make_manifest(method, **(bundle_overrides or {}))
        provider = _make_provider(method)
        if method in {"RID", "HSTR"}:
            provider.state_source = "bundle"
        provider.selected_pattern_sha256 = "provider_target_sha"
        provider.watermark_mask_sha256 = "provider_mask_sha"

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        from PIL import Image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            Image.new("RGB", (64, 64)).save(img_dir / f"img_{i}.png")

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        for i in range(2):
            role_dir = out_dir / "samples" / "watermarked" / str(i)
            role_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(role_dir / "output.png")

        # Build metadata CSV with all required Fourier provenance fields
        metadata_fields = [
            "run_id", "role", "method",
            f"{prefix}_bundle_dir", f"{prefix}_bundle_config_sha256",
            f"{prefix}_selected_pattern_sha256", f"{prefix}_mask_sha256",
            f"{prefix}_key_index", f"{prefix}_protocol_mode",
            "watermark_target_sha256", "watermark_mask_sha256",
        ]
        metadata_csv = tmp_path / "metadata.csv"
        csv_lines = [",".join(metadata_fields)]
        for i in range(2):
            rid = str(i)
            csv_lines.append(",".join([
                rid, "watermarked", method,
                str(bundle_dir), manifest["bundle_config_sha256"],
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                str(manifest.get("selected_key_index", 0)),
                manifest.get("profile_name", f"{prefix}_official"),
                "provider_target_sha", "provider_mask_sha",
            ]))
        metadata_csv.write_text("\n".join(csv_lines) + "\n")

        records = []
        for i in range(2):
            rec = {
                "run_id": str(i), "role": "watermarked", "method": method,
                "input_path": str(img_dir / f"img_{i}.png"),
                "output_path": str(out_dir / "samples" / "watermarked" / str(i) / "output.png"),
                "prompt": "", "attack_seed": 59,
                "planned_flow_dx_image_px": 24.0,
                "planned_flow_dy_image_px": -24.0,
                "effective_source_flow_dx_image_px": 24.0,
                "effective_source_flow_dy_image_px": -24.0,
                "debug_info_path": "", "debug_info_retained": False,
            }
            rec.update(record_overrides or {})
            records.append(rec)

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        return out_dir, records, extract_mod, provider, manifest, metadata_csv


class TestOrchestratorSuccess:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_real_adapter_full_orchestrator_success(self, method, monkeypatch,
                                                     tmp_path):
        """evaluate_detector with REAL load_state/score_image succeeds."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        out_dir, records, extract_mod, provider, manifest, metadata_csv = \
            _OrchestratorFixtures._build_orchestrator_env(
                tmp_path, method, monkeypatch)

        # Align provider's bundle manifest with the metadata values
        # so that target/mask SHA validation in score_image passes
        if method == "HSQR":
            provider.bundle.manifest = {
                "selected_pattern_sha256": "provider_target_sha",
                "mask_sha256": "provider_mask_sha",
            }
            provider.watermark_mask_sha256 = "provider_mask_sha"
        provider.selected_pattern_sha256 = "provider_target_sha"
        provider.watermark_mask_sha256 = "provider_mask_sha"

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)

        assert result["stage"] == "detector"
        # With only watermarked records, status is completed_with_errors
        # (no clean calibration cohort).  Must not be skipped.
        assert result["status"] == "completed_with_errors"
        assert result["scored_count"] == 4  # 2 records × 2 cohorts
        assert result["failed_count"] == 0

        # Verify detector records were written with method-specific labels
        det_path = out_dir / "evaluation" / "detector_records.jsonl"
        assert det_path.is_file()
        lines = det_path.read_text().strip().split("\n")
        assert len(lines) == 4

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_real_adapter_clean_and_wm_cohorts(self, method, monkeypatch,
                                                tmp_path):
        """Full orchestrator with clean+watermarked records → completed."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        prefix = method.lower()
        manifest = _make_manifest(method)
        provider = _make_provider(method)
        if method in {"RID", "HSTR"}:
            provider.state_source = "bundle"

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        from PIL import Image
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Build metadata CSV
        metadata_fields = [
            "run_id", "role", "method",
            f"{prefix}_bundle_dir", f"{prefix}_bundle_config_sha256",
            f"{prefix}_selected_pattern_sha256", f"{prefix}_mask_sha256",
            f"{prefix}_key_index", f"{prefix}_protocol_mode",
            "watermark_target_sha256", "watermark_mask_sha256",
        ]
        csv_lines = [",".join(metadata_fields)]

        records = []
        for role in ("clean", "watermarked"):
            for i in range(2):
                rid = f"{role}_{i}"
                img = tmp_path / f"img_{rid}.png"
                Image.new("RGB", (64, 64)).save(img)
                role_dir = out_dir / "samples" / role / rid
                role_dir.mkdir(parents=True)
                Image.new("RGB", (64, 64)).save(role_dir / "output.png")

                records.append({
                    "run_id": rid, "role": role, "method": method,
                    "input_path": str(img),
                    "output_path": str(role_dir / "output.png"),
                    "prompt": "", "attack_seed": 59,
                    "planned_flow_dx_image_px": 24.0,
                    "planned_flow_dy_image_px": -24.0,
                    "effective_source_flow_dx_image_px": 24.0,
                    "effective_source_flow_dy_image_px": -24.0,
                    "debug_info_path": "", "debug_info_retained": False,
                })
                csv_lines.append(",".join([
                    rid, role, method,
                    str(bundle_dir), manifest["bundle_config_sha256"],
                    manifest["selected_pattern_sha256"],
                    manifest.get("mask_sha256", ""),
                    str(manifest.get("selected_key_index", 0)),
                    manifest.get("profile_name", f"{prefix}_official"),
                    "provider_target_sha", "provider_mask_sha",
                ]))

        metadata_csv = tmp_path / "metadata.csv"
        metadata_csv.write_text("\n".join(csv_lines) + "\n")

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        # Align provider identity with CSV provenance for HSQR
        if method == "HSQR":
            provider.bundle.manifest = {
                "selected_pattern_sha256": "provider_target_sha",
                "mask_sha256": "provider_mask_sha",
            }
            provider.watermark_mask_sha256 = "provider_mask_sha"
        provider.selected_pattern_sha256 = "provider_target_sha"
        provider.watermark_mask_sha256 = "provider_mask_sha"

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)
        assert result["status"] in ("completed", "completed_with_errors")


class TestOrchestratorFailures:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_bundle_mismatch_fails_stage(self, method, monkeypatch, tmp_path):
        """Bundle SHA mismatch → failed_state_validation stage."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        prefix = method.lower()
        manifest = _make_manifest(method)
        provider = _make_provider(method)
        if method in {"RID", "HSTR"}:
            provider.state_source = "bundle"

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        from PIL import Image
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Metadata CSV with WRONG bundle_config_sha256
        metadata_fields = [
            "run_id", "role", "method",
            f"{prefix}_bundle_dir", f"{prefix}_bundle_config_sha256",
            f"{prefix}_selected_pattern_sha256", f"{prefix}_mask_sha256",
            f"{prefix}_key_index", f"{prefix}_protocol_mode",
            "watermark_target_sha256", "watermark_mask_sha256",
        ]
        csv_lines = [",".join(metadata_fields)]

        records = []
        for i in range(2):
            rid = str(i)
            img = tmp_path / f"img_{rid}.png"
            Image.new("RGB", (64, 64)).save(img)
            role_dir = out_dir / "samples" / "watermarked" / rid
            role_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(role_dir / "output.png")

            records.append({
                "run_id": rid, "role": "watermarked", "method": method,
                "input_path": str(img),
                "output_path": str(role_dir / "output.png"),
                "prompt": "", "attack_seed": 59,
                "planned_flow_dx_image_px": 24.0,
                "planned_flow_dy_image_px": -24.0,
                "effective_source_flow_dx_image_px": 24.0,
                "effective_source_flow_dy_image_px": -24.0,
                "debug_info_path": "", "debug_info_retained": False,
            })
            csv_lines.append(",".join([
                rid, "watermarked", method,
                str(bundle_dir), "wrong_bundle_sha",
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                str(manifest.get("selected_key_index", 0)),
                manifest.get("profile_name", f"{prefix}_official"),
                "provider_target_sha", "provider_mask_sha",
            ]))

        metadata_csv = tmp_path / "metadata.csv"
        metadata_csv.write_text("\n".join(csv_lines) + "\n")

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        # Make fourier_bundle_manifest validate the SHA and reject mismatches
        def _validating_fbm(row, identifier, m):
            row_sha = str(row.get(f"{prefix}_bundle_config_sha256", ""))
            if row_sha != manifest["bundle_config_sha256"]:
                raise RuntimeError(
                    f"{method} bundle/source "
                    f"{prefix}_bundle_config_sha256 mismatch: "
                    f"source={row_sha!r} "
                    f"bundle={manifest['bundle_config_sha256']!r}")
            return (bundle_dir, manifest)
        extract_mod.fourier_bundle_manifest = mock.MagicMock(
            side_effect=_validating_fbm)

        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)
        assert result["status"] == "failed_state_validation"

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_wrong_key_index_fails_stage(self, method, monkeypatch, tmp_path):
        """Wrong key_index → failed_state_validation stage."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        prefix = method.lower()
        manifest = _make_manifest(method, selected_key_index=0)
        provider = _make_provider(method)
        if method in {"RID", "HSTR"}:
            provider.state_source = "bundle"

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        from PIL import Image
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        metadata_fields = [
            "run_id", "role", "method",
            f"{prefix}_bundle_dir", f"{prefix}_bundle_config_sha256",
            f"{prefix}_selected_pattern_sha256", f"{prefix}_mask_sha256",
            f"{prefix}_key_index", f"{prefix}_protocol_mode",
            "watermark_target_sha256", "watermark_mask_sha256",
        ]
        csv_lines = [",".join(metadata_fields)]

        records = []
        for i in range(2):
            rid = str(i)
            img = tmp_path / f"img_{rid}.png"
            Image.new("RGB", (64, 64)).save(img)
            role_dir = out_dir / "samples" / "watermarked" / rid
            role_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(role_dir / "output.png")

            records.append({
                "run_id": rid, "role": "watermarked", "method": method,
                "input_path": str(img),
                "output_path": str(role_dir / "output.png"),
                "prompt": "", "attack_seed": 59,
                "planned_flow_dx_image_px": 24.0,
                "planned_flow_dy_image_px": -24.0,
                "effective_source_flow_dx_image_px": 24.0,
                "effective_source_flow_dy_image_px": -24.0,
                "debug_info_path": "", "debug_info_retained": False,
            })
            # WRONG key index in metadata
            csv_lines.append(",".join([
                rid, "watermarked", method,
                str(bundle_dir), manifest["bundle_config_sha256"],
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                "99",
                manifest.get("profile_name", f"{prefix}_official"),
                "provider_target_sha", "provider_mask_sha",
            ]))

        metadata_csv = tmp_path / "metadata.csv"
        metadata_csv.write_text("\n".join(csv_lines) + "\n")

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)
        assert result["status"] == "failed_state_validation"


# ===========================================================================
# 12. State-source gates (preserved from v1)
# ===========================================================================
class TestStateSourceGates:

    def test_rid_rejects_non_bundle_state_source(self, monkeypatch, bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider("RID", state_source="random")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")
        with pytest.raises(DetectorStateValidationError, match="state_source"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_hsqr_accepts_no_state_source(self, monkeypatch, bundle_dir):
        from raven.detectors import fourier_detector

        provider = _make_provider("HSQR")
        del provider.state_source
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")
        result = fourier_detector.load_state([record], "cpu", method="HSQR")
        assert result["method"] == "HSQR"

    def test_hsqr_rejects_missing_bundle(self, monkeypatch, bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider("HSQR", has_bundle=False)
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")
        with pytest.raises(DetectorStateValidationError,
                           match="no persisted bundle"):
            fourier_detector.load_state([record], "cpu", method="HSQR")


# ===========================================================================
# 13. Mixed cohort + edge cases
# ===========================================================================
class TestMixedCohortRejection:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_bundle_dir_rejected(self, method, monkeypatch, bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        other_dir = Path(str(bundle_dir) + "_other")
        other_dir.mkdir(parents=True, exist_ok=True)
        (other_dir / "manifest.json").write_text(
            json.dumps(_make_manifest(method)))

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method=method)
        r2 = _make_record(other_dir, "2", method=method)

        with pytest.raises(DetectorStateValidationError, match="mixed bundle_dir"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)


class TestEdgeCases:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_empty_records(self, method):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        with pytest.raises(DetectorMissingStateError, match="no records"):
            fourier_detector.load_state([], "cpu", method=method)

    def test_unknown_method(self):
        from raven.detectors import fourier_detector

        with pytest.raises(ValueError, match="Unknown Fourier method"):
            fourier_detector.load_state(
                [_make_record(Path("/tmp"), "1", method="RID")],
                "cpu", method="TR")

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_bundle_dir(self, method, monkeypatch):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        monkeypatch.setattr(fourier_detector, "_ensure_paths", lambda: None)

        record = _make_record(Path("/nonexistent"), "1", method=method,
                              **{f"{method.lower()}_bundle_dir": ""})

        with pytest.raises(DetectorMissingStateError):
            fourier_detector.load_state([record], "cpu", method=method)


# ===========================================================================
# 14. Metadata preflight — None value
# ===========================================================================
class TestMetadataPreflightNone:

    def test_none_field_is_missing_state(self, monkeypatch, bundle_dir):
        """None value for required field → DetectorMissingStateError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID",
                              rid_protocol_mode=None)

        with pytest.raises(DetectorMissingStateError, match="is None"):
            fourier_detector.load_state([record], "cpu", method="RID")
