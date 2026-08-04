"""Issue #24 regression tests — method-specific Fourier bundle validation
for RID, HSTR, and HSQR unified detectors.

All tests use mocks.  No real bundles downloaded, no real inversion.
Integration tests use the REAL adapter (load_state + score_image +
aggregate) with only external construction mocked.

Protocol mode and provider profile are distinct identities and are never
conflated in fixtures.

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
# Real canonical constants — never re-invented in fixtures
# ---------------------------------------------------------------------------
from raven.pairing_provenance import (  # noqa: E402
    RID_SHARED_TR_CLEAN_MODE,
    HSTR_SHARED_TR_CLEAN_MODE,
    HSQR_SHARED_TR_CLEAN_MODE,
)

_METHOD_PROTOCOL_MODES = {
    "RID": RID_SHARED_TR_CLEAN_MODE,
    "HSTR": HSTR_SHARED_TR_CLEAN_MODE,
    "HSQR": HSQR_SHARED_TR_CLEAN_MODE,
}

# Real provider profile names from eval_bench_wm providers
_RID_PROFILES = ("legacy", "official_sd21", "paper_shift_ablation")
_HSTR_PROFILES = ("legacy_raven", "official_sfwmark_sd21",
                  "official_math_shared_tr_clean")
_HSQR_PROFILES = ("legacy_raven", "official_sfwmark_sd21")

_METHOD_PROFILES = {
    "RID": _RID_PROFILES,
    "HSTR": _HSTR_PROFILES,
    "HSQR": _HSQR_PROFILES,
}

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
# Canonical test data builders — real schema semantics
# ===========================================================================
def _default_profile(method="RID") -> str:
    """Default provider profile for a method (first of the real list)."""
    return _METHOD_PROFILES[method][0]


def _make_manifest(method_name="RID", **overrides):
    """Manifest matching real provider/bundle schema for each method.

    Both canonical bundle formats (RidBundle and SfwBundle) write a
    ``method`` tag into their manifests — the fixture always carries the
    canonical per-method ``method`` value unless overridden.
    """
    method = method_name
    profile = _default_profile(method)
    base = {
        "method": method,
        "bundle_config_sha256": "abc123_bundle",
        "selected_pattern_sha256": "abc123_pattern",
        "mask_sha256": "abc123_mask",
        "selected_key_index": 0,
        "profile_name": profile,
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "main",
        "resolution": 512,
    }
    if method == "RID":
        base.update({
            "schema": "rid_bundle_v1",
            "rng_seed": 42, "rng_device": "cpu", "rng_dtype": "float32",
            "channel_min": 0, "ring_value_range": 1,
            "quantization_levels": 16, "ring_width": 1,
            "assigned_keys": 4, "fix_gt": 1,
            "spatial_shift": 1, "spatial_shift_factor": 1.0,
            "spatial_shift_factor_semantics": "periodic",
            "torch_dtype": "float32",
            "inversion_guidance_scale": 2.5, "inversion_steps": 50,
            "vae_sample": True, "vae_scaling_factor": 0.18215,
            "scheduler": "DDIM",
            "profile_is_official": profile == "official_sd21",
            "profile_overrides": {},
        })
    elif method == "HSTR":
        base.update({
            "schema": "sfw_bundle_v1",
            "latent_shape": [1, 4, 64, 64],
            "center_slice": [1, 3], "wm_capacity": 256,
            "scheduler_type": "DDIM",
        })
    elif method == "HSQR":
        base.update({
            "schema": "sfw_bundle_v1",
            "scheduler_type": "DDIM",
        })
    base.update(overrides)
    return base


def _make_record(bundle_dir, run_id="1", method="RID", **overrides):
    """Record with real protocol mode constant and required metadata."""
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
        f"{prefix}_protocol_mode": _METHOD_PROTOCOL_MODES[method],
        "watermark_target_sha256": "provider_target_sha",
        "watermark_mask_sha256": "provider_mask_sha",
        "input_path": f"/tmp/in_{run_id}.png",
        "output_path": f"/tmp/out/watermarked/{run_id}/output.png",
        "prompt": "",
        "attack_seed": 59,
    }
    base.update(overrides)
    return base


def _make_provider(method="RID", state_source="bundle", has_bundle=True,
                   profile=None):
    """Mock provider with method-appropriate attributes and real profile."""
    p = mock.MagicMock()
    p.get_wm_type.return_value = method
    if has_bundle:
        p.bundle = mock.MagicMock()
        p.bundle.manifest = _make_manifest(method)
    else:
        p.bundle = None
    if method in {"RID", "HSTR"}:
        p.state_source = state_source
    p.profile = profile if profile is not None else _default_profile(method)
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
    profile = str(manifest.get("profile_name", _default_profile(method)))

    def _fake_fbm(row, identifier, m):
        bd = bundle_dir or Path(str(row.get(f"{method.lower()}_bundle_dir", "/tmp")))
        return bd, manifest

    mod.fourier_bundle_manifest = mock.MagicMock(side_effect=_fake_fbm)

    mod.rid_provider_kwargs_from_bundle = mock.MagicMock(return_value={
        "rid_profile": profile,
        "rid_bundle_dir": "/tmp/bundles/rid",
        "rid_create_bundle": False, "rid_key_index": 0, "rid_key_seed": 42,
        "rid_key_rng_device": "cpu", "rid_key_rng_dtype": "float32",
        "channel_min": 0, "ring_value_range": 1, "quantization_levels": 16,
        "ring_width": 1, "assigned_keys": 4, "fix_gt": 1, "time_shift": 1,
        "time_shift_factor": 1.0, "rid_shift_semantics": "periodic",
        "rid_torch_dtype": "float32", "rid_inversion_prompt": "",
        "rid_inversion_guidance": 2.5, "rid_inversion_steps": 50,
        "rid_vae_sample": True, "rid_vae_scaling_factor": 0.18215,
        "rid_profile_is_official": profile == "official_sd21",
        "rid_profile_overrides": {},
        "modelid_target": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "main", "scheduler_target": "DDIM", "resolution": 512,
    })

    mod.hstr_provider_kwargs_from_bundle = mock.MagicMock(return_value={
        "hstr_profile": profile,
        "hstr_bundle_dir": "/tmp/bundles/hstr",
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
    _MOCK_RINGID.call_count = 0
    _MOCK_HSTR.call_count = 0
    _MOCK_HSQR.call_count = 0
    _MOCK_PIPE_UTILS.get_pipe_provider.call_count = 0
    _MOCK_PIPE_UTILS.get_pipe_provider.reset_mock()


@pytest.fixture
def bundle_dir():
    """Real temp dir with manifest.json."""
    with tempfile.TemporaryDirectory() as td:
        bd = Path(td)
        manifest = _make_manifest()
        (bd / "manifest.json").write_text(json.dumps(manifest))
        yield bd


# ===========================================================================
# 1. Protocol mode vs provider profile — distinct identities
# ===========================================================================
class TestProtocolProfileSeparation:

    def test_real_rid_protocol_and_profile_are_distinct_but_valid(self,
                                                                   monkeypatch,
                                                                   bundle_dir):
        """RID: protocol=official_math_shared_tr_clean, profile=legacy is valid."""
        from raven.detectors import fourier_detector

        # Profile "legacy", protocol "official_math_shared_tr_clean" — distinct
        # but both valid for their own identity dimension.
        manifest = _make_manifest("RID", profile_name="legacy",
                                  schema="rid_bundle_v1")
        provider = _make_provider("RID", state_source="bundle", profile="legacy")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")

        result = fourier_detector.load_state([record], "cpu", method="RID")
        assert result["method"] == "RID"

    def test_rid_wrong_protocol_rejected(self, monkeypatch, bundle_dir):
        """RID protocol != canonical constant → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", profile_name="legacy")
        provider = _make_provider("RID", state_source="bundle", profile="legacy")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID",
                              rid_protocol_mode="wrong_protocol")

        with pytest.raises(DetectorStateValidationError,
                           match="protocol mode"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_rid_wrong_provider_profile_rejected(self, monkeypatch, bundle_dir):
        """manifest profile_name != provider.profile → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", profile_name="legacy")
        # Provider profile disagrees with manifest
        provider = _make_provider("RID", state_source="bundle",
                                  profile="official_sd21")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorStateValidationError, match="profile"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_rid_kwargs_profile_mismatch_rejected(self, monkeypatch, bundle_dir):
        """manifest profile_name != kwargs rid_profile → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", profile_name="legacy")
        provider = _make_provider("RID", state_source="bundle", profile="legacy")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        # Force kwargs profile mismatch
        extract_mod.rid_provider_kwargs_from_bundle.return_value["rid_profile"] = \
            "official_sd21"
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorStateValidationError, match="profile"):
            fourier_detector.load_state([record], "cpu", method="RID")

    def test_hstr_protocol_and_profile_validated_separately(self, monkeypatch,
                                                             bundle_dir):
        """HSTR: protocol constant + real profile validated independently."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSTR", profile_name="official_sfwmark_sd21")
        provider = _make_provider("HSTR", state_source="bundle",
                                  profile="official_sfwmark_sd21")
        extract_mod = _build_extract_module("HSTR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSTR", provider, extract_mod)

        # Valid combination passes
        record = _make_record(bundle_dir, "1", method="HSTR")
        result = fourier_detector.load_state([record], "cpu", method="HSTR")
        assert result["method"] == "HSTR"

        # Wrong protocol rejected
        bad = _make_record(bundle_dir, "2", method="HSTR",
                           hstr_protocol_mode="wrong")
        with pytest.raises(DetectorStateValidationError,
                           match="protocol mode"):
            fourier_detector.load_state([bad], "cpu", method="HSTR")

        # Wrong profile rejected
        provider2 = _make_provider("HSTR", state_source="bundle",
                                   profile="legacy_raven")
        extract_mod2 = _build_extract_module("HSTR", provider=provider2,
                                              manifest=manifest,
                                              bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSTR", provider2, extract_mod2)
        record2 = _make_record(bundle_dir, "3", method="HSTR")
        with pytest.raises(DetectorStateValidationError, match="profile"):
            fourier_detector.load_state([record2], "cpu", method="HSTR")

    def test_hsqr_protocol_and_profile_validated_separately(self, monkeypatch,
                                                             bundle_dir):
        """HSQR: protocol constant + real profile validated independently."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSQR", profile_name="official_sfwmark_sd21")
        provider = _make_provider("HSQR", profile="official_sfwmark_sd21")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")
        result = fourier_detector.load_state([record], "cpu", method="HSQR")
        assert result["method"] == "HSQR"

        bad = _make_record(bundle_dir, "2", method="HSQR",
                           hsqr_protocol_mode="wrong")
        with pytest.raises(DetectorStateValidationError,
                           match="protocol mode"):
            fourier_detector.load_state([bad], "cpu", method="HSQR")

        provider2 = _make_provider("HSQR", profile="legacy_raven")
        extract_mod2 = _build_extract_module("HSQR", provider=provider2,
                                              manifest=manifest,
                                              bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider2, extract_mod2)
        record2 = _make_record(bundle_dir, "3", method="HSQR")
        with pytest.raises(DetectorStateValidationError, match="profile"):
            fourier_detector.load_state([record2], "cpu", method="HSQR")


# ===========================================================================
# 2. All-row artifact path preflight
# ===========================================================================
class TestAllRowPathPreflight:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_second_row_missing_bundle_dir(self, method, monkeypatch,
                                            bundle_dir):
        """Row 2 bundle dir missing → DetectorMissingStateError, no provider."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method=method)
        r2 = _make_record(Path("/nonexistent/row2"), "2", method=method)

        with pytest.raises(DetectorMissingStateError, match="run_id=2"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

        # Provider / pipe must never be constructed
        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0
        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_second_row_missing_manifest(self, method, monkeypatch, bundle_dir):
        """Row 2 manifest missing → DetectorMissingStateError, no provider."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        # Row 2 bundle dir exists but has no manifest.json
        empty_dir = Path(str(bundle_dir) + "_empty")
        empty_dir.mkdir(parents=True, exist_ok=True)

        r1 = _make_record(bundle_dir, "1", method=method)
        r2 = _make_record(empty_dir, "2", method=method)

        with pytest.raises(DetectorMissingStateError, match="run_id=2"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0
        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0


# ===========================================================================
# 3. Mixed-cohort validation before provider construction
# ===========================================================================
class TestMixedCohortBeforeProvider:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_key_index_no_provider(self, method, monkeypatch, bundle_dir):
        """Mixed key_index across rows → error, zero provider constructions."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        prefix = method.lower()
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method=method)
        r2 = _make_record(bundle_dir, "2", method=method,
                          **{f"{prefix}_key_index": "5"})

        with pytest.raises(DetectorStateValidationError, match="mixed cohort"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0
        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_mixed_bundle_dir_no_provider(self, method, monkeypatch, bundle_dir):
        """Mixed bundle_dir across rows → error, zero provider constructions."""
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

        with pytest.raises(DetectorStateValidationError, match="mixed cohort"):
            fourier_detector.load_state([r1, r2], "cpu", method=method)

        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0
        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0


# ===========================================================================
# 4. score_image requires resolved record
# ===========================================================================
class TestScoreRequiresRecord:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_score_image_without_record_fails_closed(self, method, tmp_path):
        """record=None → DetectorMissingStateError, provenance never optional."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        manifest = _make_manifest(method)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }

        with pytest.raises(DetectorMissingStateError,
                           match="resolved source metadata"):
            fourier_detector.score_image(pinfo, str(img_path), record=None)


# ===========================================================================
# 5. Pipe profile — no fallback
# ===========================================================================
class TestPipeProfileNoFallback:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_hsqr_missing_scheduler_does_not_fallback_to_ddim(self, method,
                                                                monkeypatch,
                                                                bundle_dir):
        """Missing scheduler → DetectorStateValidationError, no DDIM fallback."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(method)
        # Remove scheduler entirely — must NOT fall back to "DDIM"
        manifest.pop("scheduler_type", None)
        manifest.pop("scheduler", None)

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorStateValidationError,
                           match="no scheduler"):
            fourier_detector.load_state([record], "cpu", method=method)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_model_revision_rejected(self, method, monkeypatch,
                                              bundle_dir):
        """Missing model_revision → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(method)
        manifest.pop("model_revision", None)

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorStateValidationError,
                           match="no model_revision"):
            fourier_detector.load_state([record], "cpu", method=method)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_pipe_receives_bundle_model_revision(self, method, monkeypatch,
                                                  bundle_dir):
        """Pipe constructor receives bundle model_revision."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest(method, model_revision="my-revision-abc")
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)
        fourier_detector.load_state([record], "cpu", method=method)

        call_kwargs = _MOCK_PIPE_UTILS.get_pipe_provider.call_args
        assert call_kwargs is not None
        _args, kwargs = call_kwargs
        assert kwargs.get("revision") == "my-revision-abc"

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_malformed_resolution_is_state_validation(self, method, monkeypatch,
                                                       bundle_dir):
        """Non-integer or non-positive resolution → DetectorStateValidationError,
        never internal error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        for bad_resolution in ("abc", 0, -512, "12.5"):
            manifest = _make_manifest(method, resolution=bad_resolution)
            provider = _make_provider(method)
            extract_mod = _build_extract_module(method, provider=provider,
                                                 manifest=manifest,
                                                 bundle_dir=bundle_dir)
            _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

            record = _make_record(bundle_dir, "1", method=method)

            with pytest.raises(DetectorStateValidationError,
                               match="resolution"):
                fourier_detector.load_state([record], "cpu", method=method)

            assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_pipe_receives_bundle_scheduler_and_resolution(self, method,
                                                            monkeypatch,
                                                            bundle_dir):
        """Pipe constructor receives bundle scheduler and resolution."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest(method,
                                  scheduler_type="DDIM",
                                  resolution=768)
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)
        fourier_detector.load_state([record], "cpu", method=method)

        call_kwargs = _MOCK_PIPE_UTILS.get_pipe_provider.call_args
        _args, kwargs = call_kwargs
        assert kwargs["schedulers_name"] == "DDIM"
        assert kwargs["resolution"] == 768


# ===========================================================================
# 6. Structured provider initialization taxonomy
# ===========================================================================
class TestProviderInitTaxonomy:

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_pipe_raises_value_error(self, method, monkeypatch, bundle_dir):
        """Pipe ValueError → DetectorProviderInitializationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorProviderInitializationError

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)
        _MOCK_PIPE_UTILS.get_pipe_provider.side_effect = ValueError("bad pipe")

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorProviderInitializationError,
                           match="pipe construction failed"):
            fourier_detector.load_state([record], "cpu", method=method)

        _MOCK_PIPE_UTILS.get_pipe_provider.side_effect = None

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_provider_raises_runtime_error(self, method, monkeypatch, bundle_dir):
        """Provider RuntimeError → DetectorProviderInitializationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorProviderInitializationError

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)
        _MOCK_RINGID.side_effect = RuntimeError("boom")
        _MOCK_HSTR.side_effect = RuntimeError("boom")

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorProviderInitializationError,
                           match="provider construction failed"):
            fourier_detector.load_state([record], "cpu", method=method)

        _MOCK_RINGID.side_effect = None
        _MOCK_HSTR.side_effect = None

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_provider_raises_type_error(self, method, monkeypatch, bundle_dir):
        """Provider TypeError → DetectorProviderInitializationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorProviderInitializationError

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)
        _MOCK_RINGID.side_effect = TypeError("bad arg")
        _MOCK_HSTR.side_effect = TypeError("bad arg")

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorProviderInitializationError,
                           match="provider construction failed"):
            fourier_detector.load_state([record], "cpu", method=method)

        _MOCK_RINGID.side_effect = None
        _MOCK_HSTR.side_effect = None


# ===========================================================================
# 7. Scoring boundary — full wrap
# ===========================================================================
def _align_provider_identity(provider, method):
    """Align provider identity with the record provenance values."""
    if method == "HSQR":
        provider.bundle.manifest = {
            "selected_pattern_sha256": "provider_target_sha",
            "mask_sha256": "provider_mask_sha",
        }
        provider.watermark_mask_sha256 = "provider_mask_sha"
    else:
        provider.selected_pattern_sha256 = "provider_target_sha"
        provider.watermark_mask_sha256 = "provider_mask_sha"
    return provider


class TestScoringBoundary:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_raw_score_failure_is_scoring_error(self, method, tmp_path,
                                                 bundle_dir):
        """raw_score failure → DetectorScoringError, not internal error."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorScoringError

        provider = _align_provider_identity(_make_provider(method), method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest)
        extract_mod.raw_score.side_effect = KeyError("l1_dist")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }
        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorScoringError, match="scoring failed"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_canonical_score_failure_is_scoring_error(self, method, tmp_path,
                                                       bundle_dir):
        """canonical_score failure → DetectorScoringError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorScoringError

        provider = _align_provider_identity(_make_provider(method), method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest)
        extract_mod.canonical_score.side_effect = ValueError("bad canonical")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }
        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorScoringError, match="scoring failed"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_nonfinite_raw_score_is_scoring_error(self, method, tmp_path,
                                                   bundle_dir):
        """Non-finite raw score → DetectorScoringError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorScoringError

        provider = _align_provider_identity(_make_provider(method), method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest)
        extract_mod.raw_score.return_value = float("nan")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }
        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorScoringError, match="non-finite"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_nonfinite_canonical_score_is_scoring_error(self, method, tmp_path,
                                                         bundle_dir):
        """Non-finite canonical score → DetectorScoringError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorScoringError

        provider = _align_provider_identity(_make_provider(method), method)
        manifest = _make_manifest(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest)
        extract_mod.canonical_score.return_value = float("inf")

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": extract_mod,
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }
        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="provider_target_sha",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorScoringError, match="non-finite"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)


# ===========================================================================
# 8. Manifest schema fail closed
# ===========================================================================
class TestManifestSchemaFailClosed:

    def test_unknown_rid_schema_rejected(self, monkeypatch, bundle_dir):
        """RID unknown schema → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("RID", schema="mystery_schema_v9")
        provider = _make_provider("RID", state_source="bundle")
        extract_mod = _build_extract_module("RID", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "RID", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="RID")

        with pytest.raises(DetectorStateValidationError,
                           match="unsupported RID bundle schema"):
            fourier_detector.load_state([record], "cpu", method="RID")

    @pytest.mark.parametrize("method", ["HSTR", "HSQR"])
    def test_unknown_sfw_schema_rejected(self, method, monkeypatch, bundle_dir):
        """HSTR/HSQR unknown schema → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(method, schema="mystery_schema_v9")
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorStateValidationError,
                           match="unsupported SFW bundle schema"):
            fourier_detector.load_state([record], "cpu", method=method)

    def test_rid_schema_used_with_sfw_method_rejected(self, monkeypatch,
                                                       bundle_dir):
        """HSTR with rid_bundle_v1 schema → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSTR", schema="rid_bundle_v1")
        provider = _make_provider("HSTR")
        extract_mod = _build_extract_module("HSTR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSTR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSTR")

        with pytest.raises(DetectorStateValidationError,
                           match="unsupported SFW bundle schema"):
            fourier_detector.load_state([record], "cpu", method="HSTR")

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_schema_is_state_validation(self, method, monkeypatch,
                                                 bundle_dir):
        """Manifest without schema → DetectorStateValidationError, no pipe."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(method)
        manifest.pop("schema", None)

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorStateValidationError,
                           match="no schema"):
            fourier_detector.load_state([record], "cpu", method=method)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0
        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0


# ===========================================================================
# 8b. Manifest method identity — canonical method tag
# ===========================================================================
class TestManifestMethodIdentity:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_manifest_method_matches_is_valid(self, method, monkeypatch,
                                               bundle_dir):
        """Canonical manifest method tag → load_state succeeds."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest(method, **{"method": method})
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)
        result = fourier_detector.load_state([record], "cpu", method=method)
        assert result["method"] == method

    def test_hstr_manifest_method_hstr_is_valid(self, monkeypatch, bundle_dir):
        """HSTR manifest method=HSTR → valid."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest("HSTR", **{"method": "HSTR"})
        provider = _make_provider("HSTR")
        extract_mod = _build_extract_module("HSTR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSTR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSTR")
        result = fourier_detector.load_state([record], "cpu", method="HSTR")
        assert result["method"] == "HSTR"

    def test_hsqr_manifest_method_hsqr_is_valid(self, monkeypatch, bundle_dir):
        """HSQR manifest method=HSQR → valid."""
        from raven.detectors import fourier_detector

        manifest = _make_manifest("HSQR", **{"method": "HSQR"})
        provider = _make_provider("HSQR")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")
        result = fourier_detector.load_state([record], "cpu", method="HSQR")
        assert result["method"] == "HSQR"

    @pytest.mark.parametrize("requested,wrong", [
        ("HSTR", "HSQR"),
        ("HSQR", "HSTR"),
    ])
    def test_wrong_method_tag_rejected_before_provider(self, requested, wrong,
                                                        monkeypatch, bundle_dir):
        """Wrong manifest method tag → DetectorStateValidationError, zero
        pipe/provider constructions."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(requested, **{"method": wrong})
        provider = _make_provider(requested)
        extract_mod = _build_extract_module(requested, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, requested, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=requested)

        with pytest.raises(DetectorStateValidationError,
                           match="manifest method"):
            fourier_detector.load_state([record], "cpu", method=requested)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0
        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0
        assert extract_mod.hsqr_provider_from_bundle.call_count == 0

    def test_hstr_rejects_hsqr_tagged_bundle_before_provider(self, monkeypatch,
                                                              bundle_dir):
        """HSQR-tagged manifest under HSTR request → state validation."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSTR", **{"method": "HSQR"})
        provider = _make_provider("HSTR")
        extract_mod = _build_extract_module("HSTR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSTR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSTR")

        with pytest.raises(DetectorStateValidationError,
                           match="manifest method"):
            fourier_detector.load_state([record], "cpu", method="HSTR")

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0
        assert _MOCK_HSTR.call_count == 0

    def test_hsqr_rejects_hstr_tagged_bundle_before_provider(self, monkeypatch,
                                                              bundle_dir):
        """HSTR-tagged manifest under HSQR request → state validation,
        never hsqr_provider_from_bundle."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest("HSQR", **{"method": "HSTR"})
        provider = _make_provider("HSQR")
        extract_mod = _build_extract_module("HSQR", provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, "HSQR", provider, extract_mod)

        record = _make_record(bundle_dir, "1", method="HSQR")

        with pytest.raises(DetectorStateValidationError,
                           match="manifest method"):
            fourier_detector.load_state([record], "cpu", method="HSQR")

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0
        assert extract_mod.hsqr_provider_from_bundle.call_count == 0

    @pytest.mark.parametrize("method", ["HSTR", "HSQR"])
    def test_sfw_manifest_missing_method_is_state_validation(self, method,
                                                              monkeypatch,
                                                              bundle_dir):
        """SFW manifest without method tag → DetectorStateValidationError."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        manifest = _make_manifest(method)
        manifest.pop("method", None)

        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)

        with pytest.raises(DetectorStateValidationError,
                           match="no method tag"):
            fourier_detector.load_state([record], "cpu", method=method)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0

    @pytest.mark.parametrize("requested,wrong", [
        ("HSTR", "HSQR"),
        ("HSQR", "HSTR"),
    ])
    def test_mixed_manifest_method_rejected_before_pipe(self, requested, wrong,
                                                         monkeypatch,
                                                         bundle_dir):
        """Row 1 manifest method HSTR + row 2 HSQR → rejected before pipe."""
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        # Row 1 manifest is correct for requested method
        manifest_ok = _make_manifest(requested, **{"method": requested})
        # Row 2 manifest disagrees — wrong method tag
        manifest_bad = _make_manifest(requested, **{"method": wrong})

        provider = _make_provider(requested)
        extract_mod = _build_extract_module(requested, provider=provider,
                                             manifest=manifest_ok,
                                             bundle_dir=bundle_dir)

        def _selective_fbm(row, identifier, m):
            if str(row.get("run_id", "")) == "2":
                return bundle_dir, manifest_bad
            return bundle_dir, manifest_ok

        extract_mod.fourier_bundle_manifest = mock.MagicMock(
            side_effect=_selective_fbm)
        _setup_adapter_mocks(monkeypatch, requested, provider, extract_mod)

        r1 = _make_record(bundle_dir, "1", method=requested)
        r2 = _make_record(bundle_dir, "2", method=requested)

        with pytest.raises(DetectorStateValidationError,
                           match="manifest method"):
            fourier_detector.load_state([r1, r2], "cpu", method=requested)

        assert _MOCK_PIPE_UTILS.get_pipe_provider.call_count == 0
        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0
        assert _MOCK_HSQR.call_count == 0


# ===========================================================================
# 9. State-source gates (preserved)
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


# ===========================================================================
# 10. Target/mask provenance (preserved fail-closed)
# ===========================================================================
class TestTargetMaskFailClosed:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_source_target_is_missing_state(self, method, tmp_path,
                                                     bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        provider = _make_provider(method)
        manifest = _make_manifest(method)

        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img_path)

        pinfo = {
            "provider": provider, "pipe": _MOCK_PIPE,
            "extract_module": mock.MagicMock(),
            "device_obj": _MOCK_TORCH.device.return_value,
            "method": method, "_manifest": manifest,
        }
        record = _make_record(bundle_dir, "1", method=method,
                              watermark_target_sha256="",
                              watermark_mask_sha256="provider_mask_sha")

        with pytest.raises(DetectorMissingStateError,
                           match="missing watermark_target_sha256"):
            fourier_detector.score_image(pinfo, str(img_path), record=record)

    def test_hsqr_mask_cannot_fallback_to_source_record(self, tmp_path,
                                                         bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = mock.MagicMock(spec=["get_wm_type", "bundle", "profile"])
        provider.get_wm_type.return_value = "HSQR"
        provider.bundle = mock.MagicMock()
        provider.bundle.manifest = {"selected_pattern_sha256": "hsqr_target"}
        provider.profile = "official_sfwmark_sd21"

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


# ===========================================================================
# 11. Method tag + key index (preserved)
# ===========================================================================
class TestMethodAndKeyIdentity:

    @pytest.mark.parametrize("eval_method,record_method", [
        ("RID", "HSTR"), ("HSTR", "RID"), ("HSQR", "RID"),
    ])
    def test_record_method_tag_must_match(self, eval_method, record_method,
                                           monkeypatch, bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorStateValidationError

        provider = _make_provider(eval_method)
        extract_mod = _build_extract_module(eval_method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, eval_method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=record_method)
        with pytest.raises(DetectorStateValidationError, match="method tag"):
            fourier_detector.load_state([record], "cpu", method=eval_method)

    @pytest.mark.parametrize("method", ["RID", "HSTR"])
    def test_key_index_must_match_bundle(self, method, monkeypatch, bundle_dir):
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
                              **{f"{prefix}_key_index": "7"})
        with pytest.raises(DetectorStateValidationError, match="key_index"):
            fourier_detector.load_state([record], "cpu", method=method)


# ===========================================================================
# 12. Score definitions (preserved)
# ===========================================================================
class TestScoreDefinitions:

    @pytest.mark.parametrize("method,expected_def", [
        ("RID", "rid_neg_channel_min_complex_l1"),
        ("HSTR", "hstr_score=-min(channel_0_l1,channel_3_l1)"),
        ("HSQR", "hsqr_negative_mean_complex_l1_distance"),
    ])
    def test_aggregate_score_definition(self, method, expected_def):
        from raven.detectors import fourier_detector

        rows = [{"run_id": "1", "evaluation_cohort": "original_watermarked",
                 "status": "scored", "canonical_score": -0.1}]
        result = fourier_detector.aggregate(rows, method=method)
        assert result["score_definition"] == expected_def


# ===========================================================================
# 13. Missing image → FileNotFoundError (preserved)
# ===========================================================================
class TestMissingImageFileNotFound:

    def test_missing_image_raises_file_not_found(self):
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
# 14. Orchestrator integration — REAL adapter, real schema
# ===========================================================================
class _OrchestratorFixtures:
    """Shared helpers for orchestrator integration tests."""

    @staticmethod
    def _build_orchestrator_env(tmp_path, method, monkeypatch, *,
                                 manifest_overrides=None,
                                 csv_overrides=None):
        """Create temp dirs, records, metadata CSV; wire up REAL adapter.

        Returns (out_dir, records, extract_mod, provider, manifest,
                 metadata_csv, bundle_dir).
        """
        from raven.detectors import fourier_detector

        prefix = method.lower()
        manifest = _make_manifest(method, **(manifest_overrides or {}))
        profile = str(manifest.get("profile_name", _default_profile(method)))
        provider = _make_provider(method, profile=profile)
        if method in {"RID", "HSTR"}:
            provider.state_source = "bundle"
        provider.selected_pattern_sha256 = "provider_target_sha"
        provider.watermark_mask_sha256 = "provider_mask_sha"
        if method == "HSQR":
            provider.bundle.manifest = dict(manifest)
            provider.bundle.manifest["selected_pattern_sha256"] = \
                "provider_target_sha"
            provider.bundle.manifest["mask_sha256"] = "provider_mask_sha"

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

        metadata_fields = [
            "run_id", "role", "method",
            f"{prefix}_bundle_dir", f"{prefix}_bundle_config_sha256",
            f"{prefix}_selected_pattern_sha256", f"{prefix}_mask_sha256",
            f"{prefix}_key_index", f"{prefix}_protocol_mode",
            "watermark_target_sha256", "watermark_mask_sha256",
        ]
        csv_lines = [",".join(metadata_fields)]
        for i in range(2):
            csv_lines.append(",".join([
                str(i), "watermarked", method,
                str(bundle_dir), manifest["bundle_config_sha256"],
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                str(manifest.get("selected_key_index", 0)),
                _METHOD_PROTOCOL_MODES[method],
                "provider_target_sha", "provider_mask_sha",
            ]))
        metadata_csv = tmp_path / "metadata.csv"
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
            records.append(rec)

        extract_mod = _build_extract_module(method, provider=provider,
                                             manifest=manifest,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        return (out_dir, records, extract_mod, provider, manifest,
                metadata_csv, bundle_dir)


class TestOrchestratorSuccess:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_valid_real_schema_success(self, method, monkeypatch, tmp_path):
        """Real-schema RID/HSTR/HSQR → completed, provider count = 1."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        (out_dir, records, extract_mod, provider, manifest,
         metadata_csv, _bundle_dir) = \
            _OrchestratorFixtures._build_orchestrator_env(
                tmp_path, method, monkeypatch)

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)

        assert result["stage"] == "detector"
        assert result["status"] in ("completed", "completed_with_errors")
        assert result["scored_count"] == 4  # 2 records × 2 cohorts
        assert result["failed_count"] == 0

        # Exactly one provider constructed (cohort-wide detector)
        provider_count = _MOCK_RINGID.call_count + _MOCK_HSTR.call_count
        if method == "HSQR":
            provider_count = extract_mod.hsqr_provider_from_bundle.call_count
        assert provider_count == 1

        # Real aggregate used — method-specific score definition present
        assert result["score_definition"] == \
            fourier_detector_score_def(method)

        # Detector records written
        det_path = out_dir / "evaluation" / "detector_records.jsonl"
        assert det_path.is_file()
        lines = det_path.read_text().strip().split("\n")
        assert len(lines) == 4

    def test_valid_rid_protocol_differs_from_profile(self, monkeypatch,
                                                      tmp_path):
        """RID protocol=official_math_shared_tr_clean + profile=legacy succeeds."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        (out_dir, records, extract_mod, provider, manifest,
         metadata_csv, _bundle_dir) = \
            _OrchestratorFixtures._build_orchestrator_env(
                tmp_path, "RID", monkeypatch,
                manifest_overrides={"profile_name": "legacy"})

        write_config(out_dir, {"method": "RID", "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": "RID", "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, "RID", device="cpu",
                                   config=eval_config)

        assert result["status"] in ("completed", "completed_with_errors")
        assert result["scored_count"] == 4


class TestOrchestratorFailures:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_second_row_missing_bundle_stage(self, method, monkeypatch,
                                              tmp_path):
        """Row 2 bundle missing → failed_missing_required_state stage."""
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
            # Row 2 points at a non-existent bundle dir
            row_bundle = str(bundle_dir) if i == 0 else "/nonexistent/row2"
            csv_lines.append(",".join([
                rid, "watermarked", method,
                row_bundle, manifest["bundle_config_sha256"],
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                str(manifest.get("selected_key_index", 0)),
                _METHOD_PROTOCOL_MODES[method],
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

        assert result["status"] == "failed_missing_required_state"

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_wrong_provider_profile_fails_stage(self, method, monkeypatch,
                                                 tmp_path):
        """manifest profile != provider profile → failed_state_validation."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        prefix = method.lower()
        manifest = _make_manifest(method)
        # Provider profile deliberately disagrees with manifest.
        # manifest default profile is the first of the real profile list;
        # use the last real profile as the contradicting provider profile.
        other_profile = _METHOD_PROFILES[method][-1]
        provider = _make_provider(method, profile=other_profile)
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
            csv_lines.append(",".join([
                rid, "watermarked", method,
                str(bundle_dir), manifest["bundle_config_sha256"],
                manifest["selected_pattern_sha256"],
                manifest.get("mask_sha256", ""),
                str(manifest.get("selected_key_index", 0)),
                _METHOD_PROTOCOL_MODES[method],
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

    @pytest.mark.parametrize("requested,wrong", [
        ("HSTR", "HSQR"),
        ("HSQR", "HSTR"),
    ])
    def test_orchestrator_wrong_manifest_method_is_failed_state_validation(
            self, requested, wrong, monkeypatch, tmp_path):
        """Wrong manifest method → failed_state_validation, exit 2 always."""
        from experiments.eval import evaluate_detector
        from experiments.eval import determine_exit_code
        from raven.experiment_io import write_config, write_record

        (out_dir, records, extract_mod, provider, manifest,
         metadata_csv, _bundle_dir) = \
            _OrchestratorFixtures._build_orchestrator_env(
                tmp_path, requested, monkeypatch)

        # Manifest carries the WRONG method tag — the canonical bundle helper
        # (mocked here) must return it so the adapter's method-identity
        # validation rejects it before any pipe/provider construction.
        bad_manifest = dict(manifest)
        bad_manifest["method"] = wrong
        (tmp_path / "bundle" / "manifest.json").write_text(
            json.dumps(bad_manifest))

        # Rebuild the extract module mock bound to the bad manifest
        from test_issue24_fourier_detector import (
            _build_extract_module as _rebuild_extract,
            _setup_adapter_mocks as _rebuild_mocks,
        )
        bad_extract = _rebuild_extract(
            requested, provider=provider, manifest=bad_manifest,
            bundle_dir=tmp_path / "bundle")
        _rebuild_mocks(monkeypatch, requested, provider, bad_extract)

        write_config(out_dir, {"method": requested, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": requested, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, requested, device="cpu",
                                   config=eval_config)

        assert result["status"] == "failed_state_validation"
        # Provider must never be constructed
        assert _MOCK_RINGID.call_count == 0
        assert _MOCK_HSTR.call_count == 0

        # Exit code policy: never allowable
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=False) == 2
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=True) == 2

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_raw_score_failure_fails_stage(self, method, monkeypatch, tmp_path):
        """raw_score failure → failed_scoring stage."""
        from experiments.eval import evaluate_detector
        from raven.experiment_io import write_config, write_record

        (out_dir, records, extract_mod, provider, manifest,
         metadata_csv, _bundle_dir) = \
            _OrchestratorFixtures._build_orchestrator_env(
                tmp_path, method, monkeypatch)

        # raw_score raises → every row fails_scoring
        extract_mod.raw_score.side_effect = KeyError("l1_dist")

        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)

        eval_config = {"method": method, "dataset": "test",
                       "metadata_path": str(metadata_csv)}
        result = evaluate_detector(records, out_dir, method, device="cpu",
                                   config=eval_config)

        assert result["status"] == "failed_scoring"
        assert result["failed_count"] == 4
        assert result["scored_count"] == 0


# ===========================================================================
# 15. Metadata preflight (preserved)
# ===========================================================================
class TestMetadataPreflight:

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_missing_metadata_checked_before_manifest_helper(self, method,
                                                              monkeypatch,
                                                              bundle_dir):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        prefix = method.lower()
        provider = _make_provider(method)
        extract_mod = _build_extract_module(method, provider=provider,
                                             bundle_dir=bundle_dir)
        _setup_adapter_mocks(monkeypatch, method, provider, extract_mod)

        record = _make_record(bundle_dir, "1", method=method)
        del record[f"{prefix}_key_index"]

        with pytest.raises(DetectorMissingStateError, match="is None"):
            fourier_detector.load_state([record], "cpu", method=method)

        extract_mod.fourier_bundle_manifest.assert_not_called()

    @pytest.mark.parametrize("method", FOURIER_METHODS)
    def test_empty_records(self, method):
        from raven.detectors import fourier_detector
        from raven.detectors import DetectorMissingStateError

        with pytest.raises(DetectorMissingStateError, match="no records"):
            fourier_detector.load_state([], "cpu", method=method)


# ---------------------------------------------------------------------------
# Small helper: expected score definition from the real adapter
# ---------------------------------------------------------------------------
def fourier_detector_score_def(method: str) -> str:
    from raven.detectors.fourier_detector import _METHOD_SCORE_DEFINITIONS
    return _METHOD_SCORE_DEFINITIONS[method]
