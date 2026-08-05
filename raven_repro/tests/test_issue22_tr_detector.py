"""Issue #22/#28 regression tests — TR detector bound to canonical
source metadata profile, with package-local complex-L1 scoring.

The detector imports its scoring math from ``raven.detectors.tr_scoring``
directly — no legacy script loading, no ``importlib.util``.

Covers:
- w_pattern_const in REQUIRED_METADATA_FIELDS, passed to provider
- Strict type validation (missing vs invalid separation)
- Canonical source schema aliases (scheduler_target, num_inference_steps_target)
- Optional source assertions vs runtime-derived identity
- Provider config hash canonical semantics (missing = legal)
- Metadata steps control canonical inversion
- VAE identity binding to the pipe
- Scheduler validation via the real pipe registry
- w_channel = -1 contract
- Strict numeric parsing taxonomy
- w_measurement == "l1_complex" contract
- Default score protocol: complex L1 mean (raw lower-is-watermarked,
  canonical -raw higher-is-watermarked, comparison >=), optional named
  log10p mode with separate provenance
- Complex-L1 formula: torch.abs(decoded - target).mean() over masked FFT
- Threshold equality / tie policy via the canonical calibration helper
- Recalibration computation errors never disguised as unavailable
- Real generator-schema integration through evaluate_detector
- Mixed model_id / model_revision rejection
- Manifest builder retains required source identity
- Real TrProvider scoring through tr_scoring (no mocked math)

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

# Full profile: includes extraction-output fields (optional source assertions).
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
    "provider_config_hash": "",
    "watermark_target_sha256": "default_target_sha_placeholder",
    "watermark_mask_sha256": "default_mask_sha_placeholder",
}

# Generator-style profile: exactly what experiments/generate_watermarked_images.py
# writes for a TR row.  NO extraction-output fields.
TR_PROFILE_GENERATOR: dict[str, str] = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "scheduler_target": "DDIM",
    "num_inference_steps_target": "50",
    "resolution": "512",
    "watermark_target_sha256": "default_target_sha_placeholder",
    "watermark_mask_sha256": "default_mask_sha_placeholder",
}

# The exact set of source-required identity fields (no aliases counted twice).
SOURCE_REQUIRED_CANONICAL = {
    "model_id", "model_revision", "scheduler", "steps", "resolution",
    "watermark_target_sha256", "watermark_mask_sha256",
}
# Extraction-output fields that must NOT be required from source rows.
EXTRACTION_ONLY_FIELDS = {
    "inverse_scheduler", "detector_dtype", "vae_id", "vae_scaling_factor",
    "provider_config_hash",
}


def _make_record(run_id="1", role="watermarked", method="TR",
                 provider_meta=None, profile=None, **kw):
    """Build a synthetic record with TR fields at top level (mimics
    MetadataResolver.enrich_record).  When the profile carries a
    provider_config_hash key, it is recomputed from the provider metadata so
    it matches canonical semantics; generator-style profiles (no hash key)
    stay absent."""
    if provider_meta is None:
        provider_meta = dict(TR_META_COMPLETE)
    if profile is None:
        profile = dict(TR_PROFILE)

    if "provider_config_hash" in profile and \
            not str(profile["provider_config_hash"]).strip():
        # Only auto-compute when the profile did not set an explicit value
        # (tests that assert a mismatch set their own).
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


def _generator_record(run_id="1", role="watermarked", method="TR",
                      provider_meta=None, profile=None, **kw):
    """Build a generator-style record: source schema fields ONLY."""
    if profile is None:
        profile = dict(TR_PROFILE_GENERATOR)
    return _make_record(run_id=run_id, role=role, method=method,
                        provider_meta=provider_meta, profile=profile, **kw)


# ===========================================================================
# Mock harness for load_state unit tests
# ===========================================================================
_FAKE_SCHEDULER_CLASSES = {
    "DDIM": (mock.MagicMock(), mock.MagicMock()),
    "DPM": (mock.MagicMock(), mock.MagicMock()),
    "DPM_DDIM_INV": (mock.MagicMock(), mock.MagicMock()),
    "MAXSIVE_DPM": (mock.MagicMock(), mock.MagicMock()),
    "Euler": (None, None),
}


@contextmanager
def _mock_load_state_deps(monkeypatch):
    """Replace heavy imports so load_state can run CPU-only."""
    import builtins

    fake_pipe_obj = mock.MagicMock()
    fake_pipe_obj.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe_obj.get_dtype.return_value = "torch.float32"
    scheduler_inv = mock.MagicMock()
    scheduler_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe_obj.scheduler_inverse = scheduler_inv
    fake_pipe_obj.pipe.vae.config.scaling_factor = 0.18215

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.SCHEDULER_CLASSES = dict(_FAKE_SCHEDULER_CLASSES)
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
    yield fake_pipe_obj, fake_tr_provider_class, fake_pipe_utils


def _patch_tensor_sha256(monkeypatch, *sha_values):
    """Patch raven.pairing_provenance.tensor_sha256."""
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
    def test_w_pattern_const_in_required_set(self):
        from raven.detectors.tr_detector import REQUIRED_METADATA_FIELDS
        assert "w_pattern_const" in REQUIRED_METADATA_FIELDS

    def test_w_pattern_const_missing_raises_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE)
        del meta["w_pattern_const"]
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            with pytest.raises(DetectorMissingStateError,
                               match="w_pattern_const"):
                load_state(records, "cpu")
            assert tr_cls.call_count == 0

    def test_w_pattern_const_blank_raises_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE, w_pattern_const="")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            with pytest.raises(DetectorMissingStateError,
                               match="w_pattern_const"):
                load_state(records, "cpu")
            assert tr_cls.call_count == 0

    def test_w_pattern_const_value_passed_to_provider(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        meta = dict(TR_META_COMPLETE, w_pattern_const="0.75")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            assert tr_cls.call_count == 1
            assert tr_cls.call_args.kwargs["w_pattern_const"] == 0.75

    def test_w_pattern_const_negative_value_allowed(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        meta = dict(TR_META_COMPLETE, w_pattern_const="-1.5")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            assert tr_cls.call_args.kwargs["w_pattern_const"] == -1.5

    def test_w_pattern_const_mixed_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", provider_meta=dict(TR_META_COMPLETE,
                                                  w_pattern_const="0.0")),
            _make_record("2", provider_meta=dict(TR_META_COMPLETE,
                                                  w_pattern_const="0.75")),
        ]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")
            assert tr_cls.call_count == 0

    def test_w_pattern_const_hash_0_0_but_metadata_0_75_rejected(self, monkeypatch):
        """A hash recorded under w_pattern_const=0.0 must not validate a
        metadata row that says 0.75 — the hash is re-derived, not copied."""
        from raven.detectors.tr_detector import load_state
        from raven.detectors import DetectorStateValidationError
        from raven.eval_protocol import provider_config_hash

        meta_0 = dict(TR_META_COMPLETE, w_pattern_const="0.0")
        hash_0 = provider_config_hash("TR", meta_0)
        record = _make_record("1", provider_meta=dict(
            TR_META_COMPLETE, w_pattern_const="0.75"))
        record["provider_config_hash"] = hash_0

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            with pytest.raises(DetectorStateValidationError,
                               match="provider_config_hash"):
                load_state([record], "cpu")
            assert tr_cls.call_count == 0

    def test_w_pattern_const_nan_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_pattern_const=str(float("nan")))
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_pattern_const"):
                load_state(records, "cpu")

    def test_w_pattern_const_inf_rejected(self, monkeypatch):
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

    def test_w_radius_zero_is_state_validation(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_radius="0")
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
# 3 — Canonical source schema: aliases for scheduler / steps
# ===========================================================================
class TestSchedulerAlias:
    def test_scheduler_target_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch) as (_pipe, _tr, pipe_utils):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["verified_profile"]["scheduler"] == "DDIM"
            call_kwargs = pipe_utils.get_pipe_provider.call_args.kwargs
            assert call_kwargs["schedulers_name"] == "DDIM"

    def test_scheduler_canonical_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["scheduler_target"]
        profile["scheduler"] = "DPM"
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, _tr, pipe_utils):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["verified_profile"]["scheduler"] == "DPM"

    def test_both_aliases_agree_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR, scheduler="DDIM")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["verified_profile"]["scheduler"] == "DDIM"

    def test_both_aliases_conflict_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, scheduler="DDPM")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="scheduler"):
                load_state(records, "cpu")

    def test_neither_alias_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["scheduler_target"]
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="scheduler"):
                load_state(records, "cpu")

    def test_mixed_scheduler_across_rows_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile_a = dict(TR_PROFILE_GENERATOR, scheduler_target="DDIM")
        profile_b = dict(TR_PROFILE_GENERATOR, scheduler_target="DPM")
        records = [
            _generator_record("1", profile=profile_a),
            _generator_record("2", profile=profile_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="scheduler"):
                load_state(records, "cpu")


class TestStepsAlias:
    def test_num_inference_steps_target_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR,
                       num_inference_steps_target="17")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["inversion_steps"] == 17
            assert result["verified_profile"]["steps"] == 17

    def test_steps_canonical_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["num_inference_steps_target"]
        profile["steps"] = "25"
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["inversion_steps"] == 25

    def test_both_aliases_conflict_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, steps="60")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="steps"):
                load_state(records, "cpu")

    def test_neither_alias_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["num_inference_steps_target"]
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="steps"):
                load_state(records, "cpu")


# ===========================================================================
# 4 — Metadata steps control canonical inversion
# ===========================================================================
class TestInversionStepsControl:
    def _patch_scoring(self, monkeypatch, raw=0.001):
        """Patch the package-local complex-L1 helper; record its call args."""
        import raven.detectors.tr_scoring as tr_scoring

        fake = mock.MagicMock(return_value={
            "score": raw,
            "decoded_abs_mean": 1.0,
            "target_abs_mean": 1.0,
            "nan": False,
            "inf": False,
        })
        monkeypatch.setattr(tr_scoring, "complex_l1_score", fake)
        return fake

    def _provider_info(self, steps):
        return {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "score_mode": "complex_l1_mean",
            "inversion_steps": steps,
        }

    def test_evaluate_image_receives_metadata_steps(self, monkeypatch, tmp_path):
        """With no caller override, the verified metadata steps (17) must
        reach the scoring helper — never the old hard-coded 50."""
        import raven.detectors.tr_scoring as tr_scoring
        from raven.detectors.tr_detector import score_image

        fake = self._patch_scoring(monkeypatch)
        info = self._provider_info(17)
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (64, 64)).save(img)

        score_image(info, str(img))

        call = fake.call_args
        assert call.args[4] == 17, f"expected steps=17, got {call.args[4]}"

    def test_caller_override_conflict_rejected(self, monkeypatch, tmp_path):
        from raven.detectors.tr_detector import score_image
        from raven.detectors import DetectorStateValidationError

        self._patch_scoring(monkeypatch)
        info = self._provider_info(17)
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (64, 64)).save(img)

        with pytest.raises(DetectorStateValidationError, match="steps"):
            score_image(info, str(img), steps=25)

    def test_caller_matching_steps_accepted(self, monkeypatch, tmp_path):
        from raven.detectors.tr_detector import score_image

        self._patch_scoring(monkeypatch, raw=0.001)
        info = self._provider_info(17)
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (64, 64)).save(img)

        result = score_image(info, str(img), steps=17)
        assert result["canonical_score"] == -0.001
        assert result["tr_score_definition"] == "complex_l1_mean"


# ===========================================================================
# 5 — Optional source assertions vs runtime-derived identity
# ===========================================================================
class TestOptionalAssertions:
    def test_absent_fields_use_runtime_without_assertion(self, monkeypatch):
        """Generator-style rows (no extraction fields) must work and record
        runtime values with source_asserted=False."""
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["tr_inverse_scheduler_source_asserted"] is False
        assert result["tr_detector_dtype_source_asserted"] is False
        assert result["tr_vae_id_source_asserted"] is False
        assert result["tr_vae_scaling_source_asserted"] is False
        assert result["tr_provider_config_hash_source_asserted"] is False
        assert result["source_provider_config_hash"] == ""
        assert result["detector_provider_config_hash"] != ""

        vp = result["verified_profile"]
        assert vp["inverse_scheduler"] == "DDIMScheduler"  # runtime-derived
        assert vp["detector_dtype"] == "torch.float32"  # runtime-derived
        assert vp["vae_id"] == "checkpoint-default"  # runtime convention
        assert vp["vae_scaling_factor"] == 0.18215  # runtime-derived

    def test_wrong_asserted_inverse_scheduler_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, inverse_scheduler="DDPMScheduler")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="inverse_scheduler"):
                load_state(records, "cpu")

    def test_wrong_asserted_dtype_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, detector_dtype="torch.float16")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="detector_dtype"):
                load_state(records, "cpu")

    def test_wrong_asserted_vae_scaling_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, vae_scaling_factor="0.5")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="vae_scaling_factor"):
                load_state(records, "cpu")


# ===========================================================================
# 5b — Partial optional assertions are rejected per row
# ===========================================================================
class TestPartialAssertions:
    def test_partial_vae_id_assertion_rejected(self, monkeypatch):
        """Row 1 asserts vae_id, row 2 does not → state validation, never a
        cohort-wide asserted label."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        row1 = _make_record("1", profile=dict(TR_PROFILE, vae_id="custom/vae"))
        row2 = _make_record("2", profile=dict(TR_PROFILE))
        del row2["vae_id"]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="vae_id"):
                load_state([row1, row2], "cpu")

    def test_partial_vae_scaling_assertion_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        row1 = _make_record("1", profile=dict(TR_PROFILE,
                                              vae_scaling_factor="0.18215"))
        row2 = _make_record("2", profile=dict(TR_PROFILE))
        del row2["vae_scaling_factor"]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="vae_scaling_factor"):
                load_state([row1, row2], "cpu")

    def test_partial_provider_hash_assertion_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        row1 = _make_record("1")
        row2 = _make_record("2")
        del row2["provider_config_hash"]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="provider_config_hash"):
                load_state([row1, row2], "cpu")

    def test_partial_assertion_error_lists_run_ids(self, monkeypatch):
        """Error message must include the run_ids of present and missing rows."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        row1 = _make_record("10", profile=dict(TR_PROFILE, vae_id="custom/vae"))
        row2 = _make_record("20", profile=dict(TR_PROFILE))
        del row2["vae_id"]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError) as excinfo:
                load_state([row1, row2], "cpu")
        message = str(excinfo.value)
        assert "10" in message
        assert "20" in message

    def test_all_absent_optional_assertion_is_false(self, monkeypatch):
        """Generator-style rows: every optional assertion absent → (None,
        False) — no fabrication of source verification."""
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1"), _generator_record("2")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["tr_vae_id_source_asserted"] is False
        assert result["tr_vae_scaling_source_asserted"] is False
        assert result["tr_provider_config_hash_source_asserted"] is False
        assert result["tr_inverse_scheduler_source_asserted"] is False
        assert result["tr_detector_dtype_source_asserted"] is False

    def test_all_present_matching_optional_assertion_is_true(self, monkeypatch):
        """Every record asserts the same value → (value, True)."""
        from raven.detectors.tr_detector import load_state

        records = [
            _make_record("1", profile=dict(TR_PROFILE, vae_id="custom/vae")),
            _make_record("2", profile=dict(TR_PROFILE, vae_id="custom/vae")),
        ]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["tr_vae_id_source_asserted"] is True
        assert result["verified_profile"]["vae_id"] == "custom/vae"


# ===========================================================================
# 6 — Provider config hash canonical semantics
# ===========================================================================
class TestProviderHashSemantics:
    def test_absent_hash_is_legal(self, monkeypatch):
        """Missing/empty recorded hash must NOT be missing state."""
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["tr_provider_config_hash_source_asserted"] is False
        assert result["detector_provider_config_hash"] != ""
        assert result["tr_provider_config_verified"] is True

    def test_matching_hash_verified(self, monkeypatch):
        from raven.detectors.tr_detector import load_state
        from raven.eval_protocol import provider_config_hash

        expected = provider_config_hash("TR", TR_META_COMPLETE)
        profile = dict(TR_PROFILE, provider_config_hash=expected)
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["tr_provider_config_hash_source_asserted"] is True
        assert result["detector_provider_config_hash"] == expected

    def test_mismatched_hash_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, provider_config_hash="deadbeef")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="provider_config_hash"):
                load_state(records, "cpu")


# ===========================================================================
# 7 — VAE identity binding
# ===========================================================================
class TestVaeBinding:
    def test_vae_id_passed_to_pipe(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE, vae_id="custom/vae")
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, _tr, pipe_utils):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

            call_kwargs = pipe_utils.get_pipe_provider.call_args.kwargs
            assert call_kwargs["vae_id"] == "custom/vae"
            assert result["verified_profile"]["vae_id"] == "custom/vae"
            assert result["tr_vae_id_source_asserted"] is True

    def test_vae_id_absent_uses_checkpoint_default(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch) as (_pipe, _tr, pipe_utils):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

            call_kwargs = pipe_utils.get_pipe_provider.call_args.kwargs
            assert "vae_id" not in call_kwargs
            assert result["verified_profile"]["vae_id"] == "checkpoint-default"
            assert result["tr_vae_id_source_asserted"] is False


# ===========================================================================
# 8 — Scheduler validation via real pipe registry
# ===========================================================================
class TestSchedulerRegistry:
    @pytest.mark.parametrize("scheduler", ["DDIM", "DPM", "DPM_DDIM_INV"])
    def test_registered_schedulers_accepted(self, monkeypatch, scheduler):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR, scheduler_target=scheduler)
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")
            assert result["verified_profile"]["scheduler"] == scheduler

    def test_unregistered_scheduler_rejected(self, monkeypatch):
        """DDPM is NOT in pipe_utils.SCHEDULER_CLASSES → state validation."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, scheduler_target="DDPM")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="scheduler"):
                load_state(records, "cpu")

    def test_unknown_scheduler_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, scheduler_target="UNKNOWN")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="scheduler"):
                load_state(records, "cpu")


# ===========================================================================
# 9 — w_channel contract (-1 = all channels)
# ===========================================================================
class TestWChannelRange:
    @pytest.mark.parametrize("channel", ["-1", "0", "3"])
    def test_valid_channels_accepted(self, monkeypatch, channel):
        from raven.detectors.tr_detector import load_state

        meta = dict(TR_META_COMPLETE, w_channel=channel)
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            assert tr_cls.call_args.kwargs["w_channel"] == int(channel)

    def test_channel_out_of_range_rejected(self, monkeypatch):
        """4-channel latent; w_channel=4 is out of range."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_channel="4")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_channel"):
                load_state(records, "cpu")

    def test_channel_below_minus_one_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_channel="-2")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_channel"):
                load_state(records, "cpu")


# ===========================================================================
# 10 — Strict numeric parsing taxonomy
# ===========================================================================
class TestStrictNumericParsing:
    @pytest.mark.parametrize("steps", ["abc", "0", "-5"])
    def test_invalid_steps_rejected(self, monkeypatch, steps):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, num_inference_steps_target=steps)
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError, match="steps"):
                load_state(records, "cpu")

    @pytest.mark.parametrize("resolution", ["abc", "0", "100", "-512"])
    def test_invalid_resolution_rejected(self, monkeypatch, resolution):
        """0 and 100 violate minimum / multiple-of-8; abc is malformed."""
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE_GENERATOR, resolution=resolution)
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="resolution"):
                load_state(records, "cpu")

    @pytest.mark.parametrize("scaling", ["abc", str(float("nan")),
                                          str(float("inf")), "0", "-0.5"])
    def test_invalid_vae_scaling_rejected(self, monkeypatch, scaling):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        profile = dict(TR_PROFILE, vae_scaling_factor=scaling)
        records = [_make_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="vae_scaling_factor"):
                load_state(records, "cpu")


# ===========================================================================
# 11 — w_measurement canonical contract
# ===========================================================================
class TestMeasurementContract:
    @pytest.mark.parametrize("measurement", ["l2_complex", "l1"])
    def test_non_canonical_measurement_rejected(self, monkeypatch, measurement):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta = dict(TR_META_COMPLETE, w_measurement=measurement)
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="w_measurement"):
                load_state(records, "cpu")

    def test_empty_measurement_is_missing_state(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE, w_measurement="")
        records = [_make_record("1", provider_meta=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError,
                               match="w_measurement"):
                load_state(records, "cpu")

    def test_l1_complex_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_make_record("1")]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            assert tr_cls.call_args.kwargs["w_measurement"] == "l1_complex"


# ===========================================================================
# 12 — Recalibration computation error never disguised as unavailable
# ===========================================================================
class TestRecalibrationError:
    def test_recal_metric_error_is_structured(self, monkeypatch):
        """All recalibration inputs present + summarize_detection raises →
        structured DetectorScoringError, never a fake unavailable block."""
        from raven.detectors.tr_detector import aggregate, DetectorScoringError
        from raven.detectors import ROW_STATUS_SCORED

        import raven.metrics as metrics
        monkeypatch.setattr(
            metrics, "summarize_detection",
            mock.MagicMock(side_effect=RuntimeError("metric bug")))

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

        with pytest.raises(DetectorScoringError, match="metric bug"):
            aggregate(rows)

    def test_primary_metric_error_is_structured(self, monkeypatch):
        """Primary cohorts present + summarize_detection raises → structured
        DetectorScoringError."""
        from raven.detectors.tr_detector import aggregate, DetectorScoringError
        from raven.detectors import ROW_STATUS_SCORED

        import raven.metrics as metrics
        monkeypatch.setattr(
            metrics, "summarize_detection",
            mock.MagicMock(side_effect=RuntimeError("primary bug")))

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
        ]

        with pytest.raises(DetectorScoringError, match="primary bug"):
            aggregate(rows)

    def test_recal_metric_error_with_missing_positive_cohort_is_unavailable(self, monkeypatch):
        """When the positive cohorts have no scored rows the recalibration
        genuinely cannot run — plain unavailability, no summarize call."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        import raven.metrics as metrics
        fake_summary = mock.MagicMock(
            side_effect=RuntimeError("should never be called"))
        monkeypatch.setattr(metrics, "summarize_detection", fake_summary)

        # attacked_clean + original_clean scored, but watermarked/attacked
        # rows all failed → recalibration inputs incomplete.
        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 3.0},
            {"status": "failed_missing_state", "evaluation_cohort": "original_watermarked",
             "canonical_score": None},
            {"status": "failed_missing_state", "evaluation_cohort": "attacked_watermarked",
             "canonical_score": None},
        ]

        result = aggregate(rows)
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False
        fake_summary.assert_not_called()

    def test_no_attacked_clean_is_plain_unavailable(self, monkeypatch):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        import raven.metrics as metrics
        fake_summary = mock.MagicMock()
        monkeypatch.setattr(metrics, "summarize_detection", fake_summary)

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
        ]

        result = aggregate(rows)
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False
        # summarize_detection is called exactly once — for the primary
        # original-threshold report — and never for the unavailable
        # recalibrated block.
        assert fake_summary.call_count == 1


# ===========================================================================
# 13 — Aggregate cohort semantics (regression)
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
# 14 — Scoring boundary taxonomy
# ===========================================================================
class TestScoringBoundary:
    def _make_provider_info(self, steps=50, score_mode="complex_l1_mean"):
        return {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "score_mode": score_mode,
            "inversion_steps": steps,
        }

    def _patch_l1(self, monkeypatch, **result):
        import raven.detectors.tr_scoring as tr_scoring

        default = {"score": 0.001, "decoded_abs_mean": 1.0,
                   "target_abs_mean": 1.0, "nan": False, "inf": False}
        default.update(result)
        fake = mock.MagicMock(return_value=default)
        monkeypatch.setattr(tr_scoring, "complex_l1_score", fake)
        return fake

    def _patch_log10p(self, monkeypatch, **result):
        import raven.detectors.tr_scoring as tr_scoring

        default = {"p_values": [0.001], "p_value_diagnostics": [
            {"log_p": -20.0, "sigma": 1.0, "lambda": 100.0,
             "statistic": 50.0, "df": 100, "p_underflow": False}]}
        default.update(result)
        fake = mock.MagicMock(return_value=default)
        monkeypatch.setattr(tr_scoring, "evaluate_log10p", fake)
        return fake

    def _make_fake_image(self, tmp_path):
        from PIL import Image
        img = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img)
        return str(img)

    def test_missing_image_raises_file_not_found(self):
        from raven.detectors.tr_detector import score_image

        with pytest.raises(FileNotFoundError):
            score_image({"fake": True, "inversion_steps": 50},
                        "/nonexistent/path.png")

    def test_scoring_failure_is_scoring_error(self, monkeypatch, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        fake = self._patch_l1(monkeypatch)
        fake.side_effect = ValueError("bad raw")

        with pytest.raises(DetectorScoringError, match="bad raw"):
            score_image(self._make_provider_info(), self._make_fake_image(tmp_path))

    def test_canonical_score_failure_is_scoring_error(self, monkeypatch, tmp_path):
        """Non-finite helper output → structured scoring error, never a row."""
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        self._patch_l1(monkeypatch, score=float("nan"))

        with pytest.raises(DetectorScoringError, match="raw_score"):
            score_image(self._make_provider_info(), self._make_fake_image(tmp_path))

    def test_nan_raw_score_is_scoring_error(self, monkeypatch, tmp_path):
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        self._patch_l1(monkeypatch, score=float("nan"), nan=True)

        with pytest.raises(DetectorScoringError, match="raw_score"):
            score_image(self._make_provider_info(), self._make_fake_image(tmp_path))

    def test_log10p_mode_missing_diagnostics_is_scoring_error(self, monkeypatch, tmp_path):
        """Explicitly named p-value mode without diagnostics fails closed."""
        from raven.detectors.tr_detector import score_image, DetectorScoringError

        self._patch_log10p(monkeypatch, p_value_diagnostics=[])

        with pytest.raises(DetectorScoringError,
                           match="p_value_diagnostics"):
            score_image(self._make_provider_info(score_mode="log10p"),
                        self._make_fake_image(tmp_path))

    def test_log10p_mode_requires_explicit_metadata(self, monkeypatch, tmp_path):
        """score_mode=log10p must come from verified metadata; the default
        protocol is complex L1 mean, never -log10(p)."""
        import raven.detectors.tr_scoring as tr_scoring
        from raven.detectors.tr_detector import score_image

        l1 = self._patch_l1(monkeypatch)
        pv = self._patch_log10p(monkeypatch)

        # default mode → complex L1 helper, p-value helper never called
        score_image(self._make_provider_info(), self._make_fake_image(tmp_path))
        assert l1.call_count == 1
        assert pv.call_count == 0
        # explicit mode → p-value helper, L1 helper never called
        score_image(self._make_provider_info(score_mode="log10p"),
                    self._make_fake_image(tmp_path))
        assert pv.call_count == 1
        assert l1.call_count == 1

    def test_no_image_io_in_score_image(self):
        """score_image must not call Image.open — canonical helper does it."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "tr_detector.py").read_text()
        assert "Image.open" not in source
        assert "ImageOps" not in source


# ===========================================================================
# 15 — Integration harness (standalone functions)
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
                       canonical_score_val=-0.001):
    """Mock only pipe, provider construction, the package-local scoring
    helper, and tensor hashing.  Everything else — load_state, score_image,
    evaluate_detector, aggregate, stage reducer — runs for real."""
    import builtins
    import raven.detectors.tr_scoring as tr_scoring

    def _fake_l1(torch, provider, pipe, path, steps):
        raw = float(raw_score_val)
        return {
            "score": raw,
            "decoded_abs_mean": 1.0,
            "target_abs_mean": 1.0,
            "nan": not math.isfinite(raw) or math.isnan(raw),
            "inf": math.isinf(raw),
        }
    fake_l1 = mock.MagicMock(side_effect=_fake_l1)
    monkeypatch.setattr(tr_scoring, "complex_l1_score", fake_l1)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"
    scheduler_inv = mock.MagicMock()
    scheduler_inv.__class__.__name__ = "DDIMScheduler"
    fake_pipe.scheduler_inverse = scheduler_inv
    fake_pipe.pipe.vae.config.scaling_factor = 0.18215

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.SCHEDULER_CLASSES = dict(_FAKE_SCHEDULER_CLASSES)
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

    import raven.pairing_provenance as pp
    monkeypatch.setattr(pp, "tensor_sha256",
                        mock.MagicMock(side_effect=[target_sha, mask_sha]))

    yield fake_pipe, fake_provider, fake_tr_provider_class, fake_pipe_utils


# ===========================================================================
# 16 — Real generator-schema integration tests
# ===========================================================================
class TestGeneratorSchemaIntegration:
    """Real evaluate_detector → load_state → score_image → aggregate pipeline
    driven by generator-style metadata rows (source schema only)."""

    def test_generator_schema_full_pipeline_completed(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_COMPLETED

        profile = dict(TR_PROFILE_GENERATOR,
                       num_inference_steps_target="17")
        rec_clean = _generator_record("1", "clean", profile=profile)
        rec_wm = _generator_record("1", "watermarked", profile=profile)
        rec_wm2 = _generator_record("2", "watermarked", profile=profile)

        with _patch_integration(monkeypatch) as (
                _pipe, _prov, tr_cls, pipe_utils):
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

        # provider constructed exactly once
        assert tr_cls.call_count == 1

        # pipe received metadata profile
        call_kwargs = pipe_utils.get_pipe_provider.call_args.kwargs
        assert call_kwargs["pretrained_model_name_or_path"] == \
            TR_PROFILE_GENERATOR["model_id"]
        assert call_kwargs["schedulers_name"] == "DDIM"
        assert call_kwargs["resolution"] == 512
        assert call_kwargs.get("revision") == \
            TR_PROFILE_GENERATOR["model_revision"]

        # provider received the complete TR config incl. w_pattern_const
        p_kwargs = tr_cls.call_args.kwargs
        for field in ("w_seed", "w_channel", "w_radius", "w_pattern",
                      "w_mask_shape", "w_measurement", "w_injection",
                      "w_pattern_const"):
            assert field in p_kwargs, f"{field} not passed to TrProvider"

    def test_generator_schema_metadata_steps_reach_scoring(self, monkeypatch):
        """num_inference_steps_target=17 must reach the scoring helper."""
        import raven.detectors.tr_scoring as tr_scoring
        from experiments.eval import evaluate_detector

        profile = dict(TR_PROFILE_GENERATOR,
                       num_inference_steps_target="17")
        rec_wm = _generator_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch) as (
                _pipe, _prov, _tr_cls, _pipe_utils):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_wm])
                evaluate_detector([rec_wm], out, "TR", device="cpu")

        steps_seen = [
            call.args[4] for call in tr_scoring.complex_l1_score.call_args_list
        ]
        assert steps_seen, "complex_l1_score was never called"
        for seen in steps_seen:
            assert seen == 17, f"scoring steps={seen}, expected 17"

    def test_detector_records_contain_assertion_flags(self, monkeypatch):
        from experiments.eval import evaluate_detector

        profile = dict(TR_PROFILE_GENERATOR)
        rec_clean = _generator_record("1", "clean", profile=profile)
        rec_wm = _generator_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm])
                evaluate_detector([rec_clean, rec_wm], out, "TR", device="cpu")

                rows = _read_detector_rows(out)
                assert rows, "detector_records.jsonl is empty"
                for row in rows:
                    assert row["status"] == "scored"
                    assert row.get("tr_provider_config_hash_source_asserted") is False
                    assert row.get("tr_inverse_scheduler_source_asserted") is False
                    assert row.get("tr_detector_dtype_source_asserted") is False
                    assert row.get("tr_vae_id_source_asserted") is False
                    assert row.get("tr_vae_scaling_source_asserted") is False
                    assert int(row.get("tr_steps", 0)) > 0
                    assert row.get("tr_model_id") == TR_PROFILE_GENERATOR["model_id"]
                    assert row.get("tr_scheduler") == "DDIM"


# ===========================================================================
# 17 — Orchestrator failure taxonomy (regression)
# ===========================================================================
class TestIntegrationFailures:
    def test_missing_profile_field_setup_failure(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_MISSING_REQUIRED_STATE

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["model_id"]
        rec_wm = _generator_record("1", "watermarked", profile=profile)

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

        profile = dict(TR_PROFILE_GENERATOR)
        del profile["watermark_target_sha256"]
        rec_wm = _generator_record("1", "watermarked", profile=profile)

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

        profile = dict(TR_PROFILE_GENERATOR,
                       watermark_target_sha256="wrong_target",
                       watermark_mask_sha256="default_mask_sha_placeholder")
        rec_wm = _generator_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch,
                                target_sha="real_target",
                                mask_sha="default_mask_sha_placeholder"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_wm])
                result = evaluate_detector(
                    [rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_mixed_provider_config_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        profile = dict(TR_PROFILE_GENERATOR)
        rec_a = _generator_record("1", "watermarked",
                                  provider_meta=dict(TR_META_COMPLETE,
                                                     w_seed="99"),
                                  profile=profile)
        rec_b = _generator_record("2", "watermarked",
                                  provider_meta=dict(TR_META_COMPLETE,
                                                     w_seed="88888"),
                                  profile=profile)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_a, rec_b])
                result = evaluate_detector(
                    [rec_a, rec_b], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_mixed_model_id_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        profile_a = dict(TR_PROFILE_GENERATOR)
        profile_b = dict(TR_PROFILE_GENERATOR, model_id="other/model")
        rec_a = _generator_record("1", "watermarked", profile=profile_a)
        rec_b = _generator_record("2", "watermarked", profile=profile_b)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_a, rec_b])
                result = evaluate_detector(
                    [rec_a, rec_b], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_mixed_model_revision_state_validation(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_STATE_VALIDATION

        profile_a = dict(TR_PROFILE_GENERATOR)
        profile_b = dict(TR_PROFILE_GENERATOR, model_revision="deadbeef")
        rec_a = _generator_record("1", "watermarked", profile=profile_a)
        rec_b = _generator_record("2", "watermarked", profile=profile_b)

        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_a, rec_b])
                result = evaluate_detector(
                    [rec_a, rec_b], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_canonical_helper_failure_scoring_error(self, monkeypatch):
        from experiments.eval import evaluate_detector
        from raven.detectors import STATUS_FAILED_SCORING

        profile = dict(TR_PROFILE_GENERATOR)
        rec_clean = _generator_record("1", "clean", profile=profile)
        rec_wm = _generator_record("1", "watermarked", profile=profile)

        with _patch_integration(monkeypatch, raw_score_val="bad_string",
                                canonical_score_val="also_bad"):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR",
                                      records=[rec_clean, rec_wm])
                result = evaluate_detector(
                    [rec_clean, rec_wm], out, "TR", device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        assert result["scored_count"] == 0


# ===========================================================================
# 18 — Complete contract & manifest compatibility
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

    def test_source_required_identity_not_extraction_output(self):
        """Source-required identity must match the generator schema; the
        extraction-only fields must be optional assertions."""
        from raven.detectors.tr_detector import (
            SOURCE_REQUIRED_IDENTITY, OPTIONAL_ASSERTION_FIELDS,
        )
        source_canonical = set(SOURCE_REQUIRED_IDENTITY)
        assert source_canonical == SOURCE_REQUIRED_CANONICAL
        assert set(OPTIONAL_ASSERTION_FIELDS) == EXTRACTION_ONLY_FIELDS
        # no overlap
        assert source_canonical.isdisjoint(EXTRACTION_ONLY_FIELDS)

    def test_all_required_fields_in_provider_kwargs(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch) as (_pipe, tr_cls, _pu):
            _patch_tensor_sha256(monkeypatch)
            load_state(records, "cpu")
            kwargs = tr_cls.call_args.kwargs
            for field in ("w_seed", "w_channel", "w_radius", "w_pattern",
                          "w_mask_shape", "w_measurement", "w_injection",
                          "w_pattern_const"):
                assert field in kwargs, f"{field} not passed to TrProvider"

    def test_verified_provenance_in_provider_info(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        for key in ("source_provider_config_hash",
                    "detector_provider_config_hash",
                    "source_watermark_target_sha256",
                    "detector_watermark_target_sha256",
                    "source_watermark_mask_sha256",
                    "detector_watermark_mask_sha256",
                    "verified_profile",
                    "inversion_steps"):
            assert key in result, f"Missing {key} in provider_info"

    def test_scored_row_has_provenance_fields(self, monkeypatch, tmp_path):
        import raven.detectors.tr_scoring as tr_scoring
        from raven.detectors.tr_detector import score_image

        monkeypatch.setattr(tr_scoring, "complex_l1_score",
                            mock.MagicMock(return_value={
                                "score": 0.001, "decoded_abs_mean": 1.0,
                                "target_abs_mean": 1.0,
                                "nan": False, "inf": False}))

        info = {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "score_mode": "complex_l1_mean",
            "inversion_steps": 17,
            "detector_provider_config_hash": "h",
            "tr_provider_config_hash_source_asserted": False,
            "source_watermark_target_sha256": "st",
            "detector_watermark_target_sha256": "dt",
            "source_watermark_mask_sha256": "sm",
            "detector_watermark_mask_sha256": "dm",
            "tr_inverse_scheduler_source_asserted": False,
            "tr_detector_dtype_source_asserted": False,
            "tr_vae_id_source_asserted": False,
            "tr_vae_scaling_source_asserted": False,
            "verified_profile": {
                "model_id": "m", "model_revision": "r",
                "scheduler": "DDIM", "inverse_scheduler": "DDIMScheduler",
                "steps": 17, "resolution": 512,
                "detector_dtype": "torch.float32", "vae_id": "checkpoint-default",
                "vae_scaling_factor": 0.18215, "w_pattern_const": 0.75,
            },
        }

        from PIL import Image
        img = tmp_path / "test.png"
        Image.new("RGB", (64, 64)).save(img)

        score = score_image(info, str(img), steps=17)

        for field in ("tr_provider_config_hash", "tr_provider_config_verified",
                      "tr_source_watermark_target_sha256",
                      "tr_detector_watermark_target_sha256",
                      "tr_target_verified",
                      "tr_source_watermark_mask_sha256",
                      "tr_detector_watermark_mask_sha256",
                      "tr_mask_verified", "tr_model_id", "tr_scheduler",
                      "tr_inverse_scheduler_source_asserted",
                      "tr_detector_dtype_source_asserted",
                      "tr_vae_id_source_asserted",
                      "tr_vae_scaling_source_asserted",
                      "tr_w_pattern_const",
                      # default protocol declaration
                      "tr_score_protocol", "tr_score_definition",
                      "tr_raw_score_direction",
                      "tr_canonical_score_direction",
                      "tr_comparison_operator",
                      "tr_decoded_abs_mean", "tr_target_abs_mean"):
            assert field in score, f"Missing {field} in score dict"

        assert score["tr_steps"] == 17
        assert score["tr_score_definition"] == "complex_l1_mean"
        assert score["tr_comparison_operator"] == ">="
        assert score["canonical_score"] == -0.001


class TestManifestCompatibility:
    def test_manifest_retains_detector_identity_fields(self):
        """The manifest builder's real schema must keep every field the TR
        detector adapter requires from source rows."""
        sys.path.insert(0, str(REPO / "raven_repro" / "scripts"))
        import build_diffusiondb_tr_manifest as mod

        fields = set(mod.MANIFEST_FIELDS)
        required = {
            "w_pattern_const",
            "model_id",
            "model_revision",
            "scheduler_target",
            "num_inference_steps_target",
            "resolution",
            "watermark_target_sha256",
            "watermark_mask_sha256",
            "w_seed",
            "w_channel",
            "w_radius",
            "w_pattern",
            "w_mask_shape",
            "w_measurement",
            "w_injection",
        }
        missing = sorted(required - fields)
        assert not missing, f"manifest schema drops required identity: {missing}"

    def test_generator_row_schema_aligns_with_adapter(self):
        """Generator-style row fields must satisfy the adapter's source-required
        identity without extraction outputs."""
        from raven.detectors.tr_detector import SOURCE_REQUIRED_IDENTITY
        # Every canonical source field must be resolvable from a
        # generator-style row (alias sets must intersect the row).
        row = dict(TR_PROFILE_GENERATOR)
        row.update(TR_META_COMPLETE)
        for canonical, aliases in SOURCE_REQUIRED_IDENTITY.items():
            assert any(alias in row for alias in aliases), (
                f"generator row cannot satisfy {canonical} via {aliases}"
            )


# ===========================================================================
# 18b — Score-mode resolution: default complex-L1, explicit log10p
# ===========================================================================
class TestScoreModeResolution:
    def test_default_mode_without_metadata(self, monkeypatch):
        """No tr_score_mode in metadata → default complex-L1 protocol."""
        from raven.detectors.tr_detector import load_state

        records = [_generator_record("1")]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["score_mode"] == "complex_l1_mean"

    def test_explicit_log10p_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        profile = dict(TR_PROFILE_GENERATOR, tr_score_mode="log10p")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            _patch_tensor_sha256(monkeypatch)
            result = load_state(records, "cpu")

        assert result["score_mode"] == "log10p"

    def test_unknown_mode_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import (
            load_state, DetectorStateValidationError)

        profile = dict(TR_PROFILE_GENERATOR, tr_score_mode="bayes")
        records = [_generator_record("1", profile=profile)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="tr_score_mode"):
                load_state(records, "cpu")

    def test_mixed_mode_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import (
            load_state, DetectorStateValidationError)

        profile_a = dict(TR_PROFILE_GENERATOR, tr_score_mode="log10p")
        profile_b = dict(TR_PROFILE_GENERATOR)
        records = [
            _generator_record("1", profile=profile_a),
            _generator_record("2", profile=profile_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="tr_score_mode"):
                load_state(records, "cpu")


# ===========================================================================
# 18c — Complex-L1 protocol: formula, directions, threshold equality
# ===========================================================================
class TestComplexL1Protocol:
    def test_formula_is_abs_mean_over_masked_complex_fft(self, tmp_path):
        """raw = torch.abs(decoded - target).mean() over masked FFT positions."""
        import torch
        from raven.detectors import tr_scoring

        torch.manual_seed(0)
        mask = torch.zeros((1, 4, 64, 64), dtype=torch.bool)
        mask[0, 3, 20:25, 20:25] = True
        target = torch.randn(1, 4, 64, 64, dtype=torch.complex64)
        latents = torch.randn(1, 4, 64, 64, dtype=torch.complex64)

        provider = mock.MagicMock()
        provider.watermarking_mask = mask
        provider.gt_patch = target
        provider.invert_images = mock.MagicMock(
            return_value={"zT_torch": latents})
        pipe = mock.MagicMock()

        from PIL import Image
        img = tmp_path / "in.png"
        Image.new("RGB", (64, 64)).save(img)

        result = tr_scoring.complex_l1_score(torch, provider, pipe, img, 50)

        recovered_fft = torch.fft.fftshift(
            torch.fft.fft2(latents), dim=(-1, -2))
        expected = float(
            torch.abs(recovered_fft[0][mask[0]] - target[0][mask[0]]).mean())
        assert result["score"] == pytest.approx(expected)

    def test_canonical_is_negative_raw(self):
        from raven.detectors import tr_scoring

        assert tr_scoring.canonical_score(0.5) == -0.5
        assert tr_scoring.SCORE_DEFINITION == "complex_l1_mean"
        assert tr_scoring.RAW_SCORE_DIRECTION == "lower_is_watermarked"
        assert tr_scoring.CANONICAL_SCORE_DIRECTION == "higher_is_watermarked"
        assert tr_scoring.COMPARISON_OPERATOR == ">="

    def test_threshold_equality_and_tie_policy(self):
        """Tied clean scores stay one group; detection uses >=."""
        from raven.metrics import calibrate_threshold, detection_rate

        clean = [1.0] * 3 + [float(v) for v in range(2, 99)]
        assert len(clean) == 100
        cal = calibrate_threshold(clean, target_fpr=0.01)
        assert cal.false_positives == 1
        assert cal.actual_fpr == pytest.approx(0.01)
        # threshold equals the top clean score; the 3-way tie at 1.0 is
        # excluded because admitting the whole group would exceed the
        # false-positive budget.
        assert cal.threshold == 98.0
        # detection uses >=: a score exactly at the threshold is detected,
        # a score epsilon below it is not.
        assert detection_rate([cal.threshold], cal.threshold) == 1.0
        assert detection_rate([cal.threshold - 1e-12], cal.threshold) == 0.0


# ===========================================================================
# 19 — Real TrProvider w_channel=-1 canonical scoring
# ===========================================================================
def _real_tr_provider_import():
    """Import the REAL TrProvider, stubbing only utils.image_utils (which
    pulls in lpips, unavailable in this environment).  All watermark math —
    FFT, mask construction, gt_patch, non-central chi-square p-value — runs
    for real."""
    import sys
    import types

    fake_image_utils = types.ModuleType("utils.image_utils")
    fake_image_utils.torch_to_PIL = lambda tensor: tensor
    sys.modules.setdefault("utils.image_utils", fake_image_utils)

    eb = str(REPO / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)

    import importlib
    return importlib.import_module("utils.wm.tr_provider").TrProvider


class TestWChannelMinusOneRealScoring:
    """Real TrProvider scoring with w_channel=-1 — no mocked get_accuracies,
    no mocked mask/gt_patch.  Only pipe inversion and image loading are
    mocked.  Scoring runs through the package-local ``tr_scoring`` module."""

    LATENT_SHAPE = (1, 4, 64, 64)

    @staticmethod
    def _build_provider(w_channel):
        import torch

        TrProvider = _real_tr_provider_import()
        return TrProvider(
            latent_shape=TestWChannelMinusOneRealScoring.LATENT_SHAPE,
            dtype=torch.float32,
            device=torch.device("cpu"),
            w_seed=99,
            w_channel=w_channel,
            w_radius=10,
            w_pattern="ring",
            w_mask_shape="circle",
            w_measurement="l1_complex",
            w_injection="complex",
            w_pattern_const=0.0,
        )

    @staticmethod
    def _mock_pipe():
        import torch
        import types

        pipe = mock.MagicMock()
        pipe.invert_images.return_value = {
            "zT_torch": torch.randn(*TestWChannelMinusOneRealScoring.LATENT_SHAPE),
        }
        return pipe

    @staticmethod
    def _make_image(tmp_path):
        from PIL import Image
        img = tmp_path / "input.png"
        Image.new("RGB", (64, 64)).save(img)
        return str(img)

    def test_w_channel_minus_one_real_provider_scoring(self, tmp_path):
        """Full canonical path: real provider, real mask/gt_patch, real
        FFT/mask math → finite complex-L1 raw/canonical scores."""
        import torch
        from raven.detectors import tr_scoring

        provider = self._build_provider(-1)
        assert provider.watermarking_mask.shape == self.LATENT_SHAPE
        # All 4 channels covered by the mask.
        for ch in range(1, self.LATENT_SHAPE[1]):
            assert bool(
                (provider.watermarking_mask[:, 0]
                 == provider.watermarking_mask[:, ch]).all()
            ), "w_channel=-1 mask must cover every channel"

        pipe = self._mock_pipe()

        result = tr_scoring.complex_l1_score(
            torch, provider, pipe, Path(self._make_image(tmp_path)), 50)

        assert not result["nan"] and not result["inf"]
        raw = result["score"]
        canonical = tr_scoring.canonical_score(raw)
        assert canonical == -raw
        assert math.isfinite(float(raw))
        assert math.isfinite(float(canonical))

        # The explicitly named p-value mode still works end-to-end through
        # the real provider's get_accuracies.
        pv = tr_scoring.evaluate_log10p(
            torch, provider, pipe, Path(self._make_image(tmp_path)), 50)
        assert "p_values" in pv
        assert len(pv["p_values"]) == 1
        assert pv["p_values"][0] >= 0.0
        assert "p_value_diagnostics" in pv
        raw_p = tr_scoring.raw_log10p_score(pv)
        assert math.isfinite(float(raw_p))
        assert math.isfinite(tr_scoring.log10p_canonical(raw_p))

    def test_w_channel_minus_one_real_get_accuracies(self, tmp_path):
        """Real get_accuracies completes and produces p-values."""
        import torch

        provider = self._build_provider(-1)
        latents = torch.randn(*self.LATENT_SHAPE)

        accuracies = provider.get_accuracies(latents)
        assert "p_values" in accuracies
        assert len(accuracies["p_values"]) == 1
        assert 0.0 <= accuracies["p_values"][0] <= 1.0

    def test_w_channel_minus_one_visualization_is_nonempty(self, tmp_path):
        """Every visualization tensor must be non-empty for w_channel=-1 —
        never an empty [:, -1:0] slice."""
        import torch

        provider = self._build_provider(-1)
        latents = torch.randn(*self.LATENT_SHAPE)

        outs = provider.get_wm_latents(latents)
        for key in ("zT_clean_fft_wchannel_torch",
                    "zT_fft_wchannel_torch",
                    "pristine_zT_fft_wchannel_torch"):
            tensor = outs[key]
            assert tensor.numel() > 0, f"{key} is empty"
            assert tensor.shape[-1] == 64, f"{key} shape {tensor.shape}"
            assert tensor.shape[1] == 4, (
                f"{key} must carry all 4 channels for w_channel=-1, got "
                f"{tensor.shape}")

        result = provider.get_accuracies(latents)
        for key in ("zT_fft_torch", "zT_fft_wchannel_torch",
                    "zT_fft_wchannel_PIL"):
            assert key in result, f"missing {key}"

    def test_w_channel_three_visualization_still_single_channel(self, tmp_path):
        """Regression: w_channel >= 0 keeps selecting exactly one channel."""
        import torch

        provider = self._build_provider(3)
        latents = torch.randn(*self.LATENT_SHAPE)

        outs = provider.get_wm_latents(latents)
        tensor = outs["zT_fft_wchannel_torch"]
        assert tensor.shape[1] == 1, f"expected 1 channel, got {tensor.shape}"


# ===========================================================================
# 20 — Orchestrator aggregate failure contract
# ===========================================================================
class TestAggregateFailureOrchestrator:
    """Real evaluate_detector with summarize_detection raising — the stage
    result must be structured failed_scoring, records preserved, exit 2 with
    and without --allow-missing-metrics."""

    def _four_cohort_records(self):
        profile = dict(TR_PROFILE_GENERATOR)
        rec_clean = _generator_record("1", "clean", profile=profile)
        rec_wm = _generator_record("1", "watermarked", profile=profile)
        return [rec_clean, rec_wm]

    def test_metric_failure_structured_stage_result(self, monkeypatch):
        import raven.metrics as metrics
        monkeypatch.setattr(
            metrics, "summarize_detection",
            mock.MagicMock(side_effect=RuntimeError("metric bug")))

        from experiments.eval import evaluate_detector
        from raven.detectors import (
            STATUS_FAILED_SCORING, FAILURE_CAUSE_SCORING_ERROR,
        )

        records = self._four_cohort_records()
        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR", records=records)
                result = evaluate_detector(records, out, "TR", device="cpu")

                assert result["status"] == STATUS_FAILED_SCORING
                assert result["dominant_failure_cause"] == FAILURE_CAUSE_SCORING_ERROR
                assert result.get("aggregate_error_type") == "DetectorScoringError"
                assert "metric bug" in result.get("aggregate_error", "")
                # stage identity preserved
                assert result["stage"] == "detector"
                assert result["method"] == "TR"

                # counts + invariant preserved
                assert result["requested_count"] == 4
                assert result["scored_count"] == 4
                assert result["failed_count"] == 0
                assert result["unscored_due_to_setup_count"] == 0
                assert result["count_invariant_satisfied"] is True

                # detector_records.jsonl preserved
                rows = _read_detector_rows(out)
                assert len(rows) == 4
                assert all(r["status"] == "scored" for r in rows)

    def test_aggregate_state_validation_failure_classified(self, monkeypatch):
        """DetectorStateValidationError from aggregate stays
        failed_state_validation, never flattened to failed_scoring."""
        import raven.detectors.tr_detector as tr_mod
        from raven.detectors import (
            DetectorStateValidationError, STATUS_FAILED_STATE_VALIDATION,
            FAILURE_CAUSE_STATE_VALIDATION,
        )

        def _raise_state_validation(*args, **kwargs):
            raise DetectorStateValidationError("state bug")
        monkeypatch.setattr(tr_mod, "aggregate", _raise_state_validation)

        from experiments.eval import evaluate_detector

        records = self._four_cohort_records()
        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR", records=records)
                result = evaluate_detector(records, out, "TR", device="cpu")

                assert result["status"] == STATUS_FAILED_STATE_VALIDATION
                assert result["dominant_failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION
                assert result.get("aggregate_error_type") == "DetectorStateValidationError"
                assert "state bug" in result.get("aggregate_error", "")
                assert result["stage"] == "detector"
                assert result["method"] == "TR"
                assert result["count_invariant_satisfied"] is True

    def test_aggregate_unknown_error_classified_internal(self, monkeypatch):
        """Unknown aggregate exception fails closed as failed_internal_error."""
        import raven.detectors.tr_detector as tr_mod
        from raven.detectors import (
            STATUS_FAILED_INTERNAL_ERROR, FAILURE_CAUSE_INTERNAL_ERROR,
        )

        def _raise_unknown(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(tr_mod, "aggregate", _raise_unknown)

        from experiments.eval import evaluate_detector

        records = self._four_cohort_records()
        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR", records=records)
                result = evaluate_detector(records, out, "TR", device="cpu")

                assert result["status"] == STATUS_FAILED_INTERNAL_ERROR
                assert result["dominant_failure_cause"] == FAILURE_CAUSE_INTERNAL_ERROR
                assert result.get("aggregate_error_type") == "RuntimeError"
                assert "boom" in result.get("aggregate_error", "")
                assert result["stage"] == "detector"
                assert result["method"] == "TR"

    def test_cli_exit_without_allow_is_2(self, monkeypatch):
        import raven.metrics as metrics
        monkeypatch.setattr(
            metrics, "summarize_detection",
            mock.MagicMock(side_effect=RuntimeError("metric bug")))

        from experiments.eval import main

        records = self._four_cohort_records()
        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR", records=records)
                rc = main([
                    "--output-dir", str(out), "--device", "cpu",
                    "--log-level", "ERROR", "--stages", "detector",
                ])
                assert rc == 2, f"expected exit 2, got {rc}"

    def test_cli_exit_with_allow_is_2(self, monkeypatch):
        import raven.metrics as metrics
        monkeypatch.setattr(
            metrics, "summarize_detection",
            mock.MagicMock(side_effect=RuntimeError("metric bug")))

        from experiments.eval import main

        records = self._four_cohort_records()
        with _patch_integration(monkeypatch):
            with tempfile.TemporaryDirectory() as td:
                out = _write_fake_run(Path(td), method="TR", records=records)
                rc = main([
                    "--output-dir", str(out), "--device", "cpu",
                    "--log-level", "ERROR", "--stages", "detector",
                    "--allow-missing-metrics",
                ])
                assert rc == 2, f"expected exit 2 with allow, got {rc}"
