"""Issue #21 — T2S per-sample state resolution, validation, and classification.

All tests use synthetic state objects and mocks. No real T2S model or state
bundle is loaded.  Canonical mode constants come from
``eval_bench_wm.utils.wm.t2s_provider`` — never duplicated here.

Orchestrator tests use the REAL ``evaluate_detector``, ``load_state``,
``score_image`` and ``aggregate``.  Only the state loader, pipe provider,
inversion, accuracies, and PIL decoding are mocked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in (str(REPO / "raven_repro"), str(REPO / "experiments"),
              str(REPO / "eval_bench_wm")):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from raven.detectors import (
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
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_ERRORS,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_SCORING,
    reduce_detector_stage_status,
    stage_status_is_allowable,
    determine_exit_code,
)
from raven.detectors import t2s_detector

# Canonical constants — authoritative source of truth
from utils.wm.t2s_provider import (  # noqa: E402
    T2S_RNG_MODES,
    T2S_INVERSION_MODES,
    T2S_SHARED_TR_CLEAN_MODE,
)
from raven.pairing_provenance import T2S_SHARED_TR_CLEAN_PROTOCOL  # noqa: E402

RNG_MODES = frozenset(T2S_RNG_MODES)
INVERSION_MODES = frozenset(T2S_INVERSION_MODES)
PROTOCOL_MODES = frozenset({T2S_SHARED_TR_CLEAN_MODE})

MODEL_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"


# ===========================================================================
# Helpers
# ===========================================================================

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _fake_png(tmp_path: Path, name: str = "img.png") -> Path:
    p = tmp_path / name
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01"
        b"\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return p


@dataclasses.dataclass
class SyntheticT2SWatermarkState:
    """Minimal synthetic T2S state with canonical mode defaults."""
    watermark_id: str = "t2s-synth-01"
    inversion_mode: str = "t2s_official"
    num_inversion_steps: int = 10
    provider_config_sha256: str = _sha256("synth-prov-config")
    rng_mode: str = "official_compatible"
    num_inference_steps: int = 50
    model_id: str | None = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    scheduler: str | None = "DDIM"
    resolution: int | None = 512
    tau: float = 0.5
    key_length: int = 16
    msg_length: int = 48
    latent_shape: list | None = None
    key_channels: list | None = None
    msg_channels: list | None = None

    def __post_init__(self):
        if self.latent_shape is None:
            self.latent_shape = [1, 4, 64, 64]
        if self.key_channels is None:
            self.key_channels = [0, 1]
        if self.msg_channels is None:
            self.msg_channels = [2, 3]

    def state_sha256(self) -> str:
        return _sha256(f"state:{self.watermark_id}:{self.inversion_mode}")


def _make_state(**overrides) -> SyntheticT2SWatermarkState:
    kw = {
        "watermark_id": "t2s-synth-01",
        "inversion_mode": "t2s_official",
        "num_inversion_steps": 10,
        "provider_config_sha256": _sha256("synth-prov-config"),
        "rng_mode": "official_compatible",
    }
    kw.update(overrides)
    return SyntheticT2SWatermarkState(**kw)


def _make_accuracies(**overrides) -> dict:
    base = {
        "t2s_score_true_key": 0.85,
        "t2s_score_control_key": 0.12,
        "t2s_score_margin": 0.73,
        "detection_success": True,
        "key_accuracy": 1.0,
        "message_accuracy": 0.92,
    }
    base.update(overrides)
    return base


def _consistent_accuracies(true_key: float, control_key: float,
                            detection: bool | None = None,
                            **overrides) -> dict:
    """Accuracies whose margin/detection agree with the keys."""
    if detection is None:
        detection = true_key > control_key
    return _make_accuracies(
        t2s_score_true_key=true_key,
        t2s_score_control_key=control_key,
        t2s_score_margin=true_key - control_key,
        detection_success=detection,
        **overrides,
    )


def _make_canonical(run_id="1", source_role="watermarked", state=None,
                    tmp_path: Path | None = None, **overrides) -> dict:
    """Build canonical metadata with an actual state file on disk."""
    if state is None:
        state = _make_state()
    if tmp_path is not None:
        sp = str(tmp_path / f"st_{run_id}_{source_role}.json")
    else:
        import tempfile as _tf
        sp = str(Path(_tf.gettempdir()) / f"t2s_test_st_{run_id}_{source_role}.json")
    Path(sp).parent.mkdir(parents=True, exist_ok=True)
    Path(sp).write_text("{}")
    base = {
        "run_id": run_id,
        "role": source_role,
        "source_role": source_role,
        "t2s_state_path": sp,
        "t2s_state_sha256": state.state_sha256(),
        "t2s_watermark_id": state.watermark_id,
        "t2s_provider_config_sha256": state.provider_config_sha256,
        "t2s_protocol_mode": T2S_SHARED_TR_CLEAN_MODE,
        "t2s_rng_mode": state.rng_mode,
        "t2s_inversion_mode": state.inversion_mode,
        "t2s_num_inversion_steps": str(state.num_inversion_steps),
        "t2s_model_id": state.model_id,
        "t2s_model_revision": state.model_revision,
        "t2s_scheduler": state.scheduler,
        "t2s_resolution": str(state.resolution or 512),
        "t2s_num_inference_steps": str(state.num_inference_steps or 50),
    }
    base.update(overrides)
    return base


def _make_entry(run_id="1", source_role="watermarked",
                cohort="original_watermarked") -> dict:
    return {
        "run_id": run_id,
        "source_role": source_role,
        "evaluation_cohort": cohort,
        "image_path": "/tmp/fake.png",
    }


def _fake_provider_info(
    state: SyntheticT2SWatermarkState | None = None,
    accuracies=None,
    inversion_side_effect=None,
    state_index: dict | None = None,
    state_cache: dict | None = None,
    missing_state_keys: set | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    scheduler: str = "DDIM",
    resolution: int = 512,
):
    """provider_info shaped exactly like real load_state output, with mocks.

    Used only by direct score_image tests; orchestrator tests use the real
    load_state with targeted mocks instead.
    """
    if state is None:
        state = _make_state()
    if accuracies is None:
        accuracies = _make_accuracies()
    if state_index is None:
        state_index = {}
    if state_cache is None:
        state_cache = {}
    if missing_state_keys is None:
        missing_state_keys = set()

    provider_mod = mock.MagicMock()
    if callable(accuracies):
        provider_mod.T2SProvider.accuracies_for_state.side_effect = accuracies
    else:
        provider_mod.T2SProvider.accuracies_for_state.return_value = accuracies

    inversion_mod = mock.MagicMock()
    if inversion_side_effect is not None:
        inversion_mod.invert_image.side_effect = inversion_side_effect
    else:
        inversion_mod.invert_image.return_value = mock.MagicMock()

    return {
        "pipe": mock.MagicMock(),
        "t2s_provider_module": provider_mod,
        "t2s_inversion_module": inversion_mod,
        "device_obj": mock.MagicMock(),
        "state_metadata_index": state_index,
        "state_cache": state_cache,
        "missing_state_keys": missing_state_keys,
        "t2s_rng_modes": RNG_MODES,
        "t2s_inversion_modes": INVERSION_MODES,
        "t2s_protocol_modes": PROTOCOL_MODES,
        "model_id": model_id,
        "model_revision": model_revision,
        "scheduler": scheduler,
        "resolution": resolution,
    }


def _score(provider_info, entry, tmp_path, name="img.png"):
    """Call score_image with fake image, PIL mocked, evaluation_entry."""
    image_path = str(_fake_png(tmp_path, name))
    entry["image_path"] = image_path
    with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
        return t2s_detector.score_image(
            provider_info, image_path,
            evaluation_entry=entry)


# ---------------------------------------------------------------------------
# Real-load-state mocking infrastructure
# ---------------------------------------------------------------------------

def _make_fake_pipe(latent_shape=(1, 4, 64, 64)):
    """Pipe mock whose get_latent_shape matches the synthetic state default."""
    pipe = mock.MagicMock()
    pipe.get_latent_shape.return_value = tuple(latent_shape)
    return pipe


def install_pipe_utils_stub(get_pipe_provider=None, latent_shape=(1, 4, 64, 64)):
    """Stub eval_bench_wm.utils.pipe.pipe_utils so real load_state can import it.

    The real module drags in lpips (unavailable in test env).  This stub
    satisfies the import; everything else in load_state stays real.  The
    default fake pipe reports the synthetic latent shape.
    """
    if get_pipe_provider is None:
        get_pipe_provider = mock.MagicMock(
            return_value=_make_fake_pipe(latent_shape))
    stub = types.ModuleType("eval_bench_wm.utils.pipe.pipe_utils")
    stub.get_pipe_provider = get_pipe_provider
    sys.modules["eval_bench_wm.utils.pipe.pipe_utils"] = stub
    # Ensure parent packages are importable
    import eval_bench_wm.utils.pipe  # noqa: F401
    return get_pipe_provider


def _provider_module():
    """The module object load_state actually imports (eval_bench_wm-named).

    ``utils.wm.t2s_provider`` and ``eval_bench_wm.utils.wm.t2s_provider`` are
    TWO distinct module objects (the file executes twice under two names),
    so the mocks must patch the one the detector holds.
    """
    import eval_bench_wm.utils.wm.t2s_provider as det_module
    return det_module


def _inversion_module():
    import eval_bench_wm.utils.wm.t2s_inversion as det_module
    return det_module


def install_state_load_mock(monkeypatch, states_by_path: dict[str, object],
                            failures_by_path: dict[str, Exception] | None = None):
    """Mock T2SWatermarkState.load on the module load_state imports."""
    failures_by_path = failures_by_path or {}

    def _load(path):
        p = str(path)
        if p in failures_by_path:
            raise failures_by_path[p]
        if p in states_by_path:
            return states_by_path[p]
        raise ValueError(f"unexpected state path in test: {p}")

    monkeypatch.setattr(
        _provider_module().T2SWatermarkState, "load",
        staticmethod(_load),
    )
    return _load


def install_accuracies_mock(monkeypatch, fn):
    """Mock T2SProvider.accuracies_for_state on the module load_state imports."""
    monkeypatch.setattr(
        _provider_module().T2SProvider, "accuracies_for_state",
        staticmethod(fn),
    )


def install_inversion_mock(monkeypatch, fn=None):
    """Mock t2s_inversion.invert_image on the module load_state imports."""
    if fn is None:
        fn = lambda *a, **kw: mock.MagicMock()  # noqa: E731
    monkeypatch.setattr(_inversion_module(), "invert_image", fn)


# ---------------------------------------------------------------------------
# Real orchestrator run builder
# ---------------------------------------------------------------------------

def _make_orch_record(run_id, role, state, state_path, tmp_path):
    return {
        "run_id": run_id, "role": role, "method": "T2S",
        "input_path": str(_fake_png(tmp_path, f"in_{run_id}_{role}.png")),
        "output_path": str(tmp_path / f"out_{run_id}_{role}.png"),
        "prompt": "", "attack_seed": int(run_id),
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -24.0,
        "source_metadata": {
            "run_id": run_id,
            "role": role,
            "t2s_state_path": str(state_path),
            "t2s_state_sha256": state.state_sha256(),
            "t2s_watermark_id": state.watermark_id,
            "t2s_provider_config_sha256": state.provider_config_sha256,
            "t2s_protocol_mode": T2S_SHARED_TR_CLEAN_MODE,
            "t2s_rng_mode": state.rng_mode,
            "t2s_inversion_mode": state.inversion_mode,
            "t2s_num_inversion_steps": str(state.num_inversion_steps),
            "t2s_model_id": state.model_id or "",
            "t2s_model_revision": state.model_revision or "",
            "t2s_scheduler": state.scheduler or "",
            "t2s_resolution": str(state.resolution or 512),
            "t2s_num_inference_steps": str(state.num_inference_steps or 50),
        },
    }


def _setup_run(tmp_path, records, metadata_path: str | None = None):
    """Create a minimal run dir with config + records + fake outputs."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl

    out = tmp_path / "run"
    out.mkdir()
    cfg = {"method": "T2S", "dataset": "test"}
    if metadata_path is not None:
        cfg["metadata_path"] = metadata_path
    write_config(out, cfg)

    for rec in records:
        role = rec["role"]
        rid = rec["run_id"]
        write_record(out, role, rid, rec)
        img = out / "samples" / role / rid / "output.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")
    rebuild_records_jsonl(out)
    return out


def _orchestrator_env(monkeypatch, states_by_path, get_pipe_provider=None,
                      accuracies_fn=None, inversion_fn=None):
    """Install all mocks needed for real load_state under evaluate_detector."""
    install_pipe_utils_stub(get_pipe_provider)
    install_state_load_mock(monkeypatch, states_by_path)
    if accuracies_fn is not None:
        install_accuracies_mock(monkeypatch, accuracies_fn)
    if inversion_fn is not None:
        install_inversion_mock(monkeypatch, inversion_fn)
    else:
        install_inversion_mock(monkeypatch)


# ===========================================================================
# 1. Canonical modes — authoritative constants
# ===========================================================================

class TestCanonicalModes:
    def test_official_compatible_rng_is_accepted(self, tmp_path):
        state = _make_state(rng_mode="official_compatible")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)
        assert result["t2s_rng_mode"] == "official_compatible"

    def test_raven_deterministic_rng_is_accepted(self, tmp_path):
        state = _make_state(rng_mode="raven_deterministic")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)
        assert result["t2s_rng_mode"] == "raven_deterministic"

    def test_t2s_official_rng_is_rejected(self, tmp_path):
        state = _make_state(rng_mode="official_compatible")
        canonical = _make_canonical(
            state=state, tmp_path=tmp_path,
            t2s_rng_mode="t2s_official")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="unknown t2s_rng_mode"):
            _score(prov, entry, tmp_path)

    def test_benchmark_ddim_inversion_is_accepted(self, tmp_path):
        state = _make_state(inversion_mode="benchmark_ddim")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)
        assert result["t2s_inversion_mode"] == "benchmark_ddim"

    def test_ddim_inversion_alias_is_rejected(self, tmp_path):
        state = _make_state(inversion_mode="t2s_official")
        canonical = _make_canonical(
            state=state, tmp_path=tmp_path,
            t2s_inversion_mode="ddim_inversion")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="unknown t2s_inversion_mode"):
            _score(prov, entry, tmp_path)

    def test_real_shared_clean_protocol_mode_is_accepted(self, tmp_path):
        assert T2S_SHARED_TR_CLEAN_MODE == "official_encoder_shared_tr_clean"
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)
        assert result["t2s_protocol_mode"] == T2S_SHARED_TR_CLEAN_MODE

    def test_protocol_name_is_not_accepted_as_protocol_mode(self, tmp_path):
        # Protocol name "t2smark_shared_tr_clean_v2" is NOT a protocol mode.
        state = _make_state()
        canonical = _make_canonical(
            state=state, tmp_path=tmp_path,
            t2s_protocol_mode=T2S_SHARED_TR_CLEAN_PROTOCOL)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state, state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="unknown t2s_protocol_mode"):
            _score(prov, entry, tmp_path)


# ===========================================================================
# 2. Fail-closed required metadata fields
# ===========================================================================

class TestRequiredMetadataFailClosed:
    def test_missing_state_sha_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_state_sha256="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError, match="t2s_state_sha256"):
            _score(prov, entry, tmp_path)

    def test_missing_watermark_id_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_watermark_id="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError, match="t2s_watermark_id"):
            _score(prov, entry, tmp_path)

    def test_missing_provider_config_sha_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_provider_config_sha256="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError,
                           match="t2s_provider_config_sha256"):
            _score(prov, entry, tmp_path)

    def test_missing_inversion_mode_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_inversion_mode="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError, match="t2s_inversion_mode"):
            _score(prov, entry, tmp_path)

    def test_missing_inversion_steps_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_num_inversion_steps="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError,
                           match="t2s_num_inversion_steps"):
            _score(prov, entry, tmp_path)

    def test_missing_rng_mode_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_rng_mode="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError, match="t2s_rng_mode"):
            _score(prov, entry, tmp_path)

    def test_missing_protocol_mode_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_protocol_mode="")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorMissingStateError, match="t2s_protocol_mode"):
            _score(prov, entry, tmp_path)


class TestRequiredMetadataMismatch:
    def test_state_sha_mismatch_raises_validation_error(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(
            state=state, tmp_path=tmp_path,
            t2s_state_sha256=_sha256("wrong"))
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="state SHA mismatch"):
            _score(prov, entry, tmp_path)

    def test_watermark_id_mismatch_raises_validation_error(self, tmp_path):
        state = _make_state(watermark_id="correct")
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_watermark_id="wrong")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="watermark_id mismatch"):
            _score(prov, entry, tmp_path)

    def test_inversion_steps_mismatch(self, tmp_path):
        state = _make_state(num_inversion_steps=10)
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_num_inversion_steps="7")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="num_inversion_steps mismatch"):
            _score(prov, entry, tmp_path)

    def test_rng_mode_mismatch(self, tmp_path):
        state = _make_state(rng_mode="official_compatible")
        canonical = _make_canonical(state=state, tmp_path=tmp_path,
                                    t2s_rng_mode="raven_deterministic")
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="rng_mode mismatch"):
            _score(prov, entry, tmp_path)


# ===========================================================================
# 3. Pipe profile — no fallback
# ===========================================================================

class TestPipeProfileNoFallback:
    def test_missing_model_id_on_state_raises_validation_error(self, tmp_path):
        state = _make_state(model_id=None)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state},
            model_id=None, model_revision=None,
            scheduler=None, resolution=None)
        with pytest.raises(DetectorStateValidationError, match="model_id"):
            _score(prov, entry, tmp_path)

    def test_missing_model_revision_on_state_raises_validation_error(self, tmp_path):
        state = _make_state(model_revision=None)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state},
            model_id=MODEL_ID, model_revision=None,
            scheduler="DDIM", resolution=512)
        with pytest.raises(DetectorStateValidationError, match="model_revision"):
            _score(prov, entry, tmp_path)

    def test_state_scheduler_conflicts_with_pipe(self, tmp_path):
        state = _make_state(scheduler="DDPM")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state},
            model_id=MODEL_ID, model_revision=MODEL_REVISION,
            scheduler="DDIM", resolution=512)
        with pytest.raises(DetectorStateValidationError, match="scheduler"):
            _score(prov, entry, tmp_path)

    def test_state_resolution_conflicts_with_pipe(self, tmp_path):
        state = _make_state(resolution=256)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state},
            model_id=MODEL_ID, model_revision=MODEL_REVISION,
            scheduler="DDIM", resolution=512)
        with pytest.raises(DetectorStateValidationError, match="resolution"):
            _score(prov, entry, tmp_path)


# ===========================================================================
# 4. Scoring contract
# ===========================================================================

class TestScoringContract:
    def test_missing_true_key_is_scoring_error(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        acc = {"t2s_score_control_key": 0.1, "t2s_score_margin": 0.5,
               "detection_success": True}
        prov = _fake_provider_info(
            state=state, accuracies=acc,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorScoringError, match="t2s_score_true_key"):
            _score(prov, entry, tmp_path)

    def test_nonfinite_score_is_scoring_error(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        acc = _make_accuracies(t2s_score_true_key=float("nan"))
        prov = _fake_provider_info(
            state=state, accuracies=acc,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorScoringError, match="non-finite"):
            _score(prov, entry, tmp_path)

    def test_margin_inconsistent_with_scores_raises_scoring_error(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        acc = _make_accuracies(
            t2s_score_true_key=0.9, t2s_score_control_key=0.1,
            t2s_score_margin=0.5)
        prov = _fake_provider_info(
            state=state, accuracies=acc,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorScoringError, match="margin"):
            _score(prov, entry, tmp_path)

    def test_detection_inconsistent_with_scores_raises_scoring_error(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        acc = _make_accuracies(
            t2s_score_true_key=0.9, t2s_score_control_key=0.1,
            t2s_score_margin=0.8, detection_success=False)
        prov = _fake_provider_info(
            state=state, accuracies=acc,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorScoringError, match="detection_success"):
            _score(prov, entry, tmp_path)


# ===========================================================================
# 5. State provenance
# ===========================================================================

class TestStateProvenance:
    def test_all_provenance_fields_in_output(self, tmp_path):
        state = _make_state(
            watermark_id="prov-test",
            model_revision="rev123")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state},
            model_id=MODEL_ID, model_revision="rev123",
            scheduler="DDIM", resolution=512)
        result = _score(prov, entry, tmp_path)

        provenance_fields = [
            "t2s_state_path", "t2s_state_sha256",
            "t2s_provider_config_sha256", "t2s_watermark_id",
            "t2s_protocol_mode", "t2s_rng_mode", "t2s_inversion_mode",
            "t2s_num_inversion_steps", "t2s_model_id", "t2s_model_revision",
            "t2s_scheduler", "t2s_resolution", "t2s_num_inference_steps",
            "t2s_state_verified",
        ]
        for f in provenance_fields:
            assert f in result, f"missing provenance field: {f}"
        assert result["t2s_state_verified"] is True
        assert result["t2s_watermark_id"] == "prov-test"
        assert result["t2s_protocol_mode"] == T2S_SHARED_TR_CLEAN_MODE

    def test_bit_accuracy_equals_message_accuracy(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)
        assert result["t2s_bit_accuracy"] == result["t2s_message_accuracy"]


# ===========================================================================
# 6. Missing image taxonomy
# ===========================================================================

class TestMissingImage:
    def test_missing_image_raises_file_not_found(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        entry["image_path"] = "/nonexistent/image.png"
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(FileNotFoundError, match="T2S image not found"):
            t2s_detector.score_image(
                prov, "/nonexistent/image.png",
                evaluation_entry=entry)

    def test_missing_evaluation_entry_raises_missing_state(self, tmp_path):
        state = _make_state()
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        image_path = str(_fake_png(tmp_path))
        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            with pytest.raises(DetectorMissingStateError,
                               match="evaluation_entry"):
                t2s_detector.score_image(
                    prov, image_path, evaluation_entry=None)


# ===========================================================================
# 7. Aggregation — finite and alignment
# ===========================================================================

class TestAggregationFiniteAlignment:
    def _make_scored_row(self, run_id, cohort, **overrides):
        base = {
            "run_id": run_id,
            "evaluation_cohort": cohort,
            "status": ROW_STATUS_SCORED,
            "t2s_score_true_key": 0.85,
            "t2s_score_control_key": 0.12,
            "t2s_score_margin": 0.73,
            "t2s_detection_success": True,
            "t2s_bit_accuracy": 0.92,
        }
        base.update(overrides)
        return base

    def test_nan_bit_accuracy_excluded_from_stats(self):
        rows = [
            self._make_scored_row("1", "original_watermarked",
                                 t2s_bit_accuracy=float("nan")),
            self._make_scored_row("2", "original_watermarked",
                                 t2s_bit_accuracy=0.5),
            self._make_scored_row("3", "original_watermarked",
                                 t2s_bit_accuracy=0.7),
        ]
        result = t2s_detector.aggregate(rows)
        stats = result["original_watermarked_bit_accuracy"]
        assert stats["bit_accuracy_count"] == 2
        assert stats["bit_accuracy_unavailable_count"] == 1
        assert stats["mean"] == 0.6

    def test_inf_bit_accuracy_excluded_from_stats(self):
        rows = [
            self._make_scored_row("1", "original_watermarked",
                                 t2s_bit_accuracy=float("inf")),
            self._make_scored_row("2", "original_watermarked",
                                 t2s_bit_accuracy=0.5),
        ]
        result = t2s_detector.aggregate(rows)
        stats = result["original_watermarked_bit_accuracy"]
        assert stats["bit_accuracy_count"] == 1
        assert stats["bit_accuracy_unavailable_count"] == 1

    def test_string_garbage_bit_accuracy_excluded(self):
        rows = [
            self._make_scored_row("1", "original_watermarked",
                                 t2s_bit_accuracy="not_a_number"),
            self._make_scored_row("2", "original_watermarked",
                                 t2s_bit_accuracy=1.0),
        ]
        result = t2s_detector.aggregate(rows)
        stats = result["original_watermarked_bit_accuracy"]
        assert stats["bit_accuracy_count"] == 1
        assert stats["bit_accuracy_unavailable_count"] == 1

    def test_row_by_row_no_cross_cohort_zip(self):
        rows = [
            self._make_scored_row("1", "original_watermarked", t2s_bit_accuracy=0.9),
            self._make_scored_row("3", "original_watermarked", t2s_bit_accuracy=0.7),
            self._make_scored_row("1", "attacked_watermarked", t2s_bit_accuracy=0.4),
            self._make_scored_row("2", "attacked_watermarked", t2s_bit_accuracy=0.3),
            self._make_scored_row("4", "attacked_watermarked", t2s_bit_accuracy=0.2),
        ]
        result = t2s_detector.aggregate(rows)
        assert result["original_watermarked_bit_accuracy"]["bit_accuracy_count"] == 2
        assert result["attacked_watermarked_bit_accuracy"]["bit_accuracy_count"] == 3


# ===========================================================================
# 8. Real load_state orchestrator — pairing
# ===========================================================================

class TestOrchestratorRealPairing:
    def test_real_orchestrator_pairs_same_run_id_by_role(self, tmp_path, monkeypatch):
        """Real load_state/score_image/aggregate under evaluate_detector.

        run_id=42: clean state (score 0.11), watermarked state (score 0.91).
        Each cohort must use its own state; states load exactly once each.
        """
        from experiments.eval import evaluate_detector

        # Same provider config (cohort-uniform); pairing proven by different
        # watermark_id + state SHA, never by divergent provider configs.
        clean_state = _make_state(
            watermark_id="clean-state-id",
            provider_config_sha256=_sha256("cohort-pc"))
        wm_state = _make_state(
            watermark_id="wm-state-id",
            provider_config_sha256=_sha256("cohort-pc"))

        clean_path = tmp_path / "clean_state.json"
        clean_path.write_text("{}")
        wm_path = tmp_path / "wm_state.json"
        wm_path.write_text("{}")

        clean_rec = _make_orch_record("42", "clean", clean_state, clean_path, tmp_path)
        wm_rec = _make_orch_record("42", "watermarked", wm_state, wm_path, tmp_path)
        out = _setup_run(tmp_path, [clean_rec, wm_rec])

        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {
            str(clean_path): clean_state,
            str(wm_path): wm_state,
        })

        def _accuracies(st, inv):
            if st.watermark_id == "wm-state-id":
                return _consistent_accuracies(0.91, 0.05, True)
            return _consistent_accuracies(0.11, 0.05, True)

        install_accuracies_mock(monkeypatch, _accuracies)
        install_inversion_mock(monkeypatch)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            detector_result = evaluate_detector(
                [clean_rec, wm_rec], out, "T2S", device="cpu")

        det_path = out / "evaluation" / "detector_records.jsonl"
        assert det_path.is_file(), f"detector_records.jsonl missing at {det_path}"
        rows = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]
        scored = [r for r in rows if r.get("status") == ROW_STATUS_SCORED]

        cohorts_found = set()
        for r in rows:
            cohorts_found.add((r["run_id"], r.get("source_role", ""),
                               r["evaluation_cohort"]))

        for r in scored:
            cohort = r["evaluation_cohort"]
            if cohort == "original_clean":
                assert r["source_role"] == "clean"
                assert r["t2s_watermark_id"] == "clean-state-id"
                assert r["t2s_score_true_key"] == 0.11
            elif cohort == "attacked_clean":
                assert r["source_role"] == "clean"
                assert r["t2s_watermark_id"] == "clean-state-id"
                assert r["t2s_score_true_key"] == 0.11
            elif cohort == "original_watermarked":
                assert r["source_role"] == "watermarked"
                assert r["t2s_watermark_id"] == "wm-state-id"
                assert r["t2s_score_true_key"] == 0.91
            elif cohort == "attacked_watermarked":
                assert r["source_role"] == "watermarked"
                assert r["t2s_watermark_id"] == "wm-state-id"
                assert r["t2s_score_true_key"] == 0.91

        expected_cohorts = {
            ("42", "clean", "original_clean"),
            ("42", "clean", "attacked_clean"),
            ("42", "watermarked", "original_watermarked"),
            ("42", "watermarked", "attacked_watermarked"),
        }
        missing = expected_cohorts - cohorts_found
        assert not missing, f"missing cohorts: {missing}; found={cohorts_found}"

        assert detector_result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS), (
            f"status={detector_result.get('status')}, "
            f"rows={[(r.get('status'), r.get('error','')) for r in rows]}")

        # Clean and watermarked never cross: every clean row carries clean IDs
        clean_rows = [r for r in scored if r["source_role"] == "clean"]
        wm_rows = [r for r in scored if r["source_role"] == "watermarked"]
        assert all(r["t2s_watermark_id"] == "clean-state-id" for r in clean_rows)
        assert all(r["t2s_watermark_id"] == "wm-state-id" for r in wm_rows)

        # Cache identity: same (run_id, source_role) state object for both cohorts
        state_ids = {}
        for r in scored:
            state_ids.setdefault((r["run_id"], r["source_role"]), set()).add(
                r["t2s_state_sha256"])
        for key, shas in state_ids.items():
            assert len(shas) == 1, f"state identity drifted for {key}"


# ===========================================================================
# 9. Real load_state orchestrator — missing / validation / scoring
# ===========================================================================

class TestOrchestratorRealMissingState:
    def test_all_missing_state_paths_gives_missing_required_state(
            self, tmp_path, monkeypatch):
        """All metadata complete, all t2s_state_path files absent."""
        from experiments.eval import evaluate_detector

        state = _make_state()
        missing_path = tmp_path / "nonexistent.json"  # file does not exist
        rec = _make_orch_record("1", "watermarked", state, missing_path, tmp_path)
        out = _setup_run(tmp_path, [rec])

        install_pipe_utils_stub()
        # State load never called — path check fires first.
        monkeypatch.setattr(
            _provider_module().T2SWatermarkState, "load",
            staticmethod(lambda p: pytest.fail("load must not be called")))

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert result.get("dominant_failure_cause") == FAILURE_CAUSE_MISSING_REQUIRED_STATE
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=False) == 2
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=True) == 0

    def test_partial_missing_state_scores_valid_rows(self, tmp_path, monkeypatch):
        """clean state exists, watermarked state missing.

        Valid rows score normally; missing rows become failed_missing_state
        (never validation or scoring).
        """
        from experiments.eval import evaluate_detector

        clean_state = _make_state(watermark_id="clean-ok")
        wm_state = _make_state(watermark_id="wm-missing")

        clean_path = tmp_path / "clean_state.json"
        clean_path.write_text("{}")
        missing_path = tmp_path / "wm_missing.json"  # absent

        clean_rec = _make_orch_record("42", "clean", clean_state, clean_path, tmp_path)
        wm_rec = _make_orch_record("42", "watermarked", wm_state, missing_path, tmp_path)
        out = _setup_run(tmp_path, [clean_rec, wm_rec])

        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(clean_path): clean_state})

        def _accuracies(st, inv):
            return _consistent_accuracies(0.85, 0.12, True)

        install_accuracies_mock(monkeypatch, _accuracies)
        install_inversion_mock(monkeypatch)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([clean_rec, wm_rec], out, "T2S", device="cpu")

        det_path = out / "evaluation" / "detector_records.jsonl"
        rows = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]

        clean_rows = [r for r in rows if r.get("source_role") == "clean"]
        wm_rows = [r for r in rows if r.get("source_role") == "watermarked"]

        # Clean cohorts scored with clean state
        assert clean_rows, "clean rows missing"
        assert all(r["status"] == ROW_STATUS_SCORED for r in clean_rows), clean_rows
        assert all(r["t2s_watermark_id"] == "clean-ok" for r in clean_rows)

        # Watermarked cohorts failed as MISSING state — never validation/scoring
        assert wm_rows, "watermarked rows missing"
        assert all(r["status"] == ROW_STATUS_FAILED_MISSING_STATE for r in wm_rows), wm_rows
        assert all(r.get("failure_cause") == FAILURE_CAUSE_MISSING_REQUIRED_STATE
                   for r in wm_rows)
        assert all(r.get("error_type") == "DetectorMissingStateError" for r in wm_rows)

        # Stage: reducer contract — primary (wm) report incomplete → missing state
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert result.get("scored_count") == len(clean_rows)
        assert result.get("failed_count") == len(wm_rows)


class TestOrchestratorRealValidation:
    def test_state_json_malformed_gives_failed_state_validation(
            self, tmp_path, monkeypatch):
        from experiments.eval import evaluate_detector

        state = _make_state()
        bad_path = tmp_path / "bad_state.json"
        bad_path.write_text("not valid json")
        rec = _make_orch_record("1", "watermarked", state, bad_path, tmp_path)
        out = _setup_run(tmp_path, [rec])

        install_pipe_utils_stub()
        # Path exists but load raises (malformed payload)
        install_state_load_mock(
            monkeypatch,
            {str(bad_path): state},
            failures_by_path={str(bad_path): ValueError("corrupt state JSON")})

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert result.get("dominant_failure_cause") == FAILURE_CAUSE_STATE_VALIDATION
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=False) == 2
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=True) == 2

    def test_state_digest_mismatch_gives_failed_state_validation(
            self, tmp_path, monkeypatch):
        from experiments.eval import evaluate_detector

        state = _make_state(watermark_id="correct-id")
        good_path = tmp_path / "good_state.json"
        good_path.write_text("{}")
        rec = _make_orch_record("1", "watermarked", state, good_path, tmp_path)
        # Tamper the recorded SHA
        rec["source_metadata"]["t2s_state_sha256"] = _sha256("tampered")
        out = _setup_run(tmp_path, [rec])

        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(good_path): state})
        install_inversion_mock(monkeypatch)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert result.get("dominant_failure_cause") == FAILURE_CAUSE_STATE_VALIDATION
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=False) == 2
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=True) == 2


class TestOrchestratorRealScoring:
    def test_runtime_inversion_failure_gives_failed_scoring(self, tmp_path, monkeypatch):
        from experiments.eval import evaluate_detector

        state = _make_state()
        good_path = tmp_path / "good_state.json"
        good_path.write_text("{}")
        rec = _make_orch_record("1", "watermarked", state, good_path, tmp_path)
        out = _setup_run(tmp_path, [rec])

        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(good_path): state})

        def _fail_inversion(*a, **kw):
            raise RuntimeError("CUDA OOM during inversion")

        install_inversion_mock(monkeypatch, _fail_inversion)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        assert result.get("dominant_failure_cause") == FAILURE_CAUSE_SCORING_ERROR
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=False) == 2
        assert determine_exit_code(
            {"stages": {"detector": result}}, allow_missing_metrics=True) == 2

    def test_valid_run_produces_records_aggregate_and_revision_passed(
            self, tmp_path, monkeypatch):
        from experiments.eval import evaluate_detector

        state = _make_state(watermark_id="valid-wm")
        good_path = tmp_path / "good_state.json"
        good_path.write_text("{}")
        rec = _make_orch_record("1", "watermarked", state, good_path, tmp_path)
        out = _setup_run(tmp_path, [rec])

        get_pipe_provider = mock.MagicMock(return_value=_make_fake_pipe())
        install_pipe_utils_stub(get_pipe_provider)
        install_state_load_mock(monkeypatch, {str(good_path): state})

        def _accuracies(st, inv):
            return _consistent_accuracies(0.85, 0.12, True)

        install_accuracies_mock(monkeypatch, _accuracies)
        install_inversion_mock(monkeypatch)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)
        assert result["count_invariant_satisfied"] is True
        assert result["scored_count"] >= 1

        # Pipe got model_id + revision + resolution + scheduler, no defaults
        call_kwargs = get_pipe_provider.call_args.kwargs
        assert call_kwargs["pretrained_model_name_or_path"] == MODEL_ID
        assert call_kwargs["revision"] == MODEL_REVISION
        assert call_kwargs["resolution"] == 512
        assert call_kwargs["schedulers_name"] == "DDIM"

        det_path = out / "evaluation" / "detector_records.jsonl"
        assert det_path.is_file()
        rows = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]
        scored = [r for r in rows if r.get("status") == ROW_STATUS_SCORED]
        assert scored, rows
        for r in scored:
            assert r["t2s_state_verified"] is True
            assert r["t2s_watermark_id"] == "valid-wm"
            assert "t2s_state_sha256" in r
            assert "t2s_model_revision" in r


# ===========================================================================
# 10. load_state direct — index and modes
# ===========================================================================

class TestLoadStateDirect:
    def test_duplicate_run_role_raises_validation_error(self, monkeypatch):
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        rec = _make_canonical(run_id="1", source_role="watermarked")
        with pytest.raises(DetectorStateValidationError, match="Duplicate"):
            t2s_detector.load_state([rec, dict(rec)], "cpu")

    def test_missing_run_id_raises_validation_error(self, monkeypatch):
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        rec = _make_canonical(run_id="1", source_role="watermarked")
        rec["run_id"] = ""
        with pytest.raises(DetectorStateValidationError, match="missing run_id"):
            t2s_detector.load_state([rec], "cpu")

    def test_missing_role_raises_validation_error(self, monkeypatch):
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        rec = _make_canonical(run_id="1")
        rec["role"] = ""
        rec.pop("source_role", None)
        with pytest.raises(DetectorStateValidationError, match="missing role"):
            t2s_detector.load_state([rec], "cpu")

    def test_all_paths_missing_raises_missing_state(self, tmp_path, monkeypatch):
        state = _make_state()
        missing = tmp_path / "no.json"  # absent
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(missing))
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorMissingStateError):
            t2s_detector.load_state([rec], "cpu")

    def test_official_compatible_plus_t2s_official_succeeds(self, tmp_path, monkeypatch):
        state = _make_state(rng_mode="official_compatible",
                            inversion_mode="t2s_official")
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        result = t2s_detector.load_state([rec], "cpu")
        assert result["state_cache"][("1", "watermarked")] is state
        assert not result["missing_state_keys"]
        assert result["model_id"] == MODEL_ID
        assert result["model_revision"] == MODEL_REVISION

    def test_official_compatible_plus_benchmark_ddim_succeeds(
            self, tmp_path, monkeypatch):
        state = _make_state(rng_mode="official_compatible",
                            inversion_mode="benchmark_ddim")
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        result = t2s_detector.load_state([rec], "cpu")
        assert result["state_cache"][("1", "watermarked")] is state

    def test_identity_mismatch_at_load_raises_validation_error(
            self, tmp_path, monkeypatch):
        state = _make_state()
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path),
                              t2s_state_sha256=_sha256("wrong"))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        with pytest.raises(DetectorStateValidationError, match="state SHA mismatch"):
            t2s_detector.load_state([rec], "cpu")

    def test_missing_pipe_profile_field_raises_validation_error(
            self, tmp_path, monkeypatch):
        state = _make_state(model_id=None)
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        with pytest.raises(DetectorStateValidationError, match="model_id"):
            t2s_detector.load_state([rec], "cpu")


# ===========================================================================
# 11. Benchmark DDIM step binding
# ===========================================================================

class TestBenchmarkStepBinding:
    def test_benchmark_ddim_uses_generation_inference_steps(self, tmp_path):
        """benchmark_ddim: invert_image gets num_inversion_steps=10 AND
        benchmark_num_inference_steps=50 (generation profile), never merged."""
        state = _make_state(
            inversion_mode="benchmark_ddim",
            num_inversion_steps=10,
            num_inference_steps=50)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)

        inv = prov["t2s_inversion_module"].invert_image
        assert inv.call_count == 1
        kwargs = inv.call_args.kwargs
        assert kwargs["inversion_mode"] == "benchmark_ddim"
        assert kwargs["num_inversion_steps"] == 10
        assert kwargs["benchmark_num_inference_steps"] == 50
        # Effective steps = generation inference steps for benchmark path
        assert result["t2s_effective_inversion_steps"] == 50
        assert result["t2s_effective_inversion_step_source"] == (
            "state.num_inference_steps")

    def test_official_inversion_uses_num_inversion_steps(self, tmp_path):
        """t2s_official: 10-step inversion; benchmark field not used."""
        state = _make_state(
            inversion_mode="t2s_official",
            num_inversion_steps=10,
            num_inference_steps=50)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)

        inv = prov["t2s_inversion_module"].invert_image
        assert inv.call_count == 1
        kwargs = inv.call_args.kwargs
        assert kwargs["num_inversion_steps"] == 10
        assert kwargs["benchmark_num_inference_steps"] == 50
        # Effective steps = official inversion steps
        assert result["t2s_effective_inversion_steps"] == 10
        assert result["t2s_effective_inversion_step_source"] == (
            "state.num_inversion_steps")

    def test_scored_provenance_matches_effective_steps(self, tmp_path):
        """Recorded provenance must never claim steps that were not run."""
        state = _make_state(
            inversion_mode="benchmark_ddim",
            num_inversion_steps=10,
            num_inference_steps=37)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        result = _score(prov, entry, tmp_path)

        assert result["t2s_num_inversion_steps"] == 10
        assert result["t2s_num_inference_steps"] == 37
        assert result["t2s_actual_official_inversion_steps"] == 10
        assert result["t2s_actual_benchmark_inference_steps"] == 37
        assert result["t2s_effective_inversion_steps"] == 37
        assert result["t2s_effective_inversion_step_source"] == (
            "state.num_inference_steps")
        assert result["t2s_latent_shape"] == [1, 4, 64, 64]


# ===========================================================================
# 12. Strict numeric state profile validation
# ===========================================================================

class TestNumericProfileValidation:
    def test_num_inference_steps_required_and_positive(self, tmp_path):
        state = _make_state(num_inference_steps=None)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="num_inference_steps"):
            _score(prov, entry, tmp_path)

    def test_zero_num_inference_steps_rejected(self, tmp_path):
        state = _make_state(num_inference_steps=0)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="positive"):
            _score(prov, entry, tmp_path)

    def test_float_num_inference_steps_rejected(self, tmp_path):
        state = _make_state(num_inference_steps=50.0)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="float"):
            _score(prov, entry, tmp_path)

    def test_bool_num_inference_steps_rejected(self, tmp_path):
        state = _make_state(num_inference_steps=True)
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="bool"):
            _score(prov, entry, tmp_path)

    def test_resolution_invalid_is_state_validation(self, tmp_path):
        state = _make_state(resolution="5.12")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError, match="resolution"):
            _score(prov, entry, tmp_path)


# ===========================================================================
# 13. Cohort state compatibility — fails before pipe construction
# ===========================================================================

class TestCohortCompatibility:
    """Two states with mixed identity configuration must fail closed.

    Proven via REAL load_state: the compatibility check runs before the pipe
    constructor is ever invoked.
    """

    def _two_state_records(self, tmp_path, state_a, state_b):
        pa = tmp_path / "a.json"
        pa.write_text("{}")
        pb = tmp_path / "b.json"
        pb.write_text("{}")
        rec_a = _make_canonical(run_id="1", source_role="clean",
                                state=state_a, tmp_path=tmp_path,
                                t2s_state_path=str(pa))
        rec_b = _make_canonical(run_id="1", source_role="watermarked",
                                state=state_b, tmp_path=tmp_path,
                                t2s_state_path=str(pb))
        return [rec_a, rec_b], {str(pa): state_a, str(pb): state_b}

    def _expect_rejected_before_pipe(self, tmp_path, monkeypatch, state_a, state_b):
        records, states_by_path = self._two_state_records(tmp_path, state_a, state_b)
        pipe_mock = _make_fake_pipe()
        get_pipe_provider = mock.MagicMock(return_value=pipe_mock)
        install_pipe_utils_stub(get_pipe_provider)
        install_state_load_mock(monkeypatch, states_by_path)
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError, match="incompatib"):
            t2s_detector.load_state(records, "cpu")
        get_pipe_provider.assert_not_called()

    def test_mixed_provider_config_rejected_before_pipe(self, tmp_path, monkeypatch):
        base = dict(latent_shape=[1, 4, 64, 64])
        a = _make_state(watermark_id="a", provider_config_sha256=_sha256("pc-a"),
                        **base)
        b = _make_state(watermark_id="b", provider_config_sha256=_sha256("pc-b"),
                        **base)
        self._expect_rejected_before_pipe(tmp_path, monkeypatch, a, b)

    def test_mixed_num_inference_steps_rejected_before_pipe(self, tmp_path, monkeypatch):
        a = _make_state(watermark_id="a", num_inference_steps=50)
        b = _make_state(watermark_id="b", num_inference_steps=25)
        self._expect_rejected_before_pipe(tmp_path, monkeypatch, a, b)

    def test_mixed_inversion_mode_rejected_before_pipe(self, tmp_path, monkeypatch):
        a = _make_state(watermark_id="a", inversion_mode="t2s_official")
        b = _make_state(watermark_id="b", inversion_mode="benchmark_ddim")
        self._expect_rejected_before_pipe(tmp_path, monkeypatch, a, b)

    def test_mixed_latent_shape_rejected_before_pipe(self, tmp_path, monkeypatch):
        a = _make_state(watermark_id="a", latent_shape=[1, 4, 64, 64])
        b = _make_state(watermark_id="b",
                        latent_shape=[1, 16, 64, 64],
                        key_channels=list(range(8)),
                        msg_channels=list(range(8, 16)))
        self._expect_rejected_before_pipe(tmp_path, monkeypatch, a, b)

    def test_mixed_rng_mode_rejected_before_pipe(self, tmp_path, monkeypatch):
        a = _make_state(watermark_id="a", rng_mode="official_compatible")
        b = _make_state(watermark_id="b", rng_mode="raven_deterministic")
        self._expect_rejected_before_pipe(tmp_path, monkeypatch, a, b)

    def test_same_config_different_watermark_id_is_accepted(self, tmp_path, monkeypatch):
        """Different watermark_id + state SHA is fine — that is the pairing
        signal.  Provider/inversion/latent config must match."""
        a = _make_state(watermark_id="a")
        b = _make_state(watermark_id="b")
        records, states_by_path = self._two_state_records(tmp_path, a, b)
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, states_by_path)
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        result = t2s_detector.load_state(records, "cpu")
        assert ("1", "clean") in result["state_cache"]
        assert ("1", "watermarked") in result["state_cache"]


# ===========================================================================
# 14. Pipe/state latent shape + channel layout
# ===========================================================================

class TestPipeLatentShape:
    def test_pipe_latent_shape_mismatch_is_state_validation(self, tmp_path, monkeypatch):
        """state latent_shape [1,16,64,64] vs pipe [1,4,64,64] fails closed
        before any inversion."""
        state = _make_state(
            latent_shape=[1, 16, 64, 64],
            key_channels=list(range(8)),
            msg_channels=list(range(8, 16)))
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub(latent_shape=(1, 4, 64, 64))
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError, match="latent shape"):
            t2s_detector.load_state([rec], "cpu")

    def test_matching_pipe_latent_shape_succeeds(self, tmp_path, monkeypatch):
        state = _make_state(latent_shape=[1, 4, 64, 64])
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub(latent_shape=(1, 4, 64, 64))
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        result = t2s_detector.load_state([rec], "cpu")
        assert result["latent_shape"] == [1, 4, 64, 64]

    def test_channel_layout_must_cover_latent(self, tmp_path, monkeypatch):
        """key ∪ msg must cover every latent channel; overlap is invalid."""
        # Channels [0,1] key + [1,2] msg — overlap and missing channel 3
        state = _make_state(key_channels=[0, 1], msg_channels=[1, 2])
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError, match="channel"):
            t2s_detector.load_state([rec], "cpu")

    def test_malformed_latent_shape_rejected(self, tmp_path, monkeypatch):
        state = _make_state(latent_shape=[1, 4])
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError, match="latent_shape"):
            t2s_detector.load_state([rec], "cpu")


# ===========================================================================
# 15. Real orchestrator benchmark regression
# ===========================================================================

class TestOrchestratorRealBenchmark:
    def test_real_orchestrator_benchmark_step_binding(self, tmp_path, monkeypatch):
        """Full evaluate_detector with benchmark_ddim: verify the exact
        invert_image kwargs and the provenance written to detector_records."""
        from experiments.eval import evaluate_detector

        state = _make_state(
            watermark_id="bench-wm",
            inversion_mode="benchmark_ddim",
            num_inversion_steps=10,
            num_inference_steps=37)
        good_path = tmp_path / "bench_state.json"
        good_path.write_text("{}")
        rec = _make_orch_record("1", "watermarked", state, good_path, tmp_path)
        out = _setup_run(tmp_path, [rec])

        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(good_path): state})

        captured = {}

        def _invert(pipe, image, **kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        install_inversion_mock(monkeypatch, _invert)

        def _accuracies(st, inv):
            return _consistent_accuracies(0.85, 0.12, True)

        install_accuracies_mock(monkeypatch, _accuracies)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        # invert_image called with the exact benchmark binding
        assert captured.get("inversion_mode") == "benchmark_ddim"
        assert captured.get("num_inversion_steps") == 10
        assert captured.get("benchmark_num_inference_steps") == 37

        det_path = out / "evaluation" / "detector_records.jsonl"
        rows = [json.loads(l) for l in det_path.read_text().splitlines() if l.strip()]
        scored = [r for r in rows if r.get("status") == ROW_STATUS_SCORED]
        assert scored, rows
        for r in scored:
            assert r["t2s_num_inversion_steps"] == 10
            assert r["t2s_num_inference_steps"] == 37
            assert r["t2s_effective_inversion_steps"] == 37
            assert r["t2s_effective_inversion_step_source"] == (
                "state.num_inference_steps")
            assert r["t2s_latent_shape"] == [1, 4, 64, 64]

        assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)
        assert result.get("scored_count", 0) >= 1


# ===========================================================================
# 16. Strict channel-layout validation — no native exception leaks
# ===========================================================================

class TestStrictChannelLayout:
    """Malformed channel lists must fail as DetectorStateValidationError
    BEFORE the pipe constructor runs — never a native ValueError/TypeError
    that the orchestrator would classify as failed_internal_error."""

    def _load_one(self, tmp_path, monkeypatch, state):
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        pipe_mock = _make_fake_pipe()
        get_pipe_provider = mock.MagicMock(return_value=pipe_mock)
        install_pipe_utils_stub(get_pipe_provider)
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        return get_pipe_provider, rec

    def test_empty_key_channels_rejected_before_pipe(self, tmp_path, monkeypatch):
        """[] would build a zero-channel T2SMark and fail at scoring as a
        raw ValueError — must be validation, before the pipe is built."""
        state = _make_state(key_channels=[], msg_channels=[0, 1, 2])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError) as exc:
            t2s_detector.load_state([rec], "cpu")
        text = str(exc.value)
        assert "key_channels" in text
        assert "at least one channel" in text
        assert "run_id=1" in text
        assert "source_role=watermarked" in text
        get_pipe_provider.assert_not_called()

    def test_empty_msg_channels_rejected_before_pipe(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0], msg_channels=[])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError) as exc:
            t2s_detector.load_state([rec], "cpu")
        text = str(exc.value)
        assert "msg_channels" in text
        assert "at least one channel" in text
        assert "run_id=1" in text
        assert "source_role=watermarked" in text
        get_pipe_provider.assert_not_called()

    def test_noncanonical_2_plus_2_layout_still_accepted(self, tmp_path, monkeypatch):
        """Coverage-only design: a valid non-canonical 2+2 layout on a
        4-channel latent remains acceptable."""
        state = _make_state(key_channels=[0, 1], msg_channels=[2, 3])
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        result = t2s_detector.load_state([rec], "cpu")
        assert ("1", "watermarked") in result["state_cache"]

    def test_key_channels_with_string_element_rejected_before_pipe(
            self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0, "abc"])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="key_channels"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_key_channels_with_none_element_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0, None])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="key_channels"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_key_channels_with_bool_element_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[True])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="key_channels"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_key_channels_with_float_element_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0, 1.5])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="key_channels"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_msg_channels_not_a_list_rejected(self, tmp_path, monkeypatch):
        state = _make_state(msg_channels=4)
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="msg_channels"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_duplicate_key_channels_rejected(self, tmp_path, monkeypatch):
        """key=[0,0] would silently pass a set-based coverage check but the
        detector later consumes the raw list — duplicates must fail."""
        state = _make_state(key_channels=[0, 0], msg_channels=[1, 2, 3])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError,
                           match="duplicate channel indices"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_duplicate_msg_channels_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0], msg_channels=[1, 1, 2, 3])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError,
                           match="duplicate channel indices"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_channel_index_out_of_range_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[4], msg_channels=[0, 1, 2])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="out of range"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_negative_channel_index_rejected(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[-1], msg_channels=[0, 1, 2])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError, match="out of range"):
            t2s_detector.load_state([rec], "cpu")
        get_pipe_provider.assert_not_called()

    def test_error_message_includes_run_id_and_field(self, tmp_path, monkeypatch):
        state = _make_state(key_channels=[0, "abc"])
        get_pipe_provider, rec = self._load_one(tmp_path, monkeypatch, state)
        with pytest.raises(DetectorStateValidationError) as exc:
            t2s_detector.load_state([rec], "cpu")
        text = str(exc.value)
        assert "run_id=1" in text
        assert "source_role=watermarked" in text
        assert "key_channels" in text
        assert "'abc'" in text


# ===========================================================================
# 17. Strict numeric profile — strings and Unicode digits rejected
# ===========================================================================

class TestStrictNumericProfile:
    def test_numeric_string_inference_steps_rejected(self, tmp_path):
        """Even plain \"50\" is rejected: state numerics are JSON ints."""
        state = _make_state(num_inference_steps="50")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="num_inference_steps"):
            _score(prov, entry, tmp_path)

    def test_unicode_digit_string_inference_steps_rejected(self, tmp_path):
        """\"²\" isdigit() True but int() raises — must still be validation
        error, never a native ValueError leak."""
        state = _make_state(num_inversion_steps="²")
        canonical = _make_canonical(state=state, tmp_path=tmp_path)
        entry = _make_entry()
        prov = _fake_provider_info(
            state=state,
            state_index={("1", "watermarked"): canonical},
            state_cache={("1", "watermarked"): state})
        with pytest.raises(DetectorStateValidationError,
                           match="num_inversion_steps"):
            _score(prov, entry, tmp_path)

    def test_unicode_digit_state_field_rejected_at_load(self, tmp_path, monkeypatch):
        """Same guarantee through the real load_state path."""
        state = _make_state(num_inference_steps="٥٠")  # Arabic-Indic digits
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub()
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError,
                           match="num_inference_steps"):
            t2s_detector.load_state([rec], "cpu")


# ===========================================================================
# 18. Pipe shape error classification
# ===========================================================================

class TestPipeShapeErrorClassification:
    def test_pipe_get_latent_shape_runtime_error_is_provider_init(
            self, tmp_path, monkeypatch):
        """Lazy model loading raising RuntimeError inside get_latent_shape is
        a provider initialization error, not internal/validation."""
        state = _make_state()
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        pipe = mock.MagicMock()
        pipe.get_latent_shape.side_effect = RuntimeError("model download failed")
        get_pipe_provider = mock.MagicMock(return_value=pipe)
        install_pipe_utils_stub(get_pipe_provider)
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorProviderInitializationError,
                           match="get_latent_shape"):
            t2s_detector.load_state([rec], "cpu")

    def test_pipe_get_latent_shape_oserror_is_provider_init(
            self, tmp_path, monkeypatch):
        state = _make_state()
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        pipe = mock.MagicMock()
        pipe.get_latent_shape.side_effect = OSError("weights missing on disk")
        get_pipe_provider = mock.MagicMock(return_value=pipe)
        install_pipe_utils_stub(get_pipe_provider)
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorProviderInitializationError,
                           match="get_latent_shape"):
            t2s_detector.load_state([rec], "cpu")

    def test_pipe_shape_mismatch_remains_state_validation(
            self, tmp_path, monkeypatch):
        """Shape obtained OK but different from state → state validation
        error, never provider error."""
        state = _make_state(latent_shape=[1, 16, 64, 64],
                            key_channels=list(range(8)),
                            msg_channels=list(range(8, 16)))
        path = tmp_path / "st.json"
        path.write_text("{}")
        rec = _make_canonical(state=state, tmp_path=tmp_path,
                              t2s_state_path=str(path))
        install_pipe_utils_stub(latent_shape=(1, 4, 64, 64))
        install_state_load_mock(monkeypatch, {str(path): state})
        monkeypatch.setattr(t2s_detector, "_ensure_paths", lambda: None)
        with pytest.raises(DetectorStateValidationError, match="latent shape"):
            t2s_detector.load_state([rec], "cpu")


# ===========================================================================
# 19. Static helpers: required fields + stage derivation
# ===========================================================================

class TestStaticContract:
    def test_required_metadata_fields_complete(self):
        required = {"t2s_state_path", "t2s_state_sha256", "t2s_watermark_id",
                     "t2s_provider_config_sha256", "t2s_protocol_mode",
                     "t2s_rng_mode", "t2s_inversion_mode",
                     "t2s_num_inversion_steps"}
        assert required.issubset(t2s_detector.REQUIRED_METADATA_FIELDS)

    def test_describe_required_artifacts(self):
        artifacts = t2s_detector.describe_required_artifacts()
        assert any("t2s_state_path" in a for a in artifacts)
        assert any("provider_config_sha256" in a for a in artifacts)
        assert any("inversion_mode" in a for a in artifacts)
        assert any("rng_mode" in a for a in artifacts)
        assert any("protocol_mode" in a for a in artifacts)

    def test_missing_state_rows_give_missing_required_state(self):
        rows = [
            {"run_id": "1", "evaluation_cohort": "original_watermarked",
             "status": ROW_STATUS_FAILED_MISSING_STATE,
             "failure_cause": FAILURE_CAUSE_MISSING_REQUIRED_STATE},
        ]
        result = reduce_detector_stage_status(rows)
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_validation_error_dominates_missing_state(self):
        rows = [
            {"run_id": "1", "evaluation_cohort": "original_watermarked",
             "status": ROW_STATUS_FAILED_MISSING_STATE,
             "failure_cause": FAILURE_CAUSE_MISSING_REQUIRED_STATE},
            {"run_id": "2", "evaluation_cohort": "attacked_watermarked",
             "status": ROW_STATUS_FAILED_STATE_VALIDATION,
             "failure_cause": FAILURE_CAUSE_STATE_VALIDATION},
        ]
        result = reduce_detector_stage_status(rows)
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_allow_policy(self):
        assert stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=True)
        assert not stage_status_is_allowable(
            STATUS_FAILED_MISSING_REQUIRED_STATE, allow_missing_metrics=False)
        assert not stage_status_is_allowable(
            STATUS_FAILED_STATE_VALIDATION, allow_missing_metrics=True)
        assert not stage_status_is_allowable(
            STATUS_FAILED_SCORING, allow_missing_metrics=True)
