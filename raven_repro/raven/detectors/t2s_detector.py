"""T2SMark detector adapter.

Every sample has its own portable state.  ``load_state`` builds a per-sample
``(run_id, role)`` metadata index, loads and validates every existing state,
and constructs the inversion pipe from one fully-validated state profile.
``score_image`` resolves the canonical metadata and the cached state by
``(evaluation_entry["run_id"], evaluation_entry["source_role"])``.

Canonical mode constants come from ``eval_bench_wm.utils.wm.t2s_provider`` —
never duplicated here.  All state identity fields fail closed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
)

# Every field here MUST be present and non-empty in the resolved metadata.
# Missing → DetectorMissingStateError.  Mismatch → DetectorStateValidationError.
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "t2s_state_path",
    "t2s_state_sha256",
    "t2s_watermark_id",
    "t2s_provider_config_sha256",
    "t2s_protocol_mode",
    "t2s_rng_mode",
    "t2s_inversion_mode",
    "t2s_num_inversion_steps",
})

# Pipe profile fields that a validated state MUST carry — no fallbacks.
PIPE_PROFILE_FIELDS = (
    "model_id",
    "model_revision",
    "scheduler",
    "resolution",
    "num_inference_steps",
)

# State fields that must agree across every state in one evaluation run,
# because a single pipe/scheduler/inversion configuration is applied to all.
# Mirrors the canonical standalone verifier's SHARED_STATE_FIELDS identity
# semantics (eval_bench_wm/run_verification.py).
T2S_COHORT_COMPAT_FIELDS = (
    "latent_shape",
    "key_channels",
    "msg_channels",
    "key_length",
    "msg_length",
    "tau",
    "rng_mode",
    "inversion_mode",
    "num_inversion_steps",
    "provider_config_sha256",
    "model_id",
    "model_revision",
    "scheduler",
    "resolution",
    "num_inference_steps",
)


def describe_required_artifacts() -> list[str]:
    return [
        "t2s_state_path (per-sample portable state file, one per run_id/role)",
        "t2s_state_sha256 (validated fail-closed against loaded state)",
        "t2s_watermark_id (validated fail-closed against loaded state)",
        "t2s_provider_config_sha256 (validated fail-closed against loaded state)",
        "t2s_protocol_mode (canonical shared-tr-clean mode)",
        "t2s_rng_mode (validated fail-closed against loaded state)",
        "t2s_inversion_mode (validated fail-closed against loaded state)",
        "t2s_num_inversion_steps (validated fail-closed against loaded state)",
        "Stable Diffusion pipe for T2S inversion (configured from validated state)",
    ]


# ---------------------------------------------------------------------------
# Scoring output contract — required fields
# ---------------------------------------------------------------------------

def _require_finite_float(accuracies: dict[str, Any], field: str,
                           run_id: str, source_role: str) -> float:
    """Extract a required finite float from accuracies, or raise."""
    if field not in accuracies:
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"missing required scoring output: {field}"
        )
    try:
        value = float(accuracies[field])
    except (ValueError, TypeError):
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} is not convertible to float: {accuracies[field]!r}"
        ) from None
    if not math.isfinite(value):
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} is non-finite: {value!r}"
        )
    return value


def _require_bool(accuracies: dict[str, Any], field: str,
                   run_id: str, source_role: str) -> bool:
    """Extract a required real bool from accuracies, or raise."""
    if field not in accuracies:
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"missing required scoring output: {field}"
        )
    if not isinstance(accuracies[field], bool):
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} must be a real bool, got "
            f"{type(accuracies[field]).__name__}: {accuracies[field]!r}"
        )
    return accuracies[field]


def _validate_optional_accuracy(accuracies: dict[str, Any], field: str,
                                 run_id: str, source_role: str) -> float | None:
    """Validate optional accuracy field: None ok, else must be finite in [0,1]."""
    if field not in accuracies or accuracies[field] is None:
        return None
    try:
        value = float(accuracies[field])
    except (ValueError, TypeError):
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} is not convertible to float: {accuracies[field]!r}"
        ) from None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} must be finite in [0, 1], got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _require_nonempty(record: dict[str, Any], field: str,
                       run_id: str, source_role: str) -> str:
    """Extract a non-empty string from record; raise MissingStateError if absent."""
    value = record.get(field)
    if value is None or str(value).strip() == "":
        raise DetectorMissingStateError(
            f"run_id={run_id} source_role={source_role}: "
            f"required metadata field {field!r} is missing or empty"
        )
    return str(value).strip()


def _validate_int_field(record: dict[str, Any], field: str,
                         run_id: str, source_role: str) -> int:
    """Extract and validate an integer metadata field."""
    raw = _require_nonempty(record, field, run_id, source_role)
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"{field} must be a valid integer, got {raw!r}"
        ) from None


# ---------------------------------------------------------------------------
# State binding validation — shared by load_state and score_image
# ---------------------------------------------------------------------------

def _validate_state_binding(
    metadata: dict[str, Any],
    state: Any,
    run_id: str,
    source_role: str,
    *,
    rng_modes: frozenset[str],
    inversion_modes: frozenset[str],
    protocol_modes: frozenset[str],
) -> None:
    """Validate every recorded state identity against the loaded state.

    Missing/empty metadata → DetectorMissingStateError.
    Mismatch or unsupported value → DetectorStateValidationError.
    """
    recorded_sha = _require_nonempty(metadata, "t2s_state_sha256",
                                      run_id, source_role)
    actual_sha = state.state_sha256()
    if recorded_sha != actual_sha:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S state SHA mismatch: "
            f"recorded={recorded_sha} actual={actual_sha}"
        )

    recorded_wm_id = _require_nonempty(metadata, "t2s_watermark_id",
                                        run_id, source_role)
    if recorded_wm_id != state.watermark_id:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S watermark_id mismatch: "
            f"recorded={recorded_wm_id} state={state.watermark_id}"
        )

    recorded_provider_sha = _require_nonempty(metadata,
                                               "t2s_provider_config_sha256",
                                               run_id, source_role)
    if recorded_provider_sha != state.provider_config_sha256:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S provider_config_sha256 mismatch: "
            f"recorded={recorded_provider_sha} "
            f"state={state.provider_config_sha256}"
        )

    recorded_inversion_mode = _require_nonempty(metadata, "t2s_inversion_mode",
                                                  run_id, source_role)
    if recorded_inversion_mode not in inversion_modes:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"unknown t2s_inversion_mode={recorded_inversion_mode!r}, "
            f"expected one of {sorted(inversion_modes)}"
        )
    if recorded_inversion_mode != state.inversion_mode:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S inversion_mode mismatch: "
            f"recorded={recorded_inversion_mode} "
            f"state={state.inversion_mode}"
        )

    recorded_steps = _validate_int_field(metadata, "t2s_num_inversion_steps",
                                          run_id, source_role)
    if recorded_steps != state.num_inversion_steps:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S num_inversion_steps mismatch: "
            f"recorded={recorded_steps} "
            f"state={state.num_inversion_steps}"
        )

    recorded_rng_mode = _require_nonempty(metadata, "t2s_rng_mode",
                                           run_id, source_role)
    if recorded_rng_mode not in rng_modes:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"unknown t2s_rng_mode={recorded_rng_mode!r}, "
            f"expected one of {sorted(rng_modes)}"
        )
    if recorded_rng_mode != state.rng_mode:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"T2S rng_mode mismatch: "
            f"recorded={recorded_rng_mode} state={state.rng_mode}"
        )

    recorded_protocol = _require_nonempty(metadata, "t2s_protocol_mode",
                                           run_id, source_role)
    if recorded_protocol not in protocol_modes:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"unknown t2s_protocol_mode={recorded_protocol!r}, "
            f"expected one of {sorted(protocol_modes)}"
        )


# ---------------------------------------------------------------------------
# Pipe profile — derived from validated state, no fallbacks
# ---------------------------------------------------------------------------

def _require_positive_state_int(state: Any, field: str,
                                run_id: str, source_role: str) -> int:
    """Extract a positive integer field from a loaded state, fail closed.

    Portable-state numeric fields are JSON integers: only a real ``int`` is
    accepted.  None, bool, float, and every string (numeric strings included)
    are rejected as DetectorStateValidationError, so no native ValueError or
    TypeError can escape — including Unicode-digit strings whose
    ``str.isdigit()`` is True but ``int()`` would raise.
    """
    value = getattr(state, field, None)
    if value is None:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state is missing required numeric field {field!r}"
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state field {field!r} must be a positive JSON integer, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state field {field!r} must be positive, got {value!r}"
        )
    return value


def _require_pipe_profile(state: Any, run_id: str, source_role: str) -> dict[str, Any]:
    """Extract the pipe profile from a validated state.  No fallbacks.

    Every PIPE_PROFILE_FIELD must be present and valid on the state;
    otherwise the evaluation cannot construct a faithful inversion pipe and
    fails closed as validation error.
    """
    profile: dict[str, Any] = {}
    for field in PIPE_PROFILE_FIELDS:
        value = getattr(state, field, None)
        if value is None or str(value).strip() == "":
            raise DetectorStateValidationError(
                f"run_id={run_id} source_role={source_role}: "
                f"state is missing required pipe profile field {field!r}; "
                f"refusing to fall back to a default profile"
            )
        if field in ("resolution", "num_inference_steps"):
            profile[field] = _require_positive_state_int(
                state, field, run_id, source_role)
        else:
            profile[field] = str(value).strip()
    return profile


def _validate_channel_layout(state: Any, run_id: str, source_role: str) -> None:
    """Fail closed on malformed channel layout before any inversion.

    Design note: this detector accepts every state-declared layout whose
    channels fully cover the latent — the canonical standalone verifier's
    semantics (eval_bench_wm/run_verification.py check_states_compatible).
    The canonical T2S layouts are 1 key + 3 message channels for a 4-channel
    latent and 4 + 12 for a 16-channel latent (t2s_provider.channel_layout),
    but a correctly-signed portable state is authoritative about its own
    layout, so the detector must not reject a state for being non-canonical.

    What is always rejected, all as DetectorStateValidationError (never a
    native ValueError/TypeError that the orchestrator would classify as
    failed_internal_error):
    - latent_shape not a 4-tuple, or non-positive/non-int channel count
    - key_channels/msg_channels not a list or tuple
    - element not a real int (bool, float, string, None rejected)
    - channel index outside [0, channels - 1]
    - duplicate index inside key_channels or inside msg_channels
    - key/msg overlap
    - union not covering every latent channel
    """
    latent_shape = getattr(state, "latent_shape", None)
    if not isinstance(latent_shape, (list, tuple)) or len(latent_shape) != 4:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state has malformed latent_shape={latent_shape!r}"
        )
    channels = latent_shape[1]
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state latent_shape channel count is invalid: {channels!r}"
        )

    for field in ("key_channels", "msg_channels"):
        value = getattr(state, field, None)
        if not isinstance(value, (list, tuple)):
            raise DetectorStateValidationError(
                f"run_id={run_id} source_role={source_role}: "
                f"state field {field!r} must be a list of channel indices, "
                f"got {type(value).__name__}: {value!r}"
            )
        if len(value) == 0:
            # An empty channel list would build a zero-channel T2SMark and
            # only fail at scoring time with a raw ValueError — reject it
            # here as validation, before the pipe is constructed.
            raise DetectorStateValidationError(
                f"run_id={run_id} source_role={source_role}: "
                f"state field {field!r} must contain at least one channel "
                f"index, got empty {type(value).__name__}"
            )
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise DetectorStateValidationError(
                    f"run_id={run_id} source_role={source_role}: "
                    f"state field {field!r} element {index} must be a real "
                    f"int channel index, got {type(item).__name__}: {item!r}"
                )
            if not 0 <= item < channels:
                raise DetectorStateValidationError(
                    f"run_id={run_id} source_role={source_role}: "
                    f"state field {field!r} element {index}={item!r} is out "
                    f"of range [0, {channels - 1}]"
                )
        if len(value) != len(set(value)):
            raise DetectorStateValidationError(
                f"run_id={run_id} source_role={source_role}: "
                f"state field {field!r} contains duplicate channel indices: "
                f"{sorted(value)}"
            )

    key_channels = list(getattr(state, "key_channels"))
    msg_channels = list(getattr(state, "msg_channels"))
    key_set = set(key_channels)
    msg_set = set(msg_channels)
    overlap = key_set & msg_set
    if overlap:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state key_channels and msg_channels overlap: {sorted(overlap)}"
        )
    declared = key_set | msg_set
    expected = set(range(channels))
    if declared != expected:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state channel layout {sorted(declared)} does not cover its "
            f"{channels}-channel latent; expected {sorted(expected)}"
        )


def _check_cohort_state_compatibility(
    states: list[tuple[tuple[str, str], Any]],
) -> None:
    """All loaded states in one run must share the identity configuration.

    Mirrors the canonical verifier: one pipe/inversion profile applies to all
    samples, so mixed provider/inversion/latent-layout states fail closed
    BEFORE the pipe is constructed.
    """
    if not states:
        raise DetectorMissingStateError(
            "T2S cohort has no loadable state; cannot build an inversion pipe"
        )
    (ref_run_id, ref_role), reference = states[0]
    for (run_id, role), state in states[1:]:
        for field in T2S_COHORT_COMPAT_FIELDS:
            expected = getattr(reference, field, None)
            actual = getattr(state, field, None)
            if expected != actual:
                raise DetectorStateValidationError(
                    f"Cohort state incompatibility at run_id={run_id} "
                    f"source_role={role}: {field} differs: "
                    f"{expected!r} vs {actual!r}"
                )


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------

def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load the T2S inversion pipe, provider modules, and per-sample states.

    Steps:
    1. Build ``(run_id, role)`` metadata index; duplicate keys fail closed.
    2. Check every ``t2s_state_path``; missing paths are recorded as
       ``missing_state_keys`` (all-missing → DetectorMissingStateError).
    3. Load every existing state into ``state_cache``; load failures →
       DetectorStateValidationError (never swallowed).
    4. Validate every loaded state's identity binding against its metadata.
    5. Build the pipe from a fully-validated state profile (no fallbacks).
    """
    import torch

    # ---- Step 1: per-sample metadata index ----
    state_index: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        run_id = str(rec.get("run_id", ""))
        if not run_id:
            raise DetectorStateValidationError(
                "T2S record missing run_id; cannot build state index"
            )
        role = str(rec.get("role", rec.get("source_role", "")))
        if not role:
            raise DetectorStateValidationError(
                f"T2S record run_id={run_id} missing role/source_role"
            )
        key = (run_id, role)
        if key in state_index:
            raise DetectorStateValidationError(
                f"Duplicate (run_id={run_id!r}, role={role!r}) in T2S cohort; "
                f"each sample must have a unique key"
            )
        for field in REQUIRED_METADATA_FIELDS:
            _require_nonempty(rec, field, run_id, role)
        state_index[key] = dict(rec)

    if not state_index:
        raise DetectorMissingStateError(
            "T2S state metadata index is empty; no records to score"
        )

    # ---- Import modules ----
    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm import t2s_provider as t2s_provider_module
        from eval_bench_wm.utils.wm import t2s_inversion as t2s_inversion_module
    except ImportError as exc:
        raise DetectorDependencyError(
            f"T2S dependencies not available: {exc}"
        ) from exc

    rng_modes = frozenset(t2s_provider_module.T2S_RNG_MODES)
    inversion_modes = frozenset(t2s_provider_module.T2S_INVERSION_MODES)
    protocol_modes = frozenset({t2s_provider_module.T2S_SHARED_TR_CLEAN_MODE})

    # ---- Step 2 & 3: per-key path check + state load ----
    state_cache: dict[tuple[str, str], Any] = {}
    missing_state_keys: set[tuple[str, str]] = set()

    for (run_id, role), metadata in state_index.items():
        state_path = metadata["t2s_state_path"]
        if not state_path or not Path(state_path).is_file():
            missing_state_keys.add((run_id, role))
            continue
        try:
            state = t2s_provider_module.T2SWatermarkState.load(Path(state_path))
        except Exception as exc:
            # Path exists but the payload is corrupt — never swallowed.
            raise DetectorStateValidationError(
                f"T2S state load failed for run_id={run_id} "
                f"source_role={role}: {type(exc).__name__}: {exc}"
            ) from exc
        state_cache[(run_id, role)] = state

    if not state_cache:
        raise DetectorMissingStateError(
            "T2S state metadata is present but every t2s_state_path is "
            "missing; all rows are failed_missing_required_state"
        )

    # ---- Step 4: validate every loaded state binding ----
    loaded_states: list[tuple[tuple[str, str], Any]] = []
    profiles: list[tuple[tuple[str, str], dict[str, Any]]] = []
    for (run_id, role), state in state_cache.items():
        metadata = state_index[(run_id, role)]
        _validate_state_binding(
            metadata, state, run_id, role,
            rng_modes=rng_modes,
            inversion_modes=inversion_modes,
            protocol_modes=protocol_modes,
        )
        _validate_channel_layout(state, run_id, role)
        loaded_states.append(((run_id, role), state))
        profiles.append(((run_id, role),
                         _require_pipe_profile(state, run_id, role)))

    # ---- Step 5: cohort-wide identity compatibility (BEFORE pipe) ----
    _check_cohort_state_compatibility(loaded_states)

    profile = profiles[0][1]
    model_id = profile["model_id"]
    model_revision = profile["model_revision"]
    scheduler = profile["scheduler"]
    resolution = profile["resolution"]
    num_inference_steps = profile["num_inference_steps"]

    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            revision=model_revision,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name=scheduler,
            disable_tqdm=True,
        )
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"T2S pipe init failed: {type(exc).__name__}: {exc}"
        ) from exc

    # ---- Step 6: runtime pipe latent shape must match the state ----
    reference_state = loaded_states[0][1]
    try:
        actual_latent_shape = tuple(pipe.get_latent_shape(batch_size=1))
    except Exception as exc:
        # get_latent_shape may trigger lazy model loading and raise OSError,
        # RuntimeError, AttributeError, TypeError or other provider-side
        # failures.  Every one of those is a provider initialization error —
        # a shape *mismatch* (below, outside this try) stays a state
        # validation error.
        raise DetectorProviderInitializationError(
            f"T2S pipe get_latent_shape(batch_size=1) failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    expected_latent_shape = tuple(reference_state.latent_shape)
    if actual_latent_shape != expected_latent_shape:
        raise DetectorStateValidationError(
            f"pipe latent shape {actual_latent_shape} does not match state "
            f"latent_shape {expected_latent_shape}; wrong model or resolution"
        )

    return {
        "pipe": pipe,
        "t2s_provider_module": t2s_provider_module,
        "t2s_inversion_module": t2s_inversion_module,
        "device_obj": device_obj,
        "state_metadata_index": state_index,
        "state_cache": state_cache,
        "missing_state_keys": missing_state_keys,
        "t2s_rng_modes": rng_modes,
        "t2s_inversion_modes": inversion_modes,
        "t2s_protocol_modes": protocol_modes,
        "model_id": model_id,
        "model_revision": model_revision,
        "scheduler": scheduler,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "latent_shape": list(reference_state.latent_shape),
    }


# ---------------------------------------------------------------------------
# score_image — per-sample scoring with cached state
# ---------------------------------------------------------------------------

def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using its per-sample T2S state.

    State is resolved from ``provider_info["state_cache"]`` by
    ``(evaluation_entry["run_id"], evaluation_entry["source_role"])``.
    Original and attacked rows of the same (run_id, source_role) share the
    same cached state object.
    """
    import torch
    from PIL import Image, ImageOps

    # ---- Missing image → FileNotFoundError ----
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"T2S image not found: {image_path}")

    if evaluation_entry is None:
        raise DetectorMissingStateError(
            "T2S requires evaluation_entry with run_id and source_role"
        )

    run_id = str(evaluation_entry.get("run_id", ""))
    source_role = str(evaluation_entry.get("source_role", ""))
    if not run_id or not source_role:
        raise DetectorMissingStateError(
            f"T2S evaluation_entry missing run_id or source_role: "
            f"run_id={run_id!r} source_role={source_role!r}"
        )

    state_index: dict[tuple[str, str], dict[str, Any]] = provider_info.get(
        "state_metadata_index", {})
    state_cache: dict[tuple[str, str], Any] = provider_info.get("state_cache", {})
    missing_state_keys: set[tuple[str, str]] = provider_info.get(
        "missing_state_keys", set())

    key = (run_id, source_role)
    canonical = state_index.get(key)
    if canonical is None:
        raise DetectorMissingStateError(
            f"T2S state metadata index has no entry for "
            f"(run_id={run_id!r}, source_role={source_role!r})"
        )
    if key in missing_state_keys or key not in state_cache:
        raise DetectorMissingStateError(
            f"run_id={run_id} source_role={source_role}: "
            f"t2s_state_path missing or not loadable: "
            f"{canonical.get('t2s_state_path')!r}"
        )

    # ---- If caller also passed a record, it must match the canonical index ----
    if record is not None:
        for field in ("run_id",):
            rec_val = str(record.get(field, ""))
            idx_val = str(canonical.get(field, ""))
            if rec_val and idx_val and rec_val != idx_val:
                raise DetectorStateValidationError(
                    f"run_id={run_id} source_role={source_role}: "
                    f"record.{field}={rec_val!r} does not match "
                    f"indexed value {idx_val!r}"
                )
        rec_role = str(record.get("source_role", record.get("role", "")))
        if rec_role and rec_role != source_role:
            raise DetectorStateValidationError(
                f"run_id={run_id} source_role={source_role}: "
                f"record role={rec_role!r} does not match "
                f"evaluation_entry source_role={source_role!r}"
            )
        for field in ("t2s_state_path", "t2s_state_sha256"):
            rec_val = str(record.get(field, ""))
            idx_val = str(canonical.get(field, ""))
            if rec_val and rec_val != idx_val:
                raise DetectorStateValidationError(
                    f"run_id={run_id} source_role={source_role}: "
                    f"record.{field}={rec_val!r} does not match "
                    f"indexed value {idx_val!r}"
                )

    t2s_provider_mod = provider_info["t2s_provider_module"]
    t2s_inversion_mod = provider_info["t2s_inversion_module"]
    pipe = provider_info["pipe"]

    state = state_cache[key]

    # Re-validate the binding (cheap, defense-in-depth; state already cached).
    _validate_state_binding(
        canonical, state, run_id, source_role,
        rng_modes=provider_info.get("t2s_rng_modes", frozenset()),
        inversion_modes=provider_info.get("t2s_inversion_modes", frozenset()),
        protocol_modes=provider_info.get("t2s_protocol_modes", frozenset()),
    )

    # ---- Pipe profile must match this state exactly ----
    pipe_model_id = provider_info.get("model_id")
    pipe_revision = provider_info.get("model_revision")
    pipe_scheduler = provider_info.get("scheduler")
    pipe_resolution = provider_info.get("resolution")

    if state.model_id is None or str(state.model_id).strip() == "":
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state is missing model_id; pipe profile must be explicit"
        )
    if state.model_revision is None or str(state.model_revision).strip() == "":
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state is missing model_revision; pipe profile must be explicit"
        )
    if state.scheduler is None or str(state.scheduler).strip() == "":
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state is missing scheduler; pipe profile must be explicit"
        )

    state_resolution = _require_positive_state_int(
        state, "resolution", run_id, source_role)

    mismatches = []
    if state.model_id != pipe_model_id:
        mismatches.append(f"model_id {state.model_id!r} != {pipe_model_id!r}")
    if state.model_revision != pipe_revision:
        mismatches.append(
            f"model_revision {state.model_revision!r} != {pipe_revision!r}")
    if state.scheduler != pipe_scheduler:
        mismatches.append(f"scheduler {state.scheduler!r} != {pipe_scheduler!r}")
    if state_resolution != int(pipe_resolution):
        mismatches.append(
            f"resolution {state_resolution!r} != {pipe_resolution!r}")
    if mismatches:
        raise DetectorStateValidationError(
            f"run_id={run_id} source_role={source_role}: "
            f"state conflicts with pipe profile: {mismatches}"
        )

    # ---- Numeric profile validation (fail closed, before inversion) ----
    official_steps = _require_positive_state_int(
        state, "num_inversion_steps", run_id, source_role)
    inference_steps = _require_positive_state_int(
        state, "num_inference_steps", run_id, source_role)

    # ---- Scoring ----
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    try:
        with torch.no_grad():
            zT = t2s_inversion_mod.invert_image(
                pipe, image,
                inversion_mode=state.inversion_mode,
                num_inversion_steps=official_steps,
                benchmark_num_inference_steps=inference_steps,
            )
            if not isinstance(zT, torch.Tensor):
                raise DetectorScoringError(
                    f"T2S inversion returned {type(zT).__name__}, "
                    f"expected torch.Tensor"
                )
            accuracies = t2s_provider_mod.T2SProvider.accuracies_for_state(
                state, zT,
            )
    except Exception as exc:
        raise DetectorScoringError(
            f"T2S scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    # ---- Validate scoring contract ----
    true_score = _require_finite_float(accuracies, "t2s_score_true_key",
                                        run_id, source_role)
    control_score = _require_finite_float(accuracies, "t2s_score_control_key",
                                           run_id, source_role)
    margin = _require_finite_float(accuracies, "t2s_score_margin",
                                    run_id, source_role)
    detection_success = _require_bool(accuracies, "detection_success",
                                       run_id, source_role)
    key_accuracy = _validate_optional_accuracy(accuracies, "key_accuracy",
                                                run_id, source_role)
    msg_accuracy = _validate_optional_accuracy(accuracies, "message_accuracy",
                                                run_id, source_role)

    expected_margin = true_score - control_score
    if not math.isclose(margin, expected_margin, rel_tol=1e-5, abs_tol=1e-9):
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"t2s_score_margin={margin!r} is not consistent with "
            f"true_key - control_key = {true_score!r} - {control_score!r} "
            f"= {expected_margin!r}"
        )

    expected_detection = true_score > control_score
    if detection_success != expected_detection:
        raise DetectorScoringError(
            f"run_id={run_id} source_role={source_role}: "
            f"detection_success={detection_success!r} is not consistent "
            f"with true_key={true_score!r} > control_key={control_score!r}"
        )

    # Effective inversion steps — the value actually used by the inversion
    # path, never a fabricated number that disagrees with the call above.
    if state.inversion_mode == "benchmark_ddim":
        effective_steps = inference_steps
        effective_source = "state.num_inference_steps"
    else:
        effective_steps = official_steps
        effective_source = "state.num_inversion_steps"

    return {
        "raw_score": true_score,
        "canonical_score": true_score,
        "t2s_score_true_key": true_score,
        "t2s_score_control_key": control_score,
        "t2s_score_margin": margin,
        "t2s_detection_success": detection_success,
        "t2s_key_accuracy": key_accuracy,
        "t2s_message_accuracy": msg_accuracy,
        "t2s_bit_accuracy": msg_accuracy,
        # State provenance — all from loaded state or validated metadata
        "t2s_state_path": canonical["t2s_state_path"],
        "t2s_state_sha256": state.state_sha256(),
        "t2s_provider_config_sha256": state.provider_config_sha256,
        "t2s_watermark_id": state.watermark_id,
        "t2s_protocol_mode": canonical["t2s_protocol_mode"],
        "t2s_rng_mode": state.rng_mode,
        "t2s_inversion_mode": state.inversion_mode,
        "t2s_num_inversion_steps": official_steps,
        "t2s_num_inference_steps": inference_steps,
        "t2s_actual_official_inversion_steps": official_steps,
        "t2s_actual_benchmark_inference_steps": inference_steps,
        "t2s_effective_inversion_steps": effective_steps,
        "t2s_effective_inversion_step_source": effective_source,
        "t2s_latent_shape": list(state.latent_shape),
        "t2s_model_id": state.model_id,
        "t2s_model_revision": state.model_revision,
        "t2s_scheduler": state.scheduler,
        "t2s_resolution": state.resolution,
        "t2s_state_verified": True,
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
        "score_direction": "higher_is_watermarked",
    }


# ---------------------------------------------------------------------------
# aggregate — row-by-row, no zip misalignment
# ---------------------------------------------------------------------------

def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate T2S detector rows by cohort.  Row-by-row to avoid zip
    misalignment.  Defensively ignores non-finite bit accuracy values."""
    from . import ROW_STATUS_SCORED

    def _cohort(name: str) -> list[dict[str, Any]]:
        return [r for r in detector_rows
                if r.get("evaluation_cohort") == name
                and r.get("status") == ROW_STATUS_SCORED]

    def _bit_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        vals = []
        unavailable = 0
        for r in rows:
            v = r.get("t2s_bit_accuracy")
            if v is None:
                unavailable += 1
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                unavailable += 1
                continue
            if not math.isfinite(fv):
                unavailable += 1
                continue
            vals.append(fv)
        result: dict[str, Any] = {
            "bit_accuracy_count": len(vals),
            "bit_accuracy_unavailable_count": unavailable,
        }
        if vals:
            arr = np.array(vals)
            result.update({
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            })
        return result if (vals or unavailable) else None

    original_wm = _cohort("original_watermarked")
    attacked_wm = _cohort("attacked_watermarked")

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored
    required = {"original_watermarked", "attacked_watermarked"}
    present = {r["evaluation_cohort"] for r in detector_rows
               if r.get("status") == ROW_STATUS_SCORED}
    missing = sorted(required - present)

    result: dict[str, Any] = {
        "method": "T2S",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(_cohort(c)) for c in present},
        "missing_cohorts": missing,
        "score_type": "t2s_score_true_key",
        "score_direction": "higher_is_watermarked",
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
    }

    def _cohort_metrics(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        bit_stats = _bit_stats(rows)
        if bit_stats:
            result[f"{name}_bit_accuracy"] = bit_stats
        detections = [bool(r["t2s_detection_success"]) for r in rows]
        result[f"{name}_detection_rate"] = sum(detections) / len(detections)
        corrupted = 0
        failed_readable = 0
        for r in rows:
            det = bool(r["t2s_detection_success"])
            ba = r.get("t2s_bit_accuracy")
            ba_val = None
            if ba is not None:
                try:
                    ba_val = float(ba)
                except (ValueError, TypeError):
                    ba_val = None
                if ba_val is not None and not math.isfinite(ba_val):
                    ba_val = None
            if det and ba_val is not None and ba_val < 1.0:
                corrupted += 1
            if not det and ba_val is not None and ba_val == 1.0:
                failed_readable += 1
        result[f"{name}_message_corrupted"] = corrupted
        result[f"{name}_detection_failed_but_readable"] = failed_readable

    _cohort_metrics("original_watermarked", original_wm)
    _cohort_metrics("attacked_watermarked", attacked_wm)

    if original_wm and attacked_wm:
        result["attack_success_rate"] = (
            1.0 - result.get("attacked_watermarked_detection_rate", 0.0)
        )

    return result
