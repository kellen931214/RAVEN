"""Tree-Ring detector adapter — package-local scoring.

Production TR scoring lives in this package: ``tr_scoring.py`` holds the
image decode, inversion, FFT convention, mask/target extraction, and the
default complex-L1 mean score.  The adapter never imports legacy scripts
(``extract_verification_scores.py`` / ``raven_nfpa_tr_eval.py``) and never
loads modules dynamically.

Default score protocol (issue #28):

    score_definition    = complex_l1_mean
    raw_score           = torch.abs(decoded_watermark - target_watermark).mean()
    raw_score_direction = lower_is_watermarked
    canonical_score     = -raw_score
    canonical_score_direction = higher_is_watermarked
    comparison_operator = >=

A p-value protocol (``-log10(p)``) remains only as an explicitly named
optional mode: metadata must declare ``tr_score_mode = log10p`` for the
cohort, and the resulting rows carry separate provenance fields.

All TR provider parameters MUST come from metadata.  Silent fallback to
defaults is forbidden — missing fields cause ``DetectorMissingStateError``,
invalid values cause ``DetectorStateValidationError``.  Mixed provider
configurations across records are rejected before scoring.

The adapter binds directly to the canonical source metadata schema written by
``experiments/generate_watermarked_images.py``:

    model_id, model_revision, scheduler_target, num_inference_steps_target,
    resolution, w_seed, w_channel, w_radius, w_pattern, w_mask_shape,
    w_measurement, w_injection, w_pattern_const,
    watermark_target_sha256, watermark_mask_sha256

Aliases are resolved explicitly (``scheduler``/``scheduler_target``,
``steps``/``num_inference_steps_target``).  Fields that only extraction
outputs produce (inverse_scheduler, detector_dtype, vae_id,
vae_scaling_factor, provider_config_hash) are optional source assertions:
when absent the runtime-derived value is used and recorded without claiming
source verification.

A uniform cohort constructs exactly one provider from the verified profile.
Pipe is built from the verified cohort profile, never hard-coded.  Metadata
steps control canonical inversion — the caller cannot override them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorScoringError,
    DetectorStateValidationError,
)
from . import tr_scoring

# ---------------------------------------------------------------------------
# Required TR provider fields — every field must be present, non-empty, and
# validated.  This matches the canonical TR_PROVIDER_FIELDS in eval_protocol.
# ---------------------------------------------------------------------------
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "w_seed",
    "w_channel",
    "w_radius",
    "w_pattern",
    "w_mask_shape",
    "w_measurement",
    "w_injection",
    "w_pattern_const",
})

# TR provider canonical allowed string values (from tr_provider.py).
_ALLOWED_W_PATTERN: frozenset[str] = frozenset({
    "seed_ring", "seed_zeros", "seed_rand", "rand",
    "zeros", "const", "ring",
})
_ALLOWED_W_MASK_SHAPE: frozenset[str] = frozenset({"circle", "square", "no"})
_ALLOWED_W_INJECTION: frozenset[str] = frozenset({"complex", "seed"})

# The canonical TR scoring path always runs the complex-L1 mean test; no
# alternate measurement implementation exists.  The formal TR profile
# therefore requires exactly this value.
REQUIRED_W_MEASUREMENT: str = "l1_complex"

# Optional metadata field naming the score protocol.  Absent → the default
# complex-L1 protocol.  Present → must be a supported mode and uniform.
SCORE_MODE_FIELD: str = "tr_score_mode"

# Source-required profile identity fields and their canonical aliases.
# ``scheduler`` and ``steps`` have generator-schema aliases; everything else
# is a single field.  Missing → DetectorMissingStateError; mixed →
# DetectorStateValidationError.
SOURCE_REQUIRED_IDENTITY: dict[str, tuple[str, ...]] = {
    "model_id": ("model_id",),
    "model_revision": ("model_revision",),
    "scheduler": ("scheduler", "scheduler_target"),
    "steps": ("steps", "num_inference_steps_target"),
    "resolution": ("resolution",),
    "watermark_target_sha256": ("watermark_target_sha256",),
    "watermark_mask_sha256": ("watermark_mask_sha256",),
}

# Fields only extraction outputs produce.  Source may assert them; when
# absent the runtime-derived value is used and the assertion is recorded as
# unavailable (never fabricated as source-verified).
OPTIONAL_ASSERTION_FIELDS: tuple[str, ...] = (
    "inverse_scheduler",
    "detector_dtype",
    "vae_id",
    "vae_scaling_factor",
    "provider_config_hash",
)


def describe_required_artifacts() -> list[str]:
    return [
        "TR provider parameters in source metadata: "
        "w_seed, w_channel, w_radius, w_pattern, w_mask_shape, "
        "w_measurement, w_injection, w_pattern_const",
        "Stable Diffusion pipe provider (pipe_utils.get_pipe_provider)",
        "TR source identity: model_id, model_revision, "
        "scheduler/scheduler_target, steps/num_inference_steps_target, "
        "resolution, watermark_target_sha256, watermark_mask_sha256",
    ]


# ---------------------------------------------------------------------------
# Strict normalisation helpers
# ---------------------------------------------------------------------------
def _require_nonempty(record: dict[str, Any], field: str,
                      record_index: int) -> str:
    """Return *field* stripped, or raise DetectorMissingStateError."""
    raw = record.get(field)
    if raw is None:
        raise DetectorMissingStateError(
            f"TR field {field!r} is None at record index {record_index} "
            f"(run_id={record.get('run_id', '?')})"
        )
    text = str(raw).strip()
    if not text:
        raise DetectorMissingStateError(
            f"TR field {field!r} is empty at record index {record_index} "
            f"(run_id={record.get('run_id', '?')})"
        )
    return text


def _resolve_profile_field(
    record: dict[str, Any],
    canonical_name: str,
    aliases: tuple[str, ...],
    *,
    record_index: int,
) -> str:
    """Resolve one canonical profile field from its alias set.

    Both aliases absent → DetectorMissingStateError.
    Both present but different → DetectorStateValidationError.
    Exactly one present → that value.
    """
    present: dict[str, str] = {}
    for alias in aliases:
        raw = record.get(alias)
        if raw is not None and str(raw).strip():
            present[alias] = str(raw).strip()
    if not present:
        raise DetectorMissingStateError(
            f"TR profile field {canonical_name!r} is missing at record index "
            f"{record_index} (run_id={record.get('run_id', '?')}); expected "
            f"one of {list(aliases)}"
        )
    unique = sorted(set(present.values()))
    if len(unique) != 1:
        raise DetectorStateValidationError(
            f"TR profile field {canonical_name!r} has conflicting values at "
            f"record index {record_index} (run_id={record.get('run_id', '?')}): "
            f"{present} — aliases {list(aliases)} must agree"
        )
    return unique[0]


def _resolve_uniform_field(
    records: list[dict[str, Any]],
    canonical_name: str,
    aliases: tuple[str, ...],
) -> str:
    """Resolve a canonical field for every record and require uniformity."""
    values = [
        _resolve_profile_field(record, canonical_name, aliases,
                               record_index=idx)
        for idx, record in enumerate(records)
    ]
    unique = sorted(set(values))
    if len(unique) != 1:
        raise DetectorStateValidationError(
            f"Mixed {canonical_name} across TR cohort "
            f"({len(unique)} distinct values across {len(records)} records): "
            f"{unique}"
        )
    return unique[0]


def _strict_int_value(text: str, field: str, *, record_index: int,
                      rid: Any, minimum: int | None = None,
                      multiple_of: int | None = None) -> int:
    """Parse a resolved text value as a strict integer (structured taxonomy)."""
    try:
        value = int(text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR {field} must be an integer at record index {record_index} "
            f"(run_id={rid}): got {text!r}"
        ) from None
    if minimum is not None and value < minimum:
        raise DetectorStateValidationError(
            f"TR {field} must be >= {minimum} at record index "
            f"{record_index} (run_id={rid}): got {value}"
        )
    if multiple_of is not None and value % multiple_of != 0:
        raise DetectorStateValidationError(
            f"TR {field} must be a multiple of {multiple_of} at record index "
            f"{record_index} (run_id={rid}): got {value}"
        )
    return value


def _strict_float_value(text: str, field: str, *, record_index: int,
                        rid: Any, minimum: float | None = None) -> float:
    """Parse a resolved text value as a strict finite float (structured)."""
    try:
        value = float(text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR {field} must be a finite float at record index "
            f"{record_index} (run_id={rid}): got {text!r}"
        ) from None
    if not math.isfinite(value):
        raise DetectorStateValidationError(
            f"TR {field} must be finite at record index {record_index} "
            f"(run_id={rid}): got {value}"
        )
    if minimum is not None and value <= minimum:
        raise DetectorStateValidationError(
            f"TR {field} must be > {minimum} at record index "
            f"{record_index} (run_id={rid}): got {value}"
        )
    return value


def _optional_uniform_assertion(
    records: list[dict[str, Any]],
    field: str,
) -> tuple[str | None, bool]:
    """Resolve an optional source-assertion field across the cohort, per row.

    Returns ``(value, asserted)``:

    - every record missing/blank → ``(None, False)`` — the runtime-derived
      value must be used and the source assertion marked unavailable.
    - every record present and identical → ``(value, True)``.
    - some records present, some missing → DetectorStateValidationError
      (the cohort must not be labelled asserted from a subset of rows).
    - every record present but differing → DetectorStateValidationError.
    """
    present: dict[int, str] = {}
    absent: list[int] = []
    for idx, record in enumerate(records):
        raw = record.get(field)
        text = str(raw).strip() if raw is not None else ""
        if text:
            present[idx] = text
        else:
            absent.append(idx)

    if not present:
        return None, False

    if absent:
        present_ids = [str(records[i].get("run_id", "?")) for i in present]
        absent_ids = [str(records[i].get("run_id", "?")) for i in absent]
        raise DetectorStateValidationError(
            f"Partial {field} across TR cohort: present at record index(es) "
            f"{sorted(present)} (run_ids={present_ids}), missing at record "
            f"index(es) {absent} (run_ids={absent_ids}).  Optional source "
            f"assertions must be present on every record or on none."
        )

    unique = sorted(set(present.values()))
    if len(unique) != 1:
        raise DetectorStateValidationError(
            f"Mixed {field} across TR cohort: {unique} — present on every "
            f"record but not uniform"
        )
    return unique[0], True


def _resolve_score_mode(records: list[dict[str, Any]]) -> str:
    """Resolve the optional ``tr_score_mode`` protocol selector.

    Absent on every record → the default complex-L1 protocol (documented
    default, never silently invented).  Present → must be a supported mode
    and uniform across the cohort; anything else fails closed.
    """
    present: dict[int, str] = {}
    absent: list[int] = []
    for idx, record in enumerate(records):
        raw = record.get(SCORE_MODE_FIELD)
        text = str(raw).strip() if raw is not None else ""
        if text:
            present[idx] = text
        else:
            absent.append(idx)
    if not present:
        return tr_scoring.SCORE_DEFINITION
    if absent:
        raise DetectorStateValidationError(
            f"Partial {SCORE_MODE_FIELD} across TR cohort: present at record "
            f"index(es) {sorted(present)}, missing at record index(es) "
            f"{absent}.  The score protocol must be uniform across the cohort."
        )
    unique = sorted(set(present.values()))
    if len(unique) != 1:
        raise DetectorStateValidationError(
            f"Mixed {SCORE_MODE_FIELD} across TR cohort: {unique}"
        )
    mode = unique[0]
    if mode not in tr_scoring.SUPPORTED_SCORE_MODES:
        raise DetectorStateValidationError(
            f"TR {SCORE_MODE_FIELD} {mode!r} is not a supported score "
            f"protocol: {sorted(tr_scoring.SUPPORTED_SCORE_MODES)}"
        )
    return mode


def _normalize_tr_provider_config(
    record: dict[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    """Extract and validate TR provider kwargs from one record.

    Missing / empty fields → DetectorMissingStateError.
    Invalid types / values → DetectorStateValidationError.
    Returns a dict suitable for TrProvider(**kwargs).
    """
    rid = record.get("run_id", "?")

    # ---- w_seed: integer ----
    w_seed_text = _require_nonempty(record, "w_seed", record_index)
    try:
        w_seed = int(w_seed_text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR w_seed must be an integer at record index {record_index} "
            f"(run_id={rid}): got {w_seed_text!r}"
        ) from None

    # ---- w_channel: -1 (all channels) or non-negative integer ----
    w_channel_text = _require_nonempty(record, "w_channel", record_index)
    try:
        w_channel = int(w_channel_text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR w_channel must be -1 or a non-negative integer at record "
            f"index {record_index} (run_id={rid}): got {w_channel_text!r}"
        ) from None
    if w_channel < -1:
        raise DetectorStateValidationError(
            f"TR w_channel must be -1 (all channels) or non-negative at "
            f"record index {record_index} (run_id={rid}): got {w_channel}"
        )

    # ---- w_radius: positive integer ----
    w_radius_text = _require_nonempty(record, "w_radius", record_index)
    try:
        w_radius = int(w_radius_text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR w_radius must be a positive integer at record index "
            f"{record_index} (run_id={rid}): got {w_radius_text!r}"
        ) from None
    if w_radius <= 0:
        raise DetectorStateValidationError(
            f"TR w_radius must be positive at record index "
            f"{record_index} (run_id={rid}): got {w_radius}"
        )

    # ---- w_pattern_const: finite float ----
    wpc_text = _require_nonempty(record, "w_pattern_const", record_index)
    try:
        w_pattern_const = float(wpc_text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR w_pattern_const must be a finite float at record index "
            f"{record_index} (run_id={rid}): got {wpc_text!r}"
        ) from None
    if not math.isfinite(w_pattern_const):
        raise DetectorStateValidationError(
            f"TR w_pattern_const must be finite at record index "
            f"{record_index} (run_id={rid}): got {w_pattern_const}"
        )

    # ---- string enum fields ----
    w_pattern = _require_nonempty(record, "w_pattern", record_index)
    if w_pattern not in _ALLOWED_W_PATTERN:
        raise DetectorStateValidationError(
            f"TR w_pattern {w_pattern!r} not in canonical allowed set "
            f"at record index {record_index} (run_id={rid}): "
            f"{sorted(_ALLOWED_W_PATTERN)}"
        )

    w_mask_shape = _require_nonempty(record, "w_mask_shape", record_index)
    if w_mask_shape not in _ALLOWED_W_MASK_SHAPE:
        raise DetectorStateValidationError(
            f"TR w_mask_shape {w_mask_shape!r} not in canonical allowed set "
            f"at record index {record_index} (run_id={rid}): "
            f"{sorted(_ALLOWED_W_MASK_SHAPE)}"
        )

    w_measurement = _require_nonempty(record, "w_measurement", record_index)
    if w_measurement != REQUIRED_W_MEASUREMENT:
        raise DetectorStateValidationError(
            f"TR w_measurement {w_measurement!r} is not supported: the "
            f"canonical TR scoring path only implements "
            f"{REQUIRED_W_MEASUREMENT!r}"
        )

    w_injection = _require_nonempty(record, "w_injection", record_index)
    if w_injection not in _ALLOWED_W_INJECTION:
        raise DetectorStateValidationError(
            f"TR w_injection {w_injection!r} not in canonical allowed set "
            f"at record index {record_index} (run_id={rid}): "
            f"{sorted(_ALLOWED_W_INJECTION)}"
        )

    return {
        "w_seed": w_seed,
        "w_channel": w_channel,
        "w_radius": w_radius,
        "w_pattern": w_pattern,
        "w_mask_shape": w_mask_shape,
        "w_measurement": w_measurement,
        "w_injection": w_injection,
        "w_pattern_const": w_pattern_const,
    }


def _validate_w_channel_range(w_channel: int, latent_shape: tuple[int, ...],
                              record_index: int) -> None:
    """Validate w_channel against the actual latent channel count."""
    channel_count = latent_shape[1]
    if w_channel != -1 and not 0 <= w_channel < channel_count:
        raise DetectorStateValidationError(
            f"TR w_channel {w_channel} is out of range for a "
            f"{channel_count}-channel latent"
        )


def assert_tensor_identity_match(source_sha: str, detector_sha: str,
                                 kind: str) -> None:
    """Fail closed when the source digest and provider-derived digest differ.

    ``kind`` is "target" or "mask"; used by load_state step 9 to bind the
    detector tensors to the generation-time hashes recorded in metadata.
    """
    if source_sha != detector_sha:
        raise DetectorStateValidationError(
            f"TR source watermark_{kind}_sha256 {source_sha!r} does not "
            f"match provider-derived {kind} SHA {detector_sha!r}"
        )


# ---------------------------------------------------------------------------
# load_state — fail-closed, cohort-consistent, canonical-schema bound
# ---------------------------------------------------------------------------
def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load TR provider and pipe.  Raises on missing/bad state, never swallows.

    Binds directly to the canonical generator metadata schema.  Validates
    every record — not just the first — so a mixed-key or mixed-profile
    cohort is rejected before any scoring happens.  A uniform cohort
    constructs exactly one provider built from the verified profile.  Pipe is
    built from cohort metadata, never hard-coded.
    """
    import sys

    import torch

    # ``eval_bench_wm`` is a namespace package (no __init__.py): the repo
    # root must be on sys.path so ``import eval_bench_wm`` resolves, and the
    # eval_bench_wm directory itself so the absolute ``from utils...``
    # imports inside eval_bench_wm resolve.
    repo = Path(__file__).resolve().parents[3]
    for entry in (str(repo), str(repo / "eval_bench_wm")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.tr_provider import TrProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"TR dependencies not available: {exc}"
        ) from exc

    if not records:
        raise DetectorMissingStateError(
            "TR provider requires at least one record with metadata. "
            "All required fields: " + ", ".join(sorted(REQUIRED_METADATA_FIELDS))
        )

    # ---- 0: resolve the optional score protocol selector ----
    score_mode = _resolve_score_mode(records)

    # ---- 1: normalise every record's TR provider config ----
    configs: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        configs.append(_normalize_tr_provider_config(record, record_index=idx))

    # Uniform provider config via canonical hash
    from raven.eval_protocol import provider_config_hash
    provider_hashes: set[str] = set()
    for idx, record in enumerate(records):
        try:
            provider_hashes.add(provider_config_hash("TR", record))
        except (ValueError, TypeError) as exc:
            raise DetectorStateValidationError(
                f"TR provider config hash failed at record index {idx} "
                f"(run_id={record.get('run_id', '?')}): {exc}"
            ) from exc

    if len(provider_hashes) != 1:
        raise DetectorStateValidationError(
            f"Mixed TR provider configurations in cohort: "
            f"{len(provider_hashes)} distinct configs across "
            f"{len(records)} records. All records must share the same "
            f"w_seed, w_channel, w_radius, w_pattern, w_mask_shape, "
            f"w_measurement, w_injection, and w_pattern_const."
        )

    computed_config_hash = next(iter(provider_hashes))
    uniform_cfg = configs[0]

    # ---- 2: resolve source-required identity fields (aliases honoured) ----
    source_identity: dict[str, str] = {}
    for canonical, aliases in SOURCE_REQUIRED_IDENTITY.items():
        source_identity[canonical] = _resolve_uniform_field(
            records, canonical, aliases)

    model_id = source_identity["model_id"]
    model_revision = source_identity["model_revision"]
    scheduler = source_identity["scheduler"]
    rid0 = records[0].get("run_id", "?")
    resolution = _strict_int_value(
        source_identity["resolution"], "resolution", record_index=0,
        rid=rid0, minimum=1, multiple_of=8)
    steps = _strict_int_value(
        source_identity["steps"], "steps", record_index=0,
        rid=rid0, minimum=1)

    # ---- 3: optional source assertions ----
    assertions: dict[str, tuple[str | None, bool]] = {}
    for field in OPTIONAL_ASSERTION_FIELDS:
        assertions[field] = _optional_uniform_assertion(records, field)

    asserted_inverse, inverse_asserted = assertions["inverse_scheduler"]
    asserted_dtype, dtype_asserted = assertions["detector_dtype"]
    asserted_vae_id, vae_id_asserted = assertions["vae_id"]
    asserted_vae_scaling, vae_scaling_asserted = assertions["vae_scaling_factor"]
    asserted_hash, hash_asserted = assertions["provider_config_hash"]

    # vae_scaling_factor strict parse when asserted
    if vae_scaling_asserted:
        assert asserted_vae_scaling is not None
        _strict_float_value(
            asserted_vae_scaling, "vae_scaling_factor", record_index=0,
            rid=rid0, minimum=0.0)

    # ---- 4: provider_config_hash source assertion (canonical semantics) ----
    # Missing or empty recorded hash is legal: the detector still saves the
    # computed hash.  Present-but-mismatched fails closed.
    if hash_asserted:
        assert asserted_hash is not None
        if asserted_hash != computed_config_hash:
            raise DetectorStateValidationError(
                f"Recorded provider_config_hash {asserted_hash!r} does not "
                f"match canonical computed hash {computed_config_hash!r}"
            )

    # ---- 5: scheduler validation against real pipe registry ----
    supported_schedulers = set(pipe_utils.SCHEDULER_CLASSES)
    if scheduler not in supported_schedulers:
        raise DetectorStateValidationError(
            f"TR scheduler {scheduler!r} is not in the pipe registry: "
            f"{sorted(supported_schedulers)}"
        )

    # ---- 6: build pipe from verified profile ----
    try:
        device_obj = torch.device(device)
        load_options = {"revision": model_revision} if model_revision else {}
        if vae_id_asserted:
            load_options["vae_id"] = asserted_vae_id
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name=scheduler,
            disable_tqdm=True,
            **load_options,
        )
    except DetectorMissingStateError:
        raise
    except DetectorStateValidationError:
        raise
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"TR pipe construction failed: {type(exc).__name__}: {exc}"
        ) from exc

    # ---- 7: verify pipe runtime against profile ----
    latent_shape = pipe.get_latent_shape()
    pipe_dtype = str(pipe.get_dtype())
    if dtype_asserted and pipe_dtype != asserted_dtype:
        raise DetectorStateValidationError(
            f"Pipe dtype {pipe_dtype!r} does not match source-asserted "
            f"detector_dtype {asserted_dtype!r}"
        )

    if latent_shape[-1] != resolution // 8:
        raise DetectorStateValidationError(
            f"Pipe latent spatial size {latent_shape[-1]} does not match "
            f"cohort resolution {resolution} (expected {resolution // 8})"
        )

    inverse_scheduler_name = type(pipe.scheduler_inverse).__name__
    if inverse_asserted and inverse_scheduler_name != asserted_inverse:
        raise DetectorStateValidationError(
            f"Pipe inverse scheduler {inverse_scheduler_name!r} does not "
            f"match source-asserted inverse_scheduler {asserted_inverse!r}"
        )

    vae_scaling = float(pipe.pipe.vae.config.scaling_factor)
    if vae_scaling_asserted:
        assert asserted_vae_scaling is not None
        if not math.isclose(vae_scaling, float(asserted_vae_scaling),
                            rel_tol=1e-9):
            raise DetectorStateValidationError(
                f"Pipe VAE scaling factor {vae_scaling} does not match "
                f"source-asserted vae_scaling_factor "
                f"{float(asserted_vae_scaling)}"
            )

    # Runtime VAE identity: checkpoint-default when source did not assert one.
    runtime_vae_id = asserted_vae_id if vae_id_asserted else "checkpoint-default"

    # ---- 8: validate w_channel against latent shape, build provider ----
    _validate_w_channel_range(uniform_cfg["w_channel"], latent_shape, 0)

    try:
        provider = TrProvider(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device_obj,
            **uniform_cfg,
        )
    except DetectorMissingStateError:
        raise
    except DetectorStateValidationError:
        raise
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"TR provider construction failed: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"TR initialization error: {type(exc).__name__}: {exc}"
        ) from exc

    # ---- 9: derive and verify target / mask identity ----
    from raven.pairing_provenance import tensor_sha256

    target = getattr(provider, "gt_patch", None)
    if target is None:
        raise DetectorStateValidationError(
            "TR provider has no gt_patch — cannot derive watermark target identity"
        )
    detector_target_sha = tensor_sha256(target)
    source_target_sha = source_identity["watermark_target_sha256"]

    mask = getattr(provider, "watermarking_mask", None)
    if mask is None:
        raise DetectorStateValidationError(
            "TR provider has no watermarking_mask — cannot derive mask identity"
        )
    detector_mask_sha = tensor_sha256(mask)
    source_mask_sha = source_identity["watermark_mask_sha256"]

    assert_tensor_identity_match(source_target_sha, detector_target_sha, "target")
    assert_tensor_identity_match(source_mask_sha, detector_mask_sha, "mask")

    # ---- 10: assemble verified provenance ----
    verified_profile: dict[str, Any] = {
        "model_id": model_id,
        "model_revision": model_revision,
        "scheduler": scheduler,
        "inverse_scheduler": inverse_scheduler_name,
        "steps": steps,
        "resolution": resolution,
        "detector_dtype": pipe_dtype,
        "vae_id": runtime_vae_id,
        "vae_scaling_factor": vae_scaling,
        "w_pattern_const": uniform_cfg["w_pattern_const"],
    }

    return {
        "provider": provider,
        "pipe": pipe,
        "score_mode": score_mode,
        "provider_kwargs": uniform_cfg,
        "device_obj": device_obj,
        # verified provenance
        "source_provider_config_hash": asserted_hash or "",
        "detector_provider_config_hash": computed_config_hash,
        "tr_provider_config_hash_source_asserted": hash_asserted,
        "tr_provider_config_verified": True,
        "source_watermark_target_sha256": source_target_sha,
        "detector_watermark_target_sha256": detector_target_sha,
        "source_watermark_mask_sha256": source_mask_sha,
        "detector_watermark_mask_sha256": detector_mask_sha,
        "verified_profile": verified_profile,
        "inversion_steps": steps,
        # source-assertion availability flags
        "tr_inverse_scheduler_source_asserted": inverse_asserted,
        "tr_detector_dtype_source_asserted": dtype_asserted,
        "tr_vae_id_source_asserted": vae_id_asserted,
        "tr_vae_scaling_source_asserted": vae_scaling_asserted,
    }


# ---------------------------------------------------------------------------
# score_image — package-local scoring in one scoring boundary
# ---------------------------------------------------------------------------
def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int | None = None) -> dict[str, Any]:
    """Score one image through the package-local TR scoring path.

    The default protocol is complex L1 mean:

        raw_score = torch.abs(decoded_watermark - target_watermark).mean()
        canonical_score = -raw_score

    When the verified cohort profile declares ``tr_score_mode = log10p`` the
    explicitly named p-value protocol runs instead and the score dict carries
    separate ``tr_log_p`` provenance fields.

    The inversion step count comes from the verified cohort profile
    (``provider_info["inversion_steps"]``) — the canonical helper is never
    called with a hard-coded default.  ``steps`` defaults to None so an
    orchestrator that does not pass it cannot override the metadata profile;
    an explicit caller-supplied ``steps`` that differs from the profile is
    rejected with ``DetectorStateValidationError``.
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    pipe = provider_info["pipe"]
    score_mode = provider_info.get("score_mode", tr_scoring.SCORE_DEFINITION)

    effective_steps = int(provider_info["inversion_steps"])
    if steps is not None and int(steps) != effective_steps:
        raise DetectorStateValidationError(
            f"TR caller-supplied steps {steps} conflicts with verified "
            f"cohort profile inversion_steps {effective_steps}"
        )

    # The canonical tr_scoring helpers already read and decode the image
    # internally, so the adapter does NOT duplicate image I/O here.  The
    # entire scoring path runs inside one exception boundary.
    try:
        if score_mode == tr_scoring.SCORE_DEFINITION:
            result = tr_scoring.complex_l1_score(
                torch, provider, pipe, path, effective_steps)
            if result["nan"] or result["inf"]:
                raise DetectorScoringError(
                    f"TR complex-L1 score is non-finite for {image_path}: "
                    f"{result['score']}"
                )
            raw = result["score"]
            canonical = tr_scoring.canonical_score(raw)
            l1_diagnostics = {
                "decoded_abs_mean": result["decoded_abs_mean"],
                "target_abs_mean": result["target_abs_mean"],
            }
        elif score_mode == tr_scoring.LOG10P_MODE:
            result = tr_scoring.evaluate_log10p(
                torch, provider, pipe, path, effective_steps)
            raw = tr_scoring.raw_log10p_score(result)
            canonical = tr_scoring.log10p_canonical(raw)
            l1_diagnostics = None
        else:
            raise DetectorStateValidationError(
                f"TR score_mode {score_mode!r} is not supported: "
                f"{sorted(tr_scoring.SUPPORTED_SCORE_MODES)}"
            )
    except DetectorScoringError:
        raise
    except DetectorStateValidationError:
        raise
    except Exception as exc:
        raise DetectorScoringError(
            f"TR scoring failed for {image_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Validate scores are finite floats
    try:
        raw_f = float(raw)
        canonical_f = float(canonical)
    except (ValueError, TypeError) as exc:
        raise DetectorScoringError(
            f"TR score is not numeric for {image_path}: "
            f"raw={raw!r} canonical={canonical!r}"
        ) from exc
    if not math.isfinite(raw_f):
        raise DetectorScoringError(
            f"TR raw_score is non-finite for {image_path}: {raw_f}"
        )
    if not math.isfinite(canonical_f):
        raise DetectorScoringError(
            f"TR canonical_score is non-finite for {image_path}: {canonical_f}"
        )

    score: dict[str, Any] = {
        "raw_score": raw_f,
        "canonical_score": canonical_f,
        # default score protocol declaration
        "tr_score_protocol": score_mode,
        "tr_score_definition": tr_scoring.SCORE_DEFINITION,
        "tr_raw_score_direction": tr_scoring.RAW_SCORE_DIRECTION,
        "tr_canonical_score_direction": tr_scoring.CANONICAL_SCORE_DIRECTION,
        "tr_comparison_operator": tr_scoring.COMPARISON_OPERATOR,
    }

    if score_mode == tr_scoring.SCORE_DEFINITION:
        assert l1_diagnostics is not None
        score["tr_decoded_abs_mean"] = l1_diagnostics["decoded_abs_mean"]
        score["tr_target_abs_mean"] = l1_diagnostics["target_abs_mean"]
    else:
        diagnostics = result.get("p_value_diagnostics") or []
        if not diagnostics:
            raise DetectorScoringError(
                f"TR p-value mode produced no p_value_diagnostics for "
                f"{image_path}"
            )
        d = diagnostics[0]
        score["tr_log_p"] = d.get("log_p")
        score["tr_sigma"] = d.get("sigma")
        score["tr_lambda"] = d.get("lambda")
        score["tr_statistic"] = d.get("statistic")
        score["tr_df"] = d.get("df")
        score["tr_p_underflow"] = d.get("p_underflow", False)

    # ---- attach verified provenance to score ----
    verified_profile = provider_info.get("verified_profile", {})
    score["tr_provider_config_hash"] = provider_info.get(
        "detector_provider_config_hash", "")
    score["tr_provider_config_verified"] = True
    score["tr_provider_config_hash_source_asserted"] = bool(
        provider_info.get("tr_provider_config_hash_source_asserted", False))
    score["tr_source_watermark_target_sha256"] = provider_info.get(
        "source_watermark_target_sha256", "")
    score["tr_detector_watermark_target_sha256"] = provider_info.get(
        "detector_watermark_target_sha256", "")
    score["tr_target_verified"] = True
    score["tr_source_watermark_mask_sha256"] = provider_info.get(
        "source_watermark_mask_sha256", "")
    score["tr_detector_watermark_mask_sha256"] = provider_info.get(
        "detector_watermark_mask_sha256", "")
    score["tr_mask_verified"] = True
    score["tr_model_id"] = verified_profile.get("model_id", "")
    score["tr_model_revision"] = verified_profile.get("model_revision", "")
    score["tr_scheduler"] = verified_profile.get("scheduler", "")
    score["tr_inverse_scheduler"] = verified_profile.get("inverse_scheduler", "")
    score["tr_inverse_scheduler_source_asserted"] = bool(
        provider_info.get("tr_inverse_scheduler_source_asserted", False))
    score["tr_steps"] = verified_profile.get("steps", "")
    score["tr_resolution"] = verified_profile.get("resolution", "")
    score["tr_detector_dtype"] = verified_profile.get("detector_dtype", "")
    score["tr_detector_dtype_source_asserted"] = bool(
        provider_info.get("tr_detector_dtype_source_asserted", False))
    score["tr_vae_id"] = verified_profile.get("vae_id", "")
    score["tr_vae_id_source_asserted"] = bool(
        provider_info.get("tr_vae_id_source_asserted", False))
    score["tr_vae_scaling_factor"] = verified_profile.get("vae_scaling_factor", "")
    score["tr_vae_scaling_source_asserted"] = bool(
        provider_info.get("tr_vae_scaling_source_asserted", False))
    score["tr_w_pattern_const"] = verified_profile.get("w_pattern_const", "")

    return score


# ---------------------------------------------------------------------------
# aggregate — cohort-aware, threshold-based
# ---------------------------------------------------------------------------
def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate TR detector rows across cohorts.

    Required cohorts for primary threshold report:
      ``original_clean``, ``original_watermarked``, ``attacked_watermarked``.

    ``attacked_clean`` controls the independent ``tr_recalibrated`` block
    and does NOT block the original-threshold report when absent.

    A recalibration metric-computation error is never disguised as data
    unavailability: it re-raises as a structured scoring failure.
    """
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    cohorts: dict[str, list[float]] = {}
    protocols: set[str] = set()
    for row in detector_rows:
        if row.get("status") != ROW_STATUS_SCORED:
            continue
        cohort = row.get("evaluation_cohort", "")
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))
        if row.get("tr_score_protocol"):
            protocols.add(str(row["tr_score_protocol"]))

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored

    result: dict[str, Any] = {
        "method": "TR",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": [],
        # default protocol declaration for the aggregate
        "score_definition": tr_scoring.SCORE_DEFINITION,
        "raw_score_direction": tr_scoring.RAW_SCORE_DIRECTION,
        "canonical_score_direction": tr_scoring.CANONICAL_SCORE_DIRECTION,
        "comparison_operator": tr_scoring.COMPARISON_OPERATOR,
    }
    if protocols:
        if len(protocols) != 1:
            raise DetectorScoringError(
                f"TR rows declare mixed score protocols: {sorted(protocols)}"
            )
        result["tr_score_protocol"] = next(iter(protocols))

    # Primary threshold report requires all three cohorts
    primary_required = {"original_clean", "original_watermarked",
                        "attacked_watermarked"}
    missing_primary = sorted(primary_required - set(cohorts))
    result["missing_cohorts"] = missing_primary

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        try:
            summary = summarize_detection(clean, watermarked, attacked,
                                          target_fpr=0.01)
        except Exception as exc:
            raise DetectorScoringError(
                f"TR primary detection metric computation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_comparison_operator": ">=",
            "original_clean_threshold": summary.calibration.threshold,
            "original_clean_target_fpr": summary.calibration.target_fpr,
            "original_clean_actual_fpr": summary.calibration.actual_fpr,
            "original_clean_false_positives": summary.calibration.false_positives,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr_at_original_threshold": summary.attacked_tpr,
            "watermarked_roc_auc": summary.watermarked_auc,
            "attacked_roc_auc": summary.attacked_auc,
            "attack_success_at_original_threshold": 1.0 - summary.attacked_tpr,
        }

    # Recalibrated: independent block gated on attacked_clean AND the
    # positive cohorts it calibrates against.  When any of those cohorts has
    # no scored rows the metrics genuinely cannot be computed — that is plain
    # unavailability, not an error.  When all inputs are present, a
    # summarize_detection failure is a real computation bug and must
    # propagate as a structured failure rather than being disguised as
    # data unavailability.
    attacked_clean = cohorts.get("attacked_clean", [])
    if attacked_clean and clean and watermarked and attacked:
        try:
            recal = summarize_detection(
                attacked_clean, watermarked, attacked, target_fpr=0.01)
        except Exception as exc:
            raise DetectorScoringError(
                f"TR recalibrated metric computation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        result["tr_recalibrated"] = {
            "recalibrated_metrics_available": True,
            "attacked_clean_count": len(attacked_clean),
            "attacked_clean_recalibrated_threshold": recal.calibration.threshold,
            "attacked_clean_target_fpr": recal.calibration.target_fpr,
            "attacked_clean_actual_fpr": recal.calibration.actual_fpr,
            "attacked_clean_false_positives": recal.calibration.false_positives,
            "attacked_watermarked_tpr_at_recalibrated_threshold": recal.attacked_tpr,
            "attack_success_at_recalibrated_threshold": 1.0 - recal.attacked_tpr,
            "recalibrated_roc_auc": recal.attacked_auc,
        }
    else:
        result["tr_recalibrated"] = {
            "recalibrated_metrics_available": False,
        }

    return result
