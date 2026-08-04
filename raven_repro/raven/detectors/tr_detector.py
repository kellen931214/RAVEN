"""Tree-Ring detector adapter.

Delegates to the canonical TR scoring in ``extract_verification_scores.py``.
Does NOT reimplement FFT / non-central chi-square / -log10(p) math.

All TR provider parameters MUST come from metadata.  Silent fallback to
defaults is forbidden — missing fields cause ``DetectorMissingStateError``,
invalid values cause ``DetectorStateValidationError``.  Mixed provider
configurations across records are rejected before scoring.

A uniform cohort constructs exactly one provider from the verified profile.
Target and mask identities are validated against provider-derived hashes.
Pipe is built from the verified cohort profile, never hard-coded.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorScoringError,
    DetectorStateValidationError,
)

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

# TR profile identity fields.  Every row MUST have a non-empty value for each
# field, and all rows MUST agree.  Missing fields → DetectorMissingStateError.
# Mixed values → DetectorStateValidationError.
TR_PROFILE_IDENTITY_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "scheduler",
    "inverse_scheduler",
    "steps",
    "resolution",
    "detector_dtype",
    "vae_id",
    "vae_scaling_factor",
    "provider_config_hash",
    "watermark_target_sha256",
    "watermark_mask_sha256",
)

_extract_module = None


def _get_extract_module():
    global _extract_module
    if _extract_module is not None:
        return _extract_module
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _extract_module = mod
    return mod


def describe_required_artifacts() -> list[str]:
    return [
        "TR provider parameters in source metadata: "
        "w_seed, w_channel, w_radius, w_pattern, w_mask_shape, "
        "w_measurement, w_injection, w_pattern_const",
        "Stable Diffusion pipe provider (pipe_utils.get_pipe_provider)",
        "TR profile identity: model_id, model_revision, scheduler, "
        "inverse_scheduler, steps, resolution, detector_dtype, vae_id, "
        "vae_scaling_factor, provider_config_hash, "
        "watermark_target_sha256, watermark_mask_sha256",
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


def _require_uniform_field(records: list[dict[str, Any]],
                           field: str) -> str:
    """Every record must have a non-empty value for *field* and all must agree.

    Returns the single canonical value.  Missing → DetectorMissingStateError,
    mixed → DetectorStateValidationError.
    """
    values: list[str] = []
    for idx, record in enumerate(records):
        values.append(_require_nonempty(record, field, idx))

    unique = sorted(set(values))
    if len(unique) != 1:
        raise DetectorStateValidationError(
            f"Mixed {field} across TR cohort ({len(unique)} distinct values "
            f"across {len(records)} records): {unique}. "
            f"All records must agree on detector profile identity fields."
        )
    return unique[0]


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

    # ---- w_channel: non-negative integer ----
    w_channel_text = _require_nonempty(record, "w_channel", record_index)
    try:
        w_channel = int(w_channel_text)
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"TR w_channel must be a non-negative integer at record index "
            f"{record_index} (run_id={rid}): got {w_channel_text!r}"
        ) from None
    if w_channel < 0:
        raise DetectorStateValidationError(
            f"TR w_channel must be non-negative at record index "
            f"{record_index} (run_id={rid}): got {w_channel}"
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


# ---------------------------------------------------------------------------
# load_state — fail-closed, cohort-consistent
# ---------------------------------------------------------------------------
def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load TR provider and pipe.  Raises on missing/bad state, never swallows.

    Validates every record — not just the first — so a mixed-key or
    mixed-profile cohort is rejected before any scoring happens.  A uniform
    cohort constructs exactly one provider built from the verified profile.
    Pipe is built from cohort metadata, never hard-coded.
    """
    import torch

    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

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

    # ---- 1: normalise every record's TR provider config ----
    configs: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        configs.append(_normalize_tr_provider_config(record, record_index=idx))

    # Check uniform provider config via canonical hash
    from raven.eval_protocol import provider_config_hash
    provider_hashes: set[str] = set()
    for idx, record in enumerate(records):
        try:
            h = provider_config_hash("TR", record)
            provider_hashes.add(h)
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

    # ---- 2: validate uniform profile identity fields ----
    profile: dict[str, str] = {}
    for field in TR_PROFILE_IDENTITY_FIELDS:
        profile[field] = _require_uniform_field(records, field)

    # ---- 3: validate recorded provider_config_hash ----
    recorded_hash = profile["provider_config_hash"]
    if recorded_hash != computed_config_hash:
        raise DetectorStateValidationError(
            f"Recorded provider_config_hash {recorded_hash!r} does not "
            f"match canonical computed hash {computed_config_hash!r}"
        )

    # ---- 4: build pipe from verified profile ----
    model_id = profile["model_id"]
    model_revision = profile["model_revision"]
    scheduler = profile["scheduler"]
    resolution = int(profile["resolution"])

    # Validate formal scheduler values
    _ALLOWED_SCHEDULERS: frozenset[str] = frozenset({"DDIM", "DDPM"})
    if scheduler not in _ALLOWED_SCHEDULERS:
        raise DetectorStateValidationError(
            f"TR scheduler {scheduler!r} not in canonical allowed set: "
            f"{sorted(_ALLOWED_SCHEDULERS)}"
        )

    try:
        device_obj = torch.device(device)
        load_options = {"revision": model_revision} if model_revision else {}
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

    # ---- 5: verify pipe runtime matches profile ----
    latent_shape = pipe.get_latent_shape()
    pipe_dtype = str(pipe.get_dtype())
    expected_dtype = profile["detector_dtype"]
    if pipe_dtype != expected_dtype:
        raise DetectorStateValidationError(
            f"Pipe dtype {pipe_dtype!r} does not match cohort profile "
            f"detector_dtype {expected_dtype!r}"
        )

    expected_steps = int(profile["steps"])
    expected_resolution = int(profile["resolution"])
    if latent_shape[-1] != expected_resolution // 8:
        raise DetectorStateValidationError(
            f"Pipe latent spatial size {latent_shape[-1]} does not match "
            f"cohort resolution {expected_resolution} (expected "
            f"{expected_resolution // 8})"
        )

    inverse_scheduler_name = type(pipe.scheduler_inverse).__name__
    expected_inverse = profile["inverse_scheduler"]
    if inverse_scheduler_name != expected_inverse:
        raise DetectorStateValidationError(
            f"Pipe inverse scheduler {inverse_scheduler_name!r} does not "
            f"match cohort profile inverse_scheduler {expected_inverse!r}"
        )

    vae_scaling = float(pipe.pipe.vae.config.scaling_factor)
    expected_vae = float(profile["vae_scaling_factor"])
    if not math.isclose(vae_scaling, expected_vae, rel_tol=1e-9):
        raise DetectorStateValidationError(
            f"Pipe VAE scaling factor {vae_scaling} does not match cohort "
            f"vae_scaling_factor {expected_vae}"
        )

    # ---- 6: build provider from uniform config ----
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

    # ---- 7: derive and verify target / mask identity ----
    from raven.pairing_provenance import tensor_sha256

    # Target: gt_patch (complex tensor)
    target = getattr(provider, "gt_patch", None)
    if target is None:
        raise DetectorStateValidationError(
            "TR provider has no gt_patch — cannot derive watermark target identity"
        )
    detector_target_sha = tensor_sha256(target)
    source_target_sha = profile["watermark_target_sha256"]

    # Mask: watermarking_mask
    mask = getattr(provider, "watermarking_mask", None)
    if mask is None:
        raise DetectorStateValidationError(
            "TR provider has no watermarking_mask — cannot derive mask identity"
        )
    detector_mask_sha = tensor_sha256(mask)
    source_mask_sha = profile["watermark_mask_sha256"]

    if source_target_sha != detector_target_sha:
        raise DetectorStateValidationError(
            f"TR source watermark_target_sha256 {source_target_sha!r} does not "
            f"match provider-derived target SHA {detector_target_sha!r}"
        )
    if source_mask_sha != detector_mask_sha:
        raise DetectorStateValidationError(
            f"TR source watermark_mask_sha256 {source_mask_sha!r} does not "
            f"match provider-derived mask SHA {detector_mask_sha!r}"
        )

    # ---- 8: assemble verified provenance ----
    verified_profile = {
        "model_id": model_id,
        "model_revision": model_revision,
        "scheduler": scheduler,
        "inverse_scheduler": inverse_scheduler_name,
        "steps": expected_steps,
        "resolution": resolution,
        "detector_dtype": pipe_dtype,
        "vae_id": profile["vae_id"],
        "vae_scaling_factor": vae_scaling,
    }

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "provider_kwargs": uniform_cfg,
        "device_obj": device_obj,
        # verified provenance
        "source_provider_config_hash": recorded_hash,
        "detector_provider_config_hash": computed_config_hash,
        "source_watermark_target_sha256": source_target_sha,
        "detector_watermark_target_sha256": detector_target_sha,
        "source_watermark_mask_sha256": source_mask_sha,
        "detector_watermark_mask_sha256": detector_mask_sha,
        "verified_profile": verified_profile,
    }


# ---------------------------------------------------------------------------
# score_image — canonical delegation, wrapped in one scoring boundary
# ---------------------------------------------------------------------------
def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using the canonical TR detection path.

    Delegates to ``extract_verification_scores.evaluate_image``,
    ``raw_score``, and ``canonical_score`` inside a single try/except so
    canonical-helper failures are ``DetectorScoringError``, never
    ``failed_internal_error``.
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    pipe = provider_info["pipe"]
    mod = provider_info["extract_module"]

    # The canonical evaluate_image helper already reads and decodes the image
    # internally, so the adapter does NOT duplicate image I/O here.  The entire
    # scoring path (evaluate → raw → canonical → diagnostics) runs inside one
    # exception boundary.
    try:
        result = mod.evaluate_image(torch, provider, pipe, path, steps)
        raw = mod.raw_score("TR", result)
        canonical = mod.canonical_score("TR", raw, result)
    except DetectorScoringError:
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
    }
    diagnostics = result.get("p_value_diagnostics") or []
    if diagnostics:
        d = diagnostics[0]
        score["tr_log_p"] = d.get("log_p")
        score["tr_sigma"] = d.get("sigma")
        score["tr_lambda"] = d.get("lambda")
        score["tr_statistic"] = d.get("statistic")
        score["tr_df"] = d.get("df")
        score["tr_p_underflow"] = d.get("p_underflow", False)
    else:
        raise DetectorScoringError(
            f"TR scoring produced no p_value_diagnostics for {image_path}"
        )

    # ---- attach verified provenance to score ----
    verified_profile = provider_info.get("verified_profile", {})
    score["tr_provider_config_hash"] = provider_info.get(
        "detector_provider_config_hash", "")
    score["tr_provider_config_verified"] = True
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
    score["tr_steps"] = verified_profile.get("steps", "")
    score["tr_resolution"] = verified_profile.get("resolution", "")
    score["tr_detector_dtype"] = verified_profile.get("detector_dtype", "")

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
    """
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != ROW_STATUS_SCORED:
            continue
        cohort = row.get("evaluation_cohort", "")
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored

    result: dict[str, Any] = {
        "method": "TR",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": [],
    }

    # Primary threshold report requires all three cohorts
    primary_required = {"original_clean", "original_watermarked",
                        "attacked_watermarked"}
    missing_primary = sorted(primary_required - set(cohorts))
    result["missing_cohorts"] = missing_primary

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked,
                                      target_fpr=0.01)
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

    # Recalibrated: independent block gated on attacked_clean
    attacked_clean = cohorts.get("attacked_clean", [])
    if attacked_clean and clean:
        try:
            recal = summarize_detection(
                attacked_clean, watermarked, attacked, target_fpr=0.01)
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
        except Exception:
            result["tr_recalibrated"] = {
                "recalibrated_metrics_available": False,
                "recalibrated_error": (
                    "metric computation failed unexpectedly — "
                    "this is a bug, not a data-availability problem"
                ),
            }
    else:
        result["tr_recalibrated"] = {
            "recalibrated_metrics_available": False,
        }

    return result
