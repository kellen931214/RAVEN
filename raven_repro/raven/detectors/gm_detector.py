"""GaussMarker detector adapter.

Bind GM evaluation to the canonical persisted bundle and provenance.
Delegates to the formal extraction path in ``extract_verification_scores.py``
(``gm_bundle_manifest``, ``gm_provider_kwargs``, ``evaluate_image``,
``raw_score``, ``canonical_score``).

Before constructing the provider every row is validated to share the same
bundle directory, bundle-config SHA, w1/w2 SHA, protocol/profile and
provider configuration.  Mixed bundles are rejected before scoring.
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
    DetectorStateValidationError,
    DetectorScoringError,
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_protocol_mode",
})

# ---------------------------------------------------------------------------
# Cohort-uniform bundle identity fields – every row must agree on these.
# ---------------------------------------------------------------------------
_COHORT_UNIFORM_FIELDS: tuple[str, ...] = (
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_protocol_mode",
)

# Per-row bundle-artifact identity carried through to the score record.
_VERIFIED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_protocol_mode",
)


def describe_required_artifacts() -> list[str]:
    return [
        "gm_bundle_dir (directory with manifest.json, w1.pth, w2.pth)",
        "gm_bundle_config_sha256, gm_w1_file_sha256, gm_w2_file_sha256",
        "gm_watermark_sha256, gm_m_sha256, gm_target_sha256, gm_mask_sha256",
        "gm_protocol_mode",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_extract_module():
    """Import ``extract_verification_scores.py`` as a module (canonical helpers)."""
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores_gm",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Cross-row validation helpers
# ---------------------------------------------------------------------------

def _validate_cohort_uniform(records: list[dict[str, Any]]) -> dict[str, str]:
    """Every row must agree on bundle identity fields; return the single value set.

    Raises ``DetectorStateValidationError`` when rows disagree, before any
    provider is constructed.
    """
    values: dict[str, set[str]] = {field: set() for field in _COHORT_UNIFORM_FIELDS}
    for row in records:
        for field in _COHORT_UNIFORM_FIELDS:
            values[field].add(str(row.get(field, "")))

    mismatched = sorted(
        field for field, vset in values.items() if len(vset) != 1
    )
    if mismatched:
        details = "; ".join(
            f"{f}={sorted(values[f])}" for f in mismatched
        )
        raise DetectorStateValidationError(
            f"GM cohort is not uniform — mixed {details}"
        )
    return {field: next(iter(vset)) for field, vset in values.items()}


def _validate_bundle_files_exist(bundle_dir: str) -> Path:
    """Check the three bundle artifacts exist on disk.

    Returns the resolved ``Path``.  Missing files → ``DetectorMissingStateError``.
    """
    resolved = Path(bundle_dir).resolve()
    if not resolved.is_dir():
        raise DetectorMissingStateError(
            f"GM bundle directory not found: {resolved}"
        )
    for name in ("manifest.json", "w1.pth", "w2.pth"):
        if not (resolved / name).is_file():
            raise DetectorMissingStateError(
                f"GM bundle artifact missing: {resolved / name}"
            )
    return resolved


def _classify_bundle_error(exc: Exception) -> type:
    """Distinguish missing-artifact errors from provenance mismatches.

    ``gm_bundle_manifest`` raises ``RuntimeError`` for both cases;
    inspect the message to route to the correct detector exception.
    """
    msg = str(exc)
    if "not found" in msg or "missing" in msg.lower():
        return DetectorMissingStateError
    return DetectorStateValidationError


# ---------------------------------------------------------------------------
# Public detector contract
# ---------------------------------------------------------------------------

def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GM provider via canonical ``gm_provider_kwargs`` from extract script.

    Validates cohort uniformity and bundle integrity before constructing
    the provider.  Returns a ``provider_info`` dict for ``score_image``.
    """
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gm_provider import GmProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"GM dependencies not available: {exc}"
        ) from exc

    if not records:
        raise DetectorMissingStateError("GM detector requires at least one record")

    # 1. Validate cohort uniformity — mixed bundles/configs fail before scoring.
    uniform = _validate_cohort_uniform(records)

    # 2. Bundle files must exist on disk (missing → failed_missing_required_state).
    bundle_dir = _validate_bundle_files_exist(uniform["gm_bundle_dir"])

    # 3. Load the canonical extraction module.
    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    first = records[0]
    identifier = str(first.get("run_id", "0"))

    # 4. Bind the bundle to the row's recorded digests via gm_bundle_manifest.
    #    SHA/config mismatches → DetectorStateValidationError.
    #    Missing artifacts inside the bundle dir → DetectorMissingStateError.
    try:
        bundle_dir_path, manifest = mod.gm_bundle_manifest(
            first, str(identifier))
        kwargs = mod.gm_provider_kwargs(first, str(identifier))
    except (DetectorMissingStateError, DetectorStateValidationError):
        raise
    except RuntimeError as exc:
        error_cls = _classify_bundle_error(exc)
        raise error_cls(
            f"GM bundle validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorStateValidationError(
            f"GM bundle validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    # 5. Construct the provider via the same canonical path as the extract script.
    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=kwargs.get("modelid_target",
                "RedbeardNZ/stable-diffusion-2-1-base"),
            resolution=kwargs.get("resolution", 512),
            device=device_obj,
            eager_loading=False,
            schedulers_name="DDIM",
            disable_tqdm=True,
        )
        latent_shape = pipe.get_latent_shape()

        provider = GmProvider(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device_obj,
            **kwargs,
        )
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"GM provider construction failed: {exc}"
        ) from exc

    # 6. Require persisted bundle as the state source.
    if provider.bundle is None or getattr(provider, "state_source", "") != "bundle":
        raise DetectorStateValidationError(
            "GM provider requires persisted bundle; "
            f"state_source={getattr(provider, 'state_source', 'unknown')}"
        )

    # 7. Derive the provider-side target and mask hashes for cross-validation.
    from raven.pairing_provenance import tensor_sha256

    provider_target_hash = tensor_sha256(
        provider.gt_patch.real.contiguous()
    ) if provider.gt_patch is not None else ""

    provider_mask_hash = tensor_sha256(
        provider.watermarking_mask.contiguous()
    ) if getattr(provider, "watermarking_mask", None) is not None else ""

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "provider_target_hash": provider_target_hash,
        "provider_mask_hash": provider_mask_hash,
        "bundle_dir": str(bundle_dir_path),
        # Verified provenance — every row already agrees on these.
        "verified_provenance": {
            field: uniform[field] for field in _VERIFIED_PROVENANCE_FIELDS
        },
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using GaussMarker provider via canonical ``evaluate_image``.

    Validates that the source target/mask recorded in the row match the
    provider-derived values.  Only verified provenance is saved into the
    score record — nothing is copied blindly from the input row.
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    mod = provider_info["extract_module"]

    # Validate source target/mask against provider-derived values.
    if record is not None:
        source_target = str(record.get("watermark_target_sha256", ""))
        detector_target = provider_info["provider_target_hash"]
        if source_target and detector_target and source_target != detector_target:
            raise DetectorStateValidationError(
                f"run_id={record.get('run_id')}: GM detector/source "
                f"target SHA mismatch: source={source_target!r} "
                f"detector={detector_target!r}"
            )

        source_mask = str(record.get("watermark_mask_sha256", ""))
        detector_mask = provider_info["provider_mask_hash"]
        if source_mask and detector_mask and source_mask != detector_mask:
            raise DetectorStateValidationError(
                f"run_id={record.get('run_id')}: GM detector/source "
                f"mask SHA mismatch: source={source_mask!r} "
                f"detector={detector_mask!r}"
            )

    # Score via the canonical extraction path.
    try:
        result = mod.evaluate_image(torch, provider, provider_info["pipe"],
                                     path, steps)
    except Exception as exc:
        raise DetectorScoringError(
            f"GM scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    raw = mod.raw_score("GM", result)
    canonical = mod.canonical_score("GM", raw, result)

    # Build score record — only verified provenance, no blind copies.
    score: dict[str, Any] = {
        "raw_score": raw,
        "canonical_score": canonical,
        # GM domain scores.
        "gm_raw_bit_accuracy": float(result.get("gm_raw_bit_accuracy", raw)),
        "gm_raw_ring_l1": float(result.get("gm_raw_ring_l1", 0)),
        "gm_restored_bit_accuracy": result.get("gm_restored_bit_accuracy"),
        "gm_classifier_probability": result.get("gm_classifier_probability"),
        # Detector self-description (from the scorer, not the row).
        "gm_report_label": str(result.get("gm_report_label", "")),
        "gm_score_definition": str(result.get("gm_score_definition", "")),
        "gm_threshold_source": str(result.get("gm_threshold_source", "")),
        "gm_comparison_operator": str(result.get("gm_comparison_operator", "")),
    }

    # Attach verified provenance from the provider (cohort-uniform, validated).
    score.update(provider_info.get("verified_provenance", {}))

    # Attach provider-side target/mask hashes (verified in this call).
    score["watermark_target_sha256"] = provider_info.get("provider_target_hash", "")
    score["watermark_mask_sha256"] = provider_info.get("provider_mask_hash", "")

    return score


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate GM detector rows."""
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
    required = {"original_watermarked", "attacked_watermarked"}
    missing = sorted(required - set(cohorts))

    result: dict[str, Any] = {
        "method": "GM",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_type": "gm_raw_bit_accuracy",
        "score_direction": "higher_is_watermarked",
        "official_ensemble_threshold_available": False,
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_type": "empirical_clean_1pct_fpr",
            "threshold_comparison_operator": ">=",
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
