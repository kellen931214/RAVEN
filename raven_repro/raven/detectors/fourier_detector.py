"""RID / HSTR / HSQR (Fourier) detector adapter.

Delegates to the canonical bundle loading and scoring in
``extract_verification_scores.py``.  Uses method-specific provider kwargs
helpers (rid_provider_kwargs_from_bundle, hstr_provider_kwargs_from_bundle,
hsqr_provider_from_bundle).

Canonical score = ``-raw_l1`` (higher is watermarked).
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

FOURIER_METHODS = frozenset({"RID", "HSTR", "HSQR"})

# Per-method required metadata prefix
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset()


def describe_required_artifacts() -> list[str]:
    return [
        "<prefix>_bundle_dir (directory with manifest.json)",
        "<prefix>_bundle_config_sha256, <prefix>_selected_pattern_sha256",
        "<prefix>_mask_sha256, <prefix>_key_index",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_extract_module():
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores_fourier",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_state(records: list[dict[str, Any]], device: str,
               method: str = "RID", **extra) -> dict[str, Any]:
    """Load Fourier provider via canonical bundle helpers from extract script."""
    import torch

    _ensure_paths()
    method = method.upper()
    if method not in FOURIER_METHODS:
        raise ValueError(f"Unknown Fourier method: {method}")
    prefix = method.lower()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
    except ImportError as exc:
        raise DetectorDependencyError(
            f"Fourier dependencies not available: {exc}"
        ) from exc

    first = records[0] if records else {}
    bundle_dir = first.get(f"{prefix}_bundle_dir", "")
    if not bundle_dir or not Path(bundle_dir).is_dir():
        raise DetectorMissingStateError(
            f"{method}: {prefix}_bundle_dir not found or not a directory"
        )

    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    identifier = str(first.get("run_id", "0"))

    # Use canonical bundle manifest and provider kwargs
    try:
        bundle_dir_path, manifest = mod.fourier_bundle_manifest(
            first, str(identifier), method)
    except Exception as exc:
        raise DetectorStateValidationError(
            f"{method} bundle validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        if method == "RID":
            kwargs = mod.rid_provider_kwargs_from_bundle(first, str(identifier))
        elif method == "HSTR":
            kwargs = mod.hstr_provider_kwargs_from_bundle(first, str(identifier))
        else:
            kwargs = {}

        device_obj = torch.device(device)
        model_id = kwargs.get("modelid_target",
                               "RedbeardNZ/stable-diffusion-2-1-base")
        resolution = kwargs.get("resolution", 512)

        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name="DDIM",
            disable_tqdm=True,
        )
        latent_shape = pipe.get_latent_shape()

        if method == "RID":
            from eval_bench_wm.utils.wm.ringid_provider import RingIDProvider
            provider = RingIDProvider(
                latent_shape=latent_shape,
                dtype=pipe.get_dtype(),
                device=device_obj,
                **kwargs,
            )
        elif method == "HSTR":
            from eval_bench_wm.utils.wm.hstr_provider import HSTRProvider
            provider = HSTRProvider(
                latent_shape=latent_shape,
                dtype=pipe.get_dtype(),
                device=device_obj,
                **kwargs,
            )
        elif method == "HSQR":
            from eval_bench_wm.utils.wm.hsqr_provider import HSQRProvider
            provider = mod.hsqr_provider_from_bundle(
                first, str(identifier), latent_shape, device_obj)
        else:
            raise DetectorProviderInitializationError(
                f"Unknown Fourier method: {method}"
            )

        # Validate state_source
        if getattr(provider, "bundle", None) is None:
            raise DetectorStateValidationError(
                f"{method} provider has no persisted bundle"
            )
        if getattr(provider, "state_source", "") != "bundle":
            raise DetectorStateValidationError(
                f"{method}: state_source is not 'bundle': "
                f"{getattr(provider, 'state_source', 'unknown')}"
            )
    except (DetectorStateValidationError, DetectorMissingStateError):
        raise
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"{method} provider construction failed: {exc}"
        ) from exc

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "method": method,
        "score_definition": f"{prefix}_score = -raw_l1",
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using the Fourier provider via canonical evaluate_image.

    Canonical score = ``-raw_l1`` (delegates to extract module's canonical_score).
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    method = provider_info["method"]
    mod = provider_info["extract_module"]

    try:
        result = mod.evaluate_image(
            torch, provider, provider_info["pipe"], path, steps)
    except Exception as exc:
        raise DetectorScoringError(
            f"{method} scoring failed for {image_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raw = mod.raw_score(method, result)
    canonical = mod.canonical_score(method, raw, result)

    return {
        "raw_score": raw,
        "canonical_score": canonical,
        "raw_l1": raw,
        "score_direction": "higher_is_watermarked (canonical = -raw_l1)",
    }


def aggregate(detector_rows: list[dict[str, Any]],
              method: str = "RID", **extra) -> dict[str, Any]:
    """Aggregate Fourier detector rows."""
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    method = method.upper()
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
        "method": method,
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_type": f"{method.lower()}_score",
        "score_direction": "higher_is_watermarked",
        "score_definition": "canonical_score = -raw_l1",
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_type": "empirical_clean_1pct_fpr",
            "threshold_score_space": "canonical_score",
            "threshold_comparison_operator": ">=",
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
