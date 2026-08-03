"""GaussMarker detector adapter.

Delegates to the canonical GM provider construction in
``extract_verification_scores.py`` (gm_bundle_manifest, gm_provider_kwargs,
evaluate_image, raw_score, canonical_score).
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


def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GM provider via canonical gm_provider_kwargs from extract script."""
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gm_provider import GmProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"GM dependencies not available: {exc}"
        ) from exc

    first = records[0] if records else {}
    bundle_dir = first.get("gm_bundle_dir", "")
    if not bundle_dir or not Path(bundle_dir).is_dir():
        raise DetectorMissingStateError(
            "gm_bundle_dir not found or not a directory"
        )

    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    identifier = str(first.get("run_id", "0"))

    # Use canonical gm_bundle_manifest and gm_provider_kwargs
    try:
        bundle_dir_path, manifest = mod.gm_bundle_manifest(
            first, str(identifier))
        kwargs = mod.gm_provider_kwargs(first, str(identifier))
    except Exception as exc:
        raise DetectorStateValidationError(
            f"GM bundle validation failed: {type(exc).__name__}: {exc}"
        ) from exc

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

    if provider.bundle is None or getattr(provider, "state_source", "") != "bundle":
        raise DetectorStateValidationError(
            "GM provider requires persisted bundle; "
            f"state_source={getattr(provider, 'state_source', 'unknown')}"
        )

    from raven.pairing_provenance import tensor_sha256
    target_hash = tensor_sha256(
        provider.gt_patch.real.contiguous()
    ) if provider.gt_patch is not None else ""

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "target_hash": target_hash,
        "bundle_dir": str(bundle_dir_path),
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using GaussMarker provider via canonical evaluate_image."""
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    mod = provider_info["extract_module"]

    try:
        result = mod.evaluate_image(torch, provider, provider_info["pipe"],
                                     path, steps)
    except Exception as exc:
        raise DetectorScoringError(
            f"GM scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    raw = mod.raw_score("GM", result)
    canonical = mod.canonical_score("GM", raw, result)

    score: dict[str, Any] = {
        "raw_score": raw,
        "canonical_score": canonical,
        "gm_raw_bit_accuracy": float(result.get("gm_raw_bit_accuracy", raw)),
        "gm_raw_ring_l1": float(result.get("gm_raw_ring_l1", 0)),
        "gm_restored_bit_accuracy": result.get("gm_restored_bit_accuracy"),
        "gm_classifier_probability": result.get("gm_classifier_probability"),
        "gm_report_label": str(result.get("gm_report_label", "")),
        "gm_score_definition": str(result.get("gm_score_definition", "")),
        "gm_threshold_source": str(result.get("gm_threshold_source", "")),
        "gm_comparison_operator": str(result.get("gm_comparison_operator", "")),
        "gm_bundle_config_sha256": record.get("gm_bundle_config_sha256", "") if record else "",
        "gm_w1_file_sha256": record.get("gm_w1_file_sha256", "") if record else "",
        "gm_w2_file_sha256": record.get("gm_w2_file_sha256", "") if record else "",
    }
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
