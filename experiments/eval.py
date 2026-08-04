#!/usr/bin/env python3
"""Offline RAVEN evaluation.

Reads ``config.json``, ``records.jsonl``, and per-sample ``output.png`` files
produced by ``main.py``.  Runs quality metrics, detector evaluation, FID, and
CLIP — all without importing or initializing ``RavenPipeline``.

    python3 experiments/eval.py --output-dir /tmp/run --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.experiment_io import (  # noqa: E402
    config_path, detector_records_path, evaluation_dir,
    output_image_path, read_config, read_records_jsonl,
)
from raven.metrics import pair_quality_metrics  # noqa: E402
from raven.detectors import (  # noqa: E402
    ALLOWABLE_STATUSES,
    DETECTOR_MODULES,
    NONZERO_STATUSES,
    ROW_STATUS_SCORED,
    ROW_STATUS_FAILED_MISSING_IMAGE,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_SCORING,
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_MISSING_DEPENDENCY,
    ROW_STATUS_FAILED_INTERNAL_ERROR,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_ERRORS,
    STATUS_SKIPPED_INSUFFICIENT_DATA,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_MISSING_DEPENDENCY,
    STATUS_FAILED_MISSING_IMAGE,
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_SCORING,
    STATUS_FAILED_INTERNAL_ERROR,
    STAGE_NONZERO_STATUSES,
    FAILURE_CAUSE_INTERNAL_ERROR,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION,
    FAILURE_CAUSE_SCORING_ERROR,
    FAILURE_CAUSE_MISSING_IMAGE,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_MISSING_DEPENDENCY,
    _ROW_STATUS_TO_FAILURE_CAUSE,
    _FAILURE_CAUSE_TO_STAGE_STATUS,
    reduce_detector_stage_status,
    stage_status_is_allowable,
    determine_exit_code,
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
    get_detector_module,
    _lazy_imports,
)

logger = logging.getLogger("raven.eval")

DEFAULT_REQUIRED_STAGES = frozenset({"quality", "detector"})

# ---------------------------------------------------------------------------
# Score validation — method-aware contract enforcement (Issue #19)
# ---------------------------------------------------------------------------
THRESHOLD_METHODS = frozenset({"TR", "GS", "GM", "RID", "HSTR", "HSQR"})


def _validate_score(score: Any, method: str) -> tuple[bool, str]:
    """Validate ``score_image`` return value against the method's contract.

    Returns ``(is_valid, error_message)``.  A valid score must carry every
    required key with a finite numeric value; anything else is a contract
    violation and the row must be ``failed_scoring``, not ``scored``.
    """
    if not isinstance(score, dict):
        return False, f"score_image returned non-dict: {type(score).__name__}"

    method_upper = str(method).upper()

    if method_upper == "T2S":
        return _validate_t2s_score(score)
    if method_upper in THRESHOLD_METHODS:
        return _validate_threshold_score(score, method_upper)

    # Unknown method — require at minimum canonical_score
    return _validate_threshold_score(score, method_upper)


def _validate_threshold_score(score: dict[str, Any], method: str) -> tuple[bool, str]:
    """Threshold-based detector contract: raw_score + canonical_score, both finite."""
    for field in ("raw_score", "canonical_score"):
        if field not in score:
            return False, f"missing required field: {field}"
        try:
            value = float(score[field])
        except (ValueError, TypeError):
            return False, f"{field} is not convertible to float: {score[field]!r}"
        if not math.isfinite(value):
            return False, f"{field} is non-finite: {value!r}"
        # Store back as float so downstream consumers see a consistent type
        score[field] = value
    return True, ""


def _validate_t2s_score(score: dict[str, Any]) -> tuple[bool, str]:
    """T2S contract: true/control keys finite, detection_success is real bool,
    optional accuracy fields in [0, 1], margin finite if present."""
    for field in ("t2s_score_true_key", "t2s_score_control_key"):
        if field not in score:
            return False, f"missing required field: {field}"
        try:
            value = float(score[field])
        except (ValueError, TypeError):
            return False, f"{field} is not convertible to float: {score[field]!r}"
        if not math.isfinite(value):
            return False, f"{field} is non-finite: {value!r}"
        score[field] = value

    # detection_success must be a real bool — "false"/1/None/[] all rejected
    if "t2s_detection_success" not in score:
        return False, "missing required field: t2s_detection_success"
    if not isinstance(score["t2s_detection_success"], bool):
        return False, (
            f"t2s_detection_success must be a real bool, got "
            f"{type(score['t2s_detection_success']).__name__}: "
            f"{score['t2s_detection_success']!r}"
        )

    # Margin: if present and not None, must be finite float
    if "t2s_score_margin" in score and score["t2s_score_margin"] is not None:
        try:
            margin = float(score["t2s_score_margin"])
        except (ValueError, TypeError):
            return False, (
                f"t2s_score_margin is not convertible to float: "
                f"{score['t2s_score_margin']!r}"
            )
        if not math.isfinite(margin):
            return False, f"t2s_score_margin is non-finite: {margin!r}"
        score["t2s_score_margin"] = margin

    for acc_field in ("t2s_key_accuracy", "t2s_message_accuracy", "t2s_bit_accuracy"):
        if acc_field in score and score[acc_field] is not None:
            try:
                val = float(score[acc_field])
            except (ValueError, TypeError):
                return False, f"{acc_field} is not convertible to float: {score[acc_field]!r}"
            if not math.isfinite(val) or not 0.0 <= val <= 1.0:
                return False, f"{acc_field} must be in [0, 1], got {val!r}"
            score[acc_field] = val

    # Normalize raw_score / canonical_score for T2S (same as true key)
    if "raw_score" not in score:
        score["raw_score"] = score["t2s_score_true_key"]
    if "canonical_score" not in score:
        score["canonical_score"] = score["t2s_score_true_key"]

    return True, ""


# ===========================================================================
# Detector cohort model
# ===========================================================================
DETECTOR_COHORTS = {
    "watermarked": {
        "original": {"evaluation_cohort": "original_watermarked", "image_source": "input"},
        "attacked": {"evaluation_cohort": "attacked_watermarked", "image_source": "output"},
    },
    "clean": {
        "original": {"evaluation_cohort": "original_clean", "image_source": "input"},
        "attacked": {"evaluation_cohort": "attacked_clean", "image_source": "output"},
    },
}


def _resolve_image_path(rec: dict[str, Any], source: str,
                         output_dir: str | Path) -> Path:
    if source == "input":
        return Path(rec.get("input_path", ""))
    return output_image_path(output_dir, rec.get("role", "watermarked"),
                              str(rec["run_id"]))


def _build_detector_image_index(
    records: list[dict[str, Any]], output_dir: str | Path,
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for rec in records:
        run_id = str(rec["run_id"])
        role = rec.get("role", "watermarked")
        cohorts = DETECTOR_COHORTS.get(role, {})
        for variant, info in cohorts.items():
            image_path = _resolve_image_path(rec, info["image_source"], output_dir)
            index.append({
                "run_id": run_id,
                "source_role": role,
                "evaluation_cohort": info["evaluation_cohort"],
                "image_path": str(image_path),
                "image_source": info["image_source"],
                "method": rec.get("method", ""),
                "prompt": rec.get("prompt", ""),
            })
    return index


def _scored_cohorts(detector_rows: list[dict[str, Any]]) -> set[str]:
    """Return cohort names that have at least one valid scored row."""
    return {
        row["evaluation_cohort"]
        for row in detector_rows
        if row.get("status") == ROW_STATUS_SCORED
        and row.get("canonical_score") is not None
    }


def _all_expected_cohorts(method: str) -> set[str]:
    """Return the full set of cohorts the image index *may* produce."""
    method_upper = str(method).upper()
    if method_upper == "T2S":
        return {"original_watermarked", "attacked_watermarked"}
    return {
        "original_clean", "original_watermarked",
        "attacked_watermarked", "attacked_clean",
    }


def _missing_scoring_cohorts(
    image_index: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    method: str,
) -> list[str]:
    """Cohorts that were requested (present in image_index) but zero rows scored."""
    requested = {entry["evaluation_cohort"] for entry in image_index}
    scored = {r["evaluation_cohort"] for r in detector_rows
              if r.get("status") == ROW_STATUS_SCORED}
    return sorted(requested - scored)


def _missing_metric_cohorts(
    metric_availability: dict[str, Any],
    method: str,
) -> list[str]:
    """Cohorts needed for primary metrics that are absent or have no valid scores."""
    method_upper = str(method).upper()
    if method_upper == "T2S":
        required = {"original_watermarked", "attacked_watermarked"}
    else:
        required = {"original_clean", "original_watermarked", "attacked_watermarked"}
    present = set(metric_availability.get("scored_cohorts", []))
    return sorted(required - present)


# Cohort classification for threshold-based methods
_PRIMARY_COHORTS = frozenset({
    "original_clean", "original_watermarked", "attacked_watermarked",
})
_OPTIONAL_COHORTS = frozenset({"attacked_clean"})


def _compute_primary_optional_counts(
    detector_rows: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    """Separate primary vs optional cohort counts for threshold methods.

    T2S has no primary/optional distinction — all required cohorts are primary.
    """
    method_upper = str(method).upper()
    if method_upper == "T2S":
        scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
        failed = len(detector_rows) - scored
        return {
            "primary_requested_count": len(detector_rows),
            "primary_scored_count": scored,
            "primary_failed_count": failed,
            "optional_requested_count": 0,
            "optional_scored_count": 0,
            "optional_failed_count": 0,
        }

    primary_rows = [r for r in detector_rows
                    if r.get("evaluation_cohort") in _PRIMARY_COHORTS]
    optional_rows = [r for r in detector_rows
                     if r.get("evaluation_cohort") in _OPTIONAL_COHORTS]

    return {
        "primary_requested_count": len(primary_rows),
        "primary_scored_count": sum(1 for r in primary_rows
                                     if r.get("status") == ROW_STATUS_SCORED),
        "primary_failed_count": sum(1 for r in primary_rows
                                     if r.get("status") != ROW_STATUS_SCORED),
        "optional_requested_count": len(optional_rows),
        "optional_scored_count": sum(1 for r in optional_rows
                                      if r.get("status") == ROW_STATUS_SCORED),
        "optional_failed_count": sum(1 for r in optional_rows
                                      if r.get("status") != ROW_STATUS_SCORED),
    }


def _compute_metric_availability(
    detector_rows: list[dict[str, Any]],
    method: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Determine which metric reports can be produced from scored cohorts.

    Returns a dict with boolean flags for each report type and a list of
    what is missing per report.

    ``recalibrated_cohorts_available`` means the required scored cohorts
    exist.  ``recalibrated_report_available`` means the aggregate actually
    contains a recalibrated result block (e.g. ``tr_recalibrated`` with
    ``recalibrated_metrics_available == True``).
    """
    method_upper = str(method).upper()
    scored_set = _scored_cohorts(detector_rows)
    cohort_counts = aggregate.get("cohort_counts", {})

    availability: dict[str, Any] = {
        "scored_cohorts": sorted(scored_set),
        "cohort_counts": cohort_counts,
    }

    if method_upper == "T2S":
        # T2S: needs original_watermarked + attacked_watermarked for
        # paired-key detection report.  Does NOT need original_clean.
        has_wm = "original_watermarked" in scored_set
        has_att = "attacked_watermarked" in scored_set
        availability["primary_report_available"] = has_wm and has_att
        availability["any_report_available"] = has_wm or has_att
        availability["primary_report"] = "paired_key_detection_report"
        availability["primary_required_cohorts"] = [
            "original_watermarked", "attacked_watermarked",
        ]
        if not availability["primary_report_available"]:
            availability["primary_missing"] = sorted(
                {"original_watermarked", "attacked_watermarked"} - scored_set,
            )
        # T2S has no threshold/recalibrated distinction
        availability["threshold_report_available"] = False
        availability["recalibrated_cohorts_available"] = False
        availability["recalibrated_report_available"] = False
        availability["threshold_report"] = None
        return availability

    # ---- Threshold-based methods ----
    has_clean = "original_clean" in scored_set
    has_wm = "original_watermarked" in scored_set
    has_att = "attacked_watermarked" in scored_set
    has_att_clean = "attacked_clean" in scored_set

    # Primary threshold report: needs original_clean + wm + attacked
    threshold_ok = has_clean and has_wm and has_att
    availability["threshold_report_available"] = threshold_ok
    availability["threshold_report"] = (
        "clean_calibrated_threshold_report"
        if threshold_ok else None
    )
    availability["threshold_required_cohorts"] = [
        "original_clean", "original_watermarked", "attacked_watermarked",
    ]
    if not threshold_ok:
        availability["threshold_missing"] = sorted(
            {"original_clean", "original_watermarked", "attacked_watermarked"}
            - scored_set,
        )

    # Recalibrated cohorts: are the required scored cohorts present?
    recal_cohorts_ok = has_att_clean and has_wm and has_att
    availability["recalibrated_cohorts_available"] = recal_cohorts_ok
    availability["recalibrated_required_cohorts"] = [
        "attacked_clean", "original_watermarked", "attacked_watermarked",
    ]

    # Recalibrated report: must actually exist in aggregate output
    availability["recalibrated_report_available"] = _check_recalibrated_report(
        aggregate, method_upper,
    )

    if recal_cohorts_ok and not availability["recalibrated_report_available"]:
        availability["recalibrated_unavailable_reason"] = (
            "scored cohorts are available but aggregate does not contain "
            "a recalibrated result block"
        )
    elif not recal_cohorts_ok and has_att_clean:
        availability["recalibrated_missing"] = sorted(
            {"attacked_clean", "original_watermarked", "attacked_watermarked"}
            - scored_set,
        )
    elif not recal_cohorts_ok:
        availability["recalibrated_unavailable_reason"] = (
            "attacked_clean cohort not present"
        )

    # Primary report = threshold report
    availability["primary_report_available"] = threshold_ok
    availability["any_report_available"] = threshold_ok or (has_wm and has_att)
    return availability


def _check_recalibrated_report(
    aggregate: dict[str, Any],
    method: str,
) -> bool:
    """Check whether aggregate actually contains a recalibrated result block.

    Only returns True when the method-specific recalibrated payload is
    present AND signals availability (e.g. ``recalibrated_metrics_available:
    True``).  Never fabricates availability from cohort presence alone.
    """
    if method == "TR":
        recal = aggregate.get("tr_recalibrated")
        if isinstance(recal, dict) and recal.get("recalibrated_metrics_available") is True:
            return True
        return False

    # GS/GM/fourier adapters do not currently emit recalibrated blocks.
    # Their aggregate output carries only detection_summary.
    # Future: if they add recalibration, add method-specific checks here.
    return False


# ===========================================================================
# Quality stage
# ===========================================================================
def evaluate_quality(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    for rec in records:
        run_id = str(rec["run_id"])
        role = rec.get("role", "watermarked")
        input_path = Path(rec.get("input_path", ""))
        out_path = output_image_path(output_dir, role, run_id)

        if not input_path.is_file() or not out_path.is_file():
            results.append({"run_id": run_id, "role": role,
                            "error": "missing input or output image",
                            "quality_available": False})
            continue

        edx = rec.get("effective_source_flow_dx_image_px")
        edy = rec.get("effective_source_flow_dy_image_px")
        if edx is None or edy is None:
            results.append({"run_id": run_id, "role": role,
                            "error": "missing effective_source_flow",
                            "quality_available": False})
            continue

        try:
            from PIL import Image
            dx, dy = float(edx), float(edy)
            if not math.isfinite(dx) or not math.isfinite(dy):
                results.append({"run_id": run_id, "role": role,
                                "error": "non-finite effective flow",
                                "quality_available": False})
                continue
            with Image.open(input_path) as ref, Image.open(out_path) as att:
                metrics = pair_quality_metrics(
                    ref.convert("RGB"), att.convert("RGB"), dx, dy)
            psnr = float(metrics.get("overlap_psnr", float("nan")))
            ssim = float(metrics.get("overlap_ssim", float("nan")))
            if math.isfinite(psnr):
                psnr_values.append(psnr)
            if math.isfinite(ssim):
                ssim_values.append(ssim)
            results.append({"run_id": run_id, "role": role,
                            "quality_available": True, **metrics})
        except Exception as exc:
            results.append({"run_id": run_id, "role": role,
                            "error": f"{type(exc).__name__}: {exc}",
                            "quality_available": False})

    qa = any(r.get("quality_available") for r in results)
    return {
        "stage": "quality",
        "status": STATUS_COMPLETED if qa else STATUS_SKIPPED_INSUFFICIENT_DATA,
        "available": qa, "count": len(results),
        "psnr_mean": sum(psnr_values) / len(psnr_values) if psnr_values else None,
        "ssim_mean": sum(ssim_values) / len(ssim_values) if ssim_values else None,
        "per_sample": results,
    }


# ===========================================================================
# Detector stage
# ===========================================================================
def _error_to_row_status(exc: Exception) -> str:
    """Map exception type to row status.  Used when an exception escapes
    ``score_image`` — the scoring loop adds ``failure_cause`` and
    ``error_type`` fields for structured downstream consumption."""
    if isinstance(exc, DetectorMissingStateError):
        return ROW_STATUS_FAILED_MISSING_STATE
    if isinstance(exc, DetectorProviderInitializationError):
        return ROW_STATUS_FAILED_PROVIDER
    if isinstance(exc, DetectorStateValidationError):
        return ROW_STATUS_FAILED_STATE_VALIDATION
    if isinstance(exc, DetectorScoringError):
        return ROW_STATUS_FAILED_SCORING
    if isinstance(exc, FileNotFoundError):
        return ROW_STATUS_FAILED_MISSING_IMAGE
    return ROW_STATUS_FAILED_SCORING


def _error_to_failure_cause(exc: Exception) -> str:
    """Map exception type to structured failure cause."""
    if isinstance(exc, DetectorMissingStateError):
        return FAILURE_CAUSE_MISSING_REQUIRED_STATE
    if isinstance(exc, DetectorDependencyError):
        return FAILURE_CAUSE_MISSING_DEPENDENCY
    if isinstance(exc, DetectorProviderInitializationError):
        return FAILURE_CAUSE_PROVIDER_INITIALIZATION
    if isinstance(exc, DetectorStateValidationError):
        return FAILURE_CAUSE_STATE_VALIDATION
    if isinstance(exc, FileNotFoundError):
        return FAILURE_CAUSE_MISSING_IMAGE
    if isinstance(exc, DetectorScoringError):
        return FAILURE_CAUSE_SCORING_ERROR
    if isinstance(exc, ImportError):
        return FAILURE_CAUSE_MISSING_DEPENDENCY
    return FAILURE_CAUSE_INTERNAL_ERROR


def _error_to_stage_status(exc: Exception) -> str:
    """Map exception type to stage status (for orchestration-level catches)."""
    cause = _error_to_failure_cause(exc)
    return _FAILURE_CAUSE_TO_STAGE_STATUS.get(cause, STATUS_FAILED_INTERNAL_ERROR)


def evaluate_detector(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    method: str,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run detector on all cohorts via method-specific detector module."""
    output_dir = Path(output_dir)
    eval_dir = evaluation_dir(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        det_mod = get_detector_module(method)
    except ValueError as exc:
        return {"stage": "detector", "method": method,
                "status": STATUS_FAILED_MISSING_DEPENDENCY, "reason": str(exc)}

    image_index = _build_detector_image_index(records, output_dir)
    if not image_index:
        return {"stage": "detector", "method": method,
                "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "No images to score."}

    # ---- Issue #25: image preflight BEFORE metadata/provider setup ----
    preflight_rows: list[dict[str, Any]] = []
    valid_entries: list[dict[str, Any]] = []
    for entry in image_index:
        image_path_obj = Path(entry["image_path"])
        if not image_path_obj.is_file():
            preflight_rows.append({
                "run_id": entry["run_id"],
                "source_role": entry["source_role"],
                "evaluation_cohort": entry["evaluation_cohort"],
                "image_path": entry["image_path"],
                "method": method,
                "status": ROW_STATUS_FAILED_MISSING_IMAGE,
                "failure_cause": FAILURE_CAUSE_MISSING_IMAGE,
                "error_type": "FileNotFoundError",
                "error": (
                    "Image file does not exist or is not a regular file: "
                    f"{entry['image_path']}"
                ),
            })
        else:
            valid_entries.append(entry)

    # ---- Setup phase: metadata + provider state ----
    setup_failure_cause: str | None = None
    setup_failure_status: str | None = None
    setup_error_type: str | None = None
    setup_error_message: str | None = None
    enriched_records: list[dict[str, Any]] = []
    provider_info = None

    # Resolve metadata
    from raven.metadata_resolver import (
        MetadataResolver, MetadataResolverError, MetadataConflictError,
        DuplicateMetadataError, AmbiguousMetadataError,
    )
    csv_path = config.get("metadata_path", "") if config else ""
    resolver = None
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            resolver = MetadataResolver.from_records_fallback(records)
            if resolver is None:
                setup_failure_cause = FAILURE_CAUSE_MISSING_REQUIRED_STATE
                setup_failure_status = STATUS_FAILED_MISSING_REQUIRED_STATE
                setup_error_type = "MetadataMissingStateError"
                setup_error_message = (
                    f"No metadata CSV found at {csv_path} "
                    "and no embedded source_metadata in records."
                )
        elif not path.is_file():
            setup_failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
            setup_failure_status = STATUS_FAILED_INTERNAL_ERROR
            setup_error_type = "MetadataInternalError"
            setup_error_message = (
                f"metadata_path exists but is not a regular file: {csv_path}. "
                "Expected a CSV file."
            )
        else:
            try:
                resolver = MetadataResolver.from_path(csv_path)
            except (DuplicateMetadataError, AmbiguousMetadataError,
                    MetadataResolverError) as exc:
                setup_failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
                setup_failure_status = STATUS_FAILED_INTERNAL_ERROR
                setup_error_type = type(exc).__name__
                setup_error_message = (
                    f"Metadata validation failed: {type(exc).__name__}: {exc}"
                )
            except ValueError as exc:
                setup_failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
                setup_failure_status = STATUS_FAILED_INTERNAL_ERROR
                setup_error_type = type(exc).__name__
                setup_error_message = f"Metadata CSV invalid: {exc}"
    else:
        resolver = MetadataResolver.from_records_fallback(records)
        if resolver is None:
            setup_failure_cause = FAILURE_CAUSE_MISSING_REQUIRED_STATE
            setup_failure_status = STATUS_FAILED_MISSING_REQUIRED_STATE
            setup_error_type = "MetadataMissingStateError"
            setup_error_message = (
                "No metadata_path in config.json "
                "and no embedded source_metadata in records."
            )

    # Enrich records with resolved metadata
    if resolver is not None and setup_failure_cause is None:
        for rec in records:
            try:
                enriched_records.append(
                    resolver.enrich_record(rec, csv_path=csv_path or None)
                )
            except MetadataResolverError as exc:
                setup_failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
                setup_failure_status = STATUS_FAILED_INTERNAL_ERROR
                setup_error_type = type(exc).__name__
                setup_error_message = (
                    f"Metadata resolution failed for "
                    f"run_id={rec.get('run_id')}: {exc}"
                )
                break

    # Load provider state
    if setup_failure_cause is None:
        try:
            if method in {"RID", "HSTR", "HSQR"}:
                provider_info = det_mod.load_state(enriched_records, device,
                                                   method=method)
            else:
                provider_info = det_mod.load_state(enriched_records, device)
        except DetectorMissingStateError as exc:
            setup_failure_cause = FAILURE_CAUSE_MISSING_REQUIRED_STATE
            setup_failure_status = STATUS_FAILED_MISSING_REQUIRED_STATE
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except DetectorDependencyError as exc:
            setup_failure_cause = FAILURE_CAUSE_MISSING_DEPENDENCY
            setup_failure_status = STATUS_FAILED_MISSING_DEPENDENCY
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except DetectorProviderInitializationError as exc:
            setup_failure_cause = FAILURE_CAUSE_PROVIDER_INITIALIZATION
            setup_failure_status = STATUS_FAILED_PROVIDER_INITIALIZATION
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except DetectorStateValidationError as exc:
            setup_failure_cause = FAILURE_CAUSE_STATE_VALIDATION
            setup_failure_status = STATUS_FAILED_STATE_VALIDATION
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except ImportError as exc:
            setup_failure_cause = FAILURE_CAUSE_MISSING_DEPENDENCY
            setup_failure_status = STATUS_FAILED_MISSING_DEPENDENCY
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except TypeError as exc:
            setup_failure_cause = FAILURE_CAUSE_PROVIDER_INITIALIZATION
            setup_failure_status = STATUS_FAILED_PROVIDER_INITIALIZATION
            setup_error_type = type(exc).__name__
            setup_error_message = str(exc)
        except Exception as exc:
            setup_failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
            setup_failure_status = STATUS_FAILED_INTERNAL_ERROR
            setup_error_type = type(exc).__name__
            setup_error_message = f"{type(exc).__name__}: {exc}"

        if provider_info is None and setup_failure_cause is None:
            setup_failure_cause = FAILURE_CAUSE_MISSING_REQUIRED_STATE
            setup_failure_status = STATUS_FAILED_MISSING_REQUIRED_STATE
            setup_error_type = "MissingProviderStateError"
            setup_error_message = (
                f"Provider state for {method} is not available."
            )

    # ---- Scoring phase: only if setup succeeded and there are valid images ----
    detector_rows: list[dict[str, Any]] = list(preflight_rows)
    unscored_due_to_setup_count = 0

    if setup_failure_cause is None and valid_entries:
        # Build record lookup from enriched records
        record_index: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in enriched_records:
            key = (str(rec["run_id"]), rec.get("role", "watermarked"))
            record_index[key] = rec

        for entry in valid_entries:
            key = (entry["run_id"], entry["source_role"])
            matched_record = record_index.get(key, {})

            score = None
            row_status = ROW_STATUS_FAILED_SCORING
            failure_cause = FAILURE_CAUSE_SCORING_ERROR
            error_type = ""
            error_msg = ""
            try:
                score = det_mod.score_image(
                    provider_info, entry["image_path"],
                    record=matched_record,
                    evaluation_entry=entry,
                )
                if score is None:
                    row_status = ROW_STATUS_FAILED_SCORING
                    failure_cause = FAILURE_CAUSE_SCORING_ERROR
                    error_type = "NoneReturn"
                    error_msg = "score_image returned None"
                elif not isinstance(score, dict):
                    row_status = ROW_STATUS_FAILED_SCORING
                    failure_cause = FAILURE_CAUSE_SCORING_ERROR
                    error_type = "NonDictReturn"
                    error_msg = (
                        f"score_image returned non-dict: {type(score).__name__}"
                    )
                else:
                    valid, validation_error = _validate_score(score, method)
                    if valid:
                        row_status = ROW_STATUS_SCORED
                        failure_cause = ""
                        error_type = ""
                    else:
                        row_status = ROW_STATUS_FAILED_SCORING
                        failure_cause = FAILURE_CAUSE_SCORING_ERROR
                        error_type = "ScoreContractViolation"
                        error_msg = (
                            f"score validation failed: {validation_error}"
                        )
            except DetectorMissingStateError as exc:
                row_status = ROW_STATUS_FAILED_MISSING_STATE
                failure_cause = FAILURE_CAUSE_MISSING_REQUIRED_STATE
                error_type = type(exc).__name__
                error_msg = str(exc)
            except DetectorProviderInitializationError as exc:
                row_status = ROW_STATUS_FAILED_PROVIDER
                failure_cause = FAILURE_CAUSE_PROVIDER_INITIALIZATION
                error_type = type(exc).__name__
                error_msg = str(exc)
            except DetectorStateValidationError as exc:
                row_status = ROW_STATUS_FAILED_STATE_VALIDATION
                failure_cause = FAILURE_CAUSE_STATE_VALIDATION
                error_type = type(exc).__name__
                error_msg = str(exc)
            except DetectorScoringError as exc:
                row_status = ROW_STATUS_FAILED_SCORING
                failure_cause = FAILURE_CAUSE_SCORING_ERROR
                error_type = type(exc).__name__
                error_msg = str(exc)
            except DetectorDependencyError as exc:
                row_status = ROW_STATUS_FAILED_MISSING_DEPENDENCY
                failure_cause = FAILURE_CAUSE_MISSING_DEPENDENCY
                error_type = type(exc).__name__
                error_msg = str(exc)
            except ImportError as exc:
                row_status = ROW_STATUS_FAILED_MISSING_DEPENDENCY
                failure_cause = FAILURE_CAUSE_MISSING_DEPENDENCY
                error_type = type(exc).__name__
                error_msg = str(exc)
            except FileNotFoundError:
                row_status = ROW_STATUS_FAILED_MISSING_IMAGE
                failure_cause = FAILURE_CAUSE_MISSING_IMAGE
                error_type = "FileNotFoundError"
                error_msg = (
                    f"Image not found inside score_image: "
                    f"{entry['image_path']}"
                )
            except TypeError as exc:
                row_status = ROW_STATUS_FAILED_INTERNAL_ERROR
                failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
                error_type = type(exc).__name__
                error_msg = str(exc)
            except Exception as exc:
                row_status = ROW_STATUS_FAILED_INTERNAL_ERROR
                failure_cause = FAILURE_CAUSE_INTERNAL_ERROR
                error_type = type(exc).__name__
                error_msg = f"{type(exc).__name__}: {exc}"

            row = {
                "run_id": entry["run_id"],
                "source_role": entry["source_role"],
                "evaluation_cohort": entry["evaluation_cohort"],
                "image_path": entry["image_path"],
                "method": method,
                "status": row_status,
            }
            if isinstance(score, dict) and row_status == ROW_STATUS_SCORED:
                row.update(score)
            if failure_cause:
                row["failure_cause"] = failure_cause
            if error_type:
                row["error_type"] = error_type
            if error_msg:
                row["error"] = error_msg
            detector_rows.append(row)
    elif setup_failure_cause is not None:
        # Setup failed — valid entries were never scored
        unscored_due_to_setup_count = len(valid_entries)

    # Write detector_records.jsonl (preflight rows + any scored rows)
    det_path = detector_records_path(output_dir)
    tmp = det_path.with_name(f".detector_records.jsonl.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in detector_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, det_path)

    # Aggregate (adapter sees only detector_rows, not full image_index)
    agg_kwargs: dict[str, Any] = {}
    if method in {"RID", "HSTR", "HSQR"}:
        agg_kwargs["method"] = method
    aggregate = det_mod.aggregate(detector_rows, **agg_kwargs)

    # ---- Issue #25: orchestrator is count source of truth ----
    # Adapter only receives detector_rows; it cannot know about entries
    # that were never scored due to setup failure.  Compute final counts
    # from image_index + detector_rows + unscored count.
    row_scored_count = sum(
        1 for row in detector_rows
        if row.get("status") == ROW_STATUS_SCORED
    )
    row_failed_count = len(detector_rows) - row_scored_count
    requested_count = len(image_index)

    aggregate["requested_count"] = requested_count
    aggregate["scored_count"] = row_scored_count
    aggregate["failed_count"] = row_failed_count
    aggregate["unscored_due_to_setup_count"] = unscored_due_to_setup_count

    # Count invariant: requested = scored + failed + unscored
    count_invariant_ok = (
        requested_count
        == row_scored_count + row_failed_count + unscored_due_to_setup_count
    )
    aggregate["count_invariant_satisfied"] = count_invariant_ok

    cohort_counts = aggregate.get("cohort_counts", {})

    # ---- Issue #19: metric availability ----
    metric_availability = _compute_metric_availability(
        detector_rows, method, aggregate,
    )
    aggregate["metric_availability"] = metric_availability
    aggregate["missing_scoring_cohorts"] = _missing_scoring_cohorts(
        image_index, detector_rows, method,
    )
    aggregate["missing_metric_cohorts"] = _missing_metric_cohorts(
        metric_availability, method,
    )

    # ---- Issue #19: primary/optional cohort counts ----
    primary_optional = _compute_primary_optional_counts(detector_rows, method)
    aggregate.update(primary_optional)

    # ---- Issue #25: single stage-status reducer ----
    primary_available = metric_availability.get("primary_report_available", False)
    optional_failed = primary_optional.get("optional_failed_count", 0)

    reducer_result = reduce_detector_stage_status(
        detector_rows,
        setup_failure=setup_failure_cause,
        primary_report_available=primary_available,
        primary_metrics_complete=primary_available,
        optional_failed_count=optional_failed,
    )
    stage_status = reducer_result["status"]

    # Optional cohort failures must not downgrade primary completion
    if stage_status == STATUS_COMPLETED and optional_failed > 0:
        aggregate["optional_metrics_incomplete"] = True

    # Merge reducer diagnostics into aggregate
    aggregate["dominant_failure_cause"] = reducer_result.get(
        "dominant_failure_cause")
    aggregate["status_reducer_reason"] = reducer_result.get(
        "status_reducer_reason")
    aggregate["row_status_counts"] = reducer_result.get("row_status_counts", {})
    aggregate["failure_cause_counts"] = reducer_result.get(
        "failure_cause_counts", {})

    # Setup failure diagnostics
    if setup_failure_cause is not None:
        aggregate["setup_failure_cause"] = setup_failure_cause
        aggregate["setup_error_type"] = setup_error_type
        aggregate["setup_error"] = setup_error_message
    aggregate["stage"] = "detector"
    aggregate["method"] = method
    aggregate["status"] = stage_status
    aggregate["available"] = reducer_result["available"]
    return aggregate


# ===========================================================================
# FID stage
# ===========================================================================
def evaluate_fid(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from raven.quality import clean_fid, FID_PRIMARY_MODE
    except ImportError:
        return {"stage": "fid", "status": STATUS_FAILED_MISSING_DEPENDENCY,
                "reason": "clean-fid not installed."}

    import hashlib, shutil, tempfile
    output_dir = Path(output_dir)
    wm_records = [r for r in records if r.get("role") == "watermarked"]
    if not wm_records:
        return {"stage": "fid", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "No watermarked records."}

    pairs: list[dict[str, Any]] = []
    for rec in wm_records:
        run_id = str(rec["run_id"])
        input_path = Path(rec.get("input_path", ""))
        out_path = output_image_path(output_dir, "watermarked", run_id)
        if input_path.is_file() and out_path.is_file():
            try:
                safe_name = f"{int(run_id):06d}"
            except (ValueError, TypeError):
                safe_name = hashlib.sha256(run_id.encode()).hexdigest()[:12]
            pairs.append({"run_id": run_id, "safe_name": safe_name,
                          "reference_path": str(input_path),
                          "attacked_path": str(out_path)})

    if len(pairs) < 2:
        return {"stage": "fid", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": f"Need 2+ paired images, got {len(pairs)}."}

    tmpdir = Path(tempfile.mkdtemp(prefix="raven_fid_"))
    try:
        ref_dir, att_dir = tmpdir / "reference", tmpdir / "attacked"
        ref_dir.mkdir(); att_dir.mkdir()
        for pair in pairs:
            shutil.copy2(pair["reference_path"], ref_dir / f"{pair['safe_name']}.png")
            shutil.copy2(pair["attacked_path"], att_dir / f"{pair['safe_name']}.png")
        result = clean_fid(str(ref_dir), str(att_dir), device=device)
        return {"stage": "fid", "status": STATUS_COMPLETED,
                "image_count": len(pairs), "fid_value": result.get("value"),
                "mode": FID_PRIMARY_MODE, "protocol": result.get("protocol", ""),
                "staged_records": pairs}
    except Exception as exc:
        return {"stage": "fid", "status": STATUS_FAILED_INTERNAL_ERROR,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# CLIP stage
# ===========================================================================
def evaluate_clip(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from raven.quality import openclip_text_image_scores
    except ImportError:
        return {"stage": "clip", "status": STATUS_FAILED_MISSING_DEPENDENCY,
                "reason": "open_clip_torch not installed."}

    output_dir = Path(output_dir)
    wm_records = [r for r in records if r.get("role") == "watermarked"]
    image_paths, prompts = [], []
    for rec in wm_records:
        out_path = output_image_path(output_dir, "watermarked", str(rec["run_id"]))
        if out_path.is_file():
            image_paths.append(str(out_path))
            prompts.append(rec.get("prompt", ""))
    if not image_paths:
        return {"stage": "clip", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "No watermarked output images."}
    if not all(prompts):
        return {"stage": "clip", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "Some records missing prompt."}
    try:
        result = openclip_text_image_scores(
            image_paths, prompts, device=device,
            model_name="ViT-bigG-14", pretrained="laion2b_s39b_b160k")
        scores = result.get("scores", [])
        import numpy as np
        return {"stage": "clip", "status": STATUS_COMPLETED,
                "image_count": len(image_paths),
                "model_name": result.get("model_name", "ViT-bigG-14"),
                "pretrained": result.get("pretrained", "laion2b_s39b_b160k"),
                "metric": result.get("metric", "prompt-image cosine similarity"),
                "count": len(scores), "mean_score": result.get("mean"),
                "std": float(np.std(scores)) if scores else None, "scores": scores}
    except Exception as exc:
        return {"stage": "clip", "status": STATUS_FAILED_INTERNAL_ERROR,
                "error": f"{type(exc).__name__}: {exc}"}


# ===========================================================================
# Orchestrator
# ===========================================================================
STAGE_RUNNERS: dict[str, Any] = {
    "quality": evaluate_quality,
    "detector": lambda r, od, dev, cfg: evaluate_detector(
        r, od, cfg.get("method", "TR"), dev, cfg),
    "fid": evaluate_fid,
    "clip": evaluate_clip,
}


def run_evaluation(
    output_dir: str | Path,
    *, device: str = "cuda", stages: list[str] | None = None,
    allow_missing_metrics: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not config_path(output_dir).is_file():
        raise FileNotFoundError(f"config.json not found in {output_dir}")
    config = read_config(output_dir)
    records = read_records_jsonl(output_dir)
    if not records:
        raise ValueError(f"No complete records in {output_dir}")
    method = config.get("method", "TR").upper()
    if stages is None:
        stages = ["quality", "detector"]

    result: dict[str, Any] = {
        "output_dir": str(output_dir), "method": method,
        "dataset": config.get("dataset", "unspecified"),
        "sample_count": len(records),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }

    for stage in stages:
        runner = STAGE_RUNNERS.get(stage)
        if runner is None:
            result["stages"][stage] = {"status": STATUS_FAILED_INTERNAL_ERROR,
                                        "reason": f"Unknown stage: {stage}"}
            continue
        logger.info("Running %s evaluation...", stage)
        try:
            result["stages"][stage] = runner(records, output_dir, device, config)
        except Exception as exc:
            logger.exception("%s evaluation failed", stage)
            result["stages"][stage] = {
                "status": _error_to_stage_status(exc),
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ---- Issue #25: unified exit code policy ----
    stage_statuses = {
        s: info.get("status", STATUS_FAILED_INTERNAL_ERROR)
        for s, info in result["stages"].items()}
    failed = {s for s, st in stage_statuses.items()
              if st in STAGE_NONZERO_STATUSES and st not in ALLOWABLE_STATUSES}
    failed_allowable = {s for s, st in stage_statuses.items()
                        if st in ALLOWABLE_STATUSES}

    # Preserve original stage statuses — allow flag must NOT rewrite them
    result["failed_stages"] = sorted(failed)
    result["skipped_stages"] = sorted(failed_allowable)

    # Mark whether each nonzero stage is allowable under current policy
    allowable_map: dict[str, bool] = {}
    for stage_name, stage_info in result["stages"].items():
        st = stage_info.get("status", STATUS_FAILED_INTERNAL_ERROR)
        allowable_map[stage_name] = stage_status_is_allowable(
            st, allow_missing_metrics=allow_missing_metrics)
    result["stages_allowable"] = allowable_map

    # Overall status reflects worst non-allowable stage, or completed
    exit_code = determine_exit_code(
        result, allow_missing_metrics=allow_missing_metrics)
    if exit_code == 0:
        result["overall_status"] = STATUS_COMPLETED
    else:
        result["overall_status"] = STATUS_COMPLETED_WITH_ERRORS

    result["allowed_by_policy"] = allow_missing_metrics

    return result


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", nargs="+",
                   choices=["quality", "detector", "fid", "clip"],
                   default=["quality", "detector"])
    p.add_argument("--allow-missing-metrics", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")
    if not args.output_dir.is_dir():
        logger.error("output-dir does not exist: %s", args.output_dir)
        return 1
    try:
        result = run_evaluation(args.output_dir, device=args.device,
                                stages=args.stages,
                                allow_missing_metrics=args.allow_missing_metrics)
    except Exception as exc:
        logger.exception("Evaluation failed")
        return 1

    result_json = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n", encoding="utf-8")
    else:
        print(result_json)

    failed = result.get("failed_stages", [])
    skipped = result.get("skipped_stages", [])
    if failed:
        logger.error("Failed stages: %s", ", ".join(failed))
    if skipped:
        logger.warning("Skipped required stages: %s", ", ".join(skipped))

    # ---- Issue #25: unified exit-code policy ----
    return determine_exit_code(
        result, allow_missing_metrics=args.allow_missing_metrics)


if __name__ == "__main__":
    raise SystemExit(main())
