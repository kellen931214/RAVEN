#!/usr/bin/env python3
"""Offline RAVEN evaluation.

Evaluates canonical RAVEN runs from ``config.json``, ``records.jsonl``, and
per-sample ``output.png`` files.  The optional ``pixel-shift`` workflow also
streams an external black-fill pixel shift through ``RavenPipeline`` before
running the same evaluation stages.

    python raven_repro/eval.py --output-dir /tmp/run --device cuda
    python raven_repro/eval.py --workflow pixel-shift --metadata data.csv \
        --output-dir /tmp/pixel-shift --magnitudes 16 24 32
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from raven.experiment_config import (  # noqa: E402
    check_config_match, config_for_pipeline, normalize_config,
)
from raven.experiment_io import (  # noqa: E402
    cleanup_intermediates, config_path, detector_records_path, evaluation_dir,
    is_sample_complete, output_image_path, prepare_output_dir, read_config,
    read_records_jsonl, rebuild_records_jsonl, write_config, write_record,
)
from raven.evaluation.metrics import pair_quality_metrics  # noqa: E402
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
    # ---- Aggregate phase exception boundary ----
    # An adapter metric-computation failure must not abort the whole stage:
    # detector_records.jsonl is already written above, and the stage result
    # stays a normal dict whose classification preserves the original
    # exception type (via _error_to_failure_cause / _error_to_stage_status).
    aggregate_failure: dict[str, Any] | None = None
    try:
        aggregate = det_mod.aggregate(detector_rows, **agg_kwargs)
    except Exception as exc:
        aggregate = {}
        aggregate_failure = {
            "aggregate_error_type": type(exc).__name__,
            "aggregate_error": str(exc),
            "aggregate_failure_cause": _error_to_failure_cause(exc),
            "aggregate_stage_status": _error_to_stage_status(exc),
        }

    # ---- Adapter aggregate failure → stage-status reducer ----
    # Adapters may signal a structured aggregate-level failure (e.g. GS
    # official policy rows that fail validation).  It participates in the
    # same precedence pool as row/setup failures, so the final detector
    # stage status is never completed when the aggregate failed closed.
    aggregate_failure_cause = aggregate.get("aggregate_failure_cause")

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
        aggregate_failure=aggregate_failure_cause,
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

    # Aggregate phase failure: the adapter could not produce a metric
    # aggregate.  detector_records.jsonl and all counts are preserved; the
    # stage identity (stage/method) is set and the classification preserves
    # the original exception type.
    if aggregate_failure is not None:
        aggregate["stage"] = "detector"
        aggregate["method"] = method
        aggregate.update(aggregate_failure)
        aggregate["dominant_failure_cause"] = aggregate_failure[
            "aggregate_failure_cause"]
        aggregate["status"] = aggregate_failure["aggregate_stage_status"]
        aggregate["available"] = row_scored_count > 0
        aggregate["status_reducer_reason"] = (
            "aggregate phase failed: "
            f"{aggregate_failure['aggregate_error_type']}: "
            f"{aggregate_failure['aggregate_error']}"
        )
        return aggregate

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
        from raven.evaluation.metrics import clean_fid, FID_PRIMARY_MODE
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
        from raven.evaluation.metrics import openclip_text_image_scores
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
# LPIPS stage
# ===========================================================================
def evaluate_lpips(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream LPIPS over watermarked input/output pairs without image caching."""
    try:
        import lpips
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        return {"stage": "lpips", "status": STATUS_FAILED_MISSING_DEPENDENCY,
                "reason": "lpips is not installed."}

    wm_records = [record for record in records if record.get("role") == "watermarked"]
    if not wm_records:
        return {"stage": "lpips", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "No watermarked records."}

    model = lpips.LPIPS(net="alex").to(device).eval()
    values: list[float] = []
    failed = 0

    def load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            for record in wm_records:
                input_path = Path(record.get("input_path", ""))
                output_path = output_image_path(output_dir, "watermarked", str(record["run_id"]))
                if not input_path.is_file() or not output_path.is_file():
                    failed += 1
                    continue
                try:
                    reference = load_image(input_path)
                    attacked = load_image(output_path)
                    if reference.shape != attacked.shape:
                        import torch.nn.functional as functional
                        attacked = functional.interpolate(
                            attacked, size=reference.shape[-2:], mode="bilinear", align_corners=False,
                        )
                    values.append(float(model(reference, attacked).item()))
                    del reference, attacked
                except Exception as exc:  # noqa: BLE001 - per-sample containment
                    logger.warning("LPIPS failed for run_id=%s: %s", record.get("run_id"), exc)
                    failed += 1
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not values:
        return {"stage": "lpips", "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
                "reason": "No valid watermarked input/output pairs.", "failed_count": failed}
    return {
        "stage": "lpips", "status": STATUS_COMPLETED, "model": "alex",
        "count": len(values), "failed_count": failed,
        "mean": sum(values) / len(values), "min": min(values), "max": max(values),
    }


# ===========================================================================
# Orchestrator
# ===========================================================================
STAGE_RUNNERS: dict[str, Any] = {
    "quality": evaluate_quality,
    "detector": lambda r, od, dev, cfg: evaluate_detector(
        r, od, cfg.get("method", "TR"), dev, cfg),
    "fid": evaluate_fid,
    "clip": evaluate_clip,
    "lpips": evaluate_lpips,
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
# Pixel-shift workflows
# ===========================================================================
PIXEL_SHIFT_STAGES = ["quality", "detector", "fid", "clip", "lpips"]


def _iter_pixel_shift_metadata(path: Path, limit: int | None):
    """Yield normalized metadata rows without retaining the full CSV in RAM."""
    from main import normalize_metadata_row

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield normalize_metadata_row(row)


def pixel_shift_with_black_fill(image, dx: int, dy: int):
    """Translate an image in pixel space, dropping exposed pixels to black."""
    from PIL import Image

    shifted = Image.new(image.mode, image.size, color=0)
    width, height = image.size
    source_left, source_top = max(0, -dx), max(0, -dy)
    source_right, source_bottom = min(width, width - dx), min(height, height - dy)
    if source_left >= source_right or source_top >= source_bottom:
        return shifted
    shifted.paste(
        image.crop((source_left, source_top, source_right, source_bottom)),
        (max(0, dx), max(0, dy)),
    )
    return shifted


def _pixel_shift_config(args: argparse.Namespace, output_dir: Path, magnitude: int) -> dict[str, Any]:
    config = normalize_config(
        diffusion_mode="ddim",
        method="TR",
        dataset=args.dataset,
        metadata_path=str(args.metadata.resolve()),
        output_dir=str(output_dir.resolve()),
        roles=["watermarked"],
        limit=args.limit,
        gpu=args.gpu,
        overwrite=args.overwrite,
        resume=args.resume,
        shift_mode="none",
        shift_magnitude_min=magnitude,
        shift_magnitude_max=magnitude,
        base_seed=args.base_seed,
        steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        shift_space="image_pixels",
        warp_mode=args.warp_mode,
        latent_sampling_mode=args.sampling,
        padding_mode=args.padding_mode,
        view_guided_attention=args.view_guided_attention,
        color_transfer=args.color_transfer == "aligned",
        prompt="",
        prompt_source="metadata",
        negative_prompt=args.negative_prompt,
        debug=False,
        save_input_copy=False,
        save_intermediates=args.save_intermediates,
        model_id=args.model_id,
        model_revision=args.model_revision,
        dtype=args.dtype,
    )
    config.update({
        "workflow": "pixel_shift",
        "pixel_shift": {
            "dx_px": magnitude,
            "dy_px": magnitude,
            "fill": "black",
            "internal_latent_shift": False,
        },
    })
    return config


def _write_workflow_result(output_dir: Path, result: dict[str, Any]) -> None:
    destination = evaluation_dir(output_dir) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_pixel_shift_attack_for_magnitude(
    args: argparse.Namespace,
    magnitude: int,
    output_dir: Path,
    pipe,
) -> int:
    """Run one pixel-shift cohort incrementally and write canonical records."""
    from PIL import Image, ImageOps
    from main import resolve_input_path
    from raven.shift_plan import compute_attack_seed

    config = _pixel_shift_config(args, output_dir, magnitude)
    prepared = prepare_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)
    if args.resume and config_path(prepared).is_file():
        stored = read_config(prepared)
        mismatches = check_config_match(stored, config)
        if mismatches:
            raise ValueError("pixel-shift resume config mismatch: " + ", ".join(sorted(mismatches)))
        config = stored
    else:
        write_config(prepared, config)

    pipeline_kwargs = config_for_pipeline(config)
    completed = 0
    for row in _iter_pixel_shift_metadata(args.metadata, args.limit):
        run_id = str(row["run_id"])
        if is_sample_complete(prepared, "watermarked", run_id):
            continue
        input_path = resolve_input_path(row, "watermarked")
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        shifted_input = pixel_shift_with_black_fill(image, magnitude, magnitude)
        sample_dir = prepared / "samples" / "watermarked" / run_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        attack_seed = compute_attack_seed(config["base_seed"], run_id)
        final_image = pipe.run(
            **pipeline_kwargs,
            input_image=shifted_input,
            output_dir=str(sample_dir),
            seed=attack_seed,
            prompt=row.get("prompt", ""),
            shift_x=0.0,
            shift_y=0.0,
        )
        output_path = output_image_path(prepared, "watermarked", run_id)
        final_image.save(output_path)
        record = {
            "run_id": run_id,
            "role": "watermarked",
            "dataset": config["dataset"],
            "method": "TR",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "prompt": row.get("prompt", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_source": "metadata",
            "attack_seed": attack_seed,
            "pixel_shift_dx_px": magnitude,
            "pixel_shift_dy_px": magnitude,
            "planned_flow_dx_image_px": 0.0,
            "planned_flow_dy_image_px": 0.0,
            "effective_source_flow_dx_image_px": 0.0,
            "effective_source_flow_dy_image_px": 0.0,
            "source_metadata": row,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_record(prepared, "watermarked", run_id, record)
        if not config.get("save_intermediates"):
            cleanup_intermediates(prepared, "watermarked", run_id)
        completed += 1
        del image, shifted_input, final_image
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    rebuild_records_jsonl(prepared)
    return completed


def _evaluate_pixel_shift_directory(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    stages = args.stages or PIXEL_SHIFT_STAGES
    result = run_evaluation(
        output_dir,
        device=args.device,
        stages=stages,
        allow_missing_metrics=args.allow_missing_metrics,
    )
    _write_workflow_result(output_dir, result)
    return result


def run_pixel_shift_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Pixel-shift images, run RAVEN once per sample, then evaluate each cohort."""
    if args.metadata is None or not args.metadata.is_file():
        raise FileNotFoundError("--metadata is required for --workflow pixel-shift")

    pipe = None
    summaries: dict[str, Any] = {}
    try:
        if not args.eval_only:
            if args.gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for pixel-shift RAVEN generation")
            from raven.pipeline_raven import RavenPipeline
            pipe = RavenPipeline(
                model_id=args.model_id,
                device="cuda",
                dtype=args.dtype,
                revision=args.model_revision,
                scheduler_mode="ddim",
            )

        for magnitude in args.magnitudes:
            directory = args.output_dir / f"pixel_shift_raven_{magnitude}px"
            if args.eval_only:
                if not config_path(directory).is_file():
                    raise FileNotFoundError(f"config.json not found for --eval-only: {directory}")
            else:
                _run_pixel_shift_attack_for_magnitude(args, magnitude, directory, pipe)
            summaries[str(magnitude)] = _evaluate_pixel_shift_directory(args, directory)
    finally:
        if pipe is not None:
            del pipe
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    result = {"workflow": "pixel_shift", "magnitudes": summaries}
    (args.output_dir / "pixel_shift_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


def rebuild_pixel_shift_records(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild canonical records from existing pixel-shift output images, then evaluate."""
    if args.metadata is None or not args.metadata.is_file():
        raise FileNotFoundError("--metadata is required for --workflow rebuild-pixel-shift")

    summaries: dict[str, Any] = {}
    for magnitude in args.magnitudes:
        directory = args.output_dir / f"pixel_shift_raven_{magnitude}px"
        if not directory.is_dir():
            raise FileNotFoundError(f"pixel-shift output directory not found: {directory}")
        if not config_path(directory).is_file():
            write_config(directory, _pixel_shift_config(args, directory, magnitude))

        rebuilt = 0
        for row in _iter_pixel_shift_metadata(args.metadata, args.limit):
            run_id = str(row["run_id"])
            watermarked_output = output_image_path(directory, "watermarked", run_id)
            watermarked_input = Path(row.get("watermarked_path", ""))
            if watermarked_output.is_file() and watermarked_input.is_file():
                write_record(directory, "watermarked", run_id, {
                    "run_id": run_id, "role": "watermarked", "dataset": args.dataset,
                    "method": "TR", "input_path": str(watermarked_input),
                    "output_path": str(watermarked_output), "prompt": row.get("prompt", ""),
                    "prompt_id": row.get("prompt_id", ""), "prompt_source": "metadata",
                    "pixel_shift_dx_px": magnitude, "pixel_shift_dy_px": magnitude,
                    "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
                    "effective_source_flow_dx_image_px": 0.0,
                    "effective_source_flow_dy_image_px": 0.0,
                    "source_metadata": row,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                rebuilt += 1

            clean_input = Path(row.get("clean_path", ""))
            if clean_input.is_file():
                clean_output = output_image_path(directory, "clean", run_id)
                clean_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clean_input, clean_output)
                write_record(directory, "clean", run_id, {
                    "run_id": run_id, "role": "clean", "dataset": args.dataset,
                    "method": "TR", "input_path": str(clean_input),
                    "output_path": str(clean_output), "prompt": row.get("prompt", ""),
                    "prompt_id": row.get("prompt_id", ""), "prompt_source": "metadata",
                    "pixel_shift_dx_px": 0, "pixel_shift_dy_px": 0,
                    "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
                    "effective_source_flow_dx_image_px": 0.0,
                    "effective_source_flow_dy_image_px": 0.0,
                    "source_metadata": row,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        rebuild_records_jsonl(directory)
        result = _evaluate_pixel_shift_directory(args, directory)
        result["rebuilt_watermarked_count"] = rebuilt
        summaries[str(magnitude)] = result

    result = {"workflow": "rebuild_pixel_shift", "magnitudes": summaries}
    (args.output_dir / "pixel_shift_rebuild_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


# ===========================================================================
# CLI
# ===========================================================================
def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workflow", default="evaluate",
        choices=["evaluate", "pixel-shift", "rebuild-pixel-shift"],
        help=("evaluate an existing run (default); generate/evaluate a pixel-space "
              "shift cohort; or rebuild/evaluate an existing cohort"),
    )
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Run directory for evaluate; output root for pixel-shift workflows")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", nargs="+",
                   choices=["quality", "detector", "fid", "clip", "lpips"],
                   default=None,
                   help="Defaults to quality/detector, or all stages for pixel-shift workflows")
    p.add_argument("--allow-missing-metrics", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    # The workflow options mirror RAVEN generation settings.  Metadata is
    # streamed and only one image/pipeline result is retained at a time.
    p.add_argument("--metadata", type=Path, default=None,
                   help="CSV with run_id, watermarked_path, clean_path, and prompt")
    p.add_argument("--dataset", default="diffusiondb")
    p.add_argument("--magnitudes", type=int, nargs="+", default=[16, 24, 32])
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many rows (streamed incrementally)")
    p.add_argument("--eval-only", action="store_true",
                   help="For pixel-shift, skip generation and evaluate existing outputs")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    p.add_argument("--model-revision",
                   default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--strength", type=float, default=0.15)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--sampling", choices=["nearest", "bilinear"], default="nearest")
    p.add_argument("--warp-mode", default="raven_paper_nfpa_gap_fill")
    p.add_argument("--padding-mode", default="reflection")
    p.add_argument("--color-transfer", choices=["aligned", "none"], default="aligned")
    p.add_argument("--view-guided-attention", type=_parse_bool, default=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--save-intermediates", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _workflow_exit_code(result: dict[str, Any], allow_missing_metrics: bool) -> int:
    """Return the worst per-magnitude evaluation exit code."""
    magnitudes = result.get("magnitudes", {})
    if not magnitudes:
        return 1
    return max(
        determine_exit_code(summary, allow_missing_metrics=allow_missing_metrics)
        for summary in magnitudes.values()
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")
    if args.workflow == "evaluate" and not args.output_dir.is_dir():
        logger.error("output-dir does not exist: %s", args.output_dir)
        return 1
    try:
        if args.workflow == "pixel-shift":
            result = run_pixel_shift_workflow(args)
            exit_code = _workflow_exit_code(result, args.allow_missing_metrics)
        elif args.workflow == "rebuild-pixel-shift":
            result = rebuild_pixel_shift_records(args)
            exit_code = _workflow_exit_code(result, args.allow_missing_metrics)
        else:
            result = run_evaluation(args.output_dir, device=args.device,
                                    stages=args.stages,
                                    allow_missing_metrics=args.allow_missing_metrics)
            exit_code = determine_exit_code(
                result, allow_missing_metrics=args.allow_missing_metrics)
    except Exception:
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
