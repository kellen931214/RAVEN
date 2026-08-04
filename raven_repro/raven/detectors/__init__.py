"""Method-specific detector adapters for unified RAVEN evaluation.

Unified contract
----------------
Every method module exposes:

    load_state(records, device, **extra) -> provider_info | None
    score_image(provider_info, image_path, *, record, evaluation_entry, steps=50) -> dict
    aggregate(detector_rows, **extra) -> dict
    describe_required_artifacts() -> list[str]
    REQUIRED_METADATA_FIELDS : frozenset[str]

``load_state`` raises specific exceptions, never swallows errors silently.
``score_image`` accepts the same keyword arguments for all 7 methods.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Error taxonomy — never swallow these
# ---------------------------------------------------------------------------
class DetectorError(Exception):
    """Base for all detector-related errors."""


class DetectorMissingStateError(DetectorError):
    """Artifact path does not exist — genuine missing data."""


class DetectorDependencyError(DetectorError):
    """Python package not importable — missing dependency."""


class DetectorProviderInitializationError(DetectorError):
    """Constructor TypeError / bad argument — provider instantiation failed."""


class DetectorStateValidationError(DetectorError):
    """SHA/config mismatch / bad bundle — state exists but is invalid."""


class DetectorScoringError(DetectorError):
    """invert_images / get_accuracies failed — scoring runtime error."""


# ---------------------------------------------------------------------------
# Row-level detector status taxonomy
# ---------------------------------------------------------------------------
ROW_STATUS_SCORED = "scored"
ROW_STATUS_FAILED_MISSING_IMAGE = "failed_missing_image"
ROW_STATUS_FAILED_MISSING_STATE = "failed_missing_state"
ROW_STATUS_FAILED_PROVIDER = "failed_provider"
ROW_STATUS_FAILED_SCORING = "failed_scoring"
ROW_STATUS_FAILED_STATE_VALIDATION = "failed_state_validation"
ROW_STATUS_FAILED_MISSING_DEPENDENCY = "failed_missing_dependency"
ROW_STATUS_FAILED_INTERNAL_ERROR = "failed_internal_error"

ROW_NONZERO_STATUSES = frozenset({
    ROW_STATUS_FAILED_MISSING_IMAGE,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_SCORING,
    ROW_STATUS_FAILED_STATE_VALIDATION,
    ROW_STATUS_FAILED_MISSING_DEPENDENCY,
    ROW_STATUS_FAILED_INTERNAL_ERROR,
})

# Row failure causes — structured, never derived from error strings
FAILURE_CAUSE_INTERNAL_ERROR = "internal_error"
FAILURE_CAUSE_STATE_VALIDATION = "state_validation_error"
FAILURE_CAUSE_PROVIDER_INITIALIZATION = "provider_initialization_error"
FAILURE_CAUSE_SCORING_ERROR = "scoring_error"
FAILURE_CAUSE_MISSING_IMAGE = "missing_image"
FAILURE_CAUSE_MISSING_REQUIRED_STATE = "missing_required_state"
FAILURE_CAUSE_MISSING_DEPENDENCY = "missing_dependency"

# Map row status → failure cause
_ROW_STATUS_TO_FAILURE_CAUSE: dict[str, str] = {
    ROW_STATUS_FAILED_MISSING_IMAGE: FAILURE_CAUSE_MISSING_IMAGE,
    ROW_STATUS_FAILED_MISSING_STATE: FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    ROW_STATUS_FAILED_PROVIDER: FAILURE_CAUSE_PROVIDER_INITIALIZATION,
    ROW_STATUS_FAILED_SCORING: FAILURE_CAUSE_SCORING_ERROR,
    ROW_STATUS_FAILED_STATE_VALIDATION: FAILURE_CAUSE_STATE_VALIDATION,
    ROW_STATUS_FAILED_MISSING_DEPENDENCY: FAILURE_CAUSE_MISSING_DEPENDENCY,
    ROW_STATUS_FAILED_INTERNAL_ERROR: FAILURE_CAUSE_INTERNAL_ERROR,
}

# Known failure causes — used to validate explicit row failure_cause values
KNOWN_FAILURE_CAUSES = frozenset({
    FAILURE_CAUSE_INTERNAL_ERROR,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION,
    FAILURE_CAUSE_SCORING_ERROR,
    FAILURE_CAUSE_MISSING_IMAGE,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_MISSING_DEPENDENCY,
})

# Stage-level status
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
STATUS_SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"
STATUS_FAILED_MISSING_REQUIRED_STATE = "failed_missing_required_state"
STATUS_FAILED_MISSING_DEPENDENCY = "failed_missing_dependency"
STATUS_FAILED_MISSING_IMAGE = "failed_missing_image"
STATUS_FAILED_PROVIDER_INITIALIZATION = "failed_provider_initialization"
STATUS_FAILED_STATE_VALIDATION = "failed_state_validation"
STATUS_FAILED_SCORING = "failed_scoring"
STATUS_FAILED_INTERNAL_ERROR = "failed_internal_error"

# Allowable by --allow-missing-metrics
ALLOWABLE_STATUSES = frozenset({
    STATUS_SKIPPED_INSUFFICIENT_DATA,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_MISSING_DEPENDENCY,
})

# Never allowable — always nonzero regardless of --allow-missing-metrics
NONZERO_STATUSES = frozenset({
    STATUS_FAILED_MISSING_IMAGE,
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_SCORING,
    STATUS_FAILED_INTERNAL_ERROR,
    STATUS_COMPLETED_WITH_ERRORS,
})

STAGE_NONZERO_STATUSES = frozenset({*NONZERO_STATUSES, *ALLOWABLE_STATUSES})

# Failure cause → stage status (for setup failures that skip scoring loop)
_FAILURE_CAUSE_TO_STAGE_STATUS: dict[str, str] = {
    FAILURE_CAUSE_INTERNAL_ERROR: STATUS_FAILED_INTERNAL_ERROR,
    FAILURE_CAUSE_STATE_VALIDATION: STATUS_FAILED_STATE_VALIDATION,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION: STATUS_FAILED_PROVIDER_INITIALIZATION,
    FAILURE_CAUSE_SCORING_ERROR: STATUS_FAILED_SCORING,
    FAILURE_CAUSE_MISSING_IMAGE: STATUS_FAILED_MISSING_IMAGE,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE: STATUS_FAILED_MISSING_REQUIRED_STATE,
    FAILURE_CAUSE_MISSING_DEPENDENCY: STATUS_FAILED_MISSING_DEPENDENCY,
}

# Optional-cohort failure causes that may be silently recorded without
# downgrading the primary report.  Only ``missing_required_state`` qualifies;
# all other causes (scoring_error, missing_image, state_validation, etc.)
# are hard failures regardless of which cohort they occur in.
OPTIONAL_SOFT_FAILURE_CAUSES = frozenset({
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
})

# Precedence order for stage-status reduction (highest first).
# Each entry is (failure_cause, stage_status_when_dominant).
_STAGE_PRECEDENCE: list[tuple[str, str]] = [
    (FAILURE_CAUSE_INTERNAL_ERROR, STATUS_FAILED_INTERNAL_ERROR),
    (FAILURE_CAUSE_STATE_VALIDATION, STATUS_FAILED_STATE_VALIDATION),
    (FAILURE_CAUSE_PROVIDER_INITIALIZATION, STATUS_FAILED_PROVIDER_INITIALIZATION),
    (FAILURE_CAUSE_SCORING_ERROR, STATUS_FAILED_SCORING),
    (FAILURE_CAUSE_MISSING_IMAGE, STATUS_FAILED_MISSING_IMAGE),
    (FAILURE_CAUSE_MISSING_REQUIRED_STATE, STATUS_FAILED_MISSING_REQUIRED_STATE),
    (FAILURE_CAUSE_MISSING_DEPENDENCY, STATUS_FAILED_MISSING_DEPENDENCY),
]


# ---------------------------------------------------------------------------
# Stage-status reducer — single deterministic entry point
# ---------------------------------------------------------------------------
def _failure_cause_for_row(row: dict[str, Any]) -> str | None:
    """Return the structured failure cause for a detector row, or None if scored.

    Explicit ``failure_cause`` on the row takes priority over the
    status-derived mapping.  An unrecognized explicit cause fails closed
    as ``internal_error``.
    """
    status = row.get("status", "")
    if status == ROW_STATUS_SCORED:
        return None

    explicit = row.get("failure_cause")
    if explicit is not None and explicit != "":
        if explicit in KNOWN_FAILURE_CAUSES:
            return explicit
        # Unknown structured cause — fail closed
        return FAILURE_CAUSE_INTERNAL_ERROR

    return _ROW_STATUS_TO_FAILURE_CAUSE.get(
        status, FAILURE_CAUSE_INTERNAL_ERROR,
    )


def _normalize_failure_cause(cause: Any) -> str:
    """Normalize a failure cause string.  Unknown causes fail closed as
    ``internal_error``."""
    if cause in KNOWN_FAILURE_CAUSES:
        return cause
    return FAILURE_CAUSE_INTERNAL_ERROR


def reduce_detector_stage_status(
    detector_rows: list[dict[str, Any]],
    *,
    setup_failure: str | None = None,
    aggregate_failure: str | None = None,
    primary_report_available: bool = False,
    primary_metrics_complete: bool = False,
    optional_failed_count: int = 0,
) -> dict[str, Any]:
    """Determine detector stage status from row-level failure causes.

    Returns a dict with:

    - ``status``: stage-level status string
    - ``dominant_failure_cause``: highest-precedence failure cause (or None)
    - ``status_reducer_reason``: human-readable explanation
    - ``available``: whether any usable output exists
    - ``failure_cause_counts``: count of rows per failure cause
    - ``row_status_counts``: count of rows per row status

    Precedence (highest first):

    1. internal_error
    2. state_validation_error
    3. provider_initialization_error
    4. scoring_error
    5. missing_image
    6. missing_required_state
    7. missing_dependency
    8. completed_with_errors
    9. completed

    ``setup_failure`` and ``aggregate_failure`` participate in the same
    precedence pool as row-level failures — neither unconditionally
    overrides row causes.  An adapter aggregate failure (e.g. GS official
    policy rows that fail validation) must keep the stage from completing.
    """
    # Count row statuses and failure causes
    row_status_counts: dict[str, int] = {}
    failure_cause_counts: dict[str, int] = {}
    scored_count = 0

    for row in detector_rows:
        st = row.get("status", ROW_STATUS_FAILED_SCORING)
        row_status_counts[st] = row_status_counts.get(st, 0) + 1
        cause = _failure_cause_for_row(row)
        if cause is not None:
            failure_cause_counts[cause] = failure_cause_counts.get(cause, 0) + 1
        elif st == ROW_STATUS_SCORED:
            scored_count += 1

    # Normalize setup failure and add to cause counts
    normalized_setup: str | None = None
    if setup_failure is not None:
        normalized_setup = _normalize_failure_cause(setup_failure)
        # Add setup failure to overall cause counts for diagnostics.
        # Use a distinct key to track "setup" vs "row" origin when needed.
        failure_cause_counts[normalized_setup] = (
            failure_cause_counts.get(normalized_setup, 0) + 1)

    # Normalize aggregate failure and add to cause counts.  Unknown causes
    # fail closed as internal_error.
    if aggregate_failure is not None:
        normalized_aggregate = _normalize_failure_cause(aggregate_failure)
        failure_cause_counts[normalized_aggregate] = (
            failure_cause_counts.get(normalized_aggregate, 0) + 1)

    # Find highest-precedence failure cause across BOTH setup and rows
    effective_failure: str | None = None
    present_causes = set(failure_cause_counts.keys())
    for cause, _stage_status in _STAGE_PRECEDENCE:
        if cause in present_causes:
            effective_failure = cause
            break

    # Determine stage status
    if effective_failure is not None:
        # Optional cohort exemption: only applicable when there is NO setup
        # failure, all failures are in optional cohorts, primary metrics
        # are complete, AND every failure cause is soft.
        total_failed = sum(failure_cause_counts.values())
        primary_failed = total_failed - optional_failed_count
        if (normalized_setup is None
                and primary_failed <= 0
                and primary_metrics_complete
                and scored_count > 0):
            hard_causes = set(failure_cause_counts.keys()) - OPTIONAL_SOFT_FAILURE_CAUSES
            if not hard_causes:
                reason_parts = []
                for cause, _st in _STAGE_PRECEDENCE:
                    cnt = failure_cause_counts.get(cause, 0)
                    if cnt > 0:
                        reason_parts.append(f"{cnt} {cause}")
                reason = ", ".join(reason_parts)
                return {
                    "status": STATUS_COMPLETED,
                    "dominant_failure_cause": effective_failure,
                    "status_reducer_reason": (
                        f"all {total_failed} soft failure(s) in optional "
                        f"cohorts; primary metrics complete: {reason}"
                    ),
                    "available": True,
                    "failure_cause_counts": failure_cause_counts,
                    "row_status_counts": row_status_counts,
                }

        # missing_required_state softens to completed_with_errors when
        # there are valid scores AND the primary report is complete
        # AND there is no setup failure (setup failures are never softened).
        if (normalized_setup is None
                and effective_failure == FAILURE_CAUSE_MISSING_REQUIRED_STATE
                and scored_count > 0
                and primary_metrics_complete):
            reason_parts: list[str] = []
            for cause, _st in _STAGE_PRECEDENCE:
                cnt = failure_cause_counts.get(cause, 0)
                if cnt > 0:
                    reason_parts.append(f"{cnt} {cause}")
            reason = ", ".join(reason_parts)
            return {
                "status": STATUS_COMPLETED_WITH_ERRORS,
                "dominant_failure_cause": effective_failure,
                "status_reducer_reason": (
                    f"primary metrics complete with {scored_count} valid "
                    f"score(s); {failure_cause_counts.get(FAILURE_CAUSE_MISSING_REQUIRED_STATE, 0)} "
                    f"missing_required_state failure(s) softened to "
                    f"completed_with_errors: {reason}"
                ),
                "available": True,
                "failure_cause_counts": failure_cause_counts,
                "row_status_counts": row_status_counts,
            }

        stage_status = _FAILURE_CAUSE_TO_STAGE_STATUS[effective_failure]
        reason_parts = []
        for cause, _st in _STAGE_PRECEDENCE:
            cnt = failure_cause_counts.get(cause, 0)
            if cnt > 0:
                reason_parts.append(f"{cnt} {cause}")
        reason = ", ".join(reason_parts)
        dominant = failure_cause_counts.get(effective_failure, 0)
        other = sum(
            v for k, v in failure_cause_counts.items()
            if k != effective_failure
        )
        if other > 0:
            reason = (
                f"{dominant} {effective_failure} takes precedence "
                f"over {other} other failure(s): {reason}"
            )
        else:
            reason = f"{dominant} {effective_failure}: {reason}"
        return {
            "status": stage_status,
            "dominant_failure_cause": effective_failure,
            "status_reducer_reason": reason,
            "available": scored_count > 0,
            "failure_cause_counts": failure_cause_counts,
            "row_status_counts": row_status_counts,
        }

    # No failures — check completeness
    if scored_count == 0:
        return {
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "dominant_failure_cause": None,
            "status_reducer_reason": "no rows scored and no failures recorded",
            "available": False,
            "failure_cause_counts": failure_cause_counts,
            "row_status_counts": row_status_counts,
        }

    if not primary_report_available:
        if primary_metrics_complete:
            return {
                "status": STATUS_COMPLETED_WITH_ERRORS,
                "dominant_failure_cause": None,
                "status_reducer_reason": (
                    "primary report not available but primary metrics "
                    "complete — completed_with_errors"
                ),
                "available": True,
                "failure_cause_counts": failure_cause_counts,
                "row_status_counts": row_status_counts,
            }
        return {
            "status": STATUS_COMPLETED_WITH_ERRORS,
            "dominant_failure_cause": None,
            "status_reducer_reason": (
                "scores exist but primary report not available"
            ),
            "available": True,
            "failure_cause_counts": failure_cause_counts,
            "row_status_counts": row_status_counts,
        }

    return {
        "status": STATUS_COMPLETED,
        "dominant_failure_cause": None,
        "status_reducer_reason": "all required rows scored, primary report complete",
        "available": True,
        "failure_cause_counts": failure_cause_counts,
        "row_status_counts": row_status_counts,
    }


# ---------------------------------------------------------------------------
# CLI exit-code policy — single deterministic entry point
# ---------------------------------------------------------------------------
def stage_status_is_allowable(
    status: str,
    *,
    allow_missing_metrics: bool,
) -> bool:
    """Return True if *status* permits exit 0 under the given allow policy.

    ``--allow-missing-metrics`` only suppresses:

    - ``skipped_insufficient_data``
    - ``failed_missing_required_state``
    - ``failed_missing_dependency``

    All other nonzero statuses remain nonzero regardless of the flag.
    """
    if status == STATUS_COMPLETED:
        return True
    if allow_missing_metrics and status in ALLOWABLE_STATUSES:
        return True
    return False


def determine_exit_code(
    evaluation_result: dict[str, Any],
    *,
    allow_missing_metrics: bool,
) -> int:
    """Compute CLI exit code from evaluation result.

    0 = all required stages completed or allowable under policy.
    Nonzero = at least one stage failed with a non-allowable status.
    """
    stages = evaluation_result.get("stages", {})
    for stage_name, stage_info in stages.items():
        status = stage_info.get("status", STATUS_FAILED_INTERNAL_ERROR)
        if not stage_status_is_allowable(status,
                                         allow_missing_metrics=allow_missing_metrics):
            return 2
    return 0


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------
DETECTOR_MODULES: dict[str, Any] = {}


def _lazy_imports():
    global DETECTOR_MODULES
    if DETECTOR_MODULES:
        return
    from . import tr_detector as _tr
    from . import gs_detector as _gs
    from . import gm_detector as _gm
    from . import t2s_detector as _t2s
    from . import fourier_detector as _fourier

    DETECTOR_MODULES.update({
        "TR": _tr,
        "GS": _gs,
        "GM": _gm,
        "T2S": _t2s,
        "RID": _fourier,
        "HSTR": _fourier,
        "HSQR": _fourier,
    })


def get_detector_module(method: str):
    _lazy_imports()
    mod = DETECTOR_MODULES.get(method.upper())
    if mod is None:
        raise ValueError(f"No detector module for method: {method}")
    return mod
