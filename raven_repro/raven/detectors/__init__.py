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

ROW_NONZERO_STATUSES = frozenset({
    ROW_STATUS_FAILED_MISSING_IMAGE,
    ROW_STATUS_FAILED_MISSING_STATE,
    ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_SCORING,
})

# Stage-level status
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
STATUS_SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"
STATUS_FAILED_MISSING_REQUIRED_STATE = "failed_missing_required_state"
STATUS_FAILED_MISSING_DEPENDENCY = "failed_missing_dependency"
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

# Never allowable — always nonzero
NONZERO_STATUSES = frozenset({
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    STATUS_FAILED_STATE_VALIDATION,
    STATUS_FAILED_SCORING,
    STATUS_FAILED_INTERNAL_ERROR,
    STATUS_COMPLETED_WITH_ERRORS,
})

STAGE_NONZERO_STATUSES = frozenset({*NONZERO_STATUSES, *ALLOWABLE_STATUSES})


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
