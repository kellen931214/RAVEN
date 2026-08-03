"""Method-specific detector adapters for unified RAVEN evaluation.

Each method module exposes:

    load_state(records, device) -> provider_info | None
    score_image(provider_info, image_path, **kwargs) -> dict | None
    aggregate(detector_rows) -> dict
    describe_required_artifacts() -> list[str]

``load_state`` returns ``None`` when required provider state (bundles,
secrets, state files) is unavailable.  The caller distinguishes "state not
available" from "initialization failed" by checking the return value and
any exception type.
"""

from __future__ import annotations

from typing import Any

from . import (  # noqa: F401
    fourier_detector,
    gm_detector,
    gs_detector,
    t2s_detector,
    tr_detector,
)

# Method → module mapping
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


# Status taxonomy for evaluation results
STATUS_COMPLETED = "completed"
STATUS_SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"
STATUS_FAILED_MISSING_REQUIRED_STATE = "failed_missing_required_state"
STATUS_FAILED_MISSING_DEPENDENCY = "failed_missing_dependency"
STATUS_FAILED_PROVIDER_INITIALIZATION = "failed_provider_initialization"
STATUS_FAILED_SCORING = "failed_scoring"
STATUS_FAILED_INTERNAL_ERROR = "failed_internal_error"
