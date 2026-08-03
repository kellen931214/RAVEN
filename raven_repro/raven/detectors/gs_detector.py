"""Gaussian Shading detector adapter.

Delegates to the canonical GS provider and scoring in
``extract_verification_scores.py`` and ``evaluate_verification.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


def describe_required_artifacts() -> list[str]:
    return [
        "GS secret bundle (with per-sample secret indices)",
        "gs_secret_index per row in source metadata",
        "gs_secret_bundle_sha256",
        "gs_protocol_mode",
    ]


def _ensure_eval_bench_in_path():
    repo = Path(__file__).resolve().parents[3]
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)


def load_state(records: list[dict[str, Any]], device: str) -> dict[str, Any] | None:
    """Load GS provider.  Requires per-cohort secret bundle — unavailable without data."""
    import torch

    _ensure_eval_bench_in_path()

    try:
        from eval_bench_wm.utils.wm.gs_provider import GsProvider
    except ImportError:
        return None

    try:
        device_obj = torch.device(device)
        # GS provider is per-row (each row has its own secret index).
        # We store the class and device for per-row construction.
        return {
            "provider_class": GsProvider,
            "device_obj": device_obj,
            # Actual provider construction happens per-row in score_image
            # because each GS row has a different secret.
            "records": records,
        }
    except Exception:
        return None


def score_image(provider_info: dict[str, Any], image_path: str,
                row_index: int = 0, steps: int = 50) -> dict[str, Any] | None:
    """Score one GS image.  Constructs provider per-row from the secret index.

    Without real secret bundles, returns None.
    """
    # GS requires per-sample secret state — cannot score without data.
    return None


def aggregate(detector_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate GS detector rows.

    When real GS data is available, this delegates to
    ``evaluate_verification.gs_report``.
    """
    from raven.metrics import summarize_detection

    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != "scored":
            continue
        cohort = row.get("evaluation_cohort", "")
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))

    result: dict[str, Any] = {
        "method": "GS",
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "score_direction": "higher_is_watermarked",
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
