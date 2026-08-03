"""GaussMarker detector adapter.

Delegates to the canonical GM provider in ``extract_verification_scores.py``.
Uses ``gm_raw_bit_accuracy`` as primary score (higher = watermarked).
Requires GM bundle with manifest.json, w1.pth, w2.pth.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


def describe_required_artifacts() -> list[str]:
    return [
        "gm_bundle_dir (directory with manifest.json, w1.pth, w2.pth)",
        "gm_bundle_config_sha256",
        "gm_w1_file_sha256",
        "gm_w2_file_sha256",
        "gm_watermark_sha256",
        "gm_m_sha256",
        "gm_target_sha256",
        "gm_mask_sha256",
        "gm_protocol_mode",
    ]


def _ensure_eval_bench_in_path():
    repo = Path(__file__).resolve().parents[3]
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)


def load_state(records: list[dict[str, Any]], device: str) -> dict[str, Any] | None:
    """Load GM provider from cohort bundle.

    Without real GM bundle, returns None.
    """
    import torch

    _ensure_eval_bench_in_path()

    try:
        from eval_bench_wm.utils.wm.gm_provider import GmProvider
    except ImportError:
        return None

    # GM needs the cohort bundle directory. The first record should carry the
    # bundle path. In practice, these come from source metadata.
    first = records[0] if records else {}
    bundle_dir = first.get("gm_bundle_dir", "")
    if not bundle_dir or not Path(bundle_dir).is_dir():
        return None

    try:
        device_obj = torch.device(device)
        provider = GmProvider(
            gm_profile=first.get("gm_protocol_mode", ""),
            gm_bundle_dir=bundle_dir,
            gm_create_bundle=False,
            gm_allow_in_memory_state=False,
            device=device_obj,
        )
        return {
            "provider": provider,
            "device_obj": device_obj,
            "score_type": "gm_raw_bit_accuracy",
            "score_direction": "higher_is_watermarked",
        }
    except Exception:
        return None


def score_image(provider_info: dict[str, Any], image_path: str,
                steps: int = 50) -> dict[str, Any] | None:
    """Score one image using GaussMarker provider.

    Without real GM bundle, returns None.
    """
    import torch
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        provider = provider_info["provider"]
        device_obj = provider_info["device_obj"]

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        with torch.no_grad():
            inversion = provider.invert_images(
                image, pipe_provider_target=None, num_inference_steps=steps,
            )
            recovered = inversion["zT_torch"]
            result = provider.get_accuracies(recovered)

        gm_bit_accuracy = float(result.get("gm_raw_bit_accuracy", 0))
        gm_ring_l1 = float(result.get("gm_raw_ring_l1", 0))

        return {
            "raw_score": gm_bit_accuracy,
            "canonical_score": gm_bit_accuracy,  # already higher-is-watermarked
            "gm_raw_bit_accuracy": gm_bit_accuracy,
            "gm_raw_ring_l1": gm_ring_l1,
            "gm_restored_bit_accuracy": result.get("gm_restored_bit_accuracy"),
            "gm_classifier_probability": result.get("gm_classifier_probability"),
        }
    except Exception:
        return None


def aggregate(detector_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate GM detector rows.

    When real GM data is available, this mirrors
    ``evaluate_verification.gm_report``.
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
        "method": "GM",
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
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
            "watermarked_roc_auc": summary.watermarked_auc,
            "attacked_roc_auc": summary.attacked_auc,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
