"""RID / HSTR / HSQR (Fourier) detector adapter.

Delegates to the canonical bundle loading and scoring in
``extract_verification_scores.py``.  Canonical score = ``-raw_l1``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

FOURIER_METHODS = frozenset({"RID", "HSTR", "HSQR"})


def describe_required_artifacts() -> list[str]:
    return [
        "<prefix>_bundle_dir (directory with manifest.json for RID/HSTR/HSQR)",
        "<prefix>_bundle_config_sha256",
        "<prefix>_selected_pattern_sha256",
        "<prefix>_mask_sha256",
        "<prefix>_key_index",
        "Stable Diffusion pipe for inversion",
    ]


def _ensure_eval_bench_in_path():
    repo = Path(__file__).resolve().parents[3]
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)


def load_state(records: list[dict[str, Any]], device: str,
               method: str = "RID") -> dict[str, Any] | None:
    """Load Fourier (RID/HSTR/HSQR) provider from cohort bundle.

    Without real bundle, returns None.
    """
    import torch

    _ensure_eval_bench_in_path()
    method = method.upper()

    try:
        from eval_bench_wm.utils.wm.ringid_provider import RingIDProvider
        from eval_bench_wm.utils.wm.hstr_provider import HSTRProvider
        from eval_bench_wm.utils.wm.hsqr_provider import HSQRProvider
        from eval_bench_wm.utils.wm import sfw_bundle
    except ImportError:
        return None

    first = records[0] if records else {}
    prefix = method.lower()
    bundle_dir = first.get(f"{prefix}_bundle_dir", "")
    if not bundle_dir or not Path(bundle_dir).is_dir():
        return None

    try:
        device_obj = torch.device(device)
        bundle = sfw_bundle.SfwBundle.load(Path(bundle_dir))

        if method == "RID":
            provider = RingIDProvider.from_bundle(bundle, device=device_obj)
        elif method == "HSTR":
            provider = HSTRProvider.from_bundle(bundle, device=device_obj)
        elif method == "HSQR":
            provider = HSQRProvider.from_bundle(bundle, device=device_obj)
        else:
            return None

        return {
            "provider": provider,
            "device_obj": device_obj,
            "method": method,
            "score_direction": "higher_is_watermarked",
            "score_definition": f"{prefix}_score = -raw_l1",
        }
    except Exception:
        return None


def score_image(provider_info: dict[str, Any], image_path: str,
                steps: int = 50) -> dict[str, Any] | None:
    """Score one image using the Fourier provider.

    Canonical score = ``-raw_l1`` (higher is watermarked).
    Without real bundle, returns None.
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

        raw_l1 = float(result.get("l1_dist", [0])[0])
        canonical = -raw_l1

        return {
            "raw_score": raw_l1,
            "canonical_score": canonical,
            "raw_l1": raw_l1,
            "score_direction": "higher_is_watermarked (canonical = -raw_l1)",
        }
    except Exception:
        return None


def aggregate(detector_rows: list[dict[str, Any]],
              method: str = "RID") -> dict[str, Any]:
    """Aggregate Fourier detector rows.

    Mirrors ``evaluate_verification.fourier_report``.
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
        "method": method,
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "score_type": f"{method.lower()}_score",
        "score_direction": "higher_is_watermarked",
        "score_definition": f"canonical_score = -raw_l1",
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
            "watermarked_roc_auc": summary.watermarked_auc,
            "attacked_roc_auc": summary.attacked_auc,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
