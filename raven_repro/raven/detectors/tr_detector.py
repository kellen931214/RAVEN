"""Tree-Ring detector adapter.

Delegates to the canonical TR scoring in ``extract_verification_scores.py``.
Does NOT re-implement FFT / non-central chi-square / -log10(p) math.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


# Re-export the canonical TR scoring helpers from extract_verification_scores.py.
# We import the module once and call its functions directly.
_extract_module = None


def _get_extract_module():
    global _extract_module
    if _extract_module is not None:
        return _extract_module

    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # Add eval_bench_wm to path (TR provider lives there)
    eval_bench = repo / "eval_bench_wm"
    if str(eval_bench) not in sys.path:
        sys.path.insert(0, str(eval_bench))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _extract_module = mod
    return mod


def describe_required_artifacts() -> list[str]:
    return [
        "TR provider parameters (from cohort metadata or eval_protocol defaults)",
        "Stable Diffusion pipe provider (pipe_utils.get_pipe_provider)",
    ]


def load_state(records: list[dict[str, Any]], device: str) -> dict[str, Any] | None:
    """Load TR provider and pipe.  Returns provider_info or None."""
    import torch

    try:
        mod = _get_extract_module()
    except Exception:
        return None

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.tr_provider import TrProvider
    except ImportError:
        return None

    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path="RedbeardNZ/stable-diffusion-2-1-base",
            resolution=512,
            device=device_obj,
            eager_loading=False,
            schedulers_name="DDIM",
            disable_tqdm=True,
        )
        latent_shape = pipe.get_latent_shape()

        # Build TR provider kwargs from records or defaults.
        # In a full implementation these come from cohort metadata.
        first = records[0] if records else {}
        kwargs = {
            "w_seed": int(first.get("w_seed", 999999)),
            "w_channel": int(first.get("w_channel", 3)),
            "w_radius": int(first.get("w_radius", 10)),
            "w_pattern": str(first.get("w_pattern", "ring")),
            "w_mask_shape": str(first.get("w_mask_shape", "circle")),
            "w_measurement": str(first.get("w_measurement", "l1_complex")),
            "w_injection": str(first.get("w_injection", "complex")),
        }

        provider = TrProvider(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device_obj,
            **kwargs,
        )
        return {
            "provider": provider,
            "pipe": pipe,
            "extract_module": mod,
            "provider_kwargs": kwargs,
            "device_obj": device_obj,
        }
    except Exception:
        return None


def score_image(provider_info: dict[str, Any], image_path: str,
                steps: int = 50) -> dict[str, Any] | None:
    """Score one image using the canonical TR detection path.

    Delegates to ``extract_verification_scores.evaluate_image``.
    """
    import torch
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        provider = provider_info["provider"]
        pipe = provider_info["pipe"]
        mod = provider_info["extract_module"]

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        result = mod.evaluate_image(torch, provider, pipe, path, steps)

        raw = mod.raw_score("TR", result)
        canonical = mod.canonical_score("TR", raw, result)

        score: dict[str, Any] = {
            "raw_score": raw,
            "canonical_score": canonical,
        }

        # Preserve formal TR diagnostic fields
        diagnostics = result.get("p_value_diagnostics") or []
        if diagnostics:
            d = diagnostics[0]
            score["tr_log_p"] = d.get("log_p")
            score["tr_sigma"] = d.get("sigma")
            score["tr_lambda"] = d.get("lambda")
            score["tr_statistic"] = d.get("statistic")
            score["tr_df"] = d.get("df")
            score["tr_p_underflow"] = d.get("p_underflow", False)

        return score
    except Exception:
        return None


def aggregate(detector_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate TR detector rows across cohorts."""
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
        "method": "TR",
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_comparison_operator": ">=",
            "original_clean_threshold": summary.calibration.threshold,
            "original_clean_target_fpr": summary.calibration.target_fpr,
            "original_clean_actual_fpr": summary.calibration.actual_fpr,
            "original_clean_false_positives": summary.calibration.false_positives,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr_at_original_threshold": summary.attacked_tpr,
            "watermarked_roc_auc": summary.watermarked_auc,
            "attacked_roc_auc": summary.attacked_auc,
            "attack_success_at_original_threshold": 1.0 - summary.attacked_tpr,
        }

    # Attacked-clean recalibration
    attacked_clean = cohorts.get("attacked_clean", [])
    if attacked_clean and clean:
        try:
            recal = summarize_detection(
                attacked_clean, watermarked, attacked, target_fpr=0.01)
            result["tr_recalibrated"] = {
                "recalibrated_metrics_available": True,
                "attacked_clean_count": len(attacked_clean),
                "attacked_clean_recalibrated_threshold": recal.calibration.threshold,
                "attacked_clean_target_fpr": recal.calibration.target_fpr,
                "attacked_clean_actual_fpr": recal.calibration.actual_fpr,
                "attacked_clean_false_positives": recal.calibration.false_positives,
                "attacked_watermarked_tpr_at_recalibrated_threshold": recal.attacked_tpr,
                "attack_success_at_recalibrated_threshold": 1.0 - recal.attacked_tpr,
                "recalibrated_roc_auc": recal.attacked_auc,
            }
        except Exception:
            result["tr_recalibrated"] = {
                "recalibrated_metrics_available": False,
                "error": "recalibration failed",
            }
    else:
        result["tr_recalibrated"] = {
            "recalibrated_metrics_available": False,
            "reason": "No attacked-clean scores available.",
        }

    return result
