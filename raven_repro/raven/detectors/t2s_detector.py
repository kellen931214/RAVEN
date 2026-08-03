"""T2SMark detector adapter.

Delegates to the canonical T2S per-sample state loading and scoring in
``extract_verification_scores.py``.  Every sample carries its own portable
state; there is no cohort-wide provider.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


def describe_required_artifacts() -> list[str]:
    return [
        "t2s_state_path (per-sample portable state file, one per run_id)",
        "t2s_state_sha256",
        "t2s_provider_config_sha256",
        "t2s_protocol_mode",
        "t2s_rng_mode",
        "t2s_inversion_mode",
        "t2s_num_inversion_steps",
        "t2s_watermark_id",
        "Stable Diffusion pipe for T2S inversion",
    ]


def _ensure_eval_bench_in_path():
    repo = Path(__file__).resolve().parents[3]
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)


def load_state(records: list[dict[str, Any]], device: str) -> dict[str, Any] | None:
    """Load the T2S inversion pipe and provider module.

    T2S state is per-sample; the actual decoding happens in ``score_image``.
    Without real T2S state files, returns None.
    """
    import torch

    _ensure_eval_bench_in_path()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm import t2s_provider as t2s_provider_module
        from eval_bench_wm.utils.wm import t2s_inversion as t2s_inversion_module
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
        return {
            "pipe": pipe,
            "t2s_provider_module": t2s_provider_module,
            "t2s_inversion_module": t2s_inversion_module,
            "device_obj": device_obj,
            "records": records,
        }
    except Exception:
        return None


def score_image(provider_info: dict[str, Any], image_path: str,
                row: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any] | None:
    """Score one image using its per-sample T2S state.

    Each T2S sample has a unique state file.  ``row`` must contain
    ``t2s_state_path``.  Without real state files, returns None.
    """
    if row is None:
        return None

    state_path = row.get("t2s_state_path", "")
    if not state_path or not Path(state_path).is_file():
        return None

    import torch
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        t2s_provider_mod = provider_info["t2s_provider_module"]
        t2s_inversion_mod = provider_info["t2s_inversion_module"]
        pipe = provider_info["pipe"]

        state = t2s_provider_mod.T2SWatermarkState.load(Path(state_path))

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        with torch.no_grad():
            zT = t2s_inversion_mod.invert_image(
                pipe, image,
                inversion_mode=state.inversion_mode,
                num_inversion_steps=state.num_inversion_steps,
                benchmark_num_inference_steps=state.num_inference_steps,
            )
            accuracies = t2s_provider_mod.T2SProvider.accuracies_for_state(
                state, {"zT_torch": zT},
            )

        return {
            "raw_score": float(accuracies.get("t2s_score_true_key", 0)),
            "canonical_score": float(accuracies.get("t2s_score_true_key", 0)),
            "t2s_score_true_key": float(accuracies.get("t2s_score_true_key", 0)),
            "t2s_score_control_key": float(accuracies.get("t2s_score_control_key", 0)),
            "t2s_score_margin": float(accuracies.get("t2s_score_margin", 0)),
            "t2s_detection_success": bool(accuracies.get("detection_success", False)),
            "t2s_key_accuracy": accuracies.get("key_accuracy"),
            "t2s_message_accuracy": accuracies.get("message_accuracy"),
            "t2s_bit_accuracy": accuracies.get("message_accuracy"),
            "t2s_state_sha256": state.state_sha256(),
            "t2s_watermark_id": state.watermark_id,
        }
    except Exception:
        return None


def aggregate(detector_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate T2S detector rows, preserving per-sample statistics.

    Mirrors ``evaluate_verification.t2s_report``.
    """
    import numpy as np

    # Per-cohort extraction
    def _cohort_rows(cohort_name: str) -> list[dict[str, Any]]:
        return [r for r in detector_rows
                if r.get("evaluation_cohort") == cohort_name
                and r.get("status") == "scored"]

    original_wm = _cohort_rows("original_watermarked")
    attacked_wm = _cohort_rows("attacked_watermarked")

    result: dict[str, Any] = {
        "method": "T2S",
        "score_type": "t2s_score_true_key",
        "score_direction": "higher_is_watermarked",
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
    }

    # Original watermarked stats
    if original_wm:
        bit_accs = [
            float(r["t2s_bit_accuracy"]) for r in original_wm
            if r.get("t2s_bit_accuracy") is not None
        ]
        detections = [
            bool(r.get("t2s_detection_success", False)) for r in original_wm
        ]
        if bit_accs:
            arr = np.array(bit_accs)
            result["original_watermarked_bit_accuracy"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(bit_accs),
            }
        if detections:
            result["original_watermarked_detection_rate"] = (
                sum(detections) / len(detections)
            )
            corrupted = sum(
                1 for d, a in zip(detections, bit_accs) if d and a < 1.0
            )
            failed_readable = sum(
                1 for d, a in zip(detections, bit_accs) if not d and a == 1.0
            )
            result["original_watermarked_message_corrupted"] = corrupted
            result["original_watermarked_detection_failed_but_readable"] = failed_readable

    # Attacked watermarked stats
    if attacked_wm:
        bit_accs = [
            float(r["t2s_bit_accuracy"]) for r in attacked_wm
            if r.get("t2s_bit_accuracy") is not None
        ]
        detections = [
            bool(r.get("t2s_detection_success", False)) for r in attacked_wm
        ]
        if bit_accs:
            arr = np.array(bit_accs)
            result["attacked_watermarked_bit_accuracy"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(bit_accs),
            }
        if detections:
            result["attacked_watermarked_detection_rate"] = (
                sum(detections) / len(detections)
            )
            corrupted = sum(
                1 for d, a in zip(detections, bit_accs) if d and a < 1.0
            )
            failed_readable = sum(
                1 for d, a in zip(detections, bit_accs) if not d and a == 1.0
            )
            result["attacked_watermarked_message_corrupted"] = corrupted
            result["attacked_watermarked_detection_failed_but_readable"] = failed_readable

        if original_wm:
            orig_rate = result.get("original_watermarked_detection_rate", 0)
            att_rate = result["attacked_watermarked_detection_rate"]
            result["attack_success_rate"] = 1.0 - att_rate

    return result
