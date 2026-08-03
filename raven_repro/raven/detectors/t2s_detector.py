"""T2SMark detector adapter.

Every sample has its own portable state.  ``score_image`` requires the
correct ``record`` (matched on run_id + role) so the right state file is used.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "t2s_state_path",
    "t2s_state_sha256",
    "t2s_provider_config_sha256",
    "t2s_protocol_mode",
})


def describe_required_artifacts() -> list[str]:
    return [
        "t2s_state_path (per-sample portable state file)",
        "t2s_state_sha256, t2s_provider_config_sha256",
        "t2s_protocol_mode, t2s_rng_mode",
        "t2s_inversion_mode, t2s_num_inversion_steps",
        "t2s_watermark_id",
        "Stable Diffusion pipe for T2S inversion",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load the T2S inversion pipe and provider modules."""
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm import t2s_provider as t2s_provider_module
        from eval_bench_wm.utils.wm import t2s_inversion as t2s_inversion_module
    except ImportError as exc:
        raise DetectorDependencyError(
            f"T2S dependencies not available: {exc}"
        ) from exc

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
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"T2S pipe init failed: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "pipe": pipe,
        "t2s_provider_module": t2s_provider_module,
        "t2s_inversion_module": t2s_inversion_module,
        "device_obj": device_obj,
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using its per-sample T2S state.

    ``record`` MUST contain t2s_state_path.  Matched on run_id + role.
    """
    import torch
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    if record is None:
        raise DetectorMissingStateError(
            "T2S requires per-sample record with t2s_state_path"
        )

    state_path = record.get("t2s_state_path", "")
    if not state_path or not Path(state_path).is_file():
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')} role={record.get('role')}: "
            f"t2s_state_path not found: {state_path}"
        )

    t2s_provider_mod = provider_info["t2s_provider_module"]
    t2s_inversion_mod = provider_info["t2s_inversion_module"]
    pipe = provider_info["pipe"]

    # Load and validate state
    try:
        state = t2s_provider_mod.T2SWatermarkState.load(Path(state_path))
    except Exception as exc:
        raise DetectorStateValidationError(
            f"T2S state load failed for run_id={record.get('run_id')}: {exc}"
        ) from exc

    recorded_sha = str(record.get("t2s_state_sha256", ""))
    actual_sha = state.state_sha256()
    if recorded_sha and recorded_sha != actual_sha:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: T2S state SHA mismatch: "
            f"recorded={recorded_sha} actual={actual_sha}"
        )

    recorded_wm_id = str(record.get("t2s_watermark_id", ""))
    if recorded_wm_id and recorded_wm_id != state.watermark_id:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: T2S watermark_id mismatch: "
            f"recorded={recorded_wm_id} state={state.watermark_id}"
        )

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    try:
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
    except Exception as exc:
        raise DetectorScoringError(
            f"T2S scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

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
        "t2s_state_sha256": actual_sha,
        "t2s_watermark_id": state.watermark_id,
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
        "score_direction": "higher_is_watermarked",
    }


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate T2S detector rows by cohort.  Row-by-row to avoid zip misalignment."""
    from . import ROW_STATUS_SCORED

    def _cohort(name: str) -> list[dict[str, Any]]:
        return [r for r in detector_rows
                if r.get("evaluation_cohort") == name
                and r.get("status") == ROW_STATUS_SCORED]

    def _bit_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        vals = []
        for r in rows:
            v = r.get("t2s_bit_accuracy")
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(vals),
        }

    original_wm = _cohort("original_watermarked")
    attacked_wm = _cohort("attacked_watermarked")

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored
    required = {"original_watermarked", "attacked_watermarked"}
    present = {r["evaluation_cohort"] for r in detector_rows
               if r.get("status") == ROW_STATUS_SCORED}
    missing = sorted(required - present)

    result: dict[str, Any] = {
        "method": "T2S",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(_cohort(c)) for c in present},
        "missing_cohorts": missing,
        "score_type": "t2s_score_true_key",
        "score_direction": "higher_is_watermarked",
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
    }

    # Original watermarked — row-by-row
    if original_wm:
        bit_stats = _bit_stats(original_wm)
        if bit_stats:
            result["original_watermarked_bit_accuracy"] = bit_stats
        detections = [bool(r.get("t2s_detection_success", False)) for r in original_wm]
        result["original_watermarked_detection_rate"] = (
            sum(detections) / len(detections)
        )
        # Row-by-row: no zip misalignment
        corrupted = 0
        failed_readable = 0
        for r in original_wm:
            det = bool(r.get("t2s_detection_success", False))
            ba = r.get("t2s_bit_accuracy")
            ba_val = float(ba) if ba is not None else None
            if det and ba_val is not None and ba_val < 1.0:
                corrupted += 1
            if not det and ba_val is not None and ba_val == 1.0:
                failed_readable += 1
        result["original_watermarked_message_corrupted"] = corrupted
        result["original_watermarked_detection_failed_but_readable"] = failed_readable

    # Attacked watermarked — row-by-row
    if attacked_wm:
        bit_stats = _bit_stats(attacked_wm)
        if bit_stats:
            result["attacked_watermarked_bit_accuracy"] = bit_stats
        detections = [bool(r.get("t2s_detection_success", False)) for r in attacked_wm]
        result["attacked_watermarked_detection_rate"] = (
            sum(detections) / len(detections)
        )
        corrupted = 0
        failed_readable = 0
        for r in attacked_wm:
            det = bool(r.get("t2s_detection_success", False))
            ba = r.get("t2s_bit_accuracy")
            ba_val = float(ba) if ba is not None else None
            if det and ba_val is not None and ba_val < 1.0:
                corrupted += 1
            if not det and ba_val is not None and ba_val == 1.0:
                failed_readable += 1
        result["attacked_watermarked_message_corrupted"] = corrupted
        result["attacked_watermarked_detection_failed_but_readable"] = failed_readable

    if original_wm and attacked_wm:
        orig_rate = result.get("original_watermarked_detection_rate", 0.0)
        att_rate = result["attacked_watermarked_detection_rate"]
        result["attack_success_rate"] = 1.0 - att_rate

    return result
