#!/usr/bin/env python3
"""Offline RAVEN evaluation.

Reads ``config.json``, ``records.jsonl``, and per-sample ``output.png`` files
produced by ``main.py``.  Runs quality metrics, detector evaluation, FID, and
CLIP — all without importing or initializing ``RavenPipeline``.

    python3 experiments/eval.py --output-dir /tmp/run --device cuda

Responsibilities
----------------
* Quality: PSNR / SSIM (paired, overlap-aware).
* Detector: method-specific dispatch for all seven watermark families.
* FID: clean-fid between reference and attacked image sets.
* CLIP: prompt-image cosine similarity.
* Aggregate summary.

This module must NOT:
* Import ``RavenPipeline``.
* Execute inversion, denoising, warp, attention, or color transfer.
* Modify any attack output.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.experiment_io import (  # noqa: E402
    config_path,
    output_image_path,
    read_config,
    read_records_jsonl,
    record_path,
)
from raven.metrics import (  # noqa: E402
    canonical_watermark_score,
    calibrate_threshold,
    detection_rate,
    pair_quality_metrics,
    roc_auc,
    summarize_detection,
)

logger = logging.getLogger("raven.eval")


# --------------------------------------------------------------------------- #
# Method dispatch
# --------------------------------------------------------------------------- #
def _canonical_score(method: str, raw_score: float) -> float:
    """Convert a raw provider score to canonical higher-is-watermarked space."""
    method = method.upper()
    # GM bit accuracy and T2S score_true_key are already higher-is-watermarked.
    if method in {"GM", "T2S", "GS"}:
        return float(raw_score)
    # TR, RID, HSTR, HSQR: canonical score = -raw (or -log10(p) for TR).
    if method in {"TR", "RID", "HSTR", "HSQR"}:
        return -float(raw_score)
    raise ValueError(f"Unsupported method for canonical score: {method}")


# --------------------------------------------------------------------------- #
# Quality stage
# --------------------------------------------------------------------------- #
def evaluate_quality(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compute per-sample PSNR/SSIM using the effective source flow overlap."""
    results: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    for rec in records:
        run_id = str(rec["run_id"])
        role = rec.get("role", "watermarked")
        input_path = Path(rec.get("input_path", ""))
        out_path = output_image_path(output_dir, role, run_id)

        if not input_path.is_file() or not out_path.is_file():
            results.append({
                "run_id": run_id,
                "role": role,
                "error": "missing input or output image",
                "quality_available": False,
            })
            continue

        try:
            from PIL import Image

            dx = float(rec.get("effective_source_flow_dx_image_px",
                               rec.get("planned_flow_dx_image_px", 0)))
            dy = float(rec.get("effective_source_flow_dy_image_px",
                               rec.get("planned_flow_dy_image_px", 0)))

            with Image.open(input_path) as ref, Image.open(out_path) as att:
                metrics = pair_quality_metrics(
                    ref.convert("RGB"), att.convert("RGB"), dx, dy,
                )

            psnr = float(metrics.get("overlap_psnr", metrics.get("raw_full_psnr", float("nan"))))
            ssim = float(metrics.get("overlap_ssim", metrics.get("raw_full_ssim", float("nan"))))

            if math.isfinite(psnr):
                psnr_values.append(psnr)
            if math.isfinite(ssim):
                ssim_values.append(ssim)

            results.append({
                "run_id": run_id,
                "role": role,
                "quality_available": True,
                **metrics,
            })
        except Exception as exc:
            results.append({
                "run_id": run_id,
                "role": role,
                "error": f"{type(exc).__name__}: {exc}",
                "quality_available": False,
            })

    return {
        "stage": "quality",
        "count": len(results),
        "available": len(psnr_values),
        "psnr_mean": sum(psnr_values) / len(psnr_values) if psnr_values else None,
        "ssim_mean": sum(ssim_values) / len(ssim_values) if ssim_values else None,
        "per_sample": results,
    }


# --------------------------------------------------------------------------- #
# Detector stage — method-specific adapters
# --------------------------------------------------------------------------- #
# Each adapter receives (records, output_dir, device) and returns a result dict.
# When real provider state is unavailable, it returns {"available": False, "reason": "..."}.

def _get_canonical_scores(
    records: list[dict[str, Any]],
    method: str,
    score_field: str,
) -> dict[str, list[float]]:
    """Extract canonical scores from records.  Returns empty lists if missing."""
    scores: dict[str, list[float]] = {"clean": [], "watermarked": [], "attacked": []}
    for rec in records:
        role = rec.get("role", "watermarked")
        value = rec.get(score_field)
        if value is not None and str(value).strip():
            try:
                scores[role].append(float(value))
            except (ValueError, TypeError):
                pass
    return scores


def evaluate_tr(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Tree-Ring detector evaluation.

    Uses the canonical score from debug_info (``effective_source_flow_*``).
    When attacked-clean records are present, computes recalibrated threshold.
    """
    # TR raw score comes from the pipeline's TR detection or pre-computed scores.
    # In the new pipeline, scores must be pre-computed and stored in records.
    scores = _get_canonical_scores(records, "TR", "canonical_score")
    if not any(scores.values()):
        return {
            "method": "TR",
            "available": False,
            "reason": "No canonical scores in records. Run TR detector first.",
        }

    target_fpr = 0.01
    clean = scores.get("clean", [])
    watermarked = scores.get("watermarked", [])
    attacked = scores.get("attacked", [])

    result: dict[str, Any] = {
        "method": "TR",
        "available": True,
        "target_fpr": target_fpr,
    }

    if watermarked and attacked:
        if clean:
            summary = summarize_detection(clean, watermarked, attacked, target_fpr)
            result["calibration"] = {
                "threshold": summary.calibration.threshold,
                "target_fpr": summary.calibration.target_fpr,
                "actual_fpr": summary.calibration.actual_fpr,
                "false_positives": summary.calibration.false_positives,
                "num_clean": summary.calibration.num_clean,
            }
            result["watermarked_tpr"] = summary.watermarked_tpr
            result["attacked_tpr_at_original_threshold"] = summary.attacked_tpr
            result["watermarked_auc"] = summary.watermarked_auc
            result["attacked_auc"] = summary.attacked_auc
            result["attack_success"] = 1.0 - summary.attacked_tpr
        else:
            result["warning"] = "No clean scores available; threshold calibration skipped."
            result["recalibrated_metrics_available"] = False

    # Check for attacked-clean recalibration
    attacked_clean = [r for r in records if r.get("role") == "clean"]
    if attacked_clean:
        result["attacked_clean_count"] = len(attacked_clean)
        result["recalibrated_metrics_available"] = True
    else:
        result["recalibrated_metrics_available"] = False

    return result


def evaluate_gs(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Gaussian Shading detector evaluation."""
    scores = _get_canonical_scores(records, "GS", "bit_accuracy")
    if not any(scores.values()):
        return {
            "method": "GS",
            "available": False,
            "reason": "No bit accuracy scores in records. Run GS detector first.",
        }

    target_fpr = 0.01
    clean = scores.get("clean", [])
    watermarked = scores.get("watermarked", [])
    attacked = scores.get("attacked", [])

    result: dict[str, Any] = {
        "method": "GS",
        "available": True,
        "target_fpr": target_fpr,
    }

    if watermarked and attacked and clean:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr)
        result["calibration"] = {
            "threshold": summary.calibration.threshold,
            "actual_fpr": summary.calibration.actual_fpr,
        }
        result["watermarked_tpr"] = summary.watermarked_tpr
        result["attacked_tpr"] = summary.attacked_tpr
        result["watermarked_auc"] = summary.watermarked_auc
        result["attacked_auc"] = summary.attacked_auc
        result["attack_success"] = 1.0 - summary.attacked_tpr

    return result


def evaluate_gm(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """GaussMarker detector evaluation.

    Uses ``gm_raw_bit_accuracy`` as the canonical score (higher = watermarked).
    Requires GM bundle state for real detection; without it, only orchestration
    is verified.
    """
    scores: dict[str, list[float]] = {"clean": [], "watermarked": [], "attacked": []}
    for rec in records:
        role = rec.get("role", "watermarked")
        value = rec.get("gm_raw_bit_accuracy")
        if value is not None:
            try:
                scores[role].append(float(value))
            except (ValueError, TypeError):
                pass

    if not any(scores.values()):
        return {
            "method": "GM",
            "available": False,
            "reason": (
                "No gm_raw_bit_accuracy scores in records. "
                "GM bundle state is required for real detection."
            ),
            "required_artifacts": [
                "gm_bundle_dir (with manifest.json, w1.pth, w2.pth)",
                "gm_bundle_config_sha256",
                "gm_w1_file_sha256",
                "gm_w2_file_sha256",
                "gm_watermark_sha256",
                "gm_m_sha256",
                "gm_target_sha256",
                "gm_mask_sha256",
            ],
        }

    target_fpr = 0.01
    clean = scores.get("clean", [])
    watermarked = scores.get("watermarked", [])
    attacked = scores.get("attacked", [])

    result: dict[str, Any] = {
        "method": "GM",
        "available": True,
        "score_type": "gm_raw_bit_accuracy",
        "score_direction": "higher_is_watermarked",
        "target_fpr": target_fpr,
    }

    if watermarked and attacked and clean:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr)
        result["calibration"] = {
            "threshold": summary.calibration.threshold,
            "actual_fpr": summary.calibration.actual_fpr,
        }
        result["watermarked_tpr"] = summary.watermarked_tpr
        result["attacked_tpr"] = summary.attacked_tpr
        result["watermarked_auc"] = summary.watermarked_auc
        result["attacked_auc"] = summary.attacked_auc
        result["attack_success"] = 1.0 - summary.attacked_tpr
    elif watermarked and attacked:
        result["watermarked_mean_score"] = (
            sum(watermarked) / len(watermarked) if watermarked else None
        )
        result["attacked_mean_score"] = (
            sum(attacked) / len(attacked) if attacked else None
        )
        result["warning"] = "No clean scores; threshold not calibrated."

    return result


def evaluate_t2s(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """T2SMark detector evaluation.

    Preserves per-sample bit accuracy, detection under paired-key comparison,
    and message corruption statistics.  Requires T2S state files for real
    detection.
    """
    # Collect T2S-specific fields from records
    bit_accuracies: list[float] = []
    detection_success: list[bool] = []
    message_accuracies: list[float] = []
    true_key_scores: dict[str, list[float]] = {
        "clean": [], "watermarked": [], "attacked": [],
    }

    for rec in records:
        role = rec.get("role", "watermarked")
        for field in ("t2s_bit_accuracy", "bit_accuracy"):
            val = rec.get(field)
            if val is not None:
                try:
                    bit_accuracies.append(float(val))
                except (ValueError, TypeError):
                    pass
                break

        det = rec.get("t2s_detection_success")
        if det is not None:
            detection_success.append(bool(det))

        msg_acc = rec.get("t2s_message_accuracy", rec.get("message_accuracy"))
        if msg_acc is not None:
            try:
                message_accuracies.append(float(msg_acc))
            except (ValueError, TypeError):
                pass

        score = rec.get("t2s_score_true_key")
        if score is not None:
            try:
                true_key_scores[role].append(float(score))
            except (ValueError, TypeError):
                pass

    has_data = bool(bit_accuracies or detection_success or any(true_key_scores.values()))

    if not has_data:
        return {
            "method": "T2S",
            "available": False,
            "reason": (
                "No T2S scores in records. "
                "T2S state files are required for real detection."
            ),
            "required_artifacts": [
                "t2s_state_path (per-sample portable state)",
                "t2s_state_sha256",
                "t2s_provider_config_sha256",
            ],
            "preserved_fields": [
                "per-sample bit accuracy",
                "mean bit accuracy",
                "median bit accuracy",
                "q25 / q75 bit accuracy",
                "detection threshold (paired_key_comparison)",
                "before detection rate",
                "attacked detection rate",
                "attack success rate",
                "detected but message corrupted",
                "detection failed but message readable",
            ],
        }

    result: dict[str, Any] = {
        "method": "T2S",
        "available": True,
        "score_type": "t2s_score_true_key",
        "score_direction": "higher_is_watermarked",
        "decision_rule": "paired_key_comparison (score_true_key > score_control_key)",
    }

    if bit_accuracies:
        import numpy as np
        arr = np.array(bit_accuracies)
        result["bit_accuracy"] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(bit_accuracies),
        }
        # Count corrupted messages (detected but bit_accuracy < 1)
        corrupted = sum(
            1 for det, acc in zip(detection_success, bit_accuracies)
            if det and acc < 1.0
        )
        result["message_corrupted_count"] = corrupted

    if detection_success:
        result["detection_rate"] = sum(detection_success) / len(detection_success)
        result["detection_count"] = len(detection_success)
        result["detected_count"] = sum(detection_success)

    if message_accuracies:
        result["mean_message_accuracy"] = (
            sum(message_accuracies) / len(message_accuracies)
        )

    # Secondary: clean-calibrated threshold on score_true_key
    if true_key_scores["watermarked"] and true_key_scores["attacked"]:
        result["mean_score_true_key_watermarked"] = (
            sum(true_key_scores["watermarked"]) / len(true_key_scores["watermarked"])
        )
        result["mean_score_true_key_attacked"] = (
            sum(true_key_scores["attacked"]) / len(true_key_scores["attacked"])
        )

    return result


def evaluate_rid(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """RingID detector evaluation."""
    return _fourier_eval(records, "RID")


def evaluate_hstr(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """HSTR detector evaluation."""
    return _fourier_eval(records, "HSTR")


def evaluate_hsqr(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """HSQR detector evaluation."""
    return _fourier_eval(records, "HSQR")


def _fourier_eval(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Shared Fourier (RID/HSTR/HSQR) detector evaluation."""
    prefix = method.lower()
    scores: dict[str, list[float]] = {"clean": [], "watermarked": [], "attacked": []}
    for rec in records:
        role = rec.get("role", "watermarked")
        for field in (f"{prefix}_canonical_score", "canonical_score", f"{prefix}_raw_l1"):
            val = rec.get(field)
            if val is not None:
                try:
                    scores[role].append(float(val))
                except (ValueError, TypeError):
                    pass
                break

    if not any(scores.values()):
        return {
            "method": method,
            "available": False,
            "reason": (
                f"No {method} scores in records. "
                f"{method} bundle state is required for real detection."
            ),
            "required_artifacts": [
                f"{prefix}_bundle_dir (with manifest.json)",
                f"{prefix}_bundle_config_sha256",
                f"{prefix}_selected_pattern_sha256",
                f"{prefix}_mask_sha256",
            ],
        }

    target_fpr = 0.01
    clean = scores.get("clean", [])
    watermarked = scores.get("watermarked", [])
    attacked = scores.get("attacked", [])

    result: dict[str, Any] = {
        "method": method,
        "available": True,
        "score_direction": "higher_is_watermarked",
        "target_fpr": target_fpr,
    }

    if watermarked and attacked and clean:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr)
        result["calibration"] = {
            "threshold": summary.calibration.threshold,
            "actual_fpr": summary.calibration.actual_fpr,
        }
        result["watermarked_tpr"] = summary.watermarked_tpr
        result["attacked_tpr"] = summary.attacked_tpr
        result["watermarked_auc"] = summary.watermarked_auc
        result["attacked_auc"] = summary.attacked_auc
        result["attack_success"] = 1.0 - summary.attacked_tpr

    return result


DETECTOR_ADAPTERS: dict[str, Any] = {
    "TR": evaluate_tr,
    "GS": evaluate_gs,
    "GM": evaluate_gm,
    "T2S": evaluate_t2s,
    "RID": evaluate_rid,
    "HSTR": evaluate_hstr,
    "HSQR": evaluate_hsqr,
}


# --------------------------------------------------------------------------- #
# Main evaluation orchestrator
# --------------------------------------------------------------------------- #
def run_evaluation(
    output_dir: str | Path,
    *,
    device: str = "cuda",
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Run all evaluation stages on a completed output directory.

    Returns a dict with per-stage results.  Stages that cannot run due to
    missing data are marked ``available: false`` with a ``reason``.
    """
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
        "output_dir": str(output_dir),
        "method": method,
        "dataset": config.get("dataset", "unspecified"),
        "sample_count": len(records),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }

    # --- Quality ---
    if "quality" in stages:
        logger.info("Running quality evaluation...")
        result["stages"]["quality"] = evaluate_quality(records, output_dir)

    # --- Detector ---
    if "detector" in stages:
        logger.info("Running detector evaluation (method=%s)...", method)
        adapter = DETECTOR_ADAPTERS.get(method)
        if adapter is None:
            result["stages"]["detector"] = {
                "method": method,
                "available": False,
                "reason": f"No detector adapter for method {method}",
            }
        else:
            try:
                result["stages"]["detector"] = adapter(records, output_dir, device)
            except Exception as exc:
                logger.exception("Detector evaluation failed")
                result["stages"]["detector"] = {
                    "method": method,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    # --- FID ---
    if "fid" in stages:
        logger.info("FID evaluation deferred — requires reference image set.")
        result["stages"]["fid"] = {
            "available": False,
            "reason": "FID requires reference and attacked image sets. Use clean-fid directly.",
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    # --- CLIP ---
    if "clip" in stages:
        logger.info("CLIP evaluation deferred — requires OpenCLIP model and prompts.")
        result["stages"]["clip"] = {
            "available": False,
            "reason": "CLIP requires OpenCLIP model. Install open_clip_torch and run separately.",
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory containing config.json, records.jsonl, and samples/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+",
                        choices=["quality", "detector", "fid", "clip"],
                        default=["quality", "detector"],
                        help="Stages to run (default: quality detector)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write evaluation result JSON to this path")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not args.output_dir.is_dir():
        logger.error("output-dir does not exist: %s", args.output_dir)
        return 1

    try:
        result = run_evaluation(
            args.output_dir,
            device=args.device,
            stages=args.stages,
        )
    except Exception as exc:
        logger.exception("Evaluation failed")
        return 1

    # Write result
    result_json = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n", encoding="utf-8")
        logger.info("Evaluation result written to %s", args.output)
    else:
        print(result_json)

    # Check for unavailable stages
    unavailable = [
        stage for stage, info in result["stages"].items()
        if not info.get("available", True)
    ]
    if unavailable:
        logger.warning("Stages not run (data unavailable): %s", ", ".join(unavailable))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
