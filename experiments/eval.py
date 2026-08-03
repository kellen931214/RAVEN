#!/usr/bin/env python3
"""Offline RAVEN evaluation.

Reads ``config.json``, ``records.jsonl``, and per-sample ``output.png`` files
produced by ``main.py``.  Runs quality metrics, detector evaluation, FID, and
CLIP — all without importing or initializing ``RavenPipeline``.

    python3 experiments/eval.py --output-dir /tmp/run --device cuda

Detector cohort model
---------------------
Attack roles (watermarked, clean) are attack input identities.  Detector
evaluation uses a separate cohort model:

    original_watermarked  → detector on input_path (watermarked input image)
    attacked_watermarked  → detector on output.png (attacked output)
    original_clean        → detector on clean input_path
    attacked_clean        → detector on clean output.png

Results are written to ``evaluation/detector_records.jsonl``.

This module must NOT:
* Import ``RavenPipeline``.
* Execute inversion, denoising, warp, attention, or color transfer.
* Modify any attack output.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.experiment_io import (  # noqa: E402
    config_path,
    detector_records_path,
    evaluation_dir,
    output_image_path,
    read_config,
    read_records_jsonl,
    record_path,
)
from raven.metrics import (  # noqa: E402
    canonical_watermark_score,
    pair_quality_metrics,
    summarize_detection,
)

logger = logging.getLogger("raven.eval")


# ===========================================================================
# Detector cohort model
# ===========================================================================
# Maps attack input roles to detector evaluation cohorts.
DETECTOR_COHORTS = {
    "watermarked": {
        "original": {
            "evaluation_cohort": "original_watermarked",
            "image_source": "input",   # read from record["input_path"]
        },
        "attacked": {
            "evaluation_cohort": "attacked_watermarked",
            "image_source": "output",  # read from output.png
        },
    },
    "clean": {
        "original": {
            "evaluation_cohort": "original_clean",
            "image_source": "input",
        },
        "attacked": {
            "evaluation_cohort": "attacked_clean",
            "image_source": "output",
        },
    },
}


def _resolve_image_path(rec: dict[str, Any], source: str,
                         output_dir: str | Path) -> Path:
    if source == "input":
        return Path(rec.get("input_path", ""))
    elif source == "output":
        return output_image_path(output_dir, rec.get("role", "watermarked"),
                                  str(rec["run_id"]))
    raise ValueError(f"Unknown image source: {source}")


# ===========================================================================
# Quality stage
# ===========================================================================
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
                "run_id": run_id, "role": role,
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
                "run_id": run_id, "role": role,
                "quality_available": True,
                **metrics,
            })
        except Exception as exc:
            results.append({
                "run_id": run_id, "role": role,
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


# ===========================================================================
# Detector execution — orchestration layer
# ===========================================================================
def _build_detector_image_index(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Build the list of (run_id, evaluation_cohort, image_path) entries.

    For each attack record, emits original (input) and attacked (output) rows.
    """
    index: list[dict[str, Any]] = []
    for rec in records:
        run_id = str(rec["run_id"])
        role = rec.get("role", "watermarked")
        cohorts = DETECTOR_COHORTS.get(role, {})
        for variant, info in cohorts.items():
            image_path = _resolve_image_path(rec, info["image_source"], output_dir)
            index.append({
                "run_id": run_id,
                "source_role": role,
                "evaluation_cohort": info["evaluation_cohort"],
                "image_path": str(image_path),
                "image_source": info["image_source"],
                "method": rec.get("method", ""),
                "prompt": rec.get("prompt", ""),
                "prompt_id": rec.get("prompt_id", ""),
            })
    return index


def _try_load_provider(method: str, records: list[dict[str, Any]],
                        device: str) -> tuple[Any, dict[str, Any]] | None:
    """Attempt to load a method-specific provider from available state.

    Returns ``(provider, provider_info)`` on success, ``None`` on failure.
    Provider info documents the artifacts that were loaded.
    """
    try:
        if method == "TR":
            from utils.wm.tr_provider import TrProvider
            from utils.pipe import pipe_utils

            pipe = pipe_utils.get_pipe_provider(
                pretrained_model_name_or_path="RedbeardNZ/stable-diffusion-2-1-base",
                resolution=512, device=torch.device(device),
                eager_loading=False, schedulers_name="DDIM", disable_tqdm=True,
            )
            latent_shape = pipe.get_latent_shape()
            provider = TrProvider(
                latent_shape=latent_shape, dtype=pipe.get_dtype(),
                device=torch.device(device),
                w_seed=999999, w_channel=3, w_radius=10,
                w_pattern="ring", w_mask_shape="circle",
                w_measurement="l1_complex", w_injection="complex",
            )
            return provider, {"pipe": pipe, "provider_type": "TrProvider"}

        # Methods that require per-cohort bundle/state — not available without data.
        if method in {"GM", "T2S", "GS", "RID", "HSTR", "HSQR"}:
            return None

    except ImportError as exc:
        logger.debug("Cannot load provider for %s: %s", method, exc)
        return None
    except Exception as exc:
        logger.debug("Provider init failed for %s: %s", method, exc)
        return None

    return None


def _score_image(method: str, provider: Any, provider_info: dict[str, Any],
                  image_path: str, torch_module) -> dict[str, Any] | None:
    """Run one image through a method-specific detector. Returns score dict or None."""
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        if method == "TR":
            pipe = provider_info["pipe"]
            with torch_module.no_grad():
                inversion = provider.invert_images(
                    image, pipe_provider_target=pipe, num_inference_steps=50,
                )
                recovered = inversion["zT_torch"]
                result = provider.get_accuracies(recovered)
                raw = float(result["p_values"][0])
                import scipy.stats
                recovered_fft = torch_module.fft.fftshift(
                    torch_module.fft.fft2(recovered), dim=(-1, -2))
                mask = provider.watermarking_mask[0]
                target = provider.gt_patch[0][mask].flatten()
                target_cat = torch_module.concatenate([target.real, target.imag])
                for latent_fft in recovered_fft:
                    observed = latent_fft[mask].flatten()
                    observed_cat = torch_module.concatenate([observed.real, observed.imag])
                    sigma = observed_cat.std()
                    log_p = float(scipy.stats.ncx2.logcdf(
                        ((observed_cat - target_cat) / sigma).square().sum().item(),
                        df=len(target_cat),
                        nc=(target_cat.square() / sigma.square()).sum().item(),
                    ))
                canonical = -log_p / math.log(10.0) if math.isfinite(log_p) else float("inf")
                return {
                    "raw_score": raw,
                    "canonical_score": canonical,
                    "tr_log_p": log_p,
                }
        return None
    except Exception as exc:
        logger.debug("Score failed for %s: %s", image_path, exc)
        return None


def _detector_available(method: str) -> tuple[bool, str]:
    """Check whether a detector can run without real provider state.

    Returns ``(available, reason)``.
    """
    # Only TR can run without external data (uses built-in defaults).
    # All other methods need per-cohort state (bundles, secrets, state files).
    if method == "TR":
        return True, "TR provider can be constructed from defaults"
    return False, (
        f"{method} requires per-cohort provider state "
        f"(bundle/secret/state files) not available on this machine"
    )


def evaluate_detector(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    method: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run detector on all cohorts and write ``evaluation/detector_records.jsonl``.

    When provider state is unavailable, returns a structured unavailable result
    documenting exactly which artifacts are needed.
    """
    output_dir = Path(output_dir)
    eval_dir = evaluation_dir(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    can_run, reason = _detector_available(method)
    if not can_run:
        return {
            "stage": "detector",
            "method": method,
            "available": False,
            "reason": reason,
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    image_index = _build_detector_image_index(records, output_dir)
    if not image_index:
        return {
            "stage": "detector",
            "method": method,
            "available": False,
            "reason": "No images to score. Run attack first.",
        }

    import torch
    provider_result = _try_load_provider(method, records, device)
    if provider_result is None:
        return {
            "stage": "detector",
            "method": method,
            "available": False,
            "reason": f"Provider for {method} could not be initialized.",
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    provider, provider_info = provider_result
    detector_rows: list[dict[str, Any]] = []

    for entry in image_index:
        score = _score_image(method, provider, provider_info,
                              entry["image_path"], torch)
        row = {
            "run_id": entry["run_id"],
            "source_role": entry["source_role"],
            "evaluation_cohort": entry["evaluation_cohort"],
            "image_path": entry["image_path"],
            "method": method,
            "status": "scored" if score else "failed",
        }
        if score:
            row.update(score)
        else:
            row["error"] = "scoring failed"
        detector_rows.append(row)

    # Write detector_records.jsonl
    det_path = detector_records_path(output_dir)
    tmp = det_path.with_name(f".detector_records.jsonl.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in detector_rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
    os.replace(tmp, det_path)

    # Aggregate by cohort
    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != "scored":
            continue
        cohort = row["evaluation_cohort"]
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))

    aggregate: dict[str, Any] = {
        "stage": "detector",
        "method": method,
        "available": True,
        "scored_count": sum(1 for r in detector_rows if r.get("status") == "scored"),
        "failed_count": sum(1 for r in detector_rows if r.get("status") != "scored"),
        "cohorts": {c: {"count": len(v)} for c, v in cohorts.items()},
    }

    # Detection summary using original_clean as negative set
    clean_scores = cohorts.get("original_clean", [])
    watermarked_scores = cohorts.get("original_watermarked", [])
    attacked_scores = cohorts.get("attacked_watermarked", [])

    if clean_scores and watermarked_scores and attacked_scores:
        try:
            summary = summarize_detection(
                clean_scores, watermarked_scores, attacked_scores,
                target_fpr=0.01,
            )
            aggregate["detection_summary"] = {
                "target_fpr": 0.01,
                "clean_calibrated_threshold": summary.calibration.threshold,
                "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
                "original_watermarked_tpr": summary.watermarked_tpr,
                "attacked_watermarked_tpr_at_original_threshold": summary.attacked_tpr,
                "watermarked_roc_auc": summary.watermarked_auc,
                "attacked_roc_auc": summary.attacked_auc,
                "attack_success": 1.0 - summary.attacked_tpr,
            }
        except Exception as exc:
            aggregate["detection_summary_error"] = f"{type(exc).__name__}: {exc}"

    # TR-specific: check for attacked-clean recalibration
    if method == "TR":
        attacked_clean_scores = cohorts.get("attacked_clean", [])
        if attacked_clean_scores and clean_scores:
            try:
                recal_summary = summarize_detection(
                    attacked_clean_scores, watermarked_scores, attacked_scores,
                    target_fpr=0.01,
                )
                aggregate["tr_recalibrated"] = {
                    "recalibrated_metrics_available": True,
                    "attacked_clean_recalibrated_threshold": recal_summary.calibration.threshold,
                    "attacked_clean_actual_fpr": recal_summary.calibration.actual_fpr,
                    "attacked_tpr_at_recalibrated_threshold": recal_summary.attacked_tpr,
                }
            except Exception as exc:
                aggregate["tr_recalibrated"] = {
                    "recalibrated_metrics_available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            aggregate["tr_recalibrated"] = {
                "recalibrated_metrics_available": False,
                "reason": "No attacked-clean scores — recalibration not possible.",
            }

    return aggregate


# ===========================================================================
# FID stage
# ===========================================================================
def evaluate_fid(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute FID between original watermarked inputs and attacked outputs.

    Uses ``clean_fid`` from ``raven.quality``.  Staging is temporary.
    Without real data, returns unavailable.
    """
    try:
        from raven.quality import clean_fid, FID_PRIMARY_MODE
    except ImportError:
        return {
            "stage": "fid",
            "available": False,
            "reason": "clean-fid not installed.",
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    import tempfile
    output_dir = Path(output_dir)

    # Collect paired (reference, attacked) paths
    pairs: list[tuple[Path, Path]] = []
    for rec in records:
        role = rec.get("role", "watermarked")
        input_path = Path(rec.get("input_path", ""))
        out_path = output_image_path(output_dir, role, str(rec["run_id"]))
        if input_path.is_file() and out_path.is_file():
            pairs.append((input_path, out_path))

    if len(pairs) < 2:
        return {
            "stage": "fid",
            "available": False,
            "reason": f"Need at least 2 paired images for FID, got {len(pairs)}.",
        }

    # Stage images in temporary directories
    tmpdir = Path(tempfile.mkdtemp(prefix="raven_fid_"))
    try:
        ref_dir = tmpdir / "reference"
        att_dir = tmpdir / "attacked"
        ref_dir.mkdir()
        att_dir.mkdir()

        width = max(6, max(len(str(rec["run_id"])) for rec in records))
        staged: list[dict[str, Any]] = []
        for i, (ref_path, att_path) in enumerate(pairs):
            run_id = str(records[i]["run_id"])
            name = f"{int(run_id):0{width}d}.png"
            import shutil
            shutil.copy2(ref_path, ref_dir / name)
            shutil.copy2(att_path, att_dir / name)
            staged.append({
                "run_id": run_id,
                "staged_name": name,
                "reference_path": str(ref_path),
                "attacked_path": str(att_path),
            })

        result = clean_fid(str(ref_dir), str(att_dir), device=device)
        return {
            "stage": "fid",
            "available": True,
            "image_count": len(pairs),
            "fid_value": result.get("value"),
            "mode": FID_PRIMARY_MODE,
            "protocol": result.get("protocol", ""),
            "staged_records": staged,
        }
    except Exception as exc:
        return {
            "stage": "fid",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# CLIP stage
# ===========================================================================
def evaluate_clip(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute CLIP prompt-image cosine similarity.

    Uses ``openclip_text_image_scores`` from ``raven.quality``.
    Image = attacked-watermarked output.png.
    Text = original generation prompt from metadata.
    """
    try:
        from raven.quality import openclip_text_image_scores
    except ImportError:
        return {
            "stage": "clip",
            "available": False,
            "reason": "open_clip_torch not installed.",
            "NOT RUN — DATA UNAVAILABLE": True,
        }

    output_dir = Path(output_dir)
    image_paths: list[str] = []
    prompts: list[str] = []

    for rec in records:
        role = rec.get("role", "watermarked")
        out_path = output_image_path(output_dir, role, str(rec["run_id"]))
        if out_path.is_file():
            image_paths.append(str(out_path))
            # Use original generation prompt from metadata record
            prompts.append(rec.get("prompt", ""))

    if not image_paths:
        return {
            "stage": "clip",
            "available": False,
            "reason": "No output images with prompts available.",
        }

    if not all(prompts):
        return {
            "stage": "clip",
            "available": False,
            "reason": "Some records missing prompt — CLIP requires original generation prompts.",
        }

    try:
        result = openclip_text_image_scores(
            image_paths, prompts, device=device,
            model_name="ViT-bigG-14",
            pretrained="laion2b_s39b_b160k",
        )
        return {
            "stage": "clip",
            "available": True,
            "image_count": len(image_paths),
            "model": result.get("model_name", "ViT-bigG-14"),
            "pretrained": result.get("pretrained", "laion2b_s39b_b160k"),
            "metric": result.get("metric", "prompt-image cosine similarity"),
            "mean_score": result.get("mean"),
            "scores": result.get("scores", []),
        }
    except Exception as exc:
        return {
            "stage": "clip",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ===========================================================================
# Main evaluation orchestrator
# ===========================================================================
STAGE_RUNNERS = {
    "quality": lambda records, od, dev, cfg: evaluate_quality(records, od),
    "detector": lambda records, od, dev, cfg: evaluate_detector(
        records, od, cfg.get("method", "TR"), dev,
    ),
    "fid": evaluate_fid,
    "clip": evaluate_clip,
}


def run_evaluation(
    output_dir: str | Path,
    *,
    device: str = "cuda",
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Run all evaluation stages on a completed output directory."""
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

    for stage in stages:
        runner = STAGE_RUNNERS.get(stage)
        if runner is None:
            result["stages"][stage] = {
                "available": False,
                "reason": f"Unknown stage: {stage}",
            }
            continue
        logger.info("Running %s evaluation...", stage)
        try:
            result["stages"][stage] = runner(records, output_dir, device, config)
        except Exception as exc:
            logger.exception("%s evaluation failed", stage)
            result["stages"][stage] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return result


# ===========================================================================
# CLI
# ===========================================================================
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

    result_json = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n", encoding="utf-8")
        logger.info("Evaluation result written to %s", args.output)
    else:
        print(result_json)

    unavailable = [
        stage for stage, info in result["stages"].items()
        if not info.get("available", True)
    ]
    if unavailable:
        logger.warning("Stages not run (data unavailable): %s", ", ".join(unavailable))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
