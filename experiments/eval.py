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
)
from raven.metrics import pair_quality_metrics  # noqa: E402

logger = logging.getLogger("raven.eval")


# ===========================================================================
# Status taxonomy
# ===========================================================================
STATUS_COMPLETED = "completed"
STATUS_SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"
STATUS_FAILED_MISSING_REQUIRED_STATE = "failed_missing_required_state"
STATUS_FAILED_MISSING_DEPENDENCY = "failed_missing_dependency"
STATUS_FAILED_PROVIDER_INITIALIZATION = "failed_provider_initialization"
STATUS_FAILED_SCORING = "failed_scoring"
STATUS_FAILED_INTERNAL_ERROR = "failed_internal_error"

# Stages whose failure/non-completion should cause nonzero exit.
# Quality and detector are required by default; FID/CLIP are optional.
DEFAULT_REQUIRED_STAGES = frozenset({"quality", "detector"})


# ===========================================================================
# Detector cohort model
# ===========================================================================
DETECTOR_COHORTS = {
    "watermarked": {
        "original": {
            "evaluation_cohort": "original_watermarked",
            "image_source": "input",
        },
        "attacked": {
            "evaluation_cohort": "attacked_watermarked",
            "image_source": "output",
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


def _build_detector_image_index(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
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
            })
    return index


# ===========================================================================
# Quality stage
# ===========================================================================
def evaluate_quality(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-sample PSNR/SSIM using effective source flow overlap.

    Requires ``effective_source_flow_*`` fields in records.
    Planned-flow fallback is forbidden.
    """
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

        # Require effective flow — planned-flow fallback is forbidden.
        edx = rec.get("effective_source_flow_dx_image_px")
        edy = rec.get("effective_source_flow_dy_image_px")
        if edx is None or edy is None:
            results.append({
                "run_id": run_id, "role": role,
                "error": "missing effective_source_flow — planned-flow fallback forbidden",
                "quality_available": False,
            })
            continue

        try:
            from PIL import Image

            dx = float(edx)
            dy = float(edy)
            if not math.isfinite(dx) or not math.isfinite(dy):
                results.append({
                    "run_id": run_id, "role": role,
                    "error": "non-finite effective flow",
                    "quality_available": False,
                })
                continue

            with Image.open(input_path) as ref, Image.open(out_path) as att:
                metrics = pair_quality_metrics(
                    ref.convert("RGB"), att.convert("RGB"), dx, dy,
                )

            psnr = float(metrics.get("overlap_psnr",
                          metrics.get("raw_full_psnr", float("nan"))))
            ssim = float(metrics.get("overlap_ssim",
                          metrics.get("raw_full_ssim", float("nan"))))

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

    quality_available = any(r.get("quality_available") for r in results)
    return {
        "stage": "quality",
        "status": STATUS_COMPLETED if quality_available else STATUS_SKIPPED_INSUFFICIENT_DATA,
        "available": quality_available,
        "count": len(results),
        "psnr_mean": sum(psnr_values) / len(psnr_values) if psnr_values else None,
        "ssim_mean": sum(ssim_values) / len(ssim_values) if ssim_values else None,
        "per_sample": results,
    }


# ===========================================================================
# Detector stage — delegates to method-specific modules
# ===========================================================================
def evaluate_detector(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    method: str,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run detector on all cohorts via method-specific detector module."""
    from raven.detectors import get_detector_module

    output_dir = Path(output_dir)
    eval_dir = evaluation_dir(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        det_mod = get_detector_module(method)
    except ValueError as exc:
        return {
            "stage": "detector",
            "method": method,
            "status": STATUS_FAILED_MISSING_DEPENDENCY,
            "reason": str(exc),
        }

    image_index = _build_detector_image_index(records, output_dir)
    if not image_index:
        return {
            "stage": "detector",
            "method": method,
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "reason": "No images to score.",
        }

    # Try loading provider state
    provider_info = None
    load_error_type = STATUS_FAILED_MISSING_REQUIRED_STATE
    try:
        if method in {"RID", "HSTR", "HSQR"}:
            provider_info = det_mod.load_state(records, device, method=method)
        elif method == "T2S":
            provider_info = det_mod.load_state(records, device)
        else:
            provider_info = det_mod.load_state(records, device)
    except ImportError as exc:
        load_error_type = STATUS_FAILED_MISSING_DEPENDENCY
        logger.debug("Detector dependency missing: %s", exc)
    except Exception as exc:
        load_error_type = STATUS_FAILED_PROVIDER_INITIALIZATION
        logger.debug("Provider init failed: %s", exc)

    if provider_info is None:
        return {
            "stage": "detector",
            "method": method,
            "status": load_error_type,
            "reason": f"Provider state for {method} is not available on this machine.",
            "required_artifacts": det_mod.describe_required_artifacts(),
        }

    # Score every image
    detector_rows: list[dict[str, Any]] = []
    row_index = 0
    for entry in image_index:
        # For T2S, pass the corresponding record for per-sample state
        extra_kwargs: dict[str, Any] = {}
        if method == "T2S":
            run_id = entry["run_id"]
            matching = [r for r in records if str(r.get("run_id", "")) == run_id]
            if matching:
                extra_kwargs["row"] = matching[0]

        score = None
        try:
            score = det_mod.score_image(
                provider_info, entry["image_path"],
                row_index=row_index, **extra_kwargs,
            )
        except Exception:
            pass

        row = {
            "run_id": entry["run_id"],
            "source_role": entry["source_role"],
            "evaluation_cohort": entry["evaluation_cohort"],
            "image_path": entry["image_path"],
            "method": method,
            "status": STATUS_COMPLETED if score else STATUS_FAILED_SCORING,
        }
        if score:
            row.update(score)
        else:
            row["error"] = "scoring failed"
        detector_rows.append(row)
        row_index += 1

    # Write detector_records.jsonl
    det_path = detector_records_path(output_dir)
    tmp = det_path.with_name(f".detector_records.jsonl.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in detector_rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
    os.replace(tmp, det_path)

    # Aggregate
    agg_kwargs: dict[str, Any] = {}
    if method in {"RID", "HSTR", "HSQR"}:
        agg_kwargs["method"] = method
    aggregate = det_mod.aggregate(detector_rows, **agg_kwargs)
    aggregate["stage"] = "detector"
    aggregate["method"] = method
    aggregate["status"] = STATUS_COMPLETED
    aggregate["available"] = True
    return aggregate


# ===========================================================================
# FID stage — watermarked role only
# ===========================================================================
def evaluate_fid(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FID: original watermarked input vs attacked watermarked output.

    Clean-role records are excluded.
    """
    try:
        from raven.quality import clean_fid, FID_PRIMARY_MODE
    except ImportError:
        return {
            "stage": "fid",
            "status": STATUS_FAILED_MISSING_DEPENDENCY,
            "reason": "clean-fid not installed.",
        }

    import hashlib
    import shutil
    import tempfile

    output_dir = Path(output_dir)

    # Watermarked role ONLY
    wm_records = [r for r in records if r.get("role") == "watermarked"]
    if not wm_records:
        return {
            "stage": "fid",
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "reason": "No watermarked records for FID.",
        }

    # Build pairs with safe deterministic filenames
    pairs: list[dict[str, Any]] = []
    for rec in wm_records:
        run_id = str(rec["run_id"])
        input_path = Path(rec.get("input_path", ""))
        out_path = output_image_path(output_dir, "watermarked", run_id)
        if input_path.is_file() and out_path.is_file():
            # Safe deterministic name: zero-pad numeric, hash non-numeric
            try:
                safe_name = f"{int(run_id):06d}"
            except (ValueError, TypeError):
                h = hashlib.sha256(run_id.encode()).hexdigest()[:12]
                safe_name = f"{h}"
            pairs.append({
                "run_id": run_id,
                "safe_name": safe_name,
                "reference_path": str(input_path),
                "attacked_path": str(out_path),
            })

    if len(pairs) < 2:
        return {
            "stage": "fid",
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "reason": f"Need at least 2 paired images, got {len(pairs)}.",
        }

    tmpdir = Path(tempfile.mkdtemp(prefix="raven_fid_"))
    try:
        ref_dir = tmpdir / "reference"
        att_dir = tmpdir / "attacked"
        ref_dir.mkdir()
        att_dir.mkdir()

        for pair in pairs:
            shutil.copy2(pair["reference_path"],
                          ref_dir / f"{pair['safe_name']}.png")
            shutil.copy2(pair["attacked_path"],
                          att_dir / f"{pair['safe_name']}.png")

        result = clean_fid(str(ref_dir), str(att_dir), device=device)
        return {
            "stage": "fid",
            "status": STATUS_COMPLETED,
            "image_count": len(pairs),
            "fid_value": result.get("value"),
            "mode": FID_PRIMARY_MODE,
            "protocol": result.get("protocol", ""),
            "staged_records": pairs,
        }
    except Exception as exc:
        return {
            "stage": "fid",
            "status": STATUS_FAILED_INTERNAL_ERROR,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# CLIP stage — watermarked role only
# ===========================================================================
def evaluate_clip(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CLIP: attacked watermarked output.png vs original generation prompt.

    Clean-role records are excluded.
    """
    try:
        from raven.quality import openclip_text_image_scores
    except ImportError:
        return {
            "stage": "clip",
            "status": STATUS_FAILED_MISSING_DEPENDENCY,
            "reason": "open_clip_torch not installed.",
        }

    output_dir = Path(output_dir)

    # Watermarked role ONLY
    wm_records = [r for r in records if r.get("role") == "watermarked"]
    image_paths: list[str] = []
    prompts: list[str] = []

    for rec in wm_records:
        out_path = output_image_path(output_dir, "watermarked", str(rec["run_id"]))
        if out_path.is_file():
            image_paths.append(str(out_path))
            prompts.append(rec.get("prompt", ""))

    if not image_paths:
        return {
            "stage": "clip",
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "reason": "No watermarked output images available.",
        }

    if not all(prompts):
        return {
            "stage": "clip",
            "status": STATUS_SKIPPED_INSUFFICIENT_DATA,
            "reason": "Some records missing prompt.",
        }

    try:
        result = openclip_text_image_scores(
            image_paths, prompts, device=device,
            model_name="ViT-bigG-14",
            pretrained="laion2b_s39b_b160k",
        )
        scores = result.get("scores", [])
        import numpy as np
        return {
            "stage": "clip",
            "status": STATUS_COMPLETED,
            "image_count": len(image_paths),
            "model_name": result.get("model_name", "ViT-bigG-14"),
            "pretrained": result.get("pretrained", "laion2b_s39b_b160k"),
            "metric": result.get("metric", "prompt-image cosine similarity"),
            "count": len(scores),
            "mean_score": result.get("mean"),
            "std": float(np.std(scores)) if scores else None,
            "scores": scores,
        }
    except Exception as exc:
        return {
            "stage": "clip",
            "status": STATUS_FAILED_INTERNAL_ERROR,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ===========================================================================
# Stage runners (uniform signature)
# ===========================================================================
STAGE_RUNNERS: dict[str, Any] = {
    "quality": evaluate_quality,
    "detector": lambda records, od, dev, cfg: evaluate_detector(
        records, od, cfg.get("method", "TR"), dev, cfg,
    ),
    "fid": evaluate_fid,
    "clip": evaluate_clip,
}

NONZERO_STATUSES = frozenset({
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_MISSING_DEPENDENCY,
    STATUS_FAILED_PROVIDER_INITIALIZATION,
    STATUS_FAILED_SCORING,
    STATUS_FAILED_INTERNAL_ERROR,
})


def run_evaluation(
    output_dir: str | Path,
    *,
    device: str = "cuda",
    stages: list[str] | None = None,
    allow_missing_metrics: bool = False,
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
                "status": STATUS_FAILED_INTERNAL_ERROR,
                "reason": f"Unknown stage: {stage}",
            }
            continue
        logger.info("Running %s evaluation...", stage)
        try:
            result["stages"][stage] = runner(records, output_dir, device, config)
        except Exception as exc:
            logger.exception("%s evaluation failed", stage)
            result["stages"][stage] = {
                "status": STATUS_FAILED_INTERNAL_ERROR,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Compute overall status
    stage_statuses = {
        s: info.get("status", STATUS_FAILED_INTERNAL_ERROR)
        for s, info in result["stages"].items()
    }
    failed = {
        s for s, status in stage_statuses.items()
        if status in NONZERO_STATUSES
    }
    skipped = {
        s for s, status in stage_statuses.items()
        if status == STATUS_SKIPPED_INSUFFICIENT_DATA
        and s in DEFAULT_REQUIRED_STAGES
        and not allow_missing_metrics
    }
    result["failed_stages"] = sorted(failed)
    result["skipped_stages"] = sorted(skipped)
    result["overall_status"] = (
        STATUS_COMPLETED if not (failed or skipped)
        else STATUS_SKIPPED_INSUFFICIENT_DATA
    )

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
    parser.add_argument("--allow-missing-metrics", action="store_true",
                        help="Exit 0 even when required stages are unavailable")
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
            allow_missing_metrics=args.allow_missing_metrics,
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

    failed = result.get("failed_stages", [])
    skipped = result.get("skipped_stages", [])
    if failed:
        logger.error("Failed stages: %s", ", ".join(failed))
    if skipped:
        logger.warning("Skipped required stages (no data): %s", ", ".join(skipped))

    if failed:
        return 2
    if skipped and not args.allow_missing_metrics:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
