#!/usr/bin/env python3
"""Unified RAVEN attack runner.

    python3 experiments/main.py \\
      --dataset synthetic \\
      --method TR \\
      --metadata /tmp/metadata.csv \\
      --output-dir /tmp/run \\
      --roles watermarked \\
      --diffusion-mode ddim \\
      --overwrite

Responsibilities
----------------
* Load metadata CSV with per-sample prompts.
* Build and record a normalized config (algorithm + execution fields).
* Initialize ``RavenPipeline`` **once** for the entire dataset.
* For each sample: resolve per-sample prompt, compute seed/shift, run the
  pipeline, save ``output.png`` and ``record.json``.
* Rebuild ``records.jsonl`` atomically from the per-sample records.

This module must NOT import or invoke any detector, FID, CLIP, PSNR/SSIM
aggregation, threshold calibration, formal validation, source manifest,
snapshot or SHA audit.  Those belong in ``eval.py``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.experiment_config import (  # noqa: E402
    check_config_match,
    config_for_pipeline,
    normalize_config,
)
from raven.experiment_io import (  # noqa: E402
    cleanup_intermediates,
    collect_incomplete_run_ids,
    is_sample_complete,
    output_image_path,
    prepare_output_dir,
    read_config,
    rebuild_records_jsonl,
    record_path,
    write_config,
    write_record,
)
from raven.shift_plan import plan_shift  # noqa: E402

logger = logging.getLogger("raven.main")


# --------------------------------------------------------------------------- #
# Metadata helpers
# --------------------------------------------------------------------------- #
def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unspecified")
    parser.add_argument(
        "--method", default="TR",
        choices=["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"],
    )
    parser.add_argument("--metadata", type=Path, required=True,
                        help="CSV with columns run_id, watermarked_path, clean_path, prompt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["watermarked"],
                        choices=["watermarked", "clean"])
    parser.add_argument(
        "--diffusion-mode", default="ddim",
        choices=["ddim", "ddpm", "ddim-ddpm"],
    )
    parser.add_argument("--sampling", default="nearest",
                        choices=["nearest", "bilinear"],
                        help="Latent sampling mode")
    parser.add_argument("--shift-mode", default="random",
                        choices=["none", "fixed", "random"])
    parser.add_argument("--shift-x", type=float, default=None)
    parser.add_argument("--shift-y", type=float, default=None)
    parser.add_argument("--shift-magnitude-min", type=int, default=24)
    parser.add_argument("--shift-magnitude-max", type=int, default=32)
    parser.add_argument("--shift-space", default="image_pixels")
    parser.add_argument("--warp-mode", default="raven_paper_nfpa_gap_fill")
    parser.add_argument("--padding-mode", default="reflection")
    parser.add_argument("--color-transfer", default="aligned",
                        choices=["aligned", "none"],
                        help="Color transfer mode: 'aligned' (paper-exact) or 'none'")
    parser.add_argument("--view-guided-attention", type=_parse_bool, default=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--prompt", default="",
                        help="Global fallback prompt (metadata prompt takes precedence)")
    parser.add_argument("--prompt-source", default="metadata",
                        choices=["metadata", "formal_config"],
                        help="metadata: use per-sample prompt from metadata CSV; "
                             "formal_config: force empty prompt regardless of metadata")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision",
                        default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-input-copy", type=_parse_bool, default=False)
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Keep pipeline intermediate artifacts (debug_info.json, etc.)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #
def load_metadata(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"no rows in metadata: {path}")
    return rows


def normalize_metadata_row(row: dict[str, str]) -> dict[str, str]:
    """Normalize core fields while preserving all original metadata columns.

    Detector-required fields (GM bundle, T2S state, GS secret, TR params,
    Fourier bundle, target/mask hashes) pass through unmodified.
    """
    normalized = dict(row)  # passthrough ALL original columns
    run_id = _first(row, "run_id", "sample_id", "id")
    if not run_id:
        raise ValueError("metadata row missing run_id")
    normalized["run_id"] = run_id
    normalized["watermarked_path"] = _first(row, "watermarked_path", "watermarked_image_path")
    normalized["clean_path"] = _first(row, "clean_path", "clean_image_path")
    normalized["prompt"] = _first(row, "prompt", "source_prompt", "caption", "text")
    normalized["prompt_id"] = _first(row, "prompt_id", "source_id")
    return normalized


def resolve_input_path(row: dict[str, str], role: str) -> Path:
    field = "watermarked_path" if role == "watermarked" else "clean_path"
    value = row.get(field, "")
    if not value:
        raise ValueError(f"run_id={row['run_id']}: missing {field}")
    path = _resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"run_id={row['run_id']}: {field} not found: {path}")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    color_transfer_bool = args.color_transfer == "aligned"

    # --- build config ---
    config = normalize_config(
        diffusion_mode=args.diffusion_mode,
        method=args.method,
        dataset=args.dataset,
        metadata_path=str(args.metadata.resolve()),
        output_dir=str(args.output_dir.resolve()),
        roles=args.roles,
        limit=args.limit,
        gpu=args.gpu,
        overwrite=args.overwrite,
        resume=args.resume,
        shift_mode=args.shift_mode,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        shift_magnitude_min=args.shift_magnitude_min,
        shift_magnitude_max=args.shift_magnitude_max,
        base_seed=args.base_seed,
        steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        shift_space=args.shift_space,
        warp_mode=args.warp_mode,
        latent_sampling_mode=args.sampling,
        padding_mode=args.padding_mode,
        view_guided_attention=args.view_guided_attention,
        color_transfer=color_transfer_bool,
        prompt=args.prompt,
        prompt_source=args.prompt_source,
        negative_prompt=args.negative_prompt,
        debug=args.debug,
        save_input_copy=args.save_input_copy,
        save_intermediates=args.save_intermediates,
        model_id=args.model_id,
        model_revision=args.model_revision,
        dtype=args.dtype,
    )

    # --- output directory ---
    output_dir = prepare_output_dir(
        args.output_dir, overwrite=args.overwrite, resume=args.resume,
    )

    # --- resume check ---
    existing_config_path = output_dir / "config.json"
    if args.resume and existing_config_path.is_file():
        stored = read_config(output_dir)
        mismatches = check_config_match(stored, config)
        if mismatches:
            logger.error(
                "Config mismatch for resume.  Differing algorithm fields: %s",
                ", ".join(sorted(mismatches)),
            )
            return 1
        config = stored  # reuse stored config exactly
        logger.info("Resume: algorithm config matches, reusing stored config.")

    write_config(output_dir, config)

    # --- load metadata ---
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    normalized_rows = [normalize_metadata_row(row) for row in rows]
    run_ids = [row["run_id"] for row in normalized_rows]
    row_by_id = {row["run_id"]: row for row in normalized_rows}

    logger.info("Dataset: %s  Method: %s  Samples: %d  Roles: %s",
                 args.dataset, args.method, len(run_ids), args.roles)

    # --- collect incomplete samples ---
    incomplete = collect_incomplete_run_ids(output_dir, args.roles, run_ids)
    if not incomplete:
        logger.info("All %d sample(s) complete — nothing to do.",
                     len(run_ids) * len(args.roles))
        rebuild_records_jsonl(output_dir)
        return 0

    logger.info("%d complete, %d incomplete — running attack.",
                 len(run_ids) * len(args.roles) - len(incomplete), len(incomplete))

    # --- GPU setup ---
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")

    # --- initialize pipeline ONCE ---
    from raven.pipeline_raven import RavenPipeline
    from raven.resource_guard import limit_cpu_threads

    limit_cpu_threads(1)
    logger.info("Initializing RavenPipeline (scheduler=%s)...", config["scheduler_mode"])
    pipe = RavenPipeline(
        model_id=config["model_id"],
        device="cuda",
        dtype=config["dtype"],
        revision=config["model_revision"],
        scheduler_mode=config["scheduler_mode"],
    )

    pipeline_kwargs = config_for_pipeline(config)
    global_fallback_prompt = config.get("prompt", "")

    # --- run attack ---
    completed = 0
    failed = 0

    try:
        for role, run_id in incomplete:
            row = row_by_id[run_id]
            try:
                input_path = resolve_input_path(row, role)
                with Image.open(input_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.load()

                dx, dy, attack_seed = plan_shift(run_id, config)
                logger.info("[%s/%s] seed=%d shift=(%g, %g)",
                             role, run_id, attack_seed, dx, dy)

                # Per-sample prompt: honour prompt_source policy.
                prompt_source = config.get("prompt_source", "metadata")
                if prompt_source == "formal_config":
                    sample_prompt = ""
                    sample_prompt_id = row.get("prompt_id", "")
                    sample_prompt_source = "formal_config"
                else:
                    sample_prompt = row.get("prompt", "") or global_fallback_prompt
                    sample_prompt_id = row.get("prompt_id", "")
                    sample_prompt_source = "metadata" if row.get("prompt", "") else "cli_fallback"

                # per-sample output dir
                sample_out = output_dir / "samples" / role / run_id
                sample_out.mkdir(parents=True, exist_ok=True)

                kwargs = dict(pipeline_kwargs)
                kwargs.update({
                    "seed": attack_seed,
                    "shift_x": dx,
                    "shift_y": dy,
                    "input_image": image,
                    "output_dir": str(sample_out),
                    "prompt": sample_prompt,
                })

                final_image = pipe.run(**kwargs)

                # Save canonical output.png
                canonical_out = output_image_path(output_dir, role, run_id)
                final_image.save(str(canonical_out))

                # Build record from debug_info
                debug_info_path = sample_out / "debug_info.json"
                debug_info: dict[str, Any] = {}
                debug_info_retained = bool(config.get("save_intermediates"))
                if debug_info_path.is_file():
                    debug_info = json.loads(debug_info_path.read_text(encoding="utf-8"))

                record = {
                    "run_id": run_id,
                    "role": role,
                    "dataset": config["dataset"],
                    "method": config["method"],
                    "attack_seed": attack_seed,
                    "planned_flow_dx_image_px": dx,
                    "planned_flow_dy_image_px": dy,
                    "diffusion_mode": config["diffusion_mode"],
                    "inversion_mode": config["inversion_mode"],
                    "scheduler_mode": config["scheduler_mode"],
                    "input_path": str(input_path),
                    "output_path": str(canonical_out),
                    "prompt": sample_prompt,
                    "prompt_id": sample_prompt_id,
                    "prompt_source": sample_prompt_source,
                    "negative_prompt": config.get("negative_prompt", ""),
                    "debug_info_path": (
                        str(debug_info_path.resolve())
                        if debug_info_retained and debug_info_path.is_file()
                        else ""
                    ),
                    "debug_info_retained": debug_info_retained,
                    "effective_source_flow_dx_image_px": debug_info.get(
                        "effective_source_flow_dx_image_px") if debug_info else None,
                    "effective_source_flow_dy_image_px": debug_info.get(
                        "effective_source_flow_dy_image_px") if debug_info else None,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                write_record(output_dir, role, run_id, record)

                # Clean up intermediates unless --save-intermediates
                if not config.get("save_intermediates"):
                    cleanup_intermediates(output_dir, role, run_id)

                completed += 1
                logger.info("[%s/%s] done.", role, run_id)

            except Exception:
                logger.exception("[%s/%s] FAILED", role, run_id)
                failed += 1
                if not args.resume:
                    raise
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # --- rebuild records.jsonl ---
    rebuild_records_jsonl(output_dir)

    logger.info("Attack complete: %d succeeded, %d failed.", completed, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
