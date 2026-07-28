#!/usr/bin/env python3
"""Generate a Gaussian Shading cohort from the canonical Tree-Ring clean source.

This is the shared-clean V2 generator. It regenerates ONLY the GS watermarked
images. The Tree-Ring clean images, the Tree-Ring watermarked images and the
Tree-Ring metadata are read-only inputs and are never rewritten, copied,
re-encoded or regenerated.

Protocol (``gaussian_shading_shared_tr_clean_v2``):

1. Rebuild each Tree-Ring base latent from its recorded ``base_latent_seed``
   using the canonical TR procedure and verify the tensor SHA-256 against both
   ``base_latent_sha256`` and ``clean_base_latent_sha256``.
2. Verify the existing TR clean image on disk against ``clean_sha256``.
3. Derive the GS uniforms as ``norm.cdf(base_latent_float64)`` — no RNG.
4. Embed with the official Gaussian quantile partition
   ``norm.ppf((u + encrypted_bit) / 2)`` via
   ``GsProvider.get_wm_latents_from_uniforms``.
5. Generate the GS watermarked image only, and record metadata that points at
   the existing TR clean image path and SHA.

Any provenance mismatch fails closed. This is NOT a byte-identical reproduction
of upstream Gaussian Shading sampling (upstream draws its own uniforms); it is
the official embedding math applied to the shared Tree-Ring clean latent.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENTS_ROOT.parent
RAVEN_ROOT = WORKSPACE / "raven_repro"
BENCH_ROOT = WORKSPACE / "eval_bench_wm"
for root in (str(EXPERIMENTS_ROOT), str(RAVEN_ROOT), str(BENCH_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from raven.eval_protocol import method_data_root, source_metadata_path
from raven.gpu_utils import (
    configure_gpu,
    finalize_gpu_logging,
    setup_run_logging,
    utc_timestamp,
    write_experiment_records,
)
from raven.pairing_provenance import (
    GS_SHARED_TR_CLEAN_MODE,
    GS_SHARED_TR_CLEAN_PROTOCOL,
    GS_UNIFORM_DERIVATION,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    audit_pairing_rows,
    audit_tr_gs_shared_clean,
    build_pairing_sha256,
    canonical_json_sha256,
    sha256_path,
    tensor_sha256,
)

# Canonical TR-source plumbing lives in one place and is shared with the
# GaussMarker and T2SMark shared-clean runners. These names are re-exported into
# this module's namespace so existing references keep working unchanged.
from shared_clean_tr import (  # noqa: E402
    LATENT_CHANNELS,
    SharedCleanError,
    append_row,
    existing_completed_rows,
    load_tr_rows,
    rebuild_shared_clean_latent,
    save_json,
    select_rows,
    shard_suffix,
    str_to_bool,
    verify_source_clean_image,
)

# Single authoritative definition of the canonical TR cohort location.
DEFAULT_TR_METADATA = source_metadata_path("TR", "diffusiondb")
DEFAULT_DATASET_NAME = "diffusiondb_shared_tr"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Gaussian Shading cohort from the canonical Tree-Ring "
            "clean latents and clean images (shared-clean protocol v2)"
        )
    )
    parser.add_argument("--tr-metadata", type=Path, default=DEFAULT_TR_METADATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to data/gs/<dataset-name>/GS",
    )
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-id", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument(
        "--model-revision", type=str, default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
    )
    parser.add_argument("--scheduler-target", type=str, default="DDIM")
    parser.add_argument("--num-inference-steps-target", type=int, default=50)
    parser.add_argument("--guidance-scale-target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--require-free-gpu", type=str_to_bool, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing metadata CSV after re-verifying every stored row",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="restrict generation to these TR run_ids (gates / smoke tests)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-before", type=str_to_bool, default=True)
    parser.add_argument("--min-cpu-mem-gb", type=float, default=64.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=96.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
    return parser.parse_args(argv)


def derive_uniforms(np, base_cpu):
    """GS uniforms from the TR float32 base latent, validated, never clamped."""
    from scipy.stats import norm

    uniforms = norm.cdf(base_cpu.numpy().astype(np.float64))
    if not np.isfinite(uniforms).all():
        raise SharedCleanError("derived uniforms contain non-finite values")
    if not (uniforms > 0.0).all():
        raise SharedCleanError("derived uniforms are not strictly greater than 0")
    if not (uniforms < 1.0).all():
        raise SharedCleanError("derived uniforms are not strictly less than 1")
    return np.ascontiguousarray(uniforms, dtype=np.float64)


def validate_resume_row(
    stored: Dict[str, str],
    *,
    run_id: int,
    tr_row: Dict[str, str],
    wm_results: Dict[str, Any],
    base_latent_sha256: str,
) -> None:
    """Re-derive the row's identity and refuse to skip on any mismatch."""
    secret = wm_results["secret_provenance_list"][0]
    checks = {
        "protocol": GS_SHARED_TR_CLEAN_PROTOCOL,
        "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
        "gs_secret_index": str(run_id),
        "gs_sampling_uniform_sha256": wm_results["sampling_uniform_sha256_list"][0],
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "watermarked_latent_sha256": tensor_sha256(wm_results["zT_torch"]),
        "watermark_target_sha256": tensor_sha256(wm_results["barcodes_torch"]),
        "base_latent_sha256": base_latent_sha256,
        "clean_base_latent_sha256": base_latent_sha256,
        "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
        "clean_path": str(tr_row["clean_path"]),
        "clean_sha256": str(tr_row["clean_sha256"]),
        "base_latent_seed": str(tr_row["base_latent_seed"]),
        "prompt_sha256": str(tr_row["prompt_sha256"]),
    }
    for field, expected in checks.items():
        if str(stored.get(field, "")) != str(expected):
            raise SharedCleanError(
                f"shared-clean resume mismatch run_id={run_id} field={field}: "
                f"stored={stored.get(field)!r} expected={expected!r}"
            )
    watermarked_path = Path(str(stored["watermarked_path"]))
    if not watermarked_path.is_file():
        raise SharedCleanError(
            f"recorded GS output missing run_id={run_id}: {watermarked_path}"
        )
    actual = sha256_path(watermarked_path)
    if actual != str(stored["watermarked_sha256"]):
        raise SharedCleanError(
            f"recorded GS output SHA drift run_id={run_id}: "
            f"expected {stored['watermarked_sha256']}, got {actual}"
        )


def run(args: argparse.Namespace, guard: Any, device: Any) -> Dict[str, Any]:
    import numpy as np
    import torch
    from utils.imprint_utils import validate
    from utils.pipe import pipe_utils
    from utils.utils import set_random_seed
    from utils.wm.gs_provider import GS_SHARED_TR_CLEAN_MODE as PROVIDER_MODE
    from utils.wm.gs_provider import GsProvider

    if PROVIDER_MODE != GS_SHARED_TR_CLEAN_MODE:
        raise SharedCleanError(
            "provider and protocol disagree on the shared-clean mode name: "
            f"{PROVIDER_MODE!r} != {GS_SHARED_TR_CLEAN_MODE!r}"
        )

    tr_metadata = Path(args.tr_metadata).resolve()
    tr_rows = load_tr_rows(tr_metadata)
    tr_metadata_sha256 = sha256_path(tr_metadata)
    selected = select_rows(
        tr_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        run_ids=args.run_ids,
        limit=args.limit,
    )

    method_dir = Path(args.output_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(args.num_shards, args.shard_index)
    metadata_csv = method_dir / f"metadata{suffix}.csv"
    completed = existing_completed_rows(metadata_csv, resume=args.resume)

    print(f"[GS-v2] loading target pipeline: {args.model_id}", flush=True)
    set_random_seed(42)
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
        revision=args.model_revision,
    )
    latent_shape = tuple(pipe_provider_target.get_latent_shape())
    pipe_dtype = pipe_provider_target.get_dtype()
    expected_shape = (1, LATENT_CHANNELS, args.resolution // 8, args.resolution // 8)
    if latent_shape != expected_shape:
        raise SharedCleanError(
            f"pipeline latent shape {latent_shape} does not match the Tree-Ring "
            f"cohort shape {expected_shape}"
        )
    if pipe_dtype != torch.float32:
        # The TR base-latent SHA is a float32 tensor hash; a different compute
        # dtype would make the shared-latent invariant unprovable.
        raise SharedCleanError(
            f"shared-clean generation requires float32 latents, got {pipe_dtype}"
        )

    generation_config = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "scheduler": args.scheduler_target,
        "num_inference_steps": args.num_inference_steps_target,
        "guidance_scale": args.guidance_scale_target,
        "resolution": args.resolution,
        "dtype": str(pipe_dtype),
    }
    generation_config_sha256 = canonical_json_sha256(generation_config)
    tr_generation_hashes = {str(row["generation_config_sha256"]) for row in tr_rows}
    if tr_generation_hashes != {generation_config_sha256}:
        raise SharedCleanError(
            "generation config does not match the Tree-Ring cohort: "
            f"gs={generation_config_sha256} tr={sorted(tr_generation_hashes)}; "
            "the shared-clean protocol requires an identical model/scheduler/"
            "steps/guidance/resolution/dtype configuration"
        )

    watermark_mask_sha256 = canonical_json_sha256(
        {"method": "GS", "mask": "not_applicable", "version": 1}
    )
    watermark_config = {
        "wm_type": "GS",
        "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
        "official_source_repository": "bsmhmmlf/Gaussian-Shading",
        "official_source_commit": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
        "official_math_claim": (
            "official Gaussian quantile-partition embedding math "
            "norm.ppf((u+encrypted_bit)/2) with the official payload/cipher layout"
        ),
        "not_claimed": (
            "NOT byte-identical to upstream Gaussian Shading sampling: upstream "
            "draws its own uniforms, this cohort derives them from the canonical "
            "Tree-Ring clean latent"
        ),
        "message_width_in_bytes": 32,
        "channel_copy": 1,
        "hw_copy": 8,
        "payload_layout": "torch.repeat(1,channel_copy,hw_copy,hw_copy)",
        "cipher": "PyCryptodome.ChaCha20",
        "key_bytes": 32,
        "nonce_bytes": 12,
        "sampling": "norm.ppf((u+encrypted_bit)/2)",
        "majority_vote": "strict_gt_half_ties_zero",
        "secret_mapping": "secret_index=run_id",
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
        "watermark_mask_sha256": watermark_mask_sha256,
    }
    watermark_config_sha256 = canonical_json_sha256(watermark_config)

    rows_written = 0
    skipped = 0
    gate_records: List[Dict[str, Any]] = []
    try:
        for local_idx, tr_row in enumerate(selected):
            run_id = int(tr_row["run_id"])
            item_dir = method_dir / f"{run_id:06d}"
            image_path = item_dir / "watermarked.png"

            guard.check(f"{args.dataset_name}/GS/run_id={run_id}/before")
            base_cpu, shared_clean_latent, base_latent_sha256 = rebuild_shared_clean_latent(
                torch, tr_row, resolution=args.resolution, device=device, dtype=pipe_dtype
            )
            clean_path = verify_source_clean_image(tr_row)
            uniforms = derive_uniforms(np, base_cpu)

            provider = GsProvider(
                latent_shape=latent_shape,
                dtype=pipe_dtype,
                device=device,
                offset=run_id,
                gs_secret_index=run_id,
                gs_protocol_mode=GS_SHARED_TR_CLEAN_MODE,
            )
            wm_results = provider.get_wm_latents_from_uniforms(uniforms, shared_clean_latent)
            clean_zT = wm_results["zT_clean_torch"]
            wm_zT = wm_results["zT_torch"]
            clean_base_latent_sha256 = tensor_sha256(clean_zT)
            if clean_base_latent_sha256 != base_latent_sha256:
                raise SharedCleanError(
                    f"provider changed the shared clean latent run_id={run_id}"
                )
            watermarked_latent_sha256 = tensor_sha256(wm_zT)
            if watermarked_latent_sha256 == base_latent_sha256:
                raise SharedCleanError(f"watermark made no latent change run_id={run_id}")
            watermark_target_sha256 = tensor_sha256(wm_results["barcodes_torch"])
            secret_provenance = wm_results["secret_provenance_list"][0]
            sampling_uniform_sha256 = wm_results["sampling_uniform_sha256_list"][0]

            if run_id in completed:
                validate_resume_row(
                    completed[run_id],
                    run_id=run_id,
                    tr_row=tr_row,
                    wm_results=wm_results,
                    base_latent_sha256=base_latent_sha256,
                )
                skipped += 1
                del base_cpu, shared_clean_latent, clean_zT, wm_zT, wm_results, provider
                gc.collect()
                continue

            if image_path.exists():
                raise SharedCleanError(
                    f"unrecorded GS output already exists for run_id={run_id}: {image_path}"
                )

            print(
                f"[GS-v2] generating {local_idx + 1}/{len(selected)} run_id={run_id} "
                f"seed={tr_row['base_latent_seed']}",
                flush=True,
            )
            generated = pipe_provider_target.generate(
                prompts=tr_row["prompt"],
                latents=wm_zT,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
            watermarked_image = generated["images_PIL"][0]
            item_dir.mkdir(parents=True, exist_ok=True)
            watermarked_image.save(image_path)
            # The clean image is an input, not an output: prove we did not touch it.
            if sha256_path(clean_path) != str(tr_row["clean_sha256"]):
                raise SharedCleanError(
                    f"TR clean image changed during generation run_id={run_id}"
                )

            if args.validate_before:
                before = validate(
                    out_dir=str(method_dir),
                    image_to_verify_PIL=watermarked_image,
                    original_PIL=watermarked_image,
                    wm_provider=provider,
                    pipe_provider_target=pipe_provider_target,
                    num_inference_steps_target=args.num_inference_steps_target,
                    step=-1,
                    message_bits_str_initial=wm_results["message_bits_str_list"][0],
                    do_psnr=False,
                    do_ssim=False,
                    do_msssim=False,
                    do_lpips=False,
                    device=str(device),
                )
                before_metric_value = before["bit_accuracy"]
                before_successful = provider.is_detection_successful(before_metric_value)
            else:
                before = {}
                before_metric_value = None
                before_successful = None
            detection_info = provider.active_detection_threshold()

            row: Dict[str, Any] = {
                "protocol": GS_SHARED_TR_CLEAN_PROTOCOL,
                "dataset_name": args.dataset_name,
                "dataset": args.dataset_name,
                "run_id": run_id,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "prompt_id": tr_row["prompt_id"],
                "prompt": tr_row["prompt"],
                "prompt_sha256": hashlib.sha256(
                    str(tr_row["prompt"]).encode("utf-8")
                ).hexdigest(),
                "source": tr_row.get("source", ""),
                "wm_type": "GS",
                "wm_name": "Gaussian Shading",
                "target_model": args.model_id,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                "detection_threshold": detection_info["threshold"],
                "detection_threshold_type": detection_info["threshold_type"],
                "detection_threshold_comparison_operator": detection_info[
                    "comparison_operator"
                ],
                "threshold_calibrated_from_current_clean_negatives": False,
                "detection_metric": "bit_accuracy",
                "before_detection_successful": before_successful,
                "before_detection_metric_value": before_metric_value,
                # --- shared-clean identity (must equal the TR row) ---
                "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
                "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
                "shared_clean_source_metadata_path": str(tr_metadata),
                "shared_clean_source_metadata_sha256": tr_metadata_sha256,
                "shared_clean_sample_sha256": base_latent_sha256,
                "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
                "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                "tr_clean_path": str(tr_row["clean_path"]),
                "tr_clean_sha256": str(tr_row["clean_sha256"]),
                "base_latent_seed": int(tr_row["base_latent_seed"]),
                "generation_seed": int(tr_row["base_latent_seed"]),
                "base_latent_sha256": base_latent_sha256,
                "clean_base_latent_sha256": clean_base_latent_sha256,
                "watermarked_base_latent_sha256": base_latent_sha256,
                "watermarked_latent_sha256": watermarked_latent_sha256,
                "watermark_target_sha256": watermark_target_sha256,
                "watermark_mask_sha256": watermark_mask_sha256,
                "generation_config_sha256": generation_config_sha256,
                "watermark_config_sha256": watermark_config_sha256,
                "injection_only_difference_verified": False,
                "pairing_relation": (
                    "shared_tr_clean_latent_quantile_partition_of_its_own_uniforms"
                ),
                "injection_max_abs_error": "",
                "clean_path": str(tr_row["clean_path"]),
                "clean_sha256": str(tr_row["clean_sha256"]),
                "watermarked_path": str(image_path.resolve()),
                "watermarked_image_path": str(image_path.resolve()),
                "watermarked_sha256": sha256_path(image_path),
                "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
                "gs_secret_index": run_id,
                "gs_message_sha256": secret_provenance["message_sha256"],
                "gs_key_sha256": secret_provenance["key_sha256"],
                "gs_nonce_sha256": secret_provenance["nonce_sha256"],
                "gs_secret_bundle_sha256": secret_provenance["secret_bundle_sha256"],
                "gs_sampling_uniform_sha256": sampling_uniform_sha256,
                "gs_payload_layout": "channel_spatial_repeat",
                "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
                "gs_official_tau_onebit": detection_info["official_tau_onebit"],
                "gs_official_tau_bits": detection_info["official_tau_bits"],
                "gs_detection_mode": detection_info["detection_mode"],
                "watermark_implementation_protocol": GS_SHARED_TR_CLEAN_MODE,
                "generation_benchmark_protocol": "shared_formal_cohort_redbeardnz_ddim",
                "upstream_official_reproduction_runner": (
                    "stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16"
                ),
            }
            row["pairing_sha256"] = build_pairing_sha256(row)
            row["before_bit_accuracy"] = before.get("bit_accuracy")
            row["before_message_bits_str_initial"] = before.get(
                "message_bits_str_initial", ""
            )
            row["before_message_bits_str_recovered"] = before.get(
                "message_bits_str_recovered", ""
            )
            append_row(metadata_csv, row)
            rows_written += 1
            gate_records.append(
                {
                    "run_id": run_id,
                    "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                    "gs_base_latent_sha256": base_latent_sha256,
                    "tr_clean_sha256": str(tr_row["clean_sha256"]),
                    "gs_clean_sha256": str(row["clean_sha256"]),
                    "gs_sampling_uniform_sha256": sampling_uniform_sha256,
                    "gs_watermarked_sha256": row["watermarked_sha256"],
                    "before_bit_accuracy": before_metric_value,
                    "before_detection_successful": before_successful,
                    "detection_threshold": detection_info["threshold"],
                    "detection_threshold_type": detection_info["threshold_type"],
                }
            )
            print(
                f"[GS-v2] run_id={run_id} bit_accuracy={before_metric_value} "
                f"detected={before_successful}",
                flush=True,
            )
            guard.check(f"{args.dataset_name}/GS/run_id={run_id}/done")
            del base_cpu, shared_clean_latent, clean_zT, wm_zT, wm_results
            del provider, watermarked_image, generated
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        try:
            pipe_provider_target.stash_pipe()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        gs_rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=True)
    cross = audit_tr_gs_shared_clean(tr_rows, gs_rows, verify_files=True)
    save_json(method_dir / f"pairing_audit{suffix}.json", audit)
    save_json(method_dir / f"cross_method_shared_clean_audit{suffix}.json", cross)
    save_json(method_dir / f"generation_config{suffix}.json", generation_config)
    save_json(method_dir / f"watermark_config{suffix}.json", watermark_config)

    accuracies = [
        float(row["before_bit_accuracy"])
        for row in gs_rows
        if str(row.get("before_bit_accuracy", "")) not in {"", "None"}
    ]
    summary = {
        "protocol": GS_SHARED_TR_CLEAN_PROTOCOL,
        "dataset_name": args.dataset_name,
        "metadata_csv": str(metadata_csv),
        "completed": len(gs_rows),
        "rows_written_this_run": rows_written,
        "rows_verified_and_skipped": skipped,
        "selected_source_rows": len(selected),
        "tr_source_metadata": str(tr_metadata),
        "tr_source_metadata_sha256": tr_metadata_sha256,
        "tr_source_rows": len(tr_rows),
        "detector_metric": "bit_accuracy",
        "score_direction": "higher_is_watermarked",
        "mean_before_bit_accuracy": (
            sum(accuracies) / len(accuracies) if accuracies else None
        ),
        "pairing_audit": audit,
        "cross_method_shared_clean_audit": {
            key: value for key, value in cross.items() if key != "rows"
        },
        "gate_records": gate_records,
        "clean_images_generated": 0,
        "clean_images_copied": 0,
        "paper_setting_note": (
            "Gaussian Shading embedded with official quantile-partition math from "
            "uniforms derived from the canonical Tree-Ring clean latent; the clean "
            "images are the existing Tree-Ring clean images, untouched."
        ),
    }
    save_json(method_dir / f"summary{suffix}.json", summary)
    print(f"[GS-v2] summary: {json.dumps(summary['pairing_audit'], sort_keys=True)}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.output_dir is None:
        args.output_dir = method_data_root("GS") / args.dataset_name / "GS"
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = shard_suffix(args.num_shards, args.shard_index)
    setup_run_logging(args.output_dir, filename=f"run{suffix}.log")
    started_at = utc_timestamp()
    gpu_record = configure_gpu(
        args.gpu, args.device, args.output_dir, require_free_gpu=args.require_free_gpu
    )

    status = "failed"
    summary: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
        import torch

        limit_cpu_threads(1)
        guard = CpuMemoryGuard(
            min_available_gib=args.min_cpu_mem_gb,
            warn_available_gib=args.warn_cpu_mem_gb,
            max_process_rss_gib=args.max_process_ram_gb,
        )
        guard.check("startup")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SharedCleanError(
                "--device cuda requested but torch.cuda.is_available() is false"
            )
        device = torch.device(args.device)
        os.environ.setdefault("TQDM_DISABLE", "1")
        summary = run(args, guard, device)
        status = "completed"
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        finished_at = utc_timestamp()
        gpu_record = finalize_gpu_logging(args.output_dir, gpu_record)
        extra: Dict[str, Any] = {"summary": summary}
        if error:
            extra["error"] = error
        write_experiment_records(
            args.output_dir,
            {key: str(value) for key, value in vars(args).items()},
            gpu_record,
            started_at,
            finished_at,
            status,
            extra,
            filename=f"results{suffix}.json",
        )


if __name__ == "__main__":
    raise SystemExit(main())
