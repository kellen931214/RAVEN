#!/usr/bin/env python3
"""Generate watermarked images with eval_bench_wm providers only.

This produces watermarked source images and validates the watermark before any
RAVEN attack. It intentionally does not run RAVEN or removal baselines.
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

from PIL import Image

_HERE = Path(__file__).resolve()
WORKSPACE = _HERE.parents[2]                         # repo root
RAVEN_ROOT = _HERE.parents[1]                        # raven_repro/
BENCH_ROOT = WORKSPACE / "eval_bench_wm"             # repo root / eval_bench_wm
for root in (str(RAVEN_ROOT), str(BENCH_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from generate.layout import method_data_root
from runtime import configure_gpu, finalize_gpu_logging, setup_run_logging, utc_timestamp, write_experiment_records
from raven.detectors.protocols import GS_PAIRING_PROTOCOL, PAIRING_PROTOCOL, tensor_sha256
from raven.protocol import canonical_json_hash
from generate.provenance import audit_pairing_rows, build_pairing_sha256, sha256_path

PAPER_WM_METHODS_IN_BENCH = ["GS", "TR", "RID", "HSTR", "HSQR"]
PAPER_WM_NAMES = {
    "GS": "Gaussian Shading",
    "TR": "Tree-Ring",
    "RID": "RingID",
    "HSTR": "HSTR",
    "HSQR": "HSQR",
}
METRIC_MAP = {
    "GS": "bit_accuracy",
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_name", type=str, default="mscoco")
    parser.add_argument("--prompts_csv", type=str, default=None)
    # Canonical layout (migration 2026-07-26): watermarked images live under
    # data/<method>/<dataset>/<METHOD>/ and clean images under data/clean/<dataset>/.
    # --output_dir defaults to the method root resolved in main() from --wm_types.
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--clean_output_dir", type=str, default=str(WORKSPACE / "data" / "clean"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--require_free_gpu", type=str_to_bool, default=True)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=64.0)
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=96.0)
    parser.add_argument("--max_process_ram_gb", type=float, default=16.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    from utils.pipe import pipe_utils
    from utils.wm.gs_provider import parser as gs_parser
    from utils.wm.hstr_provider import parser as hstr_parser
    from utils.wm.hsqr_provider import parser as hsqr_parser
    from utils.wm.ringid_provider import parser as ringid_parser
    from utils.wm.tr_provider import parser as tr_parser

    parent_parsers = [base_parser(), gs_parser, tr_parser, ringid_parser, hstr_parser, hsqr_parser]
    parser = argparse.ArgumentParser(
        description="Generate paper-overlap watermarked images with eval_bench_wm providers",
        parents=parent_parsers,
        conflict_handler="resolve",
    )
    parser.add_argument("--wm_types", nargs="+", default=PAPER_WM_METHODS_IN_BENCH, choices=PAPER_WM_METHODS_IN_BENCH)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--modelid_target", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision", type=str, default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--validate_before", type=str_to_bool, default=True)
    return parser.parse_args()


def method_dataset_dir(output_dir: Any, dataset_name: str, wm_type: str) -> Path:
    """Canonical dataset directory for one method's watermarked output.

    Without an explicit ``--output_dir`` each method writes under its own canonical
    data root (``data/tr/<dataset>`` / ``data/gs/<dataset>``), so a multi-method run
    can never collide two methods into one directory.
    """
    if output_dir:
        return Path(output_dir) / dataset_name
    return method_data_root(wm_type) / dataset_name


def deterministic_gs_sampling_seed(base_seed: int, run_id: int) -> int:
    # Per-row GS sampling seed = base_seed + run_id, matching the Tree-Ring
    # per-row seed schedule (base_latent_seed = base_seed + run_id) so GS and TR
    # consume the same numeric seed per run_id for apples-to-apples comparison.
    # This only seeds the GS uniform draw; payload/cipher/latent construction
    # in gs_provider.py are unchanged.
    return int(base_seed) + int(run_id)


def validate_gs_resume_provenance(
    stored: Dict[str, Any],
    *,
    run_id: int,
    sampling_seed: int,
    wm_results: Dict[str, Any],
) -> Dict[str, str]:
    secret = wm_results["secret_provenance_list"][0]
    checks = {
        "gs_secret_index": str(run_id),
        "gs_sampling_seed": str(sampling_seed),
        "gs_sampling_uniform_sha256": wm_results["sampling_uniform_sha256_list"][0],
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "watermarked_latent_sha256": tensor_sha256(wm_results["zT_torch"]),
    }
    for field, expected in checks.items():
        if str(stored.get(field, "")) != str(expected):
            raise RuntimeError(
                f"GS resume provenance mismatch run_id={run_id} field={field}"
            )
    return checks


def bool_for_json(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_prompts(
    path: Path,
    start_index: int,
    num_pairs: int,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prompt CSV has no header: {path}")
        prompt_field = "prompt" if "prompt" in reader.fieldnames else reader.fieldnames[0]
        for row_index, row in enumerate(reader):
            if row_index < start_index:
                continue
            prompt = (row.get(prompt_field) or "").strip()
            if not prompt:
                continue
            selected.append({
                "run_id": str(row_index),
                "prompt": prompt,
                "prompt_id": str(row.get("id", row_index)),
                "source": str(row.get("source", "")),
            })
            if len(selected) >= num_pairs:
                break
    if len(selected) < num_pairs:
        raise ValueError(f"Only found {len(selected)} prompts in {path}; requested {num_pairs}")
    return [row for row in selected if int(row["run_id"]) % num_shards == shard_index]


def shard_suffix(num_shards: int, shard_index: int) -> str:
    if num_shards == 1:
        return ""
    return f".shard-{shard_index:03d}-of-{num_shards:03d}"


def existing_completed_rows(csv_path: Path) -> dict[int, dict[str, str]]:
    if not csv_path.exists():
        return {}
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    # Partial resumes are allowed only after every existing row passes the full
    # provenance and file audit. Legacy metadata fails here by design.
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return {int(row["run_id"]): row for row in rows}


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    desired_fields = list(row.keys())
    if exists:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            stored_fields = list(reader.fieldnames or [])
            stored_rows = list(reader)
        if set(stored_fields) != set(desired_fields):
            raise ValueError(
                f"metadata schema field mismatch for {csv_path}: "
                f"missing={sorted(set(desired_fields) - set(stored_fields))} "
                f"extra={sorted(set(stored_fields) - set(desired_fields))}"
            )
        if stored_fields != desired_fields:
            if stored_rows:
                audit_pairing_rows(
                    stored_rows, expected_count=len(stored_rows), verify_files=True
                )
            temp = csv_path.with_name(f".{csv_path.name}.schema-{os.getpid()}.tmp")
            with temp.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=desired_fields, extrasaction="raise"
                )
                writer.writeheader()
                writer.writerows(stored_rows)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=desired_fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def detection_value(results: Dict[str, Any], wm_type: str) -> Any:
    return results[METRIC_MAP[wm_type]]


def compact_results(results: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bit_accuracy",
        "p_value",
        "value",
        "l1_dist",
        "detection_success",
        "log_message",
        "prc_threshold",
        "message_bits_str_initial",
        "message_bits_str_recovered",
    ]
    return {key: results.get(key) for key in keys}


def generate_watermarked(pipe_provider_target: Any, wm_provider: Any, prompt: str, wm_zT: Any, args: argparse.Namespace) -> Image.Image:
    if hasattr(wm_provider, "generate"):
        generated = wm_provider.generate(
            pipe_provider_target=pipe_provider_target,
            prompts=prompt,
            latents=wm_zT,
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
    else:
        generated = pipe_provider_target.generate(
            prompts=prompt,
            latents=wm_zT,
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
    return generated["images_PIL"][0]


def generate_clean(pipe_provider_target: Any, prompt: str, base_zT: Any, args: argparse.Namespace) -> Image.Image:
    generated = pipe_provider_target.generate(
        prompts=prompt,
        latents=base_zT,
        num_inference_steps=args.num_inference_steps_target,
        guidance_scale=args.guidance_scale_target,
    )
    return generated["images_PIL"][0]


def verify_tree_ring_injection(torch, provider: Any, base_latent: Any, watermarked_latent: Any) -> float:
    """Verify the provider output is exactly base latent plus configured FFT injection."""
    expected_fft = torch.fft.fftshift(torch.fft.fft2(base_latent.float()), dim=(-1, -2))
    expected_fft[provider.watermarking_mask] = provider.gt_patch[provider.watermarking_mask].clone()
    expected = torch.fft.ifft2(torch.fft.ifftshift(expected_fft, dim=(-1, -2))).real
    max_abs = float((expected - watermarked_latent.float()).abs().max().detach().cpu().item())
    if max_abs > 1e-6:
        raise RuntimeError(f"Tree-Ring injection reconstruction mismatch: max_abs={max_abs}")
    return max_abs


def summarize_metadata(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"completed": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"completed": 0}

    def parse_bool(value: str) -> Optional[bool]:
        if str(value).lower() in {"true", "1"}:
            return True
        if str(value).lower() in {"false", "0"}:
            return False
        return None

    successes = [parse_bool(row.get("before_detection_successful", "")) for row in rows]
    successes_valid = [x for x in successes if x is not None]
    return {
        "completed": len(rows),
        "before_detection_rate": (sum(successes_valid) / len(successes_valid)) if successes_valid else None,
        "metadata_csv": str(csv_path),
    }


def run_method(args: argparse.Namespace, wm_type: str, prompt_rows: List[Dict[str, str]], guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from utils.imprint_utils import validate
    from utils.pipe import pipe_utils
    from utils.utils import (
        check_if_detection_successful,
        describe_legacy_detection_threshold,
        get_detection_threshold,
        seed_everything,
        set_random_seed,
    )
    from utils.wm.wm_utils import WmProviders

    wm_provider_cls = WmProviders[wm_type].value
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    # Each method writes under its own canonical data root unless --output_dir is set.
    dataset_dir = method_dataset_dir(args.output_dir, args.dataset_name, wm_type)
    method_dir = dataset_dir / wm_type
    method_dir.mkdir(parents=True, exist_ok=True)
    clean_root = Path(args.clean_output_dir) / args.dataset_name
    if wm_type == "GS":
        # GS clean latents are method-specific (shared sampling uniforms), so they are
        # kept apart from the TR-paired clean images of the same dataset name.
        clean_root = clean_root / "GS"
    clean_root.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(args.num_shards, args.shard_index)
    metadata_csv = method_dir / f"metadata{suffix}.csv"
    summary_json = method_dir / f"summary{suffix}.json"
    completed = existing_completed_rows(metadata_csv)
    detection_threshold = get_detection_threshold(wm_type, args.modelid_target)
    legacy_threshold = describe_legacy_detection_threshold(wm_type, args.modelid_target)

    print(f"[{wm_type}] loading target pipeline: {args.modelid_target}", flush=True)
    set_random_seed(args.seed)
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
        revision=args.model_revision,
    )
    provider_kwargs = vars(args).copy()
    for reserved_key in ("latent_shape", "dtype", "device"):
        provider_kwargs.pop(reserved_key, None)
    if wm_type not in {"TR", "GS"}:
        raise ValueError("paired formal generation currently supports only TR and GS")
    if wm_type == "GS" and args.gs_protocol_mode != "official_compatible":
        raise ValueError(
            "formal GS generation requires --gs_protocol_mode official_compatible; "
            "legacy remains available only through explicitly legacy-labeled runners"
        )
    wm_provider = None
    if wm_type == "TR":
        wm_provider = wm_provider_cls(
            latent_shape=pipe_provider_target.get_latent_shape(),
            dtype=pipe_provider_target.get_dtype(),
            device=device,
            **provider_kwargs,
        )
        watermark_target_sha256 = tensor_sha256(wm_provider.gt_patch)
        watermark_mask_sha256 = tensor_sha256(wm_provider.watermarking_mask)
    else:
        watermark_target_sha256 = None
        watermark_mask_sha256 = canonical_json_hash(
            {"method": "GS", "mask": "not_applicable", "version": 1}
        )
    generation_config = {
        "model_id": args.modelid_target,
        "model_revision": args.model_revision,
        "scheduler": args.scheduler_target,
        "num_inference_steps": args.num_inference_steps_target,
        "guidance_scale": args.guidance_scale_target,
        "resolution": args.resolution,
        "dtype": str(pipe_provider_target.get_dtype()),
    }
    generation_config_sha256 = canonical_json_hash(generation_config)
    if wm_type == "TR":
        watermark_config = {
            "wm_type": "TR",
            "w_seed": int(args.w_seed),
            "w_channel": int(args.w_channel),
            "w_pattern": str(args.w_pattern),
            "w_mask_shape": str(args.w_mask_shape),
            "w_radius": int(args.w_radius),
            "w_measurement": str(args.w_measurement),
            "w_injection": str(args.w_injection),
            "w_pattern_const": float(args.w_pattern_const),
            "watermark_target_sha256": watermark_target_sha256,
            "watermark_mask_sha256": watermark_mask_sha256,
        }
    else:
        watermark_config = {
            "wm_type": "GS",
            "gs_protocol_mode": "official_compatible",
            "official_source_repository": "bsmhmmlf/Gaussian-Shading",
            "official_source_commit": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
            "message_width_in_bytes": int(args.message_width_in_bytes),
            "channel_copy": int(args.gs_channel_copy),
            "hw_copy": int(args.gs_hw_copy),
            "payload_layout": "torch.repeat(1,channel_copy,hw_copy,hw_copy)",
            "cipher": "PyCryptodome.ChaCha20",
            "key_bytes": 32,
            "nonce_bytes": 12,
            "sampling": "norm.ppf((u+encrypted_bit)/2)",
            "majority_vote": "strict_gt_half_ties_zero",
            "secret_mapping": "secret_index=run_id",
            "sampling_seed_policy": "base_seed+run_id",
            "watermark_mask_sha256": watermark_mask_sha256,
        }
    watermark_config_sha256 = canonical_json_hash(watermark_config)
    message_bits_str_initial = None

    rows_written = 0
    try:
        for local_idx, prompt_row in enumerate(prompt_rows):
            run_id = int(prompt_row["run_id"])
            item_dir = method_dir / f"{run_id:06d}"
            image_path = item_dir / "watermarked.png"
            clean_path = clean_root / f"{run_id:06d}.png"
            run_provider = wm_provider
            gs_results = None
            if wm_type == "GS":
                gs_sampling_seed = deterministic_gs_sampling_seed(args.seed, run_id)
                run_kwargs = provider_kwargs.copy()
                run_kwargs.update({
                    "offset": run_id,
                    "gs_secret_index": run_id,
                    "gs_sampling_seed": gs_sampling_seed,
                    "gs_protocol_mode": "official_compatible",
                })
                run_provider = wm_provider_cls(
                    latent_shape=pipe_provider_target.get_latent_shape(),
                    dtype=pipe_provider_target.get_dtype(),
                    device=device,
                    **run_kwargs,
                )
                gs_results = run_provider.get_wm_latents()
            if run_id in completed:
                if wm_type == "GS":
                    validate_gs_resume_provenance(
                        completed[run_id],
                        run_id=run_id,
                        sampling_seed=gs_sampling_seed,
                        wm_results=gs_results,
                    )
                del gs_results, run_provider
                if local_idx == 0 or (local_idx + 1) % 25 == 0:
                    print(f"[{wm_type}] skip existing run_id={run_id}", flush=True)
                continue

            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/before")
            if wm_type in ["PRC", "SPH"]:
                seed_everything(args.seed)

            if image_path.exists() or clean_path.exists():
                raise FileExistsError(
                    f"unrecorded paired output exists for run_id={run_id}: "
                    f"clean={clean_path.exists()} watermarked={image_path.exists()}"
                )
            if wm_type == "TR":
                base_latent_seed = int(args.seed) + run_id
                generator = torch.Generator(device="cpu").manual_seed(base_latent_seed)
                base_cpu = torch.randn(
                    tuple(pipe_provider_target.get_latent_shape()),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                base_latent = base_cpu.to(device=device, dtype=pipe_provider_target.get_dtype())
                wm_results = run_provider.get_wm_latents(latents_clean=base_latent)
            else:
                base_latent_seed = gs_sampling_seed
                wm_results = gs_results
                base_cpu = wm_results["zT_clean_torch"].detach().to("cpu", dtype=torch.float32)
                base_latent = wm_results["zT_clean_torch"]
            clean_zT = wm_results["zT_clean_torch"]
            wm_zT = wm_results["zT_torch"]
            base_latent_sha256 = tensor_sha256(base_latent)
            clean_base_latent_sha256 = tensor_sha256(clean_zT)
            watermarked_base_latent_sha256 = tensor_sha256(base_latent)
            if clean_base_latent_sha256 != base_latent_sha256:
                raise RuntimeError(f"provider changed clean base latent run_id={run_id}")
            if wm_type == "TR":
                injection_max_abs_error = verify_tree_ring_injection(
                    torch, run_provider, clean_zT, wm_zT
                )
            else:
                injection_max_abs_error = None
                watermark_target_sha256 = tensor_sha256(wm_results["barcodes_torch"])
                secret_provenance = wm_results["secret_provenance_list"][0]
            watermarked_latent_sha256 = tensor_sha256(wm_zT)
            if watermarked_latent_sha256 == base_latent_sha256:
                raise RuntimeError(f"watermark made no latent change run_id={run_id}")

            print(
                f"[{wm_type}] generating paired {local_idx + 1}/{len(prompt_rows)} "
                f"run_id={run_id} base_seed={base_latent_seed}",
                flush=True,
            )
            clean_image = generate_clean(
                pipe_provider_target, prompt_row["prompt"], clean_zT, args
            )
            watermarked_image = generate_watermarked(
                pipe_provider_target, run_provider, prompt_row["prompt"], wm_zT, args
            )
            item_dir.mkdir(parents=True, exist_ok=True)
            clean_image.save(clean_path)
            watermarked_image.save(image_path)

            if args.validate_before:
                before = validate(
                    out_dir=str(method_dir),
                    image_to_verify_PIL=watermarked_image,
                    original_PIL=watermarked_image,
                    wm_provider=run_provider,
                    pipe_provider_target=pipe_provider_target,
                    num_inference_steps_target=args.num_inference_steps_target,
                    step=-1,
                    message_bits_str_initial=(
                        wm_results.get("message_bits_str_list", [message_bits_str_initial])[0]
                    ),
                    do_psnr=False,
                    do_ssim=False,
                    do_msssim=False,
                    do_lpips=False,
                    device=str(device),
                )
                before_metric_value = detection_value(before, wm_type)
                if wm_type == "GS":
                    # Official Gaussian Shading detection (default gs_detection_mode
                    # official_onebit, beta-tail tau_onebit, ">="). The generation
                    # model/scheduler stay the shared formal cohort (RedbeardNZ + DDIM);
                    # only the detection-success threshold family is official here.
                    before_successful = run_provider.is_detection_successful(before_metric_value)
                else:
                    before_successful = check_if_detection_successful(
                        wm_type=wm_type,
                        threshold=detection_threshold,
                        value=before_metric_value,
                    )
                before_compact = compact_results(before)
            else:
                before_successful = None
                before_metric_value = None
                before_compact = {}

            if wm_type == "GS":
                gs_detection_info = run_provider.active_detection_threshold()
                row_detection_threshold = gs_detection_info["threshold"]
                row_detection_threshold_type = gs_detection_info["threshold_type"]
            else:
                gs_detection_info = None
                row_detection_threshold = detection_threshold
                row_detection_threshold_type = legacy_threshold["threshold_type"]
            row = {
                "protocol": GS_PAIRING_PROTOCOL if wm_type == "GS" else PAIRING_PROTOCOL,
                "dataset_name": args.dataset_name,
                "dataset": args.dataset_name,
                "run_id": run_id,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "prompt_id": prompt_row["prompt_id"],
                "prompt": prompt_row["prompt"],
                "prompt_sha256": hashlib.sha256(prompt_row["prompt"].encode("utf-8")).hexdigest(),
                "source": prompt_row["source"],
                "wm_type": wm_type,
                "wm_name": PAPER_WM_NAMES[wm_type],
                "target_model": args.modelid_target,
                "model_id": args.modelid_target,
                "model_revision": args.model_revision,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                "detection_threshold": row_detection_threshold,
                "detection_threshold_type": row_detection_threshold_type,
                "threshold_calibrated_from_current_clean_negatives": False,
                "detection_metric": METRIC_MAP[wm_type],
                "before_detection_successful": bool_for_json(before_successful),
                "before_detection_metric_value": bool_for_json(before_metric_value),
                "base_latent_seed": base_latent_seed,
                "generation_seed": base_latent_seed,
                "base_latent_sha256": base_latent_sha256,
                "clean_base_latent_sha256": clean_base_latent_sha256,
                "watermarked_base_latent_sha256": watermarked_base_latent_sha256,
                "watermarked_latent_sha256": watermarked_latent_sha256,
                "watermark_target_sha256": watermark_target_sha256,
                "watermark_mask_sha256": watermark_mask_sha256,
                "generation_config_sha256": generation_config_sha256,
                "watermark_config_sha256": watermark_config_sha256,
                "injection_only_difference_verified": wm_type == "TR",
                "pairing_relation": (
                    "shared_sampling_uniforms_distribution_preserving_partition"
                    if wm_type == "GS" else "shared_base_latent_fft_injection_only"
                ),
                "injection_max_abs_error": injection_max_abs_error,
                "clean_path": str(clean_path.resolve()),
                "clean_sha256": sha256_path(clean_path),
                "watermarked_path": str(image_path.resolve()),
                "watermarked_image_path": str(image_path.resolve()),
                "watermarked_sha256": sha256_path(image_path),
            }
            if wm_type == "TR":
                row.update({
                    "w_seed": int(args.w_seed),
                    "w_channel": int(args.w_channel),
                    "w_pattern": str(args.w_pattern),
                    "w_mask_shape": str(args.w_mask_shape),
                    "w_radius": int(args.w_radius),
                    "w_measurement": str(args.w_measurement),
                    "w_injection": str(args.w_injection),
                    "w_pattern_const": float(args.w_pattern_const),
                })
            else:
                row.update({
                    "gs_protocol_mode": "official_compatible",
                    "gs_secret_index": run_id,
                    "gs_message_sha256": secret_provenance["message_sha256"],
                    "gs_key_sha256": secret_provenance["key_sha256"],
                    "gs_nonce_sha256": secret_provenance["nonce_sha256"],
                    "gs_secret_bundle_sha256": secret_provenance["secret_bundle_sha256"],
                    "gs_sampling_seed": gs_sampling_seed,
                    "gs_sampling_uniform_sha256": wm_results["sampling_uniform_sha256_list"][0],
                    "gs_payload_layout": "channel_spatial_repeat",
                    "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
                    "gs_official_tau_onebit": gs_detection_info["official_tau_onebit"],
                    "gs_official_tau_bits": gs_detection_info["official_tau_bits"],
                    "gs_detection_mode": gs_detection_info["detection_mode"],
                    "detection_threshold_comparison_operator": gs_detection_info["comparison_operator"],
                    # Three explicitly-distinct protocol layers (see docs):
                    # 1) watermark implementation protocol, 2) generation benchmark
                    # protocol (this run), 3) the separate upstream reproduction runner.
                    "watermark_implementation_protocol": "official_compatible",
                    "generation_benchmark_protocol": "shared_formal_cohort_redbeardnz_ddim",
                    "upstream_official_reproduction_runner": (
                        "stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16"
                    ),
                })
            row["pairing_sha256"] = build_pairing_sha256(row)
            row.update({f"before_{key}": bool_for_json(value) for key, value in before_compact.items()})
            append_row(metadata_csv, row)
            rows_written += 1

            print(
                f"[{wm_type}] run_id={run_id} before={before_successful} "
                f"metric={before_metric_value}",
                flush=True,
            )
            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/done")
            del base_cpu, base_latent, clean_zT, wm_zT, wm_results, clean_image, watermarked_image
            if wm_type == "GS":
                del run_provider, gs_results
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
        final_rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(final_rows, expected_count=len(prompt_rows), verify_files=True)
    save_json(method_dir / f"pairing_audit{suffix}.json", audit)
    save_json(method_dir / f"generation_config{suffix}.json", generation_config)
    save_json(method_dir / f"watermark_config{suffix}.json", watermark_config)
    summary = summarize_metadata(metadata_csv)
    summary.update({
        "wm_type": wm_type,
        "wm_name": PAPER_WM_NAMES[wm_type],
        "rows_written_this_run": rows_written,
        "target_images_requested": len(prompt_rows),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "pairing_audit": audit,
        "paper_setting_note": (
            "Per-sample official-compatible Gaussian Shading with shared sampling uniforms; "
            "RAVEN/removal not run in this step."
            if wm_type == "GS" else
            "Per-sample paired clean/Tree-Ring generation; RAVEN/removal not run in this step."
        ),
    })
    save_json(summary_json, summary)
    print(f"[{wm_type}] summary: {summary}", flush=True)
    return summary


def main() -> int:
    first = base_parser().parse_known_args()[0]
    # Run-level logging lives with the first requested method's canonical dataset dir.
    first_method = next(
        (
            value.upper()
            for index, value in enumerate(sys.argv)
            if index and sys.argv[index - 1] == "--wm_types"
        ),
        PAPER_WM_METHODS_IN_BENCH[0],
    )
    dataset_dir = method_dataset_dir(first.output_dir, first.dataset_name, first_method)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(first.num_shards, first.shard_index)
    setup_run_logging(dataset_dir, filename=f"run{suffix}.log")
    started_at = utc_timestamp()
    gpu_record = configure_gpu(first.gpu, first.device, dataset_dir, require_free_gpu=first.require_free_gpu)

    status = "failed"
    summaries: Dict[str, Any] = {}
    args_dict: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        args = parse_args()
        args_dict = vars(args).copy()
        os.environ.setdefault("TQDM_DISABLE", "1")

        if args.num_shards <= 0:
            raise ValueError("--num_shards must be positive")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard_index must be in [0, num_shards)")

        from generate.runtime import CpuMemoryGuard
        from raven.runtime import limit_cpu_threads
        import torch

        limit_cpu_threads(1)
        guard = CpuMemoryGuard(
            min_available_gib=args.min_cpu_mem_gb,
            warn_available_gib=args.warn_cpu_mem_gb,
            max_process_rss_gib=args.max_process_ram_gb,
        )
        guard.check("startup")

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        if not args.prompts_csv:
            raise ValueError(
                "--prompts_csv is required; dataset prompt lists live with the clean "
                "data they define, e.g. data/clean/<dataset>/inputs/<prompts>.csv"
            )
        prompt_rows = load_prompts(
            Path(args.prompts_csv),
            args.start_index,
            args.num_pairs,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        if not prompt_rows:
            raise ValueError(
                f"shard {args.shard_index}/{args.num_shards} has no selected prompts"
            )

        settings = {
            "paper_dataset_size_used_for_detection": args.num_pairs,
            "shard_sample_count": len(prompt_rows),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "model_setup": {
                "model_id_requested_by_paper": "stabilityai/stable-diffusion-2-1-base",
                "model_id_used": args.modelid_target,
                "resolution": args.resolution,
                "cfg_scale": args.guidance_scale_target,
                "steps": args.num_inference_steps_target,
                "scheduler": args.scheduler_target,
            },
            "watermark_methods": args.wm_types,
            "paper_watermark_methods_not_available_in_eval_bench_wm": [
                "DwtDct", "DwtDctSvd", "RivaGAN", "StegaStamp", "StableSignature",
                "Zodiac", "ROBIN", "TrustMark", "VINE",
            ],
            "stage": "generate_watermarked_images_only",
        }
        save_json(dataset_dir / f"paper_settings{suffix}.json", settings)

        for wm_type in args.wm_types:
            summaries[wm_type] = run_method(args, wm_type, prompt_rows, guard, device)
            guard.check(f"{args.dataset_name}/{wm_type}/method_complete")
        status = "completed"
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        finished_at = utc_timestamp()
        gpu_record = finalize_gpu_logging(dataset_dir, gpu_record)
        extra = {"method_summaries": summaries}
        if error:
            extra["error"] = error
        write_experiment_records(
            dataset_dir,
            args_dict or vars(first),
            gpu_record,
            started_at,
            finished_at,
            status,
            extra,
            filename=f"results{suffix}.json",
        )


if __name__ == "__main__":
    raise SystemExit(main())
