#!/usr/bin/env python3
"""Generic RID/HSTR/HSQR shared Tree-Ring clean runner plumbing.

This module owns orchestration only. Watermark algorithms remain in the
method-specific providers under ``eval_bench_wm/utils/wm``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENTS_ROOT.parent
RAVEN_ROOT = WORKSPACE / "raven_repro"
BENCH_ROOT = WORKSPACE / "eval_bench_wm"
for _root in (str(EXPERIMENTS_ROOT), str(RAVEN_ROOT), str(BENCH_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from raven.eval_protocol import method_data_root, source_metadata_path  # noqa: E402
from raven.gpu_utils import (  # noqa: E402
    configure_gpu,
    finalize_gpu_logging,
    setup_run_logging,
    utc_timestamp,
    write_experiment_records,
)
from raven.pairing_provenance import (  # noqa: E402
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    SHARED_FOURIER_METHOD_CONFIG,
    audit_pairing_rows,
    audit_shared_clean_cohorts,
    build_pairing_sha256,
)
from shared_clean_tr import (  # noqa: E402
    LATENT_CHANNELS,
    CleanImageGuard,
    SharedCleanError,
    add_common_cli_args,
    append_row,
    assert_recorded_output,
    assert_resume_fields,
    canonical_json_sha256,
    entrypoint_provenance,
    existing_completed_rows,
    finalize_run_manifest,
    git_provenance,
    load_tr_rows,
    preflight_run_manifest,
    rebuild_shared_clean_latent,
    run_manifest_path,
    save_json,
    select_rows,
    sha256_path,
    shard_suffix,
    str_to_bool,
    tensor_sha256,
    verify_generation_config,
    verify_source_clean_image,
    verify_source_prompt,
)

DEFAULT_TR_METADATA = source_metadata_path("TR", "diffusiondb")
DEFAULT_DATASET_NAME = "diffusiondb_shared_tr"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    wm_name: str
    protocol: str
    protocol_mode: str
    provider_module: str
    provider_class: str
    provider_entrypoint_field: str
    bundle_arg: str
    create_bundle_arg: str
    shared_profile: str
    pairing_relation: str
    official_source_repository: str
    official_source_commit: str
    official_math_claim: str
    not_claimed: str


def _base_parser(spec: MethodSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Generate {spec.wm_name} watermarked images from canonical TR clean latents"
    )
    add_common_cli_args(parser, default_tr_metadata=DEFAULT_TR_METADATA)
    parser.add_argument(f"--{spec.method.lower()}-bundle-dir", dest=spec.bundle_arg, type=Path, required=True)
    parser.add_argument(f"--{spec.method.lower()}-create-bundle", dest=spec.create_bundle_arg, type=str_to_bool, default=False)
    return parser


def parse_fourier_args(spec: MethodSpec, argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = _base_parser(spec)
    if spec.method == "RID":
        parser.add_argument("--rid-profile", default="legacy", choices=["legacy", "official_sd21", "paper_shift_ablation"])
        parser.add_argument("--rid-key-index", default=628, type=int)
        parser.add_argument("--rid-key-seed", default=42, type=int)
        parser.add_argument("--rid-shift-semantics", default="official_code_exact")
        parser.add_argument("--rid-torch-dtype", default="float32")
    elif spec.method == "HSTR":
        parser.add_argument("--hstr-profile", default=spec.protocol_mode)
        parser.add_argument("--hstr-key-index", default=1, type=int)
        parser.add_argument("--hstr-rng-device", default="cpu", choices=["cpu", "cuda"])
        parser.add_argument("--hstr-save-full-keybook", action="store_true", default=False)
    elif spec.method == "HSQR":
        parser.add_argument("--hsqr-profile", default="official_sfwmark_sd21")
        parser.add_argument("--hsqr-key-index", default=0, type=int)
        parser.add_argument("--hsqr-save-keybook", action="store_true", default=False)
        parser.add_argument("--hsqr-key-policy", default="fixed", choices=["fixed", "per_sample"])
    else:
        raise ValueError(f"unsupported spec {spec.method}")
    return parser.parse_args(argv)


def _import_provider(spec: MethodSpec):
    module = __import__(f"utils.wm.{spec.provider_module}", fromlist=[spec.provider_class])
    return module, getattr(module, spec.provider_class)


def _provider_for_spec(spec: MethodSpec, Provider, args, *, device, latent_shape, pipe_dtype):
    bundle_dir = Path(getattr(args, spec.bundle_arg)).resolve()
    create_bundle = bool(getattr(args, spec.create_bundle_arg))
    common = dict(
        latent_shape=latent_shape,
        device=device,
        modelid_target=args.model_id,
        model_revision=args.model_revision,
        scheduler_target=args.scheduler_target,
        resolution=args.resolution,
    )
    if spec.method == "RID":
        return Provider(
            **common,
            rid_profile=args.rid_profile,
            rid_bundle_dir=str(bundle_dir),
            rid_key_index=args.rid_key_index,
            rid_key_seed=args.rid_key_seed,
            rid_key_rng_device="cpu",
            rid_key_rng_dtype="float32",
            rid_torch_dtype="float32",
            rid_shift_semantics=args.rid_shift_semantics,
            rid_create_bundle=create_bundle,
        )
    if spec.method == "HSTR":
        return Provider(
            **common,
            dtype=pipe_dtype,
            hstr_profile=args.hstr_profile,
            hstr_key_index=args.hstr_key_index,
            hstr_bundle_dir=str(bundle_dir),
            hstr_create_bundle=create_bundle,
            hstr_save_full_keybook=args.hstr_save_full_keybook,
            hstr_rng_device=args.hstr_rng_device,
        )
    if spec.method == "HSQR":
        from utils.wm import sfw_bundle
        bundle = None
        if bundle_dir.exists() and not create_bundle:
            bundle = sfw_bundle.SfwBundle.load(bundle_dir)
            return Provider.from_bundle(bundle, latent_shape=latent_shape, device=device)
        provider = Provider(
            **common,
            hsqr_profile=args.hsqr_profile,
            hsqr_key_index=args.hsqr_key_index,
            hsqr_key_policy=args.hsqr_key_policy,
            hsqr_torch_dtype="float32",
        )
        provider.create_bundle(bundle_dir, save_keybook=args.hsqr_save_keybook)
        return provider
    raise ValueError(spec.method)


def _bundle_state(spec: MethodSpec, provider: Any, args: argparse.Namespace) -> Dict[str, Any]:
    bundle_dir = str(Path(getattr(args, spec.bundle_arg)).resolve())
    manifest = provider.bundle.public_manifest() if getattr(provider, "bundle", None) is not None else {}
    return {
        f"{spec.method.lower()}_bundle_dir": bundle_dir,
        f"{spec.method.lower()}_bundle_config_sha256": str(manifest.get("bundle_config_sha256", "")),
        f"{spec.method.lower()}_state_source": str(getattr(provider, "state_source", "bundle" if manifest else "in_memory")),
        f"{spec.method.lower()}_selected_pattern_sha256": str(
            manifest.get("selected_pattern_sha256")
            or getattr(provider, "selected_pattern_sha256", None)
            or (provider.pattern_sha256() if hasattr(provider, "pattern_sha256") else "")
        ),
        f"{spec.method.lower()}_mask_sha256": str(
            manifest.get("mask_sha256")
            or getattr(provider, "watermark_mask_sha256", None)
            or getattr(provider, "rid_mask_sha256", None)
            or (provider.mask_sha256() if hasattr(provider, "mask_sha256") else "")
        ),
        f"{spec.method.lower()}_key_index": int(
            getattr(provider, "key_index", getattr(provider, "selected_key_index", 0))
        ),
    }


def _watermark_config(spec: MethodSpec, provider: Any, state: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    provider_config = provider.provider_config() if hasattr(provider, "provider_config") else provider.detector_config()
    provider_config_sha = provider.provider_config_sha256() if hasattr(provider, "provider_config_sha256") else canonical_json_sha256(provider_config)
    return {
        "wm_type": spec.method,
        "protocol_mode": spec.protocol_mode,
        f"{spec.method.lower()}_protocol_mode": spec.protocol_mode,
        "official_source_repository": spec.official_source_repository,
        "official_source_commit": spec.official_source_commit,
        "official_math_claim": spec.official_math_claim,
        "not_claimed": spec.not_claimed,
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "provider_config": provider_config,
        f"{spec.method.lower()}_provider_config_sha256": provider_config_sha,
        "watermark_target_sha256": state[f"{spec.method.lower()}_selected_pattern_sha256"],
        "watermark_mask_sha256": state[f"{spec.method.lower()}_mask_sha256"],
        "provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
    }


def _resume_checks(spec: MethodSpec, row: Mapping[str, Any], tr_row: Mapping[str, Any], *,
                   prompt_sha256: str, tr_metadata_sha256: str, base_latent_sha256: str,
                   watermarked_latent_sha256: str, watermark_config_sha256: str,
                   generation_config_sha256: str, state: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = spec.method.lower()
    checks = {
        "protocol": spec.protocol,
        f"{prefix}_protocol_mode": spec.protocol_mode,
        f"{prefix}_bundle_config_sha256": state[f"{prefix}_bundle_config_sha256"],
        f"{prefix}_selected_pattern_sha256": state[f"{prefix}_selected_pattern_sha256"],
        f"{prefix}_mask_sha256": state[f"{prefix}_mask_sha256"],
        f"{prefix}_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
        "watermarked_latent_sha256": watermarked_latent_sha256,
        "watermark_target_sha256": state[f"{prefix}_selected_pattern_sha256"],
        "watermark_mask_sha256": state[f"{prefix}_mask_sha256"],
        "watermark_config_sha256": watermark_config_sha256,
        "generation_config_sha256": generation_config_sha256,
        "base_latent_sha256": base_latent_sha256,
        "clean_base_latent_sha256": base_latent_sha256,
        "watermark_pre_injection_base_latent_sha256": base_latent_sha256,
        "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
        "clean_path": str(tr_row["clean_path"]),
        "clean_sha256": str(tr_row["clean_sha256"]),
        "base_latent_seed": str(tr_row["base_latent_seed"]),
        "prompt_sha256": prompt_sha256,
        "shared_clean_source_metadata_sha256": tr_metadata_sha256,
    }
    return checks


def run_fourier_shared_clean(spec: MethodSpec, args: argparse.Namespace, guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from utils.pipe import pipe_utils
    from utils.utils import set_random_seed

    provider_module, Provider = _import_provider(spec)
    tr_metadata = Path(args.tr_metadata).resolve()
    tr_rows = load_tr_rows(tr_metadata)
    tr_metadata_sha256 = sha256_path(tr_metadata)
    selected = select_rows(tr_rows, num_shards=args.num_shards, shard_index=args.shard_index,
                           run_ids=args.run_ids, limit=args.limit)
    selected_run_ids = sorted(int(row["run_id"]) for row in selected)

    method_dir = Path(args.output_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(args.num_shards, args.shard_index)
    metadata_csv = method_dir / f"metadata{suffix}.csv"
    manifest_path = run_manifest_path(method_dir, suffix)
    provenance = entrypoint_provenance(Path(sys.argv[0] or __file__), provider_module)
    git = git_provenance(WORKSPACE)
    prefix = spec.method.lower()

    stored_manifest = preflight_run_manifest(
        manifest_path,
        {
            "shared_clean_source_metadata_sha256": tr_metadata_sha256,
            "selected_run_ids": json.dumps(selected_run_ids),
            "entrypoint_sha256": provenance["entrypoint_sha256"],
            spec.provider_entrypoint_field: provenance["provider_entrypoint_sha256"],
            "smoke_only": bool(args.smoke_only),
        },
        resume=args.resume,
    )
    completed = existing_completed_rows(metadata_csv, resume=args.resume)
    if (completed or stored_manifest is not None) and bool(getattr(args, spec.create_bundle_arg)):
        setattr(args, spec.create_bundle_arg, False)
        print(f"[{spec.method}-v2] existing run found; bundle creation disabled", flush=True)

    print(f"[{spec.method}-v2] loading target pipeline: {args.model_id}", flush=True)
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
        raise SharedCleanError(f"pipeline latent shape {latent_shape} does not match TR {expected_shape}")
    if pipe_dtype != torch.float32:
        raise SharedCleanError(f"shared-clean generation requires float32 latents, got {pipe_dtype}")

    generation_config = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "scheduler": args.scheduler_target,
        "num_inference_steps": args.num_inference_steps_target,
        "guidance_scale": args.guidance_scale_target,
        "resolution": args.resolution,
        "dtype": str(pipe_dtype),
    }
    generation_config_sha256 = verify_generation_config(tr_rows, generation_config)

    provider = _provider_for_spec(spec, Provider, args, device=device, latent_shape=latent_shape, pipe_dtype=pipe_dtype)
    if getattr(provider, "bundle", None) is None:
        raise SharedCleanError(f"{spec.method} shared-clean generation requires a persisted bundle")
    state = _bundle_state(spec, provider, args)
    watermark_config = _watermark_config(spec, provider, state, provenance)
    watermark_config_sha256 = canonical_json_sha256(watermark_config)
    run_manifest = finalize_run_manifest(
        manifest_path,
        stored_manifest,
        {
            "protocol": spec.protocol,
            "method": spec.method,
            "dataset_name": args.dataset_name,
            "shared_clean_source_metadata_path": str(tr_metadata),
            "shared_clean_source_metadata_sha256": tr_metadata_sha256,
            "selected_run_ids": json.dumps(selected_run_ids),
            "selected_run_id_count": len(selected_run_ids),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "generation_config_sha256": generation_config_sha256,
            "watermark_config_sha256": watermark_config_sha256,
            f"{prefix}_bundle_config_sha256": state[f"{prefix}_bundle_config_sha256"],
            f"{prefix}_protocol_mode": spec.protocol_mode,
            "entrypoint_path": provenance["entrypoint_path"],
            "entrypoint_sha256": provenance["entrypoint_sha256"],
            spec.provider_entrypoint_field: provenance["provider_entrypoint_sha256"],
            "git_branch": git["git_branch"],
            "git_commit": git["git_commit"],
            "smoke_only": bool(args.smoke_only),
            "incomplete": bool(args.smoke_only),
            "formal_output_eligible": not bool(args.smoke_only),
        },
    )

    clean_guard = CleanImageGuard()
    rows_written = 0
    skipped = 0
    gate_records: List[Dict[str, Any]] = []
    try:
        for local_idx, tr_row in enumerate(selected):
            run_id = int(tr_row["run_id"])
            item_dir = method_dir / f"{run_id:06d}"
            image_path = item_dir / "watermarked.png"

            guard.check(f"{args.dataset_name}/{spec.method}/run_id={run_id}/before")
            _, shared_clean_latent, base_latent_sha256 = rebuild_shared_clean_latent(
                torch, tr_row, resolution=args.resolution, device=device, dtype=pipe_dtype
            )
            prompt_sha256 = verify_source_prompt(tr_row)
            clean_path = verify_source_clean_image(tr_row)
            clean_guard.snapshot(clean_path, expected_sha256=str(tr_row["clean_sha256"]))
            wm_results = provider.get_wm_latents_from_base_latent(shared_clean_latent)
            watermarked_latent_sha256 = tensor_sha256(wm_results["zT_torch"])
            if str(wm_results[f"{prefix}_pre_injection_latent_sha256"]) != base_latent_sha256:
                raise SharedCleanError(f"{spec.method} provider did not consume canonical latent run_id={run_id}")

            if run_id in completed:
                assert_resume_fields(
                    completed[run_id],
                    _resume_checks(spec, completed[run_id], tr_row, prompt_sha256=prompt_sha256,
                                   tr_metadata_sha256=tr_metadata_sha256,
                                   base_latent_sha256=base_latent_sha256,
                                   watermarked_latent_sha256=watermarked_latent_sha256,
                                   watermark_config_sha256=watermark_config_sha256,
                                   generation_config_sha256=generation_config_sha256,
                                   state=state, provenance=provenance),
                    run_id=run_id,
                )
                assert_recorded_output(completed[run_id], run_id=run_id, label=spec.method)
                skipped += 1
                del shared_clean_latent, wm_results
                gc.collect()
                continue

            if image_path.exists():
                raise SharedCleanError(f"unrecorded {spec.method} output already exists run_id={run_id}: {image_path}")
            print(f"[{spec.method}-v2] generating {local_idx + 1}/{len(selected)} run_id={run_id}", flush=True)
            generated = pipe_provider_target.generate(
                prompts=tr_row["prompt"],
                latents=wm_results["zT_torch"],
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
            watermarked_image = generated["images_PIL"][0]
            item_dir.mkdir(parents=True, exist_ok=True)
            watermarked_image.save(image_path)
            clean_guard.assert_unchanged(clean_path)
            watermarked_sha256 = sha256_path(image_path)
            row: Dict[str, Any] = {
                "protocol": spec.protocol,
                "dataset_name": args.dataset_name,
                "dataset": args.dataset_name,
                "run_id": run_id,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "prompt_id": tr_row["prompt_id"],
                "prompt": tr_row["prompt"],
                "prompt_sha256": prompt_sha256,
                "source": tr_row.get("source", ""),
                "wm_type": spec.method,
                "wm_name": spec.wm_name,
                "target_model": args.model_id,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
                "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
                "shared_clean_source_metadata_path": str(tr_metadata),
                "shared_clean_source_metadata_sha256": tr_metadata_sha256,
                "shared_clean_sample_sha256": base_latent_sha256,
                "shared_clean_profile": spec.shared_profile,
                "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                "tr_clean_path": str(tr_row["clean_path"]),
                "tr_clean_sha256": str(tr_row["clean_sha256"]),
                "base_latent_seed": int(tr_row["base_latent_seed"]),
                "generation_seed": int(tr_row["base_latent_seed"]),
                "base_latent_sha256": base_latent_sha256,
                "clean_base_latent_sha256": base_latent_sha256,
                "watermarked_base_latent_sha256": base_latent_sha256,
                "watermark_pre_injection_base_latent_sha256": base_latent_sha256,
                "watermarked_latent_sha256": watermarked_latent_sha256,
                "watermark_target_sha256": state[f"{prefix}_selected_pattern_sha256"],
                "watermark_mask_sha256": state[f"{prefix}_mask_sha256"],
                "generation_config_sha256": generation_config_sha256,
                "watermark_config_sha256": watermark_config_sha256,
                "injection_only_difference_verified": False,
                "pairing_relation": spec.pairing_relation,
                "injection_max_abs_error": "",
                "clean_path": str(tr_row["clean_path"]),
                "clean_sha256": str(tr_row["clean_sha256"]),
                "watermarked_path": str(image_path.resolve()),
                "watermarked_image_path": str(image_path.resolve()),
                "watermarked_sha256": watermarked_sha256,
                f"{prefix}_protocol_mode": spec.protocol_mode,
                f"{prefix}_state_source": state[f"{prefix}_state_source"],
                f"{prefix}_bundle_dir": state[f"{prefix}_bundle_dir"],
                f"{prefix}_bundle_config_sha256": state[f"{prefix}_bundle_config_sha256"],
                f"{prefix}_selected_pattern_sha256": state[f"{prefix}_selected_pattern_sha256"],
                f"{prefix}_mask_sha256": state[f"{prefix}_mask_sha256"],
                f"{prefix}_key_index": state[f"{prefix}_key_index"],
                f"{prefix}_pre_injection_latent_sha256": wm_results[f"{prefix}_pre_injection_latent_sha256"],
                f"{prefix}_post_injection_latent_sha256": wm_results[f"{prefix}_post_injection_latent_sha256"],
                f"{prefix}_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
                f"{prefix}_provider_entrypoint_path": provenance["provider_entrypoint_path"],
                "entrypoint_path": provenance["entrypoint_path"],
                "entrypoint_sha256": provenance["entrypoint_sha256"],
                "git_branch": git["git_branch"],
                "git_commit": git["git_commit"],
                "git_dirty": git["git_dirty"],
                "smoke_only": bool(args.smoke_only),
                "incomplete": bool(args.smoke_only),
                "formal_output_eligible": not bool(args.smoke_only),
                "run_config_sha256": run_manifest["run_config_sha256"],
                "watermark_implementation_protocol": spec.protocol_mode,
                "generation_benchmark_protocol": "shared_formal_cohort_redbeardnz_ddim",
                "upstream_official_reproduction_runner": spec.not_claimed,
            }
            for key, value in wm_results.items():
                if key.startswith(prefix + "_") and key not in row and not hasattr(value, "shape"):
                    row[key] = value
            row["pairing_sha256"] = build_pairing_sha256(row)
            append_row(metadata_csv, row)
            rows_written += 1
            gate_records.append({
                "run_id": run_id,
                "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                f"{prefix}_base_latent_sha256": base_latent_sha256,
                "tr_clean_sha256": str(tr_row["clean_sha256"]),
                f"{prefix}_clean_sha256": row["clean_sha256"],
                f"{prefix}_watermarked_latent_sha256": watermarked_latent_sha256,
                f"{prefix}_watermarked_sha256": watermarked_sha256,
            })
            guard.check(f"{args.dataset_name}/{spec.method}/run_id={run_id}/done")
            del shared_clean_latent, wm_results, watermarked_image, generated
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
        method_rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(method_rows, expected_count=len(selected), verify_files=True)
    cross = audit_shared_clean_cohorts(
        tr_rows,
        {spec.method: method_rows},
        verify_files=True,
        require_methods=(spec.method,),
        expected_run_ids=selected_run_ids,
        tr_metadata_path=tr_metadata,
    )
    clean_report = clean_guard.assert_unchanged()
    save_json(method_dir / f"pairing_audit{suffix}.json", audit)
    save_json(method_dir / f"cross_method_shared_clean_audit{suffix}.json", cross)
    save_json(method_dir / f"generation_config{suffix}.json", generation_config)
    save_json(method_dir / f"watermark_config{suffix}.json", watermark_config)
    save_json(method_dir / f"clean_source_integrity{suffix}.json", {"checked": len(clean_report), "clean_images": clean_report})
    summary = {
        "protocol": spec.protocol,
        "shared_clean_profile": spec.shared_profile,
        "dataset_name": args.dataset_name,
        "metadata_csv": str(metadata_csv),
        "completed": len(method_rows),
        "rows_written_this_run": rows_written,
        "rows_verified_and_skipped": skipped,
        "selected_source_rows": len(selected),
        "tr_source_metadata": str(tr_metadata),
        "tr_source_metadata_sha256": tr_metadata_sha256,
        "tr_source_rows": len(tr_rows),
        "bundle": state,
        "entrypoint": provenance,
        "git": git,
        "run_manifest_path": str(manifest_path),
        "run_config_sha256": run_manifest["run_config_sha256"],
        "selected_run_ids": selected_run_ids,
        "smoke_only": bool(args.smoke_only),
        "incomplete": bool(args.smoke_only),
        "formal_output_eligible": not bool(args.smoke_only),
        "pairing_audit": audit,
        "cross_method_shared_clean_audit": {key: value for key, value in cross.items() if key != "rows"},
        "clean_images_verified_unchanged": len(clean_report),
        "clean_images_generated": 0,
        "clean_images_copied": 0,
        "gate_records": gate_records,
        "paper_setting_note": spec.not_claimed,
    }
    save_json(method_dir / f"summary{suffix}.json", summary)
    print(f"[{spec.method}-v2] summary: {json.dumps(summary['pairing_audit'], sort_keys=True)}", flush=True)
    return summary


def main_for_spec(spec: MethodSpec, argv: Optional[List[str]] = None) -> int:
    args = parse_fourier_args(spec, argv)
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.output_dir is None:
        args.output_dir = method_data_root(spec.method) / args.dataset_name / spec.method
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(args.num_shards, args.shard_index)
    setup_run_logging(args.output_dir, filename=f"run{suffix}.log")
    started_at = utc_timestamp()
    gpu_record = configure_gpu(args.gpu, args.device, args.output_dir, require_free_gpu=args.require_free_gpu)
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
            raise SharedCleanError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        os.environ.setdefault("TQDM_DISABLE", "1")
        summary = run_fourier_shared_clean(spec, args, guard, device)
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
