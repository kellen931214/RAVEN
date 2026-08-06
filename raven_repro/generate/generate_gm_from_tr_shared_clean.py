#!/usr/bin/env python3
"""Generate a GaussMarker cohort from the canonical Tree-Ring clean source.

Protocol: ``gaussmarker_shared_tr_clean_v2``.

This runner regenerates ONLY the GM watermarked images. The Tree-Ring clean
images, the Tree-Ring watermarked images and the Tree-Ring metadata are
read-only inputs; they are never rewritten, copied, re-encoded, renamed or
regenerated, and every clean image this run reads is re-hashed afterwards to
prove it.

Per Tree-Ring source row:

1. rebuild the canonical base latent from ``base_latent_seed`` with the
   repository's canonical CPU float32 procedure and verify it against both
   ``base_latent_sha256`` and ``clean_base_latent_sha256``;
2. verify ``prompt`` / ``prompt_sha256``, the clean image against
   ``clean_sha256``, and the full generation configuration against
   ``generation_config_sha256``;
3. hand that exact tensor to the authoritative provider entrypoint
   ``GmProvider.get_wm_latents_from_base_latent``, which applies the official
   ChaCha20 encrypted message through the truncated-Gaussian quantile partition
   and then the official Tree-Ring ring injection;
4. fail closed unless the provider demonstrably consumed the supplied latent —
   same tensor storage for the clean side, and the recovered uniforms must
   re-derive to the supplied ones;
5. generate and save only ``watermarked.png``.

RAVEN shared-clean profile, NOT end-to-end official GaussMarker parity
----------------------------------------------------------------------
Official ``gaussmarker_gen.py`` draws its own per-sample latent with
``truncnorm.rvs`` from the legacy global NumPy RNG, under SD 2.1 fp16 with the
DPMSolver scheduler. This cohort keeps the GaussMarker *math* unchanged (the same
bundle identity, ChaCha20 state, encrypted message, ring target and mask, the
same quantile partition and the same complex ring injection) but

* takes the truncated-Gaussian draw deterministically from the canonical
  Tree-Ring latent instead of the legacy RNG, and
* runs the Tree-Ring generation configuration (RedbeardNZ SD 2.1-base, DDIM, 50
  steps, guidance 7.5, float32) because the shared clean images were produced by
  it.

Both differences are recorded in every row and in ``watermark_config.json``. The
result must never be described as official GaussMarker generation parity.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
GENERATE_ROOT = _HERE.parent                      # raven_repro/generate/
RAVEN_ROOT = _HERE.parents[1]                      # raven_repro/
WORKSPACE = _HERE.parents[2]                       # repo root
BENCH_ROOT = WORKSPACE / "eval_bench_wm"           # repo/eval_bench_wm
for _root in (str(GENERATE_ROOT), str(RAVEN_ROOT), str(BENCH_ROOT)):
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
    GM_SHARED_TR_CLEAN_MODE,
    GM_SHARED_TR_CLEAN_PROTOCOL,
    GM_UNIFORM_DERIVATION,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
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

#: Maximum tolerated error when re-deriving the supplied uniforms from the
#: provider's pre-injection latent.
#:
#: The partition is computed in float64 but the pre-injection latent is stored as
#: float32 (the cohort's compute dtype), so the round trip carries one float32
#: rounding: |du| <= pdf(z) * |z| * eps32 which is under 1e-6 for any latent this
#: cohort can contain. This is still an identity check and not a similarity
#: threshold — a latent that did not drive the partition gives an O(0.1) error,
#: five orders of magnitude away from the bound.
UNIFORM_ROUNDTRIP_TOLERANCE = 1e-6


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a GaussMarker cohort from the canonical Tree-Ring clean "
            "latents and clean images (shared-clean protocol v2)"
        )
    )
    add_common_cli_args(parser, default_tr_metadata=DEFAULT_TR_METADATA)
    parser.add_argument(
        "--gm-bundle-dir",
        type=Path,
        required=True,
        help=(
            "persisted GM bundle (manifest.json + w1.pth + w2.pth). The cohort's "
            "ChaCha20 state and ring target come from here and are never redrawn."
        ),
    )
    parser.add_argument(
        "--create-bundle",
        type=str_to_bool,
        default=False,
        help=(
            "create the bundle when it does not exist yet. Only honoured for a "
            "brand-new cohort: once a metadata CSV exists the bundle must already "
            "be complete, so a resume can never leave a stray bundle behind."
        ),
    )
    parser.add_argument("--gm-watermark-bits-seed", type=int, default=None)
    parser.add_argument("--gm-channel-copy", type=int, default=1)
    parser.add_argument("--gm-w-copy", type=int, default=8)
    parser.add_argument("--gm-h-copy", type=int, default=8)
    parser.add_argument("--gm-w-seed", type=int, default=999999)
    parser.add_argument("--gm-w-channel", type=int, default=3)
    parser.add_argument("--gm-w-pattern", type=str, default="ring")
    parser.add_argument("--gm-w-mask-shape", type=str, default="circle")
    parser.add_argument("--gm-w-radius", type=int, default=4)
    parser.add_argument("--gm-w-measurement", type=str, default="l1_complex")
    parser.add_argument("--gm-w-injection", type=str, default="complex")
    return parser.parse_args(argv)


def build_provider(GmProvider, args, *, device, dtype_name: str):
    """Instantiate the authoritative provider; no algorithm lives in this file."""
    return GmProvider(
        latent_shape=(1, LATENT_CHANNELS, args.resolution // 8, args.resolution // 8),
        device=device,
        gm_profile="legacy",
        gm_bundle_dir=str(args.gm_bundle_dir),
        gm_create_bundle=bool(args.create_bundle),
        gm_allow_in_memory_state=False,
        gm_torch_dtype=dtype_name,
        gm_channel_copy=args.gm_channel_copy,
        gm_w_copy=args.gm_w_copy,
        gm_h_copy=args.gm_h_copy,
        gm_watermark_bits_seed=args.gm_watermark_bits_seed,
        gm_use_gnr=False,
        gm_gnr_path=None,
        gm_use_classifier=False,
        gm_classifier_path=None,
        modelid_target=args.model_id,
        model_revision=args.model_revision,
        scheduler_target=args.scheduler_target,
        resolution=args.resolution,
        w_seed=args.gm_w_seed,
        w_channel=args.gm_w_channel,
        w_pattern=args.gm_w_pattern,
        w_mask_shape=args.gm_w_mask_shape,
        w_radius=args.gm_w_radius,
        w_measurement=args.gm_w_measurement,
        w_injection=args.gm_w_injection,
    )


def assert_provider_consumed_latent(
    np,
    torch,
    wm_results: Dict[str, Any],
    *,
    shared_clean_latent,
    base_latent_sha256: str,
    run_id: int,
) -> Dict[str, Any]:
    """Fail closed unless the provider embedded into the *supplied* latent.

    Three independent proofs, because "the SHA in the row matches" is exactly the
    thing an unsound run would also report:

    1. ``zT_clean_torch`` must be the supplied tensor's own storage;
    2. its SHA-256 must still be the canonical one;
    3. the uniforms implied by the provider's pre-injection latent must
       re-derive to ``norm.cdf`` of the supplied latent, which is only possible
       if that latent drove the truncated-Gaussian partition.
    """
    from scipy.stats import norm

    clean = wm_results["zT_clean_torch"]
    if clean.data_ptr() != shared_clean_latent.data_ptr():
        raise SharedCleanError(
            f"GM provider replaced the supplied clean latent run_id={run_id}: "
            "zT_clean_torch is not the tensor that was passed in"
        )
    actual = tensor_sha256(clean)
    if actual != base_latent_sha256:
        raise SharedCleanError(
            f"GM provider changed the shared clean latent run_id={run_id}: "
            f"expected {base_latent_sha256}, got {actual}"
        )

    pre = wm_results["gm_pre_frequency_latent"].detach().cpu().numpy().astype(np.float64)
    bits = (pre > 0).astype(np.float64)
    recovered = 2.0 * norm.cdf(pre) - bits
    expected = norm.cdf(shared_clean_latent.detach().cpu().numpy().astype(np.float64))
    max_error = float(np.max(np.abs(recovered - expected)))
    if not np.isfinite(max_error) or max_error > UNIFORM_ROUNDTRIP_TOLERANCE:
        raise SharedCleanError(
            f"GM pre-injection latent was not derived from the supplied base latent "
            f"run_id={run_id}: max uniform round-trip error {max_error}"
        )

    watermarked_latent_sha256 = tensor_sha256(wm_results["zT_torch"])
    if watermarked_latent_sha256 == base_latent_sha256:
        raise SharedCleanError(f"watermark made no latent change run_id={run_id}")
    return {
        "watermarked_latent_sha256": watermarked_latent_sha256,
        "uniform_roundtrip_max_abs_error": max_error,
    }


def run(args: argparse.Namespace, guard: Any, device: Any) -> Dict[str, Any]:
    import numpy as np
    import torch
    from utils.pipe import pipe_utils
    from utils.utils import set_random_seed
    from utils.wm import gm_provider as gm_provider_module
    from utils.wm.gm_provider import GM_SHARED_TR_CLEAN_MODE as PROVIDER_MODE
    from utils.wm.gm_provider import GM_UNIFORM_DERIVATION as PROVIDER_DERIVATION
    from utils.wm.gm_provider import GmProvider

    if PROVIDER_MODE != GM_SHARED_TR_CLEAN_MODE:
        raise SharedCleanError(
            "provider and protocol disagree on the shared-clean mode name: "
            f"{PROVIDER_MODE!r} != {GM_SHARED_TR_CLEAN_MODE!r}"
        )
    if PROVIDER_DERIVATION != GM_UNIFORM_DERIVATION:
        raise SharedCleanError(
            "provider and protocol disagree on the uniform derivation: "
            f"{PROVIDER_DERIVATION!r} != {GM_UNIFORM_DERIVATION!r}"
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
    manifest_path = run_manifest_path(method_dir, suffix)
    selected_run_ids = sorted(int(row["run_id"]) for row in selected)
    provenance = entrypoint_provenance(Path(__file__), gm_provider_module)
    git = git_provenance(WORKSPACE)

    # --- gate 1: run manifest, before any pipeline is loaded or bundle built ---
    stored_manifest = preflight_run_manifest(
        manifest_path,
        {
            "shared_clean_source_metadata_sha256": tr_metadata_sha256,
            "selected_run_ids": json.dumps(selected_run_ids),
            "entrypoint_sha256": provenance["entrypoint_sha256"],
            "gm_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
            "smoke_only": bool(args.smoke_only),
        },
        resume=args.resume,
    )

    # --- gate 2: existing cohort re-audit; never triggers bundle creation ---
    completed = existing_completed_rows(metadata_csv, resume=args.resume)
    create_bundle = bool(args.create_bundle)
    if (completed or stored_manifest is not None) and create_bundle:
        create_bundle = False
        print(
            "[GM-v2] existing run found; bundle creation disabled for this run",
            flush=True,
        )
    args.create_bundle = create_bundle

    print(f"[GM-v2] loading target pipeline: {args.model_id}", flush=True)
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
    generation_config_sha256 = verify_generation_config(tr_rows, generation_config)

    provider = build_provider(GmProvider, args, device=device, dtype_name="float32")
    if provider.bundle is None:
        raise SharedCleanError("GM shared-clean generation requires a persisted bundle")
    if provider.state_source not in {"bundle", "bundle_created"}:
        raise SharedCleanError(
            f"GM watermark state must come from a bundle, got {provider.state_source!r}"
        )
    bundle_manifest = provider.bundle.public_manifest()

    gm_state = {
        "gm_bundle_dir": str(Path(args.gm_bundle_dir).resolve()),
        "gm_bundle_config_sha256": str(bundle_manifest["bundle_config_sha256"]),
        "gm_w1_file_sha256": str(bundle_manifest["w1_file_sha256"]),
        "gm_w2_file_sha256": str(bundle_manifest["w2_file_sha256"]),
        "gm_state_source": str(provider.state_source),
    }

    watermark_mask_sha256 = tensor_sha256(provider.watermarking_mask)
    watermark_target_sha256 = tensor_sha256(provider.gt_patch.real.contiguous())
    watermark_config = {
        "wm_type": "GM",
        "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
        "official_source_repository": "SunnierLee/GaussMarker",
        "official_source_commit": "4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155",
        "official_math_claim": (
            "official GaussMarker dual-domain embedding: the bundle's ChaCha20 "
            "encrypted message through the truncated-Gaussian quantile partition "
            "norm.ppf((u+bit)/2), followed by the official complex ring injection"
        ),
        "not_claimed": (
            "NOT end-to-end official GaussMarker generation parity: official "
            "gaussmarker_gen.py draws its own truncnorm latent from the legacy "
            "NumPy RNG under SD2.1 fp16 + DPMSolver. This cohort derives the "
            "truncated-Gaussian draw from the canonical Tree-Ring clean latent "
            "and runs the Tree-Ring float32 DDIM configuration."
        ),
        "gm_channel_copy": int(args.gm_channel_copy),
        "gm_w_copy": int(args.gm_w_copy),
        "gm_h_copy": int(args.gm_h_copy),
        "w_seed": int(args.gm_w_seed),
        "w_channel": int(args.gm_w_channel),
        "w_pattern": args.gm_w_pattern,
        "w_mask_shape": args.gm_w_mask_shape,
        "w_radius": int(args.gm_w_radius),
        "w_measurement": args.gm_w_measurement,
        "w_injection": args.gm_w_injection,
        "cipher": "PyCryptodome.ChaCha20",
        "sampling": "norm.ppf((u+encrypted_bit)/2)",
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "gm_uniform_derivation": GM_UNIFORM_DERIVATION,
        "gm_bundle_watermark_sha256": str(bundle_manifest.get("watermark_sha256", "")),
        "gm_bundle_m_sha256": str(bundle_manifest.get("m_sha256", "")),
        "gm_bundle_config_sha256": gm_state["gm_bundle_config_sha256"],
        "watermark_mask_sha256": watermark_mask_sha256,
        "watermark_target_sha256": watermark_target_sha256,
        "provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
    }
    watermark_config_sha256 = canonical_json_sha256(watermark_config)

    # --- gate 3: the full run identity, now that the bundle and pipeline exist ---
    run_manifest = finalize_run_manifest(
        manifest_path,
        stored_manifest,
        {
            "protocol": GM_SHARED_TR_CLEAN_PROTOCOL,
            "method": "GM",
            "dataset_name": args.dataset_name,
            "shared_clean_source_metadata_path": str(tr_metadata),
            "shared_clean_source_metadata_sha256": tr_metadata_sha256,
            "selected_run_ids": json.dumps(selected_run_ids),
            "selected_run_id_count": len(selected_run_ids),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "generation_config_sha256": generation_config_sha256,
            "watermark_config_sha256": watermark_config_sha256,
            "gm_bundle_config_sha256": gm_state["gm_bundle_config_sha256"],
            "gm_w1_file_sha256": gm_state["gm_w1_file_sha256"],
            "gm_w2_file_sha256": gm_state["gm_w2_file_sha256"],
            "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
            "entrypoint_path": provenance["entrypoint_path"],
            "entrypoint_sha256": provenance["entrypoint_sha256"],
            "gm_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
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

            guard.check(f"{args.dataset_name}/GM/run_id={run_id}/before")
            base_cpu, shared_clean_latent, base_latent_sha256 = rebuild_shared_clean_latent(
                torch, tr_row, resolution=args.resolution, device=device, dtype=pipe_dtype
            )
            prompt_sha256 = verify_source_prompt(tr_row)
            clean_path = verify_source_clean_image(tr_row)
            clean_guard.snapshot(clean_path, expected_sha256=str(tr_row["clean_sha256"]))

            wm_results = provider.get_wm_latents_from_base_latent(shared_clean_latent)
            consumption = assert_provider_consumed_latent(
                np,
                torch,
                wm_results,
                shared_clean_latent=shared_clean_latent,
                base_latent_sha256=base_latent_sha256,
                run_id=run_id,
            )
            watermarked_latent_sha256 = consumption["watermarked_latent_sha256"]
            wm_zT = wm_results["zT_torch"]

            if run_id in completed:
                assert_resume_fields(
                    completed[run_id],
                    {
                        "protocol": GM_SHARED_TR_CLEAN_PROTOCOL,
                        "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
                        "gm_uniform_derivation": GM_UNIFORM_DERIVATION,
                        "gm_bundle_config_sha256": gm_state["gm_bundle_config_sha256"],
                        "gm_w1_file_sha256": gm_state["gm_w1_file_sha256"],
                        "gm_w2_file_sha256": gm_state["gm_w2_file_sha256"],
                        "gm_watermark_sha256": wm_results["gm_watermark_sha256"],
                        "gm_m_sha256": wm_results["gm_m_sha256"],
                        "gm_target_sha256": wm_results["gm_target_sha256"],
                        "gm_mask_sha256": wm_results["gm_mask_sha256"],
                        "gm_pre_injection_latent_sha256": wm_results[
                            "gm_pre_injection_latent_sha256"
                        ],
                        "gm_post_injection_latent_sha256": wm_results[
                            "gm_post_injection_latent_sha256"
                        ],
                        "gm_sampling_uniform_sha256": wm_results["gm_sampling_uniform_sha256"],
                        "gm_provider_entrypoint_sha256": provenance[
                            "provider_entrypoint_sha256"
                        ],
                        "watermarked_latent_sha256": watermarked_latent_sha256,
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
                    },
                    run_id=run_id,
                )
                assert_recorded_output(completed[run_id], run_id=run_id, label="GM")
                skipped += 1
                del base_cpu, shared_clean_latent, wm_zT, wm_results
                gc.collect()
                continue

            if image_path.exists():
                raise SharedCleanError(
                    f"unrecorded GM output already exists for run_id={run_id}: {image_path}"
                )

            print(
                f"[GM-v2] generating {local_idx + 1}/{len(selected)} run_id={run_id} "
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
            clean_guard.assert_unchanged(clean_path)

            row: Dict[str, Any] = {
                "protocol": GM_SHARED_TR_CLEAN_PROTOCOL,
                "dataset_name": args.dataset_name,
                "dataset": args.dataset_name,
                "run_id": run_id,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "prompt_id": tr_row["prompt_id"],
                "prompt": tr_row["prompt"],
                "prompt_sha256": prompt_sha256,
                "source": tr_row.get("source", ""),
                "wm_type": "GM",
                "wm_name": "GaussMarker",
                "target_model": args.model_id,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                # --- shared-clean identity (must equal the TR row) ---
                "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
                "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
                "shared_clean_source_metadata_path": str(tr_metadata),
                "shared_clean_source_metadata_sha256": tr_metadata_sha256,
                "shared_clean_sample_sha256": base_latent_sha256,
                "shared_clean_profile": "raven_shared_tr_clean_v2_not_official_gm_generation",
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
                "watermark_target_sha256": watermark_target_sha256,
                "watermark_mask_sha256": watermark_mask_sha256,
                "generation_config_sha256": generation_config_sha256,
                "watermark_config_sha256": watermark_config_sha256,
                "injection_only_difference_verified": False,
                "pairing_relation": (
                    "shared_tr_clean_latent_gaussmarker_quantile_partition_plus_ring"
                ),
                "injection_max_abs_error": "",
                "uniform_roundtrip_max_abs_error": consumption[
                    "uniform_roundtrip_max_abs_error"
                ],
                "clean_path": str(tr_row["clean_path"]),
                "clean_sha256": str(tr_row["clean_sha256"]),
                "watermarked_path": str(image_path.resolve()),
                "watermarked_image_path": str(image_path.resolve()),
                "watermarked_sha256": sha256_path(image_path),
                # --- GaussMarker state / bundle identity ---
                "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
                "gm_uniform_derivation": GM_UNIFORM_DERIVATION,
                "gm_state_source": gm_state["gm_state_source"],
                "gm_bundle_dir": gm_state["gm_bundle_dir"],
                "gm_bundle_config_sha256": gm_state["gm_bundle_config_sha256"],
                "gm_w1_file_sha256": gm_state["gm_w1_file_sha256"],
                "gm_w2_file_sha256": gm_state["gm_w2_file_sha256"],
                "gm_watermark_sha256": wm_results["gm_watermark_sha256"],
                "gm_m_sha256": wm_results["gm_m_sha256"],
                "gm_target_sha256": wm_results["gm_target_sha256"],
                "gm_mask_sha256": wm_results["gm_mask_sha256"],
                "gm_pre_injection_latent_sha256": wm_results["gm_pre_injection_latent_sha256"],
                "gm_post_injection_latent_sha256": wm_results["gm_post_injection_latent_sha256"],
                "gm_sampling_uniform_sha256": wm_results["gm_sampling_uniform_sha256"],
                "gm_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
                "gm_provider_entrypoint_path": provenance["provider_entrypoint_path"],
                "entrypoint_path": provenance["entrypoint_path"],
                "entrypoint_sha256": provenance["entrypoint_sha256"],
                "git_branch": git["git_branch"],
                "git_commit": git["git_commit"],
                "git_dirty": git["git_dirty"],
                "smoke_only": bool(args.smoke_only),
                "incomplete": bool(args.smoke_only),
                "formal_output_eligible": not bool(args.smoke_only),
                "run_config_sha256": run_manifest["run_config_sha256"],
                "watermark_implementation_protocol": GM_SHARED_TR_CLEAN_MODE,
                "generation_benchmark_protocol": "shared_formal_cohort_redbeardnz_ddim",
                "upstream_official_reproduction_runner": (
                    "stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16"
                ),
            }
            row["pairing_sha256"] = build_pairing_sha256(row)
            append_row(metadata_csv, row)
            rows_written += 1
            gate_records.append(
                {
                    "run_id": run_id,
                    "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                    "gm_base_latent_sha256": base_latent_sha256,
                    "tr_clean_sha256": str(tr_row["clean_sha256"]),
                    "gm_clean_sha256": str(row["clean_sha256"]),
                    "gm_watermarked_latent_sha256": watermarked_latent_sha256,
                    "gm_watermarked_sha256": row["watermarked_sha256"],
                    "uniform_roundtrip_max_abs_error": consumption[
                        "uniform_roundtrip_max_abs_error"
                    ],
                }
            )
            print(
                f"[GM-v2] run_id={run_id} watermarked_sha256={row['watermarked_sha256'][:16]}…",
                flush=True,
            )
            guard.check(f"{args.dataset_name}/GM/run_id={run_id}/done")
            del base_cpu, shared_clean_latent, wm_zT, wm_results
            del watermarked_image, generated
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
        gm_rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(gm_rows, expected_count=len(selected), verify_files=True)
    # Coverage is explicit: this run must have produced a row for exactly the
    # run_ids it selected — no missing, extra or duplicated rows.
    cross = audit_shared_clean_cohorts(
        tr_rows,
        {"GM": gm_rows},
        verify_files=True,
        require_methods=("GM",),
        expected_run_ids=selected_run_ids,
        tr_metadata_path=tr_metadata,
    )
    clean_report = clean_guard.assert_unchanged()
    save_json(method_dir / f"pairing_audit{suffix}.json", audit)
    save_json(method_dir / f"cross_method_shared_clean_audit{suffix}.json", cross)
    save_json(method_dir / f"generation_config{suffix}.json", generation_config)
    save_json(method_dir / f"watermark_config{suffix}.json", watermark_config)
    save_json(
        method_dir / f"clean_source_integrity{suffix}.json",
        {"checked": len(clean_report), "clean_images": clean_report},
    )

    summary = {
        "protocol": GM_SHARED_TR_CLEAN_PROTOCOL,
        "shared_clean_profile": "raven_shared_tr_clean_v2_not_official_gm_generation",
        "dataset_name": args.dataset_name,
        "metadata_csv": str(metadata_csv),
        "completed": len(gm_rows),
        "rows_written_this_run": rows_written,
        "rows_verified_and_skipped": skipped,
        "selected_source_rows": len(selected),
        "tr_source_metadata": str(tr_metadata),
        "tr_source_metadata_sha256": tr_metadata_sha256,
        "tr_source_rows": len(tr_rows),
        "gm_bundle": gm_state,
        "entrypoint": provenance,
        "git": git,
        "run_manifest_path": str(manifest_path),
        "run_config_sha256": run_manifest["run_config_sha256"],
        "selected_run_ids": selected_run_ids,
        "smoke_only": bool(args.smoke_only),
        "incomplete": bool(args.smoke_only),
        "formal_output_eligible": not bool(args.smoke_only),
        "pairing_audit": audit,
        "cross_method_shared_clean_audit": {
            key: value for key, value in cross.items() if key != "rows"
        },
        "clean_images_verified_unchanged": len(clean_report),
        "clean_images_generated": 0,
        "clean_images_copied": 0,
        "gate_records": gate_records,
        "paper_setting_note": (
            "GaussMarker embedded with the official ChaCha20 message, quantile "
            "partition and ring injection, driven by the canonical Tree-Ring clean "
            "latent; the clean images are the existing Tree-Ring clean images, "
            "untouched. This is the RAVEN shared-clean profile, not official "
            "end-to-end GaussMarker generation parity."
        ),
    }
    save_json(method_dir / f"summary{suffix}.json", summary)
    print(f"[GM-v2] summary: {json.dumps(summary['pairing_audit'], sort_keys=True)}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.output_dir is None:
        args.output_dir = method_data_root("GM") / args.dataset_name / "GM"
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
