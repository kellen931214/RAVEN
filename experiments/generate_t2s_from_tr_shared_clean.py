#!/usr/bin/env python3
"""Generate a T2SMark cohort from the canonical Tree-Ring clean source.

Protocol: ``t2smark_shared_tr_clean_v2``.

This runner regenerates ONLY the T2S watermarked images. The Tree-Ring clean
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
   ``T2SProvider.new_sample(base_latent=...)``, which splits it across the key
   and message channel groups and uses it as the tail-truncated Gaussian source;
4. fail closed unless the provider demonstrably consumed the supplied latent:
   ``state.base_latent_sha256`` must be the canonical SHA and the multiset of
   ``|z|`` must be unchanged by encoding (T2S only reorders and re-signs its
   source's magnitudes, so this is an exact invariant);
5. generate and save only ``watermarked.png``, plus the portable per-sample T2S
   state JSON that standalone verification needs.

RAVEN shared-clean profile, NOT end-to-end official T2S parity
--------------------------------------------------------------
Upstream ``run.py`` always draws its own ``z`` with ``torch.randn``. This cohort
keeps the T2S encoder unchanged — same key pattern, same session key and message
lifecycle, same repeated-bit codeword, same tail/central magnitude split, same
``official_compatible`` RNG semantics for the key/message/noise-sign draws — but
supplies the canonical Tree-Ring latent as the Gaussian source, and runs the
Tree-Ring generation configuration. Both differences are recorded in every row
and in ``watermark_config.json``.
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
    T2S_SHARED_TR_CLEAN_MODE,
    T2S_SHARED_TR_CLEAN_PROTOCOL,
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
    git_provenance,
    load_tr_rows,
    rebuild_shared_clean_latent,
    save_json,
    select_rows,
    sha256_path,
    shard_suffix,
    tensor_sha256,
    verify_generation_config,
    verify_source_clean_image,
    verify_source_prompt,
)

DEFAULT_TR_METADATA = source_metadata_path("TR", "diffusiondb")
DEFAULT_DATASET_NAME = "diffusiondb_shared_tr"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a T2SMark cohort from the canonical Tree-Ring clean latents "
            "and clean images (shared-clean protocol v2)"
        )
    )
    add_common_cli_args(parser, default_tr_metadata=DEFAULT_TR_METADATA)
    parser.add_argument("--t2s-key-channel-idx", type=int, default=0)
    parser.add_argument("--t2s-key-length", type=int, default=16)
    parser.add_argument("--t2s-msg-length", type=int, default=256)
    parser.add_argument("--t2s-tau", type=float, default=0.674)
    parser.add_argument(
        "--t2s-rng-mode",
        type=str,
        default="official_compatible",
        choices=["official_compatible", "raven_deterministic"],
    )
    parser.add_argument(
        "--t2s-inversion-mode",
        type=str,
        default="t2s_official",
        choices=["t2s_official", "benchmark_ddim"],
    )
    parser.add_argument("--t2s-num-inversion-steps", type=int, default=10)
    return parser.parse_args(argv)


def bits_sha256(bits: str) -> str:
    return hashlib.sha256(str(bits).encode("utf-8")).hexdigest()


def assert_provider_consumed_latent(
    t2s_provider_module,
    state,
    latents,
    *,
    shared_clean_latent,
    base_latent_sha256: str,
    run_id: int,
) -> Dict[str, Any]:
    """Fail closed unless the encoder embedded into the *supplied* latent.

    ``state.base_latent_sha256`` proves which tensor was handed in;
    the ``|z|`` multiset digest proves it was actually used, because T2S rebuilds
    its output purely by reordering and re-signing its Gaussian source's
    magnitudes. A fresh ``torch.randn`` could not reproduce that digest.
    """
    if state.base_latent_sha256 is None:
        raise SharedCleanError(
            f"T2S state has no base_latent_sha256 run_id={run_id}: the provider was "
            "not given the canonical latent"
        )
    if str(state.base_latent_sha256) != base_latent_sha256:
        raise SharedCleanError(
            f"T2S bound the wrong base latent run_id={run_id}: "
            f"expected {base_latent_sha256}, got {state.base_latent_sha256}"
        )
    if list(state.latent_shape) != list(shared_clean_latent.shape):
        raise SharedCleanError(
            f"T2S latent shape mismatch run_id={run_id}: state={state.latent_shape} "
            f"shared={list(shared_clean_latent.shape)}"
        )

    expected = t2s_provider_module.abs_magnitude_multiset_sha256(shared_clean_latent)
    actual = t2s_provider_module.abs_magnitude_multiset_sha256(latents)
    if actual != expected:
        raise SharedCleanError(
            f"T2S encoder did not consume the supplied base latent run_id={run_id}: "
            f"|z| multiset {actual} != {expected}"
        )

    watermarked_latent_sha256 = tensor_sha256(latents)
    if watermarked_latent_sha256 == base_latent_sha256:
        raise SharedCleanError(f"watermark made no latent change run_id={run_id}")
    if str(state.watermarked_latent_sha256) != watermarked_latent_sha256:
        raise SharedCleanError(
            f"T2S state watermarked_latent_sha256 disagrees with the tensor "
            f"run_id={run_id}"
        )
    return {
        "watermarked_latent_sha256": watermarked_latent_sha256,
        "abs_magnitude_sha256": actual,
    }


def run(args: argparse.Namespace, guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from utils.pipe import pipe_utils
    from utils.utils import set_random_seed
    from utils.wm import t2s_provider as t2s_provider_module
    from utils.wm.t2s_provider import T2S_SHARED_TR_CLEAN_MODE as PROVIDER_MODE
    from utils.wm.t2s_provider import T2SProvider

    if PROVIDER_MODE != T2S_SHARED_TR_CLEAN_MODE:
        raise SharedCleanError(
            "provider and protocol disagree on the shared-clean mode name: "
            f"{PROVIDER_MODE!r} != {T2S_SHARED_TR_CLEAN_MODE!r}"
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
    state_dir = method_dir / "watermark_state"
    completed = existing_completed_rows(metadata_csv, resume=args.resume)

    print(f"[T2S-v2] loading target pipeline: {args.model_id}", flush=True)
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

    provider = T2SProvider(
        latent_shape=latent_shape,
        dtype=pipe_dtype,
        device=device,
        t2s_key_channel_idx=args.t2s_key_channel_idx,
        t2s_key_length=args.t2s_key_length,
        t2s_msg_length=args.t2s_msg_length,
        t2s_tau=args.t2s_tau,
        t2s_fix_key=False,
        t2s_rng_mode=args.t2s_rng_mode,
        t2s_inversion_mode=args.t2s_inversion_mode,
        t2s_num_inversion_steps=args.t2s_num_inversion_steps,
    )
    provider_config = provider.provider_config()
    provider_config_sha256 = provider.provider_config_sha256()
    if provider.t2s_channels != LATENT_CHANNELS:
        raise SharedCleanError(
            f"T2S channel layout mismatch: provider={provider.t2s_channels} "
            f"cohort={LATENT_CHANNELS}"
        )
    provenance = entrypoint_provenance(Path(__file__), t2s_provider_module)

    watermark_mask_sha256 = canonical_json_sha256(
        {
            "method": "T2S",
            "mask": "key_pattern_derived_per_sample",
            "key_channels": list(provider.key_channels),
            "msg_channels": list(provider.msg_channels),
            "version": 1,
        }
    )
    watermark_config = {
        "wm_type": "T2S",
        "t2s_protocol_mode": T2S_SHARED_TR_CLEAN_MODE,
        "official_source_repository": "0xD009/T2SMark",
        "official_source_commit": "0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
        "official_math_claim": (
            "unchanged T2S tail-truncated encoder: upstream key pattern, session "
            "key and message lifecycle, repeated-bit codeword, and tail/central "
            "magnitude split"
        ),
        "not_claimed": (
            "NOT end-to-end upstream T2S generation parity: upstream run.py always "
            "draws its own z with torch.randn. This cohort supplies the canonical "
            "Tree-Ring clean latent as the Gaussian source and runs the Tree-Ring "
            "float32 DDIM configuration."
        ),
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "t2s_provider_config": provider_config,
        "t2s_provider_config_sha256": provider_config_sha256,
        "watermark_mask_sha256": watermark_mask_sha256,
        "provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
    }
    watermark_config_sha256 = canonical_json_sha256(watermark_config)

    clean_guard = CleanImageGuard()
    git = git_provenance(WORKSPACE)
    rows_written = 0
    skipped = 0
    gate_records: List[Dict[str, Any]] = []
    try:
        for local_idx, tr_row in enumerate(selected):
            run_id = int(tr_row["run_id"])
            item_dir = method_dir / f"{run_id:06d}"
            image_path = item_dir / "watermarked.png"
            state_path = state_dir / f"{run_id:06d}.json"

            guard.check(f"{args.dataset_name}/T2S/run_id={run_id}/before")
            base_cpu, shared_clean_latent, base_latent_sha256 = rebuild_shared_clean_latent(
                torch, tr_row, resolution=args.resolution, device=device, dtype=pipe_dtype
            )
            prompt_sha256 = verify_source_prompt(tr_row)
            clean_path = verify_source_clean_image(tr_row)
            clean_guard.snapshot(clean_path, expected_sha256=str(tr_row["clean_sha256"]))

            # The T2S sample seed is the canonical TR base-latent seed, so the
            # key/message lifecycle is reproducible from the shared source row.
            sample_seed = int(tr_row["base_latent_seed"])
            latents, state = provider.new_sample(
                sample_seed=sample_seed,
                watermark_id=f"t2s-shared-tr-clean-{run_id:06d}",
                base_latent=shared_clean_latent,
                model_id=args.model_id,
                model_revision=args.model_revision,
                scheduler=args.scheduler_target,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
                resolution=args.resolution,
                prompt_sha256=prompt_sha256,
            )
            consumption = assert_provider_consumed_latent(
                t2s_provider_module,
                state,
                latents,
                shared_clean_latent=shared_clean_latent,
                base_latent_sha256=base_latent_sha256,
                run_id=run_id,
            )
            watermarked_latent_sha256 = consumption["watermarked_latent_sha256"]
            if str(state.provider_config_sha256) != provider_config_sha256:
                raise SharedCleanError(
                    f"T2S provider config drift run_id={run_id}: "
                    f"{state.provider_config_sha256} != {provider_config_sha256}"
                )
            if state.rng_mode != args.t2s_rng_mode or state.inversion_mode != args.t2s_inversion_mode:
                raise SharedCleanError(
                    f"T2S RNG/inversion profile drift run_id={run_id}"
                )
            target_bits_sha256 = canonical_json_sha256(
                {
                    "session_key_bits": state.expected_session_key_bits,
                    "message_bits": state.expected_message_bits,
                }
            )

            if run_id in completed:
                # The stored state was signed *after* the image existed, so the
                # re-derived state must be bound to the recorded image SHA before
                # its digest can be compared.
                resume_state = dataclass_replace(
                    state, image_sha256=str(completed[run_id]["watermarked_sha256"])
                )
                assert_resume_fields(
                    completed[run_id],
                    {
                        "protocol": T2S_SHARED_TR_CLEAN_PROTOCOL,
                        "t2s_protocol_mode": T2S_SHARED_TR_CLEAN_MODE,
                        "t2s_rng_mode": args.t2s_rng_mode,
                        "t2s_inversion_mode": args.t2s_inversion_mode,
                        "t2s_watermark_id": state.watermark_id,
                        "t2s_state_sha256": resume_state.state_sha256(),
                        "t2s_provider_config_sha256": provider_config_sha256,
                        "t2s_base_latent_sha256": base_latent_sha256,
                        "t2s_abs_magnitude_sha256": consumption["abs_magnitude_sha256"],
                        "t2s_master_key_sha256": bits_sha256(state.master_key_bits),
                        "t2s_session_key_sha256": bits_sha256(
                            state.expected_session_key_bits
                        ),
                        "t2s_message_sha256": bits_sha256(state.expected_message_bits),
                        "t2s_provider_entrypoint_sha256": provenance[
                            "provider_entrypoint_sha256"
                        ],
                        "watermarked_latent_sha256": watermarked_latent_sha256,
                        "watermark_target_sha256": target_bits_sha256,
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
                assert_recorded_output(completed[run_id], run_id=run_id, label="T2S")
                stored_state_path = Path(str(completed[run_id]["t2s_state_path"]))
                if not stored_state_path.is_file():
                    raise SharedCleanError(
                        f"recorded T2S state artifact missing run_id={run_id}: "
                        f"{stored_state_path}"
                    )
                # The artifact itself must still be the one the row was written
                # for; T2SWatermarkState.load already fails closed on a tampered
                # or unsigned file.
                on_disk = type(state).load(stored_state_path)
                if on_disk.state_sha256() != resume_state.state_sha256():
                    raise SharedCleanError(
                        f"stored T2S state artifact drifted run_id={run_id}: "
                        f"{on_disk.state_sha256()} != {resume_state.state_sha256()}"
                    )
                skipped += 1
                del base_cpu, shared_clean_latent, latents
                gc.collect()
                continue

            for existing, label in ((image_path, "image"), (state_path, "state")):
                if existing.exists():
                    raise SharedCleanError(
                        f"unrecorded T2S {label} already exists for run_id={run_id}: "
                        f"{existing}"
                    )

            print(
                f"[T2S-v2] generating {local_idx + 1}/{len(selected)} run_id={run_id} "
                f"seed={sample_seed}",
                flush=True,
            )
            generated = pipe_provider_target.generate(
                prompts=tr_row["prompt"],
                latents=latents,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
            watermarked_image = generated["images_PIL"][0]
            item_dir.mkdir(parents=True, exist_ok=True)
            watermarked_image.save(image_path)
            clean_guard.assert_unchanged(clean_path)

            watermarked_sha256 = sha256_path(image_path)
            # The portable state records the image it belongs to, so standalone
            # verification cannot be pointed at the wrong file.
            state = dataclass_replace(state, image_sha256=watermarked_sha256)
            state.save(state_path)
            state_sha256 = state.state_sha256()

            row: Dict[str, Any] = {
                "protocol": T2S_SHARED_TR_CLEAN_PROTOCOL,
                "dataset_name": args.dataset_name,
                "dataset": args.dataset_name,
                "run_id": run_id,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "prompt_id": tr_row["prompt_id"],
                "prompt": tr_row["prompt"],
                "prompt_sha256": prompt_sha256,
                "source": tr_row.get("source", ""),
                "wm_type": "T2S",
                "wm_name": "T2SMark",
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
                "shared_clean_profile": "raven_shared_tr_clean_v2_not_official_t2s_generation",
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
                "watermark_target_sha256": target_bits_sha256,
                "watermark_mask_sha256": watermark_mask_sha256,
                "generation_config_sha256": generation_config_sha256,
                "watermark_config_sha256": watermark_config_sha256,
                "injection_only_difference_verified": False,
                "pairing_relation": (
                    "shared_tr_clean_latent_t2s_tail_truncated_reordering"
                ),
                "injection_max_abs_error": "",
                "clean_path": str(tr_row["clean_path"]),
                "clean_sha256": str(tr_row["clean_sha256"]),
                "watermarked_path": str(image_path.resolve()),
                "watermarked_image_path": str(image_path.resolve()),
                "watermarked_sha256": watermarked_sha256,
                # --- T2S state / profile identity ---
                "t2s_protocol_mode": T2S_SHARED_TR_CLEAN_MODE,
                "t2s_rng_mode": args.t2s_rng_mode,
                "t2s_inversion_mode": args.t2s_inversion_mode,
                "t2s_num_inversion_steps": int(args.t2s_num_inversion_steps),
                "t2s_watermark_id": state.watermark_id,
                "t2s_state_path": str(state_path.resolve()),
                "t2s_state_sha256": state_sha256,
                "t2s_provider_config_sha256": provider_config_sha256,
                "t2s_base_latent_sha256": base_latent_sha256,
                "t2s_abs_magnitude_sha256": consumption["abs_magnitude_sha256"],
                "t2s_master_key_sha256": bits_sha256(state.master_key_bits),
                "t2s_session_key_sha256": bits_sha256(state.expected_session_key_bits),
                "t2s_message_sha256": bits_sha256(state.expected_message_bits),
                "t2s_sample_seed": sample_seed,
                "t2s_provider_entrypoint_sha256": provenance["provider_entrypoint_sha256"],
                "t2s_provider_entrypoint_path": provenance["provider_entrypoint_path"],
                "entrypoint_path": provenance["entrypoint_path"],
                "entrypoint_sha256": provenance["entrypoint_sha256"],
                "git_branch": git["git_branch"],
                "git_commit": git["git_commit"],
                "git_dirty": git["git_dirty"],
                "smoke_only": bool(args.smoke_only),
                "formal_output_eligible": not bool(args.smoke_only),
                "watermark_implementation_protocol": T2S_SHARED_TR_CLEAN_MODE,
                "generation_benchmark_protocol": "shared_formal_cohort_redbeardnz_ddim",
                "upstream_official_reproduction_runner": (
                    "T2SMark run.py with its own torch.randn base latent"
                ),
            }
            row["pairing_sha256"] = build_pairing_sha256(row)
            append_row(metadata_csv, row)
            rows_written += 1
            gate_records.append(
                {
                    "run_id": run_id,
                    "tr_base_latent_sha256": str(tr_row["base_latent_sha256"]),
                    "t2s_base_latent_sha256": base_latent_sha256,
                    "tr_clean_sha256": str(tr_row["clean_sha256"]),
                    "t2s_clean_sha256": str(row["clean_sha256"]),
                    "t2s_watermarked_latent_sha256": watermarked_latent_sha256,
                    "t2s_watermarked_sha256": watermarked_sha256,
                    "t2s_abs_magnitude_sha256": consumption["abs_magnitude_sha256"],
                    "t2s_state_sha256": state_sha256,
                }
            )
            print(
                f"[T2S-v2] run_id={run_id} watermarked_sha256={watermarked_sha256[:16]}…",
                flush=True,
            )
            guard.check(f"{args.dataset_name}/T2S/run_id={run_id}/done")
            del base_cpu, shared_clean_latent, latents
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
        t2s_rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(t2s_rows, expected_count=len(t2s_rows), verify_files=True)
    cross = audit_shared_clean_cohorts(
        tr_rows, {"T2S": t2s_rows}, verify_files=True, require_methods=("T2S",)
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
        "protocol": T2S_SHARED_TR_CLEAN_PROTOCOL,
        "shared_clean_profile": "raven_shared_tr_clean_v2_not_official_t2s_generation",
        "dataset_name": args.dataset_name,
        "metadata_csv": str(metadata_csv),
        "completed": len(t2s_rows),
        "rows_written_this_run": rows_written,
        "rows_verified_and_skipped": skipped,
        "selected_source_rows": len(selected),
        "tr_source_metadata": str(tr_metadata),
        "tr_source_metadata_sha256": tr_metadata_sha256,
        "tr_source_rows": len(tr_rows),
        "t2s_provider_config_sha256": provider_config_sha256,
        "t2s_state_dir": str(state_dir),
        "entrypoint": provenance,
        "git": git,
        "smoke_only": bool(args.smoke_only),
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
            "T2SMark embedded with the unchanged upstream tail-truncated encoder, "
            "using the canonical Tree-Ring clean latent as the Gaussian source; the "
            "clean images are the existing Tree-Ring clean images, untouched. This "
            "is the RAVEN shared-clean profile, not official end-to-end T2S "
            "generation parity."
        ),
    }
    save_json(method_dir / f"summary{suffix}.json", summary)
    print(f"[T2S-v2] summary: {json.dumps(summary['pairing_audit'], sort_keys=True)}", flush=True)
    return summary


def dataclass_replace(state, **updates):
    """``dataclasses.replace`` on the provider's frozen state (no local schema)."""
    import dataclasses

    return dataclasses.replace(state, **updates)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    if args.output_dir is None:
        args.output_dir = method_data_root("T2S") / args.dataset_name / "T2S"
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
