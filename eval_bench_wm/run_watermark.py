"""
Script to run the watermark.
"""

import PIL
import PIL.Image
import torch
import torch.nn.functional as F

import pandas as pd

from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.prc_provider import parser as prc_parser
from utils.wm.tag_provider import parser as tag_parser
from utils.wm.ringid_provider import parser as ringid_parser
from utils.wm.hstr_provider import parser as hstr_parser
from utils.wm.hsqr_provider import parser as hsqr_parser
from utils.wm.sph_provider import parser as sph_parser
from utils.wm.t2s_provider import parser as t2s_parser
from utils.wm.maxsive_provider import parser as maxsive_parser
from utils.wm.shallow_provider import parser as shallow_parser
from utils.wm.gm_provider import parser as gm_parser

from utils.pipe import pipe_utils
from utils.imprint_utils import invert_image, validate
from utils.image_utils import distort_images, check_flag
from utils.bit_accuracy import format_bit_accuracy_value
# from utils.finger_utils import decode_tensors, decode_tensors_reverse#, spd_dist_latents, matched_spd_distance
from utils.utils import get_detection_threshold, check_if_detection_successful
from utils.utils import set_random_seed, seed_everything

from utils.prompt_utils import get_text_prompts

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from utils.canonical import sha256_path
from utils.logger import get_logger


model_id = ["CompVis/stable-diffusion-v1-4",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1-base",
            "RedbeardNZ/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            "cagliostrolab/animagine-xl-3.0",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            "THUDM/CogView4-6B"]

model_flux = ["black-forest-labs/FLUX.1-dev",
              "stabilityai/stable-diffusion-3-medium-diffusers",
              "stabilityai/stable-diffusion-3.5-medium",
              "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
              "THUDM/CogView4-6B"]

model_name_mapping = {
    "CompVis/stable-diffusion-v1-4": "sd14",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "stabilityai/stable-diffusion-2-1-base": "sd21",
    "RedbeardNZ/stable-diffusion-2-1-base": "sd21",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "pixart",
    "PixArt-alpha/PixArt-XL-2-512x512": "pixart-xl",
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "stabilityai/stable-diffusion-3.5-medium": "sd35m",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers": "sana",
    "THUDM/CogView4-6B": "cogview4",
}

parent_parsers = [
    gs_parser, tr_parser, prc_parser,
    tag_parser, ringid_parser, hstr_parser,
    hsqr_parser, sph_parser, t2s_parser,
    maxsive_parser, shallow_parser,
    gm_parser,
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
import argparse


def build_parser():
    """Construct the run_watermark argument parser.

    Kept side-effect free (no model loading) so tests can build and parse a
    command line without importing/running the full generation body.
    """
    parser = argparse.ArgumentParser(description="test_watermark", parents=parent_parsers)

    parser.add_argument("--out_dir", type=str, default="out/watermark_gen/")
    parser.add_argument("--target_prompt", type=str, default="cat standing on a rock in front of a crowd of cats, backlighting, digital art, trending on pixiv, fanart")

    # target model
    parser.add_argument("--modelid_target",
                        type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        choices=[model for model in model_id])
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 20 for FLUX, 28 for SD3, 20 for Pix
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3, 4.5 for Pix

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--wm_type",
                        type=str,
                        default="GS",
                        choices=[wm.name for wm in WmProviders])
    parser.add_argument("--distort", action="store_true", default=False)

    # dataset
    parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--logger", action="store_true", default=False)
    return parser


def run_gm_generation(args, argv):
    """Official-compatible GaussMarker generation.

    One command produces ``--num`` watermarked images. The bundle identity
    (watermark bits, encrypted message, key/nonce, ring target) is fixed for the
    whole run while every sample independently samples its *complete* initial
    latent from a deterministic per-sample seed, exactly as official
    ``gaussmarker_gen.py`` does.

    All algorithms live in ``utils/wm/gm_provider.py``; this function only does
    seeding bookkeeping, IO and provenance.
    """
    from utils.wm import gm_bundle, gm_runtime
    from utils.wm.gm_provider import apply_arg_defaults as gm_apply_profile

    profile_info = gm_apply_profile(args, argv)
    print(f"[GM] profile: {profile_info}", flush=True)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images" / "watermarked"
    prompts_dir = out_dir / "prompts"
    metadata_dir = out_dir / "sample_metadata"
    results_path = out_dir / "results.jsonl"
    manifest_path = out_dir / "run_manifest.json"

    gpu_info = gm_runtime.gpu_preflight(DEVICE)
    print(f"[GM] device preflight: {gpu_info}", flush=True)

    # A run that resumes into an existing output directory must not be able to
    # create anything before its manifest has been validated. If the manifest
    # exists, the bundle it was produced with must exist too and is loaded
    # read-only; a new bundle may only be created for a genuinely new run.
    resuming = manifest_path.exists()
    if resuming:
        bundle_dir = getattr(args, "gm_bundle_dir", None)
        if bundle_dir is None or not gm_bundle.GmBundle(Path(bundle_dir)).complete():
            raise RuntimeError(
                f"{manifest_path} already exists, so this run resumes an existing cohort and must "
                f"reuse the bundle that produced it, but --gm_bundle_dir ({bundle_dir}) is not a "
                "complete bundle. Nothing was modified."
            )

    pipe_provider_target = gm_runtime.build_pipe_provider(args, DEVICE)
    wm_provider = gm_runtime.build_provider(
        args,
        latent_shape=pipe_provider_target.get_latent_shape(),
        device=DEVICE,
        create_bundle=not resuming,
    )
    if wm_provider.bundle is None:
        raise RuntimeError(
            "GM generation requires --gm_bundle_dir so that w1/w2 and the manifest persist"
        )
    print(
        f"[GM] bundle {wm_provider.bundle.dir} ({wm_provider.state_source}), "
        f"config {wm_provider.bundle.manifest['bundle_config_sha256'][:16]}",
        flush=True,
    )

    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
    provenance = gm_runtime.run_provenance(args, wm_provider, pipe_provider_target)
    run_config = dict(provenance)
    run_config.update({"base_seed": int(args.seed), "num_samples": int(args.num),
                       "dataset_id": args.dataset_id})
    # ``created_utc`` is volatile, ``gm_state_source`` records whether *this* process
    # created or reloaded the bundle, and ``num_samples`` only says how far the cohort
    # was extended. None of them is part of a *sample's* configuration identity, which
    # is what the resume gate compares; each sample is still bound to its own seed,
    # prompt, bundle and model configuration.
    volatile_fields = ("created_utc", "gm_state_source", "num_samples")
    run_config_sha256 = gm_bundle.canonical_sha256(
        {k: v for k, v in run_config.items() if k not in volatile_fields}
    )

    # The run manifest is the resume gate for the whole output directory and is
    # validated *before* anything is written. An incompatible manifest stops the
    # run with the output directory byte-for-byte untouched; a compatible one is
    # kept verbatim so `created_utc` and the recorded provenance stay those of the
    # run that actually produced the existing samples.
    manifest = gm_runtime.assert_run_manifest_compatible(manifest_path, run_config_sha256)
    if manifest is None:
        manifest = dict(run_config)
        manifest["run_config_sha256"] = run_config_sha256
        manifest["entrypoint"] = "eval_bench_wm/run_watermark.py:run_gm_generation"
        manifest["report_label"] = (
            "official_profile_raw_scores" if wm_provider.profile_is_official else "legacy_or_ablation_mode"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(gm_bundle.canonical_json(manifest) + "\n", encoding="utf-8")

    for directory in (images_dir, prompts_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_id, target_prompt in tqdm(
        list(enumerate(target_prompts)), total=len(target_prompts), desc="GM generation"
    ):
        # Official semantics: seed = i + gen_seed.
        sample_seed = int(args.seed) + int(sample_id)
        name = f"{sample_id:06d}"
        image_path = images_dir / f"{name}.png"
        prompt_path = prompts_dir / f"{name}.txt"
        metadata_path = metadata_dir / f"{name}.json"
        prompt_sha256 = gm_bundle.sha256_text(target_prompt)

        if metadata_path.exists() or image_path.exists():
            # Resume: never overwrite and never silently mix incompatible runs.
            if not (metadata_path.exists() and image_path.exists()):
                raise RuntimeError(
                    f"GM sample {name} is half-written ({image_path.exists()=}, "
                    f"{metadata_path.exists()=}); use a fresh --out_dir"
                )
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            gm_runtime.assert_resumable(
                name,
                existing,
                {
                    "sample_seed": sample_seed,
                    "prompt_sha256": prompt_sha256,
                    "run_config_sha256": run_config_sha256,
                    "gm_bundle_config_sha256": wm_provider.bundle.manifest["bundle_config_sha256"],
                },
            )
            actual_image_sha = gm_bundle.sha256_file(image_path)
            if existing.get("image_sha256") != actual_image_sha:
                raise RuntimeError(
                    f"GM sample {name} image hash changed on disk; refusing to resume"
                )
            rows.append(existing)
            continue

        sample = wm_provider.build_sample_latents(sample_seed)
        generated = wm_provider.generate(
            pipe_provider_target=pipe_provider_target,
            prompts=target_prompt,
            latents=sample["latent"],
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
        image = generated["images_PIL"][0]
        image.save(image_path)
        prompt_path.write_text(target_prompt, encoding="utf-8")

        row = {
            "sample_id": sample_id,
            "sample_name": name,
            "sample_seed": sample_seed,
            "prompt": target_prompt,
            "prompt_sha256": prompt_sha256,
            "image_path": image_path.as_posix(),
            "image_sha256": gm_bundle.sha256_file(image_path),
            "pre_injection_latent_sha256": sample["pre_injection_latent_sha256"],
            "post_injection_latent_sha256": sample["post_injection_latent_sha256"],
            "gm_watermark_sha256": wm_provider.bundle.manifest["watermark_sha256"],
            "gm_m_sha256": wm_provider.bundle.manifest["m_sha256"],
            "gm_target_sha256": wm_provider.bundle.manifest["w2_tensor_sha256"],
            "gm_mask_sha256": gm_bundle.sha256_tensor(wm_provider.watermarking_mask),
            "gm_bundle_config_sha256": wm_provider.bundle.manifest["bundle_config_sha256"],
            "run_config_sha256": run_config_sha256,
        }
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})
        row["created_utc"] = gm_bundle.utc_now()
        metadata_path.write_text(gm_bundle.canonical_json(row) + "\n", encoding="utf-8")
        gm_bundle.append_jsonl(results_path, row)
        rows.append(row)

    latent_hashes = {row["post_injection_latent_sha256"] for row in rows}
    if len(rows) > 1 and len(latent_hashes) != len(rows):
        raise RuntimeError(
            "GM generation produced repeated complete initial latents across samples; "
            "this invalidates the run for independent-sample evaluation"
        )
    print(
        f"[GM] generated {len(rows)} sample(s) into {out_dir}; "
        f"{len(latent_hashes)} distinct complete initial latents; "
        f"watermark identity {wm_provider.bundle.manifest['watermark_sha256'][:16]} (constant)",
        flush=True,
    )
    return rows


def run_hsqr_generation(args, argv):
    """Official-compatible HSQR (SFWMark) generation.

    One command produces ``--num`` non-overwriting watermarked images. Every
    sample independently samples its *complete* base latent from a deterministic
    per-sample seed and the HSQR pattern is injected into a clone of that
    latent, so two samples never share a complete initial latent. The watermark
    identity (key index, seed, payload, pattern) is persisted in a reusable
    bundle that a fresh process can verify against.

    All algorithms live in ``utils/wm/hsqr_provider.py``; this function only does
    seeding bookkeeping, IO and provenance.
    """
    from utils.wm import sfw_bundle, sfw_runtime
    from utils.wm.hsqr_provider import OFFICIAL_PROFILE_NAME
    from utils.wm.hsqr_provider import apply_arg_defaults as hsqr_apply_profile

    argv = list(argv or [])
    # Standalone HSQR reproduction runner: adopt the immutable official profile
    # unless the user asked for another one. The formal cohort generators
    # (experiments/generate_watermarked_images.py, run_removal.py) keep the
    # parser default (legacy_raven) and are unaffected.
    if not any(token == "--hsqr_profile" or token.startswith("--hsqr_profile=") for token in argv):
        args.hsqr_profile = OFFICIAL_PROFILE_NAME
    profile_info = hsqr_apply_profile(args, argv)
    print(f"[HSQR] profile: {profile_info}", flush=True)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images" / "watermarked"
    clean_images_dir = out_dir / "images" / "no_watermark"
    prompts_dir = out_dir / "prompts"
    metadata_dir = out_dir / "sample_metadata"
    results_path = out_dir / "results.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    if not args.hsqr_bundle_dir:
        args.hsqr_bundle_dir = (out_dir / "hsqr_bundle").as_posix()
    bundle_dir = Path(args.hsqr_bundle_dir)

    gpu_info = sfw_runtime.gpu_preflight(DEVICE)
    print(f"[HSQR] device preflight: {gpu_info}", flush=True)

    # A run that resumes into an existing output directory must not create
    # anything before its manifest has been validated; the bundle that produced
    # the existing samples must exist and is then loaded read-only.
    resuming = manifest_path.exists()
    if resuming and not sfw_bundle.SfwBundle(bundle_dir).complete():
        raise RuntimeError(
            f"{manifest_path} already exists, so this run resumes an existing cohort and must "
            f"reuse the bundle that produced it, but {bundle_dir} is not a complete bundle. "
            "Nothing was modified."
        )

    pipe_provider_target = sfw_runtime.build_pipe_provider(args, DEVICE)
    latent_shape = pipe_provider_target.get_latent_shape()

    if resuming:
        wm_provider = sfw_runtime.load_provider_from_bundle(args, latent_shape, DEVICE)
    else:
        wm_provider = sfw_runtime.build_provider(args, latent_shape, DEVICE)

    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)

    key_mapping = None
    if wm_provider.key_policy == "per_sample":
        key_mapping = [wm_provider.sample_key_index(sample_id) for sample_id in range(args.num)]

    if not resuming:
        wm_provider.create_bundle(
            bundle_dir,
            save_keybook=args.hsqr_save_keybook,
            key_mapping=key_mapping,
        )
    print(
        f"[HSQR] bundle {wm_provider.bundle.dir} "
        f"(key_index={wm_provider.selected_key_index}, "
        f"payload={wm_provider.payload_text(wm_provider.selected_key_index)}, "
        f"config {wm_provider.bundle.manifest['bundle_config_sha256'][:16]})",
        flush=True,
    )

    provenance = sfw_runtime.run_provenance(args, wm_provider, pipe_provider_target)
    run_config = dict(provenance)
    run_config.update({"base_seed": int(args.seed), "num_samples": int(args.num),
                       "dataset_id": args.dataset_id, "key_policy": wm_provider.key_policy,
                       "paired": bool(args.hsqr_paired)})
    # ``created_utc`` is volatile, ``pattern_source`` records whether *this* process
    # derived or reloaded the pattern, and ``num_samples`` only says how far the
    # cohort was extended. None of them is part of a *sample's* configuration
    # identity, which is what the resume gate compares; each sample is still bound
    # to its own seed, prompt, key/pattern hash, bundle and model configuration.
    volatile_fields = ("created_utc", "pattern_source", "num_samples")
    run_config_sha256 = sfw_bundle.canonical_sha256(
        {k: v for k, v in run_config.items() if k not in volatile_fields}
    )

    manifest = sfw_runtime.assert_run_manifest_compatible(
        manifest_path, run_config_sha256, method="HSQR"
    )
    if manifest is None:
        manifest = dict(run_config)
        manifest["run_config_sha256"] = run_config_sha256
        manifest["entrypoint"] = "eval_bench_wm/run_watermark.py:run_hsqr_generation"
        manifest["entrypoint_sha256"] = sha256_path(Path(__file__))
        manifest["report_label"] = (
            "official_profile_raw_scores" if wm_provider.profile_is_official
            else "legacy_or_ablation_mode"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(sfw_bundle.canonical_json(manifest) + "\n", encoding="utf-8")

    directories = [images_dir, prompts_dir, metadata_dir]
    if args.hsqr_paired:
        directories.append(clean_images_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_id, target_prompt in tqdm(
        list(enumerate(target_prompts)), total=len(target_prompts), desc="HSQR generation"
    ):
        sample_seed = int(args.seed) + int(sample_id)
        name = f"{sample_id:06d}"
        image_path = images_dir / f"{name}.png"
        clean_image_path = clean_images_dir / f"{name}.png"
        prompt_path = prompts_dir / f"{name}.txt"
        metadata_path = metadata_dir / f"{name}.json"
        prompt_sha256 = sfw_bundle.sha256_text(target_prompt)

        key_index = key_mapping[sample_id] if key_mapping else wm_provider.selected_key_index
        pattern = (
            wm_provider.keybook()[key_index] if key_mapping else wm_provider.gt_patch
        )

        if metadata_path.exists() or image_path.exists():
            # Resume: never overwrite and never silently mix incompatible runs.
            if not (metadata_path.exists() and image_path.exists()):
                raise RuntimeError(
                    f"HSQR sample {name} is half-written ({image_path.exists()=}, "
                    f"{metadata_path.exists()=}); use a fresh --out_dir"
                )
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            sfw_runtime.assert_resumable(
                name,
                existing,
                {
                    "sample_seed": sample_seed,
                    "prompt_sha256": prompt_sha256,
                    "run_config_sha256": run_config_sha256,
                    "hsqr_bundle_config_sha256": wm_provider.bundle.manifest["bundle_config_sha256"],
                    "selected_pattern_sha256": wm_provider.pattern_sha256(pattern),
                },
                method="HSQR",
            )
            if existing.get("image_sha256") != sha256_path(image_path):
                raise RuntimeError(
                    f"HSQR sample {name} image hash changed on disk; refusing to resume"
                )
            rows.append(existing)
            continue

        sample = wm_provider.build_sample_latents(sample_seed, pattern=pattern)
        generated = pipe_provider_target.generate(
            prompts=target_prompt,
            latents=sample["watermarked_latent"],
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
        generated["images_PIL"][0].save(image_path)
        prompt_path.write_text(target_prompt, encoding="utf-8")

        clean_sha = None
        if args.hsqr_paired:
            clean_generated = pipe_provider_target.generate(
                prompts=target_prompt,
                latents=sample["clean_latent"],
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
            clean_generated["images_PIL"][0].save(clean_image_path)
            clean_sha = sha256_path(clean_image_path)

        row = {
            "sample_id": sample_id,
            "sample_name": name,
            "sample_seed": sample_seed,
            "prompt": target_prompt,
            "prompt_sha256": prompt_sha256,
            "image_path": image_path.as_posix(),
            "image_sha256": sha256_path(image_path),
            "clean_image_path": clean_image_path.as_posix() if args.hsqr_paired else None,
            "clean_image_sha256": clean_sha,
            "base_latent_sha256": sample["base_latent_sha256"],
            "clean_base_latent_sha256": sample["clean_base_latent_sha256"],
            "watermark_pre_injection_base_latent_sha256":
                sample["watermark_pre_injection_base_latent_sha256"],
            "watermarked_latent_sha256": sample["watermarked_latent_sha256"],
            "selected_key_index": key_index,
            "selected_key_seed": wm_provider.key_seed(key_index),
            "payload_text": wm_provider.payload_text(key_index),
            "selected_pattern_sha256": wm_provider.pattern_sha256(pattern),
            "hsqr_bundle_config_sha256": wm_provider.bundle.manifest["bundle_config_sha256"],
            "run_config_sha256": run_config_sha256,
        }
        # The pairing invariant is asserted, not assumed.
        if row["clean_base_latent_sha256"] != row["watermark_pre_injection_base_latent_sha256"]:
            raise RuntimeError(
                f"HSQR sample {name}: clean and watermark pre-injection base latents differ"
            )
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})
        row["created_utc"] = sfw_bundle.utc_now()
        metadata_path.write_text(sfw_bundle.canonical_json(row) + "\n", encoding="utf-8")
        sfw_bundle.append_jsonl(results_path, row)
        rows.append(row)

    latent_hashes = {row["base_latent_sha256"] for row in rows}
    if len(rows) > 1 and len(latent_hashes) != len(rows):
        raise RuntimeError(
            "HSQR generation produced repeated complete base latents across samples; "
            "this invalidates the run for independent-sample evaluation"
        )
    print(
        f"[HSQR] generated {len(rows)} sample(s) into {out_dir}; "
        f"{len(latent_hashes)} distinct complete base latents; "
        f"key policy {wm_provider.key_policy}",
        flush=True,
    )
    return rows



def run_hstr_generation(args, argv):
    """Official-compatible HSTR/SFWMark generation.

    HSTR algorithmic operations stay in ``utils/wm/hstr_provider.py``. This
    runner only handles profile defaults, shared SFW bundle persistence, indexed
    paired outputs, per-sample seeds, and non-overwriting resume checks.
    """
    from utils.wm import sfw_bundle, sfw_runtime
    from utils.wm.hstr_provider import apply_arg_defaults as hstr_apply_profile

    profile_info = hstr_apply_profile(args, argv)
    print(f"[HSTR] profile: {profile_info}", flush=True)
    sfw_runtime.require_official_generation_profile(args)

    out_dir = Path(args.out_dir)
    if not getattr(args, "hstr_bundle_dir", None):
        args.hstr_bundle_dir = (out_dir / "hstr_bundle").as_posix()
    images_wm_dir = out_dir / "images" / "watermarked"
    images_clean_dir = out_dir / "images" / "no_watermark"
    prompts_dir = out_dir / "prompts"
    metadata_dir = out_dir / "sample_metadata"
    results_path = out_dir / "results.jsonl"
    manifest_path = out_dir / "run_manifest.json"

    gpu_info = sfw_runtime.gpu_preflight(DEVICE)
    print(f"[HSTR] device preflight: {gpu_info}", flush=True)

    latent_shape = (1, 4, int(args.resolution) // 8, int(args.resolution) // 8)
    resuming = manifest_path.exists()
    if resuming and not sfw_bundle.SfwBundle(args.hstr_bundle_dir).complete():
        raise RuntimeError(
            f"{manifest_path} already exists, so this run must reuse a complete HSTR bundle at "
            f"--hstr_bundle_dir={args.hstr_bundle_dir}; nothing was modified."
        )

    dry_bundle_dir = args.hstr_bundle_dir
    if not resuming:
        args.hstr_bundle_dir = None
    dry_provider = sfw_runtime.build_provider(
        args,
        latent_shape=latent_shape,
        device=DEVICE,
        create_bundle=False,
    )
    args.hstr_bundle_dir = dry_bundle_dir
    run_config = {
        "official_reference_repo": sfw_bundle.OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": sfw_bundle.OFFICIAL_SFWMARK_COMMIT,
        "entrypoint": "eval_bench_wm/run_watermark.py:run_hstr_generation",
        "entrypoint_sha256": sha256_path(Path(__file__)),
        "hstr_profile": dry_provider.profile,
        "provider_config_sha256": dry_provider.provider_config_sha256(),
        "base_sample_seed": int(args.seed),
        "num_samples": int(args.num),
        "dataset_id": args.dataset_id,
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "paired_clean_watermarked_outputs": True,
    }
    run_config_sha256 = sfw_bundle.canonical_sha256(
        {k: v for k, v in run_config.items() if k != "num_samples"}
    )
    manifest = sfw_runtime.assert_run_manifest_compatible(
        manifest_path, run_config_sha256, method="HSTR"
    )

    provider = sfw_runtime.build_provider(
        args,
        latent_shape=latent_shape,
        device=DEVICE,
        create_bundle=not resuming,
    )
    if provider.bundle is None:
        raise RuntimeError("HSTR generation requires a persisted --hstr_bundle_dir")

    pipe_provider_target = sfw_runtime.build_pipe_provider(args, DEVICE)
    provenance = sfw_runtime.run_provenance(args, provider, pipe_provider_target)
    if manifest is None:
        manifest = dict(run_config)
        manifest.update({k: v for k, v in provenance.items() if k != "created_utc"})
        manifest["run_config_sha256"] = run_config_sha256
        manifest["report_label"] = "official_profile_raw_scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(sfw_bundle.canonical_json(manifest) + "\n", encoding="utf-8")

    for directory in (images_wm_dir, images_clean_dir, prompts_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
    rows = []
    for sample_id, target_prompt in tqdm(
        list(enumerate(target_prompts)), total=len(target_prompts), desc="HSTR generation"
    ):
        sample_seed = int(args.seed) + int(sample_id)
        name = f"{sample_id:06d}"
        image_wm_path = images_wm_dir / f"{name}.png"
        image_clean_path = images_clean_dir / f"{name}.png"
        prompt_path = prompts_dir / f"{name}.txt"
        metadata_path = metadata_dir / f"{name}.json"
        prompt_sha256 = sfw_bundle.sha256_text(target_prompt)

        if metadata_path.exists() or image_wm_path.exists() or image_clean_path.exists():
            if not (metadata_path.exists() and image_wm_path.exists() and image_clean_path.exists()):
                raise RuntimeError(f"HSTR sample {name} is half-written; use a fresh --out_dir")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            sfw_runtime.assert_resumable(
                name,
                existing,
                {
                    "sample_seed": sample_seed,
                    "prompt_sha256": prompt_sha256,
                    "run_config_sha256": run_config_sha256,
                    "hstr_bundle_config_sha256": provider.bundle.manifest["bundle_config_sha256"],
                },
                method="HSTR",
            )
            if existing.get("watermarked_image_sha256") != sfw_bundle.sha256_file(image_wm_path):
                raise RuntimeError(f"HSTR sample {name} watermarked image hash changed on disk")
            if existing.get("clean_image_sha256") != sfw_bundle.sha256_file(image_clean_path):
                raise RuntimeError(f"HSTR sample {name} clean image hash changed on disk")
            rows.append(existing)
            continue

        clean_latent = provider.sample_base_latent(sample_seed)
        wm_sample = provider.get_wm_latents(latents_clean=clean_latent)
        wm_latent = wm_sample["zT_torch"]
        generated = provider.generate(
            pipe_provider_target=pipe_provider_target,
            prompts=[target_prompt, target_prompt],
            latents=torch.cat([clean_latent, wm_latent], dim=0),
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
        clean_image, wm_image = generated["images_PIL"][0], generated["images_PIL"][1]
        clean_image.save(image_clean_path)
        wm_image.save(image_wm_path)
        prompt_path.write_text(target_prompt, encoding="utf-8")

        row = {
            "sample_id": sample_id,
            "sample_name": name,
            "sample_seed": sample_seed,
            "prompt": target_prompt,
            "prompt_sha256": prompt_sha256,
            "clean_image_path": image_clean_path.as_posix(),
            "clean_image_sha256": sfw_bundle.sha256_file(image_clean_path),
            "watermarked_image_path": image_wm_path.as_posix(),
            "watermarked_image_sha256": sfw_bundle.sha256_file(image_wm_path),
            "pre_injection_latent_sha256": sfw_bundle.sha256_tensor(clean_latent),
            "post_injection_latent_sha256": sfw_bundle.sha256_tensor(wm_latent),
            "hstr_bundle_config_sha256": provider.bundle.manifest["bundle_config_sha256"],
            "hstr_bundle_sha256": sfw_bundle.sha256_file(provider.bundle.manifest_path),
            "selected_key_index": provider.key_index,
            "selected_key_seed": provider.selected_key_seed,
            "selected_pattern_sha256": provider.selected_pattern_sha256,
            "run_config_sha256": run_config_sha256,
            "protocol": "hstr_official_sfwmark_paired_direct_generation",
        }
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})
        row["created_utc"] = sfw_bundle.utc_now()
        metadata_path.write_text(sfw_bundle.canonical_json(row) + "\n", encoding="utf-8")
        sfw_bundle.append_jsonl(results_path, row)
        rows.append(row)
        del clean_latent, wm_latent, wm_sample, generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latent_hashes = {row["post_injection_latent_sha256"] for row in rows}
    if len(rows) > 1 and len(latent_hashes) != len(rows):
        raise RuntimeError("HSTR generation produced repeated complete watermarked latents across samples")
    print(
        f"[HSTR] generated {len(rows)} paired clean/watermarked sample(s) into {out_dir}; "
        f"{len(latent_hashes)} distinct watermarked latents; bundle {provider.bundle.dir}",
        flush=True,
    )
    return rows

def value_log_label(wm_type):
    if wm_type == "PRC":
        return "PRC value"
    if wm_type == "T2S":
        return "T2S norm1_w"
    if wm_type == "MAXSIVE":
        return "MAXSIVE value"
    if wm_type == "GM":
        return "GM value"
    return "value"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args, unknown_args = build_parser().parse_known_args(argv)

    if args.modelid_target == "stabilityai/stable-diffusion-3.5-medium":
        if "--num_inference_steps_target" not in sys.argv:
            args.num_inference_steps_target = 40
        if "--guidance_scale_target" not in sys.argv:
            args.guidance_scale_target = 4.5

    # GaussMarker has its own single-command generation flow: a persistent bundle,
    # a deterministic per-sample seed, an independently sampled complete initial
    # latent per sample and indexed, never-overwritten outputs.
    if args.wm_type == "GM":
        return run_gm_generation(args, argv)

    # HSQR has the same single-command flow: a persistent SFWMark bundle, a
    # deterministic per-sample seed, an independently sampled complete base
    # latent per sample and indexed, never-overwritten outputs.
    if args.wm_type == "HSQR":
        return run_hsqr_generation(args, argv)

    if args.wm_type == "HSTR" and getattr(args, "hstr_profile", None) == "official_sfwmark_sd21":
        return run_hstr_generation(args, argv)

    wm_provider_cls = WmProviders[args.wm_type].value
    if args.wm_type == "GS" and int(args.num) != 1:
        raise RuntimeError(
            "Generic GS runners allow only an explicitly single-sample legacy/debug run; "
            "use experiments/generate_watermarked_images.py for auditable per-run GS state"
        )
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    # Standalone GS reproduction runner: adopt official-inspired upstream generation
    # defaults (stabilityai SD2.1-base + DPM + fp16 weight revision + an
    # official-inspired inversion approximation) unless the user specified them
    # explicitly. The official-inspired inversion note is only emitted when actually on
    # the official model + DPM (see apply_official_reproduction_defaults). This does NOT
    # affect other methods or the formal generator (which keeps the shared
    # RedbeardNZ + DDIM cohort).
    if args.wm_type == "GS":
        from utils.wm.gs_provider import apply_official_reproduction_defaults
        _gs_official_defaults = apply_official_reproduction_defaults(args, argv)
        print(f"[GS] official reproduction defaults applied: {_gs_official_defaults}", flush=True)

    # set seeds
    set_random_seed(args.seed)

    # logger
    logger = None
    if args.logger:
        log_name = f"{model_name_mapping[args.modelid_target]}_{args.wm_type}"
        logger = get_logger(args.out_dir, log_name)

    # retrieve the detection threshold for the settings
    detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)
    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
    # target_prompts = target_prompts[1:]

    # pipe_provider used by the target model (SDXL, PixArt, FLUX)
    pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                        resolution=args.resolution,
                                                        device=DEVICE,
                                                        eager_loading=True if args.modelid_target in model_flux else False,
                                                        schedulers_name=args.scheduler_target,
                                                        disable_tqdm=True,
                                                        revision=getattr(args, "model_revision", None),)

    # generate a watermarked latent zT
    # This way like it is done here is a simple way to obtain a watermark provider for a simple test run.
    # If you want to do mass experiments and have batch_sizes > 1, plz have look at the utils.wm_provider.WmProvider.generate_providers method
    wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), device=DEVICE, **vars(args))

    # T2S draws a fresh base latent, session key and message for every sample, so
    # its latent must be built inside the loop. Every other provider keeps the
    # pre-existing behaviour of reusing one latent for the whole run.
    per_sample_latents = args.wm_type == "T2S"
    if per_sample_latents:
        wm_initial_results = None
        wm_zT = None
        message_bits_str_initial = None
        t2s_state_dir = os.path.join(args.out_dir, "t2s_state") if args.t2s_state_dir is None else args.t2s_state_dir
        t2s_image_dir = os.path.join(args.out_dir, "images")
    else:
        wm_initial_results = wm_provider.get_wm_latents()
        wm_zT = wm_initial_results["zT_torch"]

        # for Gaussian Shading, we also get an initial message
        message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

    metric_map = {
        "PRC": "value",
        "TR": "p_value",
        "RID": "l1_dist",
        "HSTR": "l1_dist",
        "HSQR": "l1_dist",
        "GS": "bit_accuracy",
        "TAG": "bit_accuracy",
        "SPH": "bit_accuracy",
        "T2S": "bit_accuracy",
        "MAXSIVE": "value",
        "SHALLOW": "l1_dist",
        "GM": "value",
    }

    for id, (target_prompt) in tqdm(enumerate(target_prompts), total=len(target_prompts), desc="Generating images"):
        print(f"\n--- Starting run {id+1}/{len(target_prompts)} ---")
        # print(f"Target prompt: {target_prompt}")
        if args.logger:
            logger.info("Test single watermarked image")

        if args.wm_type in ["PRC", "SPH"]:
            seed_everything(args.seed)

        # T2S: one independent base latent, session key, message and portable
        # state per sample. Only the master key may persist (--t2s_fix_key).
        t2s_state = None
        if per_sample_latents:
            sample_seed = args.seed + id
            wm_zT, t2s_state = wm_provider.new_sample(
                sample_seed=sample_seed,
                watermark_id=f"{model_name_mapping[args.modelid_target]}-t2s-{id:05d}",
                model_id=args.modelid_target,
                model_revision=getattr(args, "model_revision", None),
                scheduler=args.scheduler_target,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
                resolution=args.resolution,
                prompt_sha256=hashlib.sha256(target_prompt.encode("utf-8")).hexdigest(),
            )
            message_bits_str_initial = t2s_state.expected_message_bits

        # generate a watermarked image with the target model
        if hasattr(wm_provider, "generate"):
            generated = wm_provider.generate(
                pipe_provider_target=pipe_provider_target,
                prompts=target_prompt,
                latents=wm_zT,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
        else:
            generated = pipe_provider_target.generate(
            prompts=target_prompt,
            latents=wm_zT,
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )

        generated_PIL_list = generated["images_PIL"]
        benign_image = generated_PIL_list[0]

        # Save every sample separately instead of overwriting one file.
        image_dir = t2s_image_dir if per_sample_latents else os.path.join(args.out_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{id:05d}.png")
        benign_image.save(image_path)

        if t2s_state is not None:
            # Bind the state to the image it actually describes, so pairing never
            # depends on filename or row ordering.
            t2s_state = dataclasses.replace(
                t2s_state, image_sha256=sha256_path(Path(image_path))
            )
            # Keep the provider's own state in sync so the compatibility
            # get_accuracies() path scores against the identical record.
            wm_provider.state = t2s_state
            state_path = os.path.join(t2s_state_dir, f"{t2s_state.watermark_id}.json")
            t2s_state.save(state_path)
            print(f"[T2S] sample {id}: image={image_path} state={state_path} "
                  f"state_sha256={t2s_state.state_sha256()}")

        # from PIL import Image
        # benign_image = Image.open("sana_vae_TR.png").convert("RGB")

        if args.distort:
            benign_image = distort_images(benign_image, jpeg_ratio=20)
            benign_image.save("distort_image.png")

        # distort param:
        # r_degree=(0, 150), jpeg_ratio=(10, 90), sp_prob_fixed=(0.05, 0.4), crop_scale_TR=(0.9 raw, 0.5),
        # random_crop_ratio=(0.1-0.5), random_drop_ratio=(0.1, 0.8), gaussian_std_fixed=(0.05, 0.4), :done
        # median_blur_k=(1,3,5,...,17), gaussian_blur_r=(2,4,6,8,10), resize_ratio=(0.1, 0.9),
        # brightness_factor=(2,16), contrast_factor,
        # vertical_shift_ratio=(0.1, 0.8), horizontal_shift_ratio=(0.1, 0.8), flip_ratio=1,

        # T2S owns its inversion protocol (see utils/wm/t2s_inversion.py) and
        # validate() invokes it; running the generic DDIM inversion here as well
        # would only waste a second inversion and mix protocols.
        if args.wm_type not in ("SHALLOW", "T2S"):
            with torch.no_grad(): # retrieve zT
                zT_retrieved = pipe_provider_target.invert_images(benign_image, num_inference_steps=args.num_inference_steps_target)["zT_torch"]
            # mse = F.mse_loss(wm_zT, zT_retrieved)
            # print("mse:", mse.item())

        # benign_image = PIL.Image.open("watermarked_image.png")

        rows = []
        results = validate(
                out_dir=args.out_dir,
                image_to_verify_PIL=benign_image,
                original_PIL=benign_image,
                wm_provider=wm_provider,
                pipe_provider_target=pipe_provider_target,
                num_inference_steps_target=args.num_inference_steps_target,
                step=-1,
                message_bits_str_initial=message_bits_str_initial,
                do_psnr=False,
                do_ssim=False,
                do_msssim=False,
                do_lpips=False,
                )

        # check if detection was successfull
        if args.wm_type == "T2S":
            # RAVEN paired_key_comparison deployment extension: the true
            # (master) key must score above a control key on the same image.
            # Upstream evaluates a cohort ROC, not a per-image rule. NOT TPR@1%FPR.
            detection_successful = results["detection_success"]
        elif args.wm_type == "SPH":
            detection_successful = None
        elif args.wm_type in ["PRC", "MAXSIVE", "SHALLOW", "GM"]:
            detection_successful = results["detection_success"]
        elif args.wm_type == "GS":
            # Official Gaussian Shading detection: default gs_detection_mode is
            # official_onebit (beta-tail tau_onebit, ">="); legacy_default must be
            # requested explicitly. Never uses the legacy fixed GS_THRESHOLDS by default.
            detection_successful = wm_provider.is_detection_successful(results[metric_map[args.wm_type]])
        else:
            detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                                threshold=detection_threshold,
                                                                value=results[metric_map[args.wm_type]])
        results["detection_successful"] = detection_successful

        rows.append({
                    "bit_accuracy": results["bit_accuracy"],
                    "p_value": results["p_value"],
                    "value": results["value"],
                    "detection_success": results["detection_success"],
                    "log_message": results["log_message"],
                    "prc_threshold": results["prc_threshold"],
                    "detection_successful": results["detection_successful"],
                    })
        value_label = value_log_label(args.wm_type)
        bit_accuracy_display = format_bit_accuracy_value(
            results["bit_accuracy"],
            results["bit_accuracy_status"],
            results["bit_accuracy_error"],
        )

        if args.logger:
            logger.info(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display}, p_value: {results['p_value']}, {value_label}: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
            if results["log_message"]:
                logger.info(results["log_message"])
        else:
            print(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display}, p_value: {results['p_value']}, {value_label}: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
            if results["log_message"]:
                print(results["log_message"])


if __name__ == "__main__":
    main()
