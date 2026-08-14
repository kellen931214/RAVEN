#!/usr/bin/env python3
"""Generate the formal released-state GaussMarker cohort, streamed one pair at a time."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eval_bench_wm"
for item in (str(ROOT / "raven_repro"), str(BENCH)):
    if item not in sys.path:
        sys.path.insert(0, item)
if "/tmp/gaussmarker-official" not in sys.path:
    sys.path.append("/tmp/gaussmarker-official")

BUNDLE = ROOT / "data/gm/diffusiondb_official_released_state/bundle"
PROMPTS = ROOT / "data/clean/diffusiondb/inputs/diffusiondb_1001_prompts.csv"
OUT = ROOT / "data/gm/diffusiondb_official_released_state/official_released_state_dpm_fp16_1001_run_20260813"
SNAPSHOT = Path("/root/.cache/huggingface/hub/models--RedbeardNZ--stable-diffusion-2-1-base/snapshots/c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
GNR = ROOT / "eval_bench_wm/GM_utils/GNR_bits256/model_final.pth"
CLASSIFIER = ROOT / "eval_bench_wm/GM_utils/sd21_cls2.pkl"
PROTOCOL = "official_gaussmarker_released_state"
FIELDS = (
    "run_id sample_id prompt prompt_sha256 source generation_seed clean_path clean_sha256 "
    "watermarked_path watermarked_image_path watermarked_sha256 clean_latent_sha256 "
    "watermarked_latent_sha256 gm_pre_injection_latent_sha256 gm_post_injection_latent_sha256 "
    "gm_bundle_dir gm_bundle_config_sha256 gm_w1_file_sha256 gm_w2_file_sha256 "
    "gm_watermark_sha256 gm_m_sha256 gm_target_sha256 gm_mask_sha256 watermark_target_sha256 watermark_mask_sha256 gm_state_source "
    "gm_protocol_mode gm_uniform_derivation gm_sampling_uniform_sha256 gm_provider_entrypoint_path "
    "gm_provider_entrypoint_sha256 w1_sha256 w2_tensor_sha256 gnr_sha256 classifier_sha256 "
    "model_id model_mirror_id model_mirror_commit model_revision scheduler scheduler_target torch_dtype "
    "resolution num_inference_steps guidance_scale protocol profile_is_official generation_algorithm "
    "generation_benchmark_protocol status error created_utc"
).split()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def complete_ids(metadata: Path) -> set[int]:
    done: set[int] = set()
    if not metadata.is_file() or not metadata.stat().st_size:
        return done
    with metadata.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("status") == "generated" and Path(row["clean_path"]).is_file() and Path(row["watermarked_path"]).is_file():
                    done.add(int(row["run_id"]))
            except (KeyError, TypeError, ValueError):
                pass
    return done


def write_row(handle, row: dict) -> None:
    csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise").writerow(row)
    handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from tr_utils import set_random_seed
    from utils.wm import gm_bundle
    from utils.wm.gm_provider import GmProvider

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for required in (BUNDLE / "manifest.json", PROMPTS, SNAPSHOT / "model_index.json", GNR, CLASSIFIER):
        if not required.is_file():
            raise FileNotFoundError(f"missing required artifact: {required}")
    bundle = json.loads((BUNDLE / "manifest.json").read_text())
    if bundle.get("profile") != "official_sd21" or bundle.get("profile_is_official") is not True:
        raise RuntimeError("refusing a non-official released-state GM bundle")
    if digest(GNR) != bundle["gnr_sha256"] or digest(CLASSIFIER) != bundle["classifier_sha256"]:
        raise RuntimeError("released GNR/classifier SHA mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    metadata = OUT / "metadata.csv"
    done = complete_ids(metadata)
    # Previous failed runner produced header only, so it contains no experiment record.
    if metadata.exists() and not done:
        metadata.unlink()
    new_csv = not metadata.exists()
    manifest_path = OUT / "run_manifest.json"
    algorithm = "official_gaussmarker_gen_released_w1_w2_adapter_parity_checked"
    manifest = {
        "method": "GM", "protocol": PROTOCOL, "status": "running", "profile_is_official": True,
        "bundle_dir": str(BUNDLE), "bundle_config_sha256": bundle["bundle_config_sha256"],
        "w1_sha256": bundle["w1_file_sha256"], "w2_tensor_sha256": bundle["w2_tensor_sha256"],
        "gnr_sha256": bundle["gnr_sha256"], "classifier_sha256": bundle["classifier_sha256"],
        "model_id": bundle["model_id"], "model_revision": None, "model_mirror_id": bundle["model_mirror_id"],
        "model_mirror_commit": bundle["model_mirror_commit"], "model_mirror_local_snapshot": str(SNAPSHOT),
        "scheduler": "DPM", "torch_dtype": "float16", "resolution": 512, "num_inference_steps": 50,
        "guidance_scale": 7.5, "batch_size": 1, "num_workers": 0, "prompt_csv": str(PROMPTS),
        "official_reference_repo": bundle["official_reference_repo"], "official_reference_commit": bundle["official_reference_commit"],
        "generation_algorithm": algorithm, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    device = torch.device("cuda")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(str(SNAPSHOT), subfolder="scheduler", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        str(SNAPSHOT), torch_dtype=torch.float16, variant="fp16", scheduler=scheduler,
        safety_checker=None, requires_safety_checker=False, local_files_only=True, low_cpu_mem_usage=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if pipe.unet.dtype != torch.float16 or pipe.vae.dtype != torch.float16:
        raise RuntimeError(f"FP16 assertion failed: UNet={pipe.unet.dtype}, VAE={pipe.vae.dtype}")
    args = {
        "gm_profile": "official_sd21", "gm_bundle_dir": str(BUNDLE), "gm_create_bundle": False,
        "gm_allow_in_memory_state": False, "gm_torch_dtype": "float16", "gm_channel_copy": 1,
        "gm_w_copy": 8, "gm_h_copy": 8, "gm_watermark_bits_seed": None, "gm_use_gnr": True,
        "gm_gnr_path": str(GNR), "gm_model_nf": 128, "gm_classifier_type": 0, "gm_use_classifier": True,
        "gm_classifier_path": str(CLASSIFIER), "modelid_target": bundle["model_id"], "model_revision": None,
        "scheduler_target": "DPM", "resolution": 512, "gm_inversion_prompt": "", "gm_inversion_guidance": 1.0,
        "gm_inversion_steps": 50, "gm_inversion_seed": 0, "gm_vae_sample": True,
        "gm_vae_scaling_factor": 0.18215, "gm_profile_is_official": True, "w_seed": 999999,
        "w_channel": 3, "w_pattern": "ring", "w_mask_shape": "circle", "w_radius": 4,
        "w_measurement": "l1_complex", "w_injection": "complex",
    }
    provider = GmProvider(latent_shape=(1, 4, 64, 64), dtype=torch.float16, device=device, **args)
    _ = provider.gnr  # strict load; no raw-bit fallback is possible afterwards
    _ = provider.classifier
    provider_path = Path(sys.modules["utils.wm.gm_provider"].__file__).resolve()
    mask_sha = gm_bundle.sha256_tensor(provider.watermarking_mask)

    count = 0
    with PROMPTS.open(newline="", encoding="utf-8") as input_f, metadata.open("a", newline="", encoding="utf-8") as output_f:
        writer = csv.DictWriter(output_f, fieldnames=FIELDS, extrasaction="raise")
        if new_csv:
            writer.writeheader(); output_f.flush(); os.fsync(output_f.fileno())
        for run_id, source in enumerate(csv.DictReader(input_f)):
            count += 1
            if run_id in done:
                continue
            prompt = source["prompt"]
            item = OUT / "GM" / f"{run_id:06d}"
            clean, watermarked = item / "clean.png", item / "watermarked.png"
            if clean.exists() or watermarked.exists():
                raise RuntimeError(f"unrecorded output exists: {item}")
            wm_sample = wm_latent = clean_latent = wm_image = clean_image = None
            try:
                set_random_seed(run_id)
                wm_sample = provider.build_sample_latents(run_id)
                wm_latent = wm_sample["latent"].to(device=device, dtype=torch.float16)
                clean_latent = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
                with torch.inference_mode():
                    wm_image = pipe(prompt, latents=wm_latent, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).images[0]
                    clean_image = pipe(prompt, latents=clean_latent, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).images[0]
                item.mkdir(parents=True, exist_ok=False)
                clean_image.save(clean); wm_image.save(watermarked)
                row = {
                    "run_id": run_id, "sample_id": source.get("id", str(run_id)), "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "source": source.get("source", ""), "generation_seed": run_id,
                    "clean_path": str(clean.resolve()), "clean_sha256": digest(clean), "watermarked_path": str(watermarked.resolve()),
                    "watermarked_image_path": str(watermarked.resolve()), "watermarked_sha256": digest(watermarked),
                    "clean_latent_sha256": gm_bundle.sha256_tensor(clean_latent), "watermarked_latent_sha256": gm_bundle.sha256_tensor(wm_latent),
                    "gm_pre_injection_latent_sha256": wm_sample["pre_injection_latent_sha256"], "gm_post_injection_latent_sha256": wm_sample["post_injection_latent_sha256"],
                    "gm_bundle_dir": str(BUNDLE.resolve()), "gm_bundle_config_sha256": bundle["bundle_config_sha256"],
                    "gm_w1_file_sha256": bundle["w1_file_sha256"], "gm_w2_file_sha256": bundle["w2_file_sha256"],
                    "gm_watermark_sha256": bundle["watermark_sha256"], "gm_m_sha256": bundle["m_sha256"], "gm_target_sha256": bundle["w2_tensor_sha256"],
                    "gm_mask_sha256": mask_sha, "watermark_target_sha256": bundle["w2_tensor_sha256"], "watermark_mask_sha256": mask_sha, "gm_state_source": provider.state_source, "gm_protocol_mode": PROTOCOL,
                    "gm_uniform_derivation": "official_truncnorm_sample_seed_plus_3", "gm_sampling_uniform_sha256": "not_materialized",
                    "gm_provider_entrypoint_path": str(provider_path), "gm_provider_entrypoint_sha256": digest(provider_path),
                    "w1_sha256": bundle["w1_file_sha256"], "w2_tensor_sha256": bundle["w2_tensor_sha256"], "gnr_sha256": bundle["gnr_sha256"],
                    "classifier_sha256": bundle["classifier_sha256"], "model_id": bundle["model_id"], "model_mirror_id": bundle["model_mirror_id"],
                    "model_mirror_commit": bundle["model_mirror_commit"], "model_revision": "", "scheduler": "DPM", "scheduler_target": "DPM",
                    "torch_dtype": "float16", "resolution": 512, "num_inference_steps": 50, "guidance_scale": 7.5,
                    "protocol": PROTOCOL, "profile_is_official": True, "generation_algorithm": algorithm,
                    "generation_benchmark_protocol": PROTOCOL, "status": "generated", "error": "", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                write_row(output_f, row)
                if run_id == 0 or (run_id + 1) % 10 == 0:
                    print(f"[GM official] {run_id + 1}/1001", flush=True)
            except Exception as exc:
                (OUT / "generation_error.json").write_text(json.dumps({"run_id": run_id, "type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}, indent=2))
                raise
            finally:
                del wm_sample, wm_latent, clean_latent, wm_image, clean_image
                gc.collect(); torch.cuda.empty_cache()
    if count != 1001 or len(complete_ids(metadata)) != 1001:
        raise RuntimeError(f"cohort incomplete: prompts={count}, generated={len(complete_ids(metadata))}")
    manifest.update({"status": "completed", "completed": 1001, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("[GM official] completed 1001/1001", flush=True)


if __name__ == "__main__":
    main()
