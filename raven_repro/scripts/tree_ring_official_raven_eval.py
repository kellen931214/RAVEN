#!/usr/bin/env python
"""Official Tree-Ring paired cohort and RAVEN evaluation for DiffusionDB."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[2]
RAVEN_ROOT = ROOT / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.metrics import pair_quality_metrics
from raven.pipeline_raven import RavenPipeline
from raven.tree_ring_official import (
    OFFICIAL_TREE_RING_COMMIT,
    TreeRingSettings,
    get_empty_text_embedding,
    image_to_official_latents,
    inject_complex_watermark,
    make_rand_watermark_target,
    make_watermark_mask,
    official_complex_l1,
    official_forward_diffusion,
    official_roc,
    rate_at_negative_l1_threshold,
    score_image,
    set_official_random_seed,
    stable_tensor_hash,
)

SETTINGS = TreeRingSettings()
MODEL_MIRROR_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_MIRROR_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
DEFAULT_MODEL_SOURCE = ROOT / ".cache" / "huggingface" / "hub" / f"models--{MODEL_MIRROR_ID.replace('/', '--')}" / "snapshots" / MODEL_MIRROR_REVISION
DEFAULT_PROMPTS = ROOT / "data" / "prompts" / "diffusiondb_1001.csv"
DEFAULT_OFFICIAL_REPO = ROOT / "tree-ring-watermark-official"
PLAN_SEED = 2026071501
ATTACK_CONFIG = {
    "steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "inversion_mode": "ddim",
    "prompt": "",
    "negative_prompt": "",
    "shift_space": "image_pixels",
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "padding_mode": "reflection",
    "latent_sampling_mode": "nearest",
    "view_guided_attention": True,
    "color_transfer": True,
    "color_transfer_mode": "paper_exact_two_stage",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_text(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd or ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], *, exclusive: bool = False) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("x" if exclusive else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def finite_stats(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("statistics require non-empty finite values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def image_quality(
    reference: Image.Image,
    attacked: Image.Image,
    requested_dx: float,
    requested_dy: float,
    effective_dx: float,
    effective_dy: float,
    alignment_mode: str,
) -> dict[str, Any]:
    """Compute formal effective-flow quality and explicit legacy diagnostics."""
    effective = pair_quality_metrics(
        reference, attacked, effective_dx, effective_dy, alignment_mode=alignment_mode
    )
    legacy = pair_quality_metrics(
        reference, attacked, requested_dx, requested_dy, alignment_mode="integer_crop"
    )
    return {
        "raw_full_image_psnr": effective["raw_full_psnr"],
        "raw_full_image_ssim": effective["raw_full_ssim"],
        "post_color_overlap_psnr": effective["overlap_psnr"],
        "post_color_overlap_ssim": effective["overlap_ssim"],
        "formal_quality_protocol": effective["overlap_protocol"],
        "effective_flow_dx_image_px": float(effective_dx),
        "effective_flow_dy_image_px": float(effective_dy),
        "valid_overlap_width": effective["valid_overlap_width"],
        "valid_overlap_height": effective["valid_overlap_height"],
        "valid_overlap_area_ratio": effective["valid_overlap_area_ratio"],
        "legacy_requested_flow_overlap_psnr": legacy["overlap_psnr"],
        "legacy_requested_flow_overlap_ssim": legacy["overlap_ssim"],
        "legacy_requested_flow_dx_image_px": float(requested_dx),
        "legacy_requested_flow_dy_image_px": float(requested_dy),
        "legacy_quality_protocol": "requested_flow_integer_crop_diagnostic_only",
    }

def resolved_model_source(path: Path) -> Path:
    source = path.resolve()
    required = [source / "model_index.json", source / "scheduler" / "scheduler_config.json"]
    if not all(item.is_file() for item in required):
        raise FileNotFoundError(f"incomplete local SD2.1-base snapshot: {source}")
    return source


def load_official_pipeline(model_source: Path, device: str = "cuda"):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        str(model_source), subfolder="scheduler", local_files_only=True
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_source),
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=os.environ.get("TQDM_DISABLE") == "1")
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    return pipe


def get_random_latents(pipe: Any):
    pipe.scheduler.set_timesteps(SETTINGS.generation_steps, device=pipe.device)
    return pipe.prepare_latents(
        1,
        pipe.unet.config.in_channels,
        SETTINGS.image_size,
        SETTINGS.image_size,
        pipe.text_encoder.dtype,
        pipe.device,
        None,
        None,
    )


def build_shift(run_id: int, plan_seed: int) -> dict[str, Any]:
    rng = random.Random(plan_seed + run_id)
    dx = rng.choice((-1, 1)) * rng.randint(24, 32)
    dy = rng.choice((-1, 1)) * rng.randint(24, 32)
    return {
        "run_id": run_id,
        "attack_seed": 42 + run_id,
        "shift_rng_seed": plan_seed + run_id,
        "flow_dx_image_px": dx,
        "flow_dy_image_px": dy,
        "visual_dx_image_px": -dx,
        "visual_dy_image_px": -dy,
        "dx_latent_equivalent": dx / 8.0,
        "dy_latent_equivalent": dy / 8.0,
    }


def protocol_hash(plan: list[dict[str, Any]]) -> str:
    payload = {"tree_ring": SETTINGS.__dict__, "attack": ATTACK_CONFIG, "shift_plan": plan}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def command_prepare(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    prompts = load_csv(args.prompts)
    if len(prompts) < args.count:
        raise ValueError(f"only {len(prompts)} prompts, requested {args.count}")
    prompts = prompts[: args.count]
    output.mkdir(parents=True)
    for relative in (
        "configs", "cohort/clean", "cohort/watermarked", "attacks/clean",
        "attacks/watermarked", "scores", "metrics", "plots", "logs", "state",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    prompt_records = []
    shift_plan = []
    for run_id, row in enumerate(prompts):
        prompt = row.get("prompt", "")
        if not prompt:
            raise ValueError(f"empty generation prompt at run_id={run_id}")
        prompt_records.append({
            "run_id": run_id,
            "prompt_id": row.get("id", str(run_id)),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "source": row.get("source", "DiffusionDB"),
            "sample_seed": SETTINGS.generation_seed + run_id,
            "watermark_seed": SETTINGS.watermark_seed,
        })
        shift_plan.append(build_shift(run_id, args.plan_seed))

    config_hash = protocol_hash(shift_plan)
    write_json(output / "configs" / "tree_ring_official.json", SETTINGS.__dict__)
    write_json(output / "configs" / "raven_attack.json", ATTACK_CONFIG)
    write_json(output / "shift_plan.json", {"plan_seed": args.plan_seed, "samples": shift_plan})
    write_csv(output / "generation_manifest.csv", prompt_records, exclusive=True)
    write_json(output / "state" / "prepared.json", {"count": args.count, "protocol_hash": config_hash})

    official_commit = run_text(["git", "rev-parse", "HEAD"], args.official_repo)
    if official_commit != OFFICIAL_TREE_RING_COMMIT:
        raise RuntimeError(f"Tree-Ring checkout commit changed: {official_commit}")
    model_source = resolved_model_source(args.model_source)
    provenance = {
        "result_name": "Tree-Ring official complex-L1 and official ROC",
        "legacy_exclusions": [
            "legacy_invalid_shared_wm_zT_cohort", "old_0.4655_after_tpr",
            "old_0.4563_deduplicated_tpr", "custom_ddim_detector", "nfpa_style_detector",
        ],
        "git_commit_at_prepare": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_at_prepare": run_text(["git", "status", "--short"]),
        "tree_ring_official_repository": str(args.official_repo.resolve()),
        "tree_ring_official_commit": official_commit,
        "official_requested_model_id": SETTINGS.model_id,
        "resolved_local_model_source": str(model_source),
        "resolved_model_mirror_id": MODEL_MIRROR_ID,
        "resolved_model_mirror_revision": MODEL_MIRROR_REVISION,
        "official_model_endpoint_status": "unavailable_401_at_audit_time",
        "official_weight_hash_parity": "not_confirmed",
        "protocol_hash": config_hash,
        "dataset": "DiffusionDB",
        "dataset_size": args.count,
        "prompts_path": str(args.prompts.resolve()),
        "prompts_sha256": sha256_path(args.prompts),
        "python": sys.version,
        "versions": {
            name: importlib.import_module(name).__version__
            for name in ("torch", "diffusers", "transformers", "numpy", "scipy", "sklearn")
        },
        "roc_implementation": "sklearn.metrics.roc_curve; tpr[np.where(fpr < 0.01)[0][-1]]",
    }
    write_json(output / "provenance.json", provenance)
    commands = [
        f"python -u {Path(__file__).resolve()} generate --output-dir {output} --model-source {model_source} --count 2",
        f"python -u {Path(__file__).resolve()} parity --output-dir {output} --model-source {model_source} --official-repo {args.official_repo.resolve()}",
        f"python -u {Path(__file__).resolve()} attack --output-dir {output} --model-source {model_source} --count 2",
        f"python -u {Path(__file__).resolve()} score --output-dir {output} --model-source {model_source} --count 2",
        f"python -u {Path(__file__).resolve()} validate --output-dir {output} --count 2",
    ]
    (output / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"prepared {args.count} paired samples at {output}", flush=True)
    return 0


def validate_existing_image(path: Path, expected_sha: str) -> None:
    if not path.is_file() or sha256_path(path) != expected_sha:
        raise ValueError(f"missing or changed completed image: {path}")


def command_generate(args: argparse.Namespace) -> int:
    import torch

    output = args.output_dir.resolve()
    prompts = load_csv(output / "generation_manifest.csv")[: args.count]
    records_path = output / "paired_latent_manifest.jsonl"
    existing = {int(row["run_id"]): row for row in load_jsonl(records_path)}
    for row in existing.values():
        validate_existing_image(Path(row["clean_image_path"]), row["clean_image_sha256"])
        validate_existing_image(Path(row["watermarked_image_path"]), row["watermarked_image_sha256"])
    open_mode = "a" if records_path.exists() else "x"

    pipe = load_official_pipeline(resolved_model_source(args.model_source), args.device)
    target_path = output / "configs" / "watermark_target.pt"
    mask_path = output / "configs" / "watermark_mask.pt"
    set_official_random_seed(SETTINGS.watermark_seed)
    gt_init = get_random_latents(pipe)
    target = make_rand_watermark_target(gt_init)
    mask = make_watermark_mask(gt_init, SETTINGS.watermark_channel, SETTINGS.watermark_radius)
    target_hash = stable_tensor_hash(target)
    mask_hash = hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if target_path.exists() or mask_path.exists():
        saved_target = torch.load(target_path, map_location=pipe.device, weights_only=True)
        saved_mask = torch.load(mask_path, map_location=pipe.device, weights_only=True)
        if stable_tensor_hash(saved_target) != target_hash or not torch.equal(saved_mask, mask):
            raise RuntimeError("saved watermark target/mask drift")
        target, mask = saved_target.to(pipe.device), saved_mask.to(pipe.device)
    else:
        torch.save(target.detach().cpu(), target_path)
        torch.save(mask.detach().cpu(), mask_path)

    seen_base_hashes = {row["base_zT_hash"] for row in existing.values()}
    with records_path.open(open_mode, encoding="utf-8") as records:
        for position, prompt_row in enumerate(prompts, start=1):
            run_id = int(prompt_row["run_id"])
            if run_id in existing:
                continue
            set_official_random_seed(int(prompt_row["sample_seed"]))
            base_zT = get_random_latents(pipe)
            clean_zT = base_zT.clone()
            wm_preinject_zT = base_zT.clone()
            hashes = {
                "base_zT_hash": stable_tensor_hash(base_zT),
                "clean_zT_hash": stable_tensor_hash(clean_zT),
                "wm_preinject_zT_hash": stable_tensor_hash(wm_preinject_zT),
            }
            if hashes["clean_zT_hash"] != hashes["wm_preinject_zT_hash"]:
                raise AssertionError(f"clean/WM pre-injection mismatch run_id={run_id}")
            if hashes["base_zT_hash"] in seen_base_hashes:
                raise AssertionError(f"duplicate base latent run_id={run_id}")
            wm_postinject_zT = inject_complex_watermark(wm_preinject_zT, mask, target)
            hashes["wm_postinject_zT_hash"] = stable_tensor_hash(wm_postinject_zT)
            if hashes["clean_zT_hash"] == hashes["wm_postinject_zT_hash"]:
                raise AssertionError(f"watermark injection made no change run_id={run_id}")

            prompt = prompt_row["prompt"]
            with torch.inference_mode():
                clean_image = pipe(
                    prompt,
                    guidance_scale=SETTINGS.generation_guidance_scale,
                    num_inference_steps=SETTINGS.generation_steps,
                    height=SETTINGS.image_size,
                    width=SETTINGS.image_size,
                    latents=clean_zT.clone(),
                ).images[0]
                watermarked_image = pipe(
                    prompt,
                    guidance_scale=SETTINGS.generation_guidance_scale,
                    num_inference_steps=SETTINGS.generation_steps,
                    height=SETTINGS.image_size,
                    width=SETTINGS.image_size,
                    latents=wm_postinject_zT.clone(),
                ).images[0]
            clean_path = output / "cohort" / "clean" / f"{run_id:06d}.png"
            wm_path = output / "cohort" / "watermarked" / f"{run_id:06d}.png"
            if clean_path.exists() or wm_path.exists():
                raise FileExistsError(f"unrecorded cohort output run_id={run_id}")
            clean_image.save(clean_path)
            watermarked_image.save(wm_path)
            row = {
                **prompt_row,
                **hashes,
                "watermark_target_hash": target_hash,
                "watermark_mask_hash": mask_hash,
                "watermark_settings": SETTINGS.__dict__,
                "clean_image_path": str(clean_path.resolve()),
                "clean_image_sha256": sha256_path(clean_path),
                "watermarked_image_path": str(wm_path.resolve()),
                "watermarked_image_sha256": sha256_path(wm_path),
            }
            append_jsonl(records, row)
            seen_base_hashes.add(hashes["base_zT_hash"])
            print(f"[generate {position}/{len(prompts)}] run_id={run_id} base={hashes['base_zT_hash'][:12]}", flush=True)
    rows = load_jsonl(records_path)
    write_csv(output / "paired_latent_manifest.csv", rows)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def load_official_source_modules(official_repo: Path):
    if run_text(["git", "rev-parse", "HEAD"], official_repo) != OFFICIAL_TREE_RING_COMMIT:
        raise RuntimeError("official Tree-Ring source checkout is not the audited commit")
    if "datasets" not in sys.modules:
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.load_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dataset loading disabled"))
        sys.modules["datasets"] = datasets_stub
    sys.path.insert(0, str(official_repo.resolve()))
    try:
        optim_utils = importlib.import_module("optim_utils")
        inverse_module = importlib.import_module("inverse_stable_diffusion")
    finally:
        sys.path.pop(0)
    return optim_utils, inverse_module


def command_parity(args: argparse.Namespace) -> int:
    import torch

    output = args.output_dir.resolve()
    cohort = load_jsonl(output / "paired_latent_manifest.jsonl")
    if not cohort:
        raise RuntimeError("generate at least one pair before parity")
    pipe = load_official_pipeline(resolved_model_source(args.model_source), args.device)
    optim_utils, inverse_module = load_official_source_modules(args.official_repo.resolve())

    official_args = types.SimpleNamespace(
        w_seed=SETTINGS.watermark_seed,
        w_channel=SETTINGS.watermark_channel,
        w_pattern=SETTINGS.watermark_pattern,
        w_mask_shape=SETTINGS.watermark_mask_shape,
        w_radius=SETTINGS.watermark_radius,
        w_measurement=SETTINGS.watermark_measurement,
        w_injection=SETTINGS.watermark_injection,
        w_pattern_const=0,
    )
    shape = (1, pipe.unet.config.in_channels, 64, 64)
    pipe.get_random_latents = types.MethodType(lambda self: get_random_latents(self), pipe)
    optim_utils.set_random_seed(SETTINGS.watermark_seed)
    official_target = optim_utils.get_watermarking_pattern(pipe, official_args, pipe.device)
    set_official_random_seed(SETTINGS.watermark_seed)
    local_gt_init = get_random_latents(pipe)
    local_target = make_rand_watermark_target(local_gt_init)
    target_max_diff = float((official_target - local_target).abs().max().detach().cpu().item())
    official_mask = optim_utils.get_watermarking_mask(local_gt_init, official_args, pipe.device)
    local_mask = make_watermark_mask(local_gt_init, SETTINGS.watermark_channel, SETTINGS.watermark_radius)
    mask_equal = bool(torch.equal(official_mask, local_mask))

    set_official_random_seed(17)
    base = torch.randn(*shape, device=pipe.device, dtype=pipe.text_encoder.dtype)
    official_injected = optim_utils.inject_watermark(base.clone(), official_mask, official_target, official_args)
    local_injected = inject_complex_watermark(base.clone(), local_mask, local_target)
    injection_max_diff = float((official_injected - local_injected).abs().max().detach().cpu().item())

    sample = cohort[0]
    image = Image.open(sample["watermarked_image_path"]).convert("RGB")
    image_latents = image_to_official_latents(pipe, image)
    embeddings = get_empty_text_embedding(pipe)

    class OfficialAdapter:
        scheduler = pipe.scheduler
        device = pipe.device
        unet = pipe.unet

        @staticmethod
        def progress_bar(iterable):
            return iterable

    official_recovered = inverse_module.InversableStableDiffusionPipeline.backward_diffusion(
        OfficialAdapter(),
        text_embeddings=embeddings,
        latents=image_latents.clone(),
        num_inference_steps=SETTINGS.detector_steps,
        guidance_scale=SETTINGS.detector_guidance_scale,
        reverse_process=True,
    )
    local_recovered, local_diagnostics = official_forward_diffusion(
        pipe, image_latents.clone(), embeddings, SETTINGS.detector_steps
    )
    recovered_max_diff = float((official_recovered - local_recovered).abs().max().detach().cpu().item())
    official_score = optim_utils.eval_watermark(
        official_recovered,
        official_recovered,
        official_mask,
        official_target,
        official_args,
    )[0]
    local_score = official_complex_l1(local_recovered, local_mask, local_target)
    score_abs_diff = abs(float(official_score) - float(local_score))
    score_rel_diff = score_abs_diff / max(abs(float(official_score)), 1e-12)
    passed = (
        target_max_diff == 0.0
        and mask_equal
        and injection_max_diff <= 1e-6
        and recovered_max_diff <= 1e-6
        and score_abs_diff <= 1e-5
    )
    report = {
        "passed": passed,
        "official_source_commit": OFFICIAL_TREE_RING_COMMIT,
        "official_target_vs_port_max_abs_difference": target_max_diff,
        "official_mask_vs_port_exact_equal": mask_equal,
        "official_injection_vs_port_max_abs_difference": injection_max_diff,
        "official_recovered_vs_port_max_abs_difference": recovered_max_diff,
        "official_score": float(official_score),
        "ported_score": float(local_score),
        "absolute_score_difference": score_abs_diff,
        "relative_score_difference": score_rel_diff,
        "official_timestep_sequence": [int(t) for t in reversed(pipe.scheduler.timesteps.detach().cpu().tolist())],
        "ported_timestep_sequence": local_diagnostics["official_inverse_timesteps"],
        "final_recovered_latent_mean": local_diagnostics["final_recovered_latent_mean"],
        "final_recovered_latent_std": local_diagnostics["final_recovered_latent_std"],
        "masked_fft_mean_magnitude": float(
            torch.fft.fftshift(torch.fft.fft2(local_recovered), dim=(-1, -2))[local_mask]
            .abs().float().mean().detach().cpu().item()
        ),
        "model_checkpoint_note": "local Redbeard mirror; official stabilityai endpoint unavailable, weight hash parity not confirmed",
    }
    write_json(output / "scores" / "official_detector_parity.json", report, exclusive=False)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("official detector parity failed; refusing later formal stages")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def read_shift_plan(output: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads((output / "shift_plan.json").read_text(encoding="utf-8"))
    return {int(row["run_id"]): row for row in payload["samples"]}


def assert_attack_debug(debug: dict[str, Any], run_id: int) -> None:
    expected = {
        "inversion_mode": "ddim",
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "padding_mode": "reflection",
        "interpolation_mode": "nearest",
        "color_transfer_mode": "paper_exact_two_stage",
        "warp_input_stage": "ddim_inversion.noisy_latents_z_tau",
        "warp_input_is_inversion_noisy_latents": True,
        "decoded_output_branch": "view_branch_index_1",
    }
    for key, value in expected.items():
        if debug.get(key) != value:
            raise RuntimeError(f"RAVEN config drift run_id={run_id}: {key}={debug.get(key)!r}, expected {value!r}")
    if int(debug.get("attention_processor_count", 0)) != 32:
        raise RuntimeError(f"expected 32 total attention processors run_id={run_id}")
    attention = debug.get("attention_debug", {})
    active_steps = len(debug.get("timesteps", []))
    if int(attention.get("self_processor_count", 0)) != 16:
        raise RuntimeError(f"expected 16 self-attention processors run_id={run_id}")
    if int(attention.get("processors_with_calls", 0)) != 16:
        raise RuntimeError(f"not all attention processors invoked run_id={run_id}")
    if int(attention.get("total_calls", 0)) != 16 * active_steps:
        raise RuntimeError(f"attention call count mismatch run_id={run_id}")


def load_completed_attack(item_dir: Path, input_sha: str, run_id: int) -> tuple[Path, dict[str, Any]] | None:
    final_path = item_dir / "final_color_corrected.png"
    debug_path = item_dir / "debug_info.json"
    input_path = item_dir / "input.png"
    if not item_dir.exists():
        return None
    if not final_path.is_file() or not debug_path.is_file() or not input_path.is_file():
        raise FileExistsError(f"partial unrecorded attack output exists: {item_dir}")
    if sha256_path(input_path) != input_sha:
        raise ValueError(f"attack input SHA mismatch: {item_dir}")
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    assert_attack_debug(debug, run_id)
    return final_path, debug


def run_attack_one(
    pipeline: RavenPipeline,
    source_path: Path,
    source_sha: str,
    item_dir: Path,
    plan: dict[str, Any],
    run_id: int,
) -> tuple[Path, dict[str, Any], float, int]:
    existing = load_completed_attack(item_dir, source_sha, run_id)
    if existing is not None:
        return existing[0], existing[1], 0.0, 0
    image = Image.open(source_path).convert("RGB")
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    pipeline.run(
        input_image=image,
        output_dir=item_dir,
        steps=ATTACK_CONFIG["steps"],
        strength=ATTACK_CONFIG["strength"],
        guidance_scale=ATTACK_CONFIG["guidance_scale"],
        shift_space=ATTACK_CONFIG["shift_space"],
        warp_mode=ATTACK_CONFIG["warp_mode"],
        padding_mode=ATTACK_CONFIG["padding_mode"],
        latent_sampling_mode=ATTACK_CONFIG["latent_sampling_mode"],
        shift_x=plan["flow_dx_image_px"],
        shift_y=plan["flow_dy_image_px"],
        view_guided_attention=True,
        color_transfer=True,
        seed=int(plan["attack_seed"]),
        prompt="",
        negative_prompt="",
        debug=False,
        inversion_mode="ddim",
    )
    final_path = item_dir / "final_color_corrected.png"
    debug = json.loads((item_dir / "debug_info.json").read_text(encoding="utf-8"))
    assert_attack_debug(debug, run_id)
    return final_path, debug, time.monotonic() - started, int(torch.cuda.max_memory_allocated())


def command_attack(args: argparse.Namespace) -> int:
    import torch

    output = args.output_dir.resolve()
    parity = json.loads((output / "scores" / "official_detector_parity.json").read_text(encoding="utf-8"))
    if not parity.get("passed"):
        raise RuntimeError("official detector parity has not passed")
    cohort = load_jsonl(output / "paired_latent_manifest.jsonl")[: args.count]
    if len(cohort) != args.count:
        raise ValueError(f"expected {args.count} generated pairs, found {len(cohort)}")
    plan = read_shift_plan(output)
    records_path = output / "attack_records.jsonl"
    existing = {int(row["run_id"]): row for row in load_jsonl(records_path)}
    for row in existing.values():
        validate_existing_image(Path(row["attacked_clean_path"]), row["attacked_clean_sha256"])
        validate_existing_image(Path(row["attacked_watermarked_path"]), row["attacked_watermarked_sha256"])
    mode = "a" if records_path.exists() else "x"
    pipeline = RavenPipeline(model_id=str(resolved_model_source(args.model_source)), device=args.device, dtype="float16")
    with records_path.open(mode, encoding="utf-8") as records:
        for position, row in enumerate(cohort, start=1):
            run_id = int(row["run_id"])
            if run_id in existing:
                continue
            shift = plan[run_id]
            clean_path = Path(row["clean_image_path"])
            wm_path = Path(row["watermarked_image_path"])
            clean_output, clean_debug, clean_runtime, clean_peak = run_attack_one(
                pipeline, clean_path, row["clean_image_sha256"],
                output / "attacks" / "clean" / f"{run_id:06d}", shift, run_id,
            )
            wm_output, wm_debug, wm_runtime, wm_peak = run_attack_one(
                pipeline, wm_path, row["watermarked_image_sha256"],
                output / "attacks" / "watermarked" / f"{run_id:06d}", shift, run_id,
            )
            if clean_debug["transform_config_hash"] != wm_debug["transform_config_hash"]:
                raise RuntimeError(f"clean/WM transform config mismatch run_id={run_id}")
            for key in ("exact_timestep", "strength", "guidance_scale", "warp_mode", "padding_mode", "interpolation_mode"):
                if clean_debug.get(key) != wm_debug.get(key):
                    raise RuntimeError(f"clean/WM attack mismatch run_id={run_id}: {key}")
            clean_attacked = Image.open(clean_output).convert("RGB")
            wm_attacked = Image.open(wm_output).convert("RGB")
            dx, dy = int(shift["flow_dx_image_px"]), int(shift["flow_dy_image_px"])
            clean_warp = clean_debug.get("nfpa_warp_metadata") or {}
            wm_warp = wm_debug.get("nfpa_warp_metadata") or {}
            for key in ("effective_flow_dx_image_px", "effective_flow_dy_image_px"):
                if key not in clean_warp or key not in wm_warp:
                    raise RuntimeError(f"missing effective-flow metadata run_id={run_id}: {key}")
                if not math.isclose(float(clean_warp[key]), float(wm_warp[key]), abs_tol=1e-6):
                    raise RuntimeError(f"clean/WM effective-flow mismatch run_id={run_id}: {key}")
            effective_dx = float(wm_warp["effective_flow_dx_image_px"])
            effective_dy = float(wm_warp["effective_flow_dy_image_px"])
            alignment_mode = (
                "fractional_grid_sample"
                if wm_warp.get("effective_flow_is_fractional")
                else "integer_crop"
            )
            record = {
                "run_id": run_id,
                "attack_seed": shift["attack_seed"],
                "flow_dx_image_px": dx,
                "flow_dy_image_px": dy,
                "clean_path": str(clean_path.resolve()),
                "clean_sha256": row["clean_image_sha256"],
                "watermarked_path": str(wm_path.resolve()),
                "watermarked_sha256": row["watermarked_image_sha256"],
                "attacked_clean_path": str(clean_output.resolve()),
                "attacked_clean_sha256": sha256_path(clean_output),
                "attacked_watermarked_path": str(wm_output.resolve()),
                "attacked_watermarked_sha256": sha256_path(wm_output),
                "transform_config_hash": clean_debug["transform_config_hash"],
                "exact_ddim_timestep": clean_debug["exact_timestep"],
                "attack_config": ATTACK_CONFIG,
                "clean_attack_runtime_seconds": clean_runtime,
                "watermarked_attack_runtime_seconds": wm_runtime,
                "peak_gpu_memory_bytes": max(clean_peak, wm_peak),
                "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "clean_quality": image_quality(
                    Image.open(clean_path), clean_attacked, dx, dy,
                    effective_dx, effective_dy, alignment_mode,
                ),
                "watermarked_quality": image_quality(
                    Image.open(wm_path), wm_attacked, dx, dy,
                    effective_dx, effective_dy, alignment_mode,
                ),
                "clean_debug_path": str((output / "attacks" / "clean" / f"{run_id:06d}" / "debug_info.json").resolve()),
                "watermarked_debug_path": str((output / "attacks" / "watermarked" / f"{run_id:06d}" / "debug_info.json").resolve()),
            }
            append_jsonl(records, record)
            print(
                f"[attack {position}/{len(cohort)}] run_id={run_id} "
                f"psnr={record['watermarked_quality']['post_color_overlap_psnr']:.3f} "
                f"gpu={record['peak_gpu_memory_bytes']/2**30:.2f}GiB",
                flush=True,
            )
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def load_target_mask(output: Path, device: str):
    import torch

    target = torch.load(output / "configs" / "watermark_target.pt", map_location=device, weights_only=True)
    mask = torch.load(output / "configs" / "watermark_mask.pt", map_location=device, weights_only=True)
    return target.to(device), mask.to(device)


def command_score(args: argparse.Namespace) -> int:
    import torch

    output = args.output_dir.resolve()
    parity = json.loads((output / "scores" / "official_detector_parity.json").read_text(encoding="utf-8"))
    if not parity.get("passed"):
        raise RuntimeError("official detector parity has not passed")
    cohort = {int(row["run_id"]): row for row in load_jsonl(output / "paired_latent_manifest.jsonl")[: args.count]}
    attacks = {int(row["run_id"]): row for row in load_jsonl(output / "attack_records.jsonl")[: args.count]}
    if len(cohort) != args.count or len(attacks) != args.count:
        raise ValueError(f"need {args.count} complete cohort and attack records")
    records_path = output / "score_records.jsonl"
    existing = {int(row["run_id"]): row for row in load_jsonl(records_path)}
    mode = "a" if records_path.exists() else "x"
    pipe = load_official_pipeline(resolved_model_source(args.model_source), args.device)
    target, mask = load_target_mask(output, args.device)
    target_hash = stable_tensor_hash(target)
    detector_provenance_written = (output / "scores" / "detector_provenance.json").is_file()

    with records_path.open(mode, encoding="utf-8") as records:
        for position, run_id in enumerate(sorted(cohort), start=1):
            if run_id in existing:
                continue
            row, attack = cohort[run_id], attacks[run_id]
            paths = {
                "clean": Path(row["clean_image_path"]),
                "watermarked": Path(row["watermarked_image_path"]),
                "attacked_clean": Path(attack["attacked_clean_path"]),
                "attacked_watermarked": Path(attack["attacked_watermarked_path"]),
            }
            expected = {
                "clean": row["clean_image_sha256"],
                "watermarked": row["watermarked_image_sha256"],
                "attacked_clean": attack["attacked_clean_sha256"],
                "attacked_watermarked": attack["attacked_watermarked_sha256"],
            }
            for stage, path in paths.items():
                validate_existing_image(path, expected[stage])
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            scores: dict[str, float] = {}
            diagnostics: dict[str, dict[str, Any]] = {}
            for stage, path in paths.items():
                score, diagnostic = score_image(
                    pipe, Image.open(path).convert("RGB"), mask, target, SETTINGS.detector_steps
                )
                if not math.isfinite(score):
                    raise ValueError(f"non-finite official score run_id={run_id} stage={stage}")
                scores[stage] = score
                diagnostics[stage] = diagnostic
            if not detector_provenance_written:
                write_json(output / "scores" / "detector_provenance.json", {
                    "score_name": "tree_ring_official_complex_l1",
                    "score_direction": "lower_is_more_watermarked",
                    "formula": "abs(fftshift(fft2(recovered))[mask] - gt_patch[mask]).mean()",
                    "scheduler": diagnostics["clean"]["scheduler"],
                    "scheduler_timesteps_descending": diagnostics["clean"]["scheduler_timesteps_descending"],
                    "official_inverse_timesteps": diagnostics["clean"]["official_inverse_timesteps"],
                    "vae_posterior": "mode",
                    "vae_sample": False,
                    "detector_prompt": "",
                    "detector_guidance_scale": 1,
                    "detector_steps": 50,
                    "watermark_target_hash": target_hash,
                    "official_source_commit": OFFICIAL_TREE_RING_COMMIT,
                })
                detector_provenance_written = True
            score_record = {
                "run_id": run_id,
                "score_name": "tree_ring_official_complex_l1",
                "score_direction": "lower_is_more_watermarked",
                "clean_l1": scores["clean"],
                "watermarked_l1": scores["watermarked"],
                "attacked_clean_l1": scores["attacked_clean"],
                "attacked_watermarked_l1": scores["attacked_watermarked"],
                "clean_path": str(paths["clean"]),
                "clean_sha256": expected["clean"],
                "watermarked_path": str(paths["watermarked"]),
                "watermarked_sha256": expected["watermarked"],
                "attacked_clean_path": str(paths["attacked_clean"]),
                "attacked_clean_sha256": expected["attacked_clean"],
                "attacked_watermarked_path": str(paths["attacked_watermarked"]),
                "attacked_watermarked_sha256": expected["attacked_watermarked"],
                "watermark_target_hash": target_hash,
                "watermark_mask_hash": row["watermark_mask_hash"],
                "transform_config_hash": attack["transform_config_hash"],
                "flow_dx_image_px": attack["flow_dx_image_px"],
                "flow_dy_image_px": attack["flow_dy_image_px"],
                "runtime_seconds": time.monotonic() - started,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "nan_count": 0,
                "inf_count": 0,
            }
            append_jsonl(records, score_record)
            print(
                f"[score {position}/{len(cohort)}] run_id={run_id} "
                f"clean={scores['clean']:.8f} wm={scores['watermarked']:.8f} "
                f"att_clean={scores['attacked_clean']:.8f} att_wm={scores['attacked_watermarked']:.8f}",
                flush=True,
            )
    score_rows = load_jsonl(records_path)
    write_csv(output / "scores" / "score_records.csv", score_rows)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def validate_records(output: Path, count: int) -> dict[str, Any]:
    cohort = load_jsonl(output / "paired_latent_manifest.jsonl")[:count]
    attacks = load_jsonl(output / "attack_records.jsonl")[:count]
    scores = load_jsonl(output / "score_records.jsonl")[:count]
    if not (len(cohort) == len(attacks) == len(scores) == count):
        raise ValueError(
            f"incomplete records for count={count}: cohort={len(cohort)} attacks={len(attacks)} scores={len(scores)}"
        )
    expected_ids = list(range(count))
    for name, rows in (("cohort", cohort), ("attacks", attacks), ("scores", scores)):
        ids = [int(row["run_id"]) for row in rows]
        if ids != expected_ids or len(set(ids)) != count:
            raise ValueError(f"{name} run_id order/uniqueness failure")
    base_hashes = [row["base_zT_hash"] for row in cohort]
    target_hashes = [row["watermark_target_hash"] for row in cohort]
    mask_hashes = [row["watermark_mask_hash"] for row in cohort]
    if len(set(base_hashes)) != count:
        raise AssertionError("base latent uniqueness failed")
    if any(row["clean_zT_hash"] != row["wm_preinject_zT_hash"] for row in cohort):
        raise AssertionError("clean/WM pre-injection equality failed")
    if any(row["clean_zT_hash"] == row["wm_postinject_zT_hash"] for row in cohort):
        raise AssertionError("watermark injection change assertion failed")
    if len(set(target_hashes)) != 1 or len(set(mask_hashes)) != 1:
        raise AssertionError("watermark target or mask is not fixed")
    for row in scores:
        values = [row[key] for key in ("clean_l1", "watermarked_l1", "attacked_clean_l1", "attacked_watermarked_l1")]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite score run_id={row['run_id']}")
    for row in attacks:
        if row["attack_config"] != ATTACK_CONFIG:
            raise AssertionError(f"attack config mismatch run_id={row['run_id']}")
        if not row.get("transform_config_hash"):
            raise AssertionError(f"missing attack config hash run_id={row['run_id']}")
    clean = [float(row["clean_l1"]) for row in scores]
    wm = [float(row["watermarked_l1"]) for row in scores]
    attacked_clean = [float(row["attacked_clean_l1"]) for row in scores]
    attacked_wm = [float(row["attacked_watermarked_l1"]) for row in scores]
    result = {
        "count": count,
        "base_latents_unique": len(set(base_hashes)),
        "clean_equals_wm_preinject": count,
        "fixed_target_count": len(set(target_hashes)),
        "fixed_mask_count": len(set(mask_hashes)),
        "clean_score": finite_stats(clean),
        "watermarked_score": finite_stats(wm),
        "attacked_clean_score": finite_stats(attacked_clean),
        "attacked_watermarked_score": finite_stats(attacked_wm),
        "wm_to_attacked_wm_change": finite_stats([b - a for a, b in zip(wm, attacked_wm)]),
        "clean_to_attacked_clean_change": finite_stats([b - a for a, b in zip(clean, attacked_clean)]),
        "paired_attacked_clean_minus_attacked_wm_gap": finite_stats([a - b for a, b in zip(attacked_clean, attacked_wm)]),
        "duplicate_image_sha": {
            "clean": count - len({row["clean_image_sha256"] for row in cohort}),
            "watermarked": count - len({row["watermarked_image_sha256"] for row in cohort}),
            "attacked_clean": count - len({row["attacked_clean_sha256"] for row in attacks}),
            "attacked_watermarked": count - len({row["attacked_watermarked_sha256"] for row in attacks}),
        },
        "nan_count": 0,
        "inf_count": 0,
        "quality": {
            "post_color_overlap_psnr": finite_stats([row["watermarked_quality"]["post_color_overlap_psnr"] for row in attacks]),
            "post_color_overlap_ssim": finite_stats([row["watermarked_quality"]["post_color_overlap_ssim"] for row in attacks]),
        },
        "formal_tpr_claim_permitted": count >= 1000,
    }
    return result


def command_validate(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    result = validate_records(output, args.count)
    write_json(output / "metrics" / f"validation_{args.count}.json", result, exclusive=False)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def histogram(values: list[float], bins: int = 40) -> dict[str, Any]:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def command_aggregate(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    validation = validate_records(output, args.expected_count)
    if args.expected_count != 1000:
        raise ValueError("formal aggregate requires exactly 1000 pairs")
    rows = load_jsonl(output / "score_records.jsonl")
    attacks = load_jsonl(output / "attack_records.jsonl")
    clean = [float(row["clean_l1"]) for row in rows]
    watermarked = [float(row["watermarked_l1"]) for row in rows]
    attacked_clean = [float(row["attacked_clean_l1"]) for row in rows]
    attacked_watermarked = [float(row["attacked_watermarked_l1"]) for row in rows]
    before = official_roc(clean, watermarked)
    after = official_roc(attacked_clean, attacked_watermarked)
    fixed_after = rate_at_negative_l1_threshold(
        attacked_watermarked, before["decision_threshold_negative_l1"]
    )
    summary = {
        "formal_result_name": "Tree-Ring official complex-L1 with official ROC at 1% FPR",
        "dataset": "DiffusionDB",
        "n_expected": 1000,
        "n_completed": len(rows),
        "n_failed": 0,
        "before": before,
        "after_attack_matched": after,
        "attack_success_rate": 1.0 - after["tpr_at_1pct_fpr"],
        "after_fixed_before_threshold_tpr": fixed_after,
        "score_statistics": {
            "clean": finite_stats(clean),
            "watermarked": finite_stats(watermarked),
            "attacked_clean": finite_stats(attacked_clean),
            "attacked_watermarked": finite_stats(attacked_watermarked),
        },
        "quality": validation["quality"],
        "nan_count": 0,
        "inf_count": 0,
        "quadrants": {},
    }
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ids = [
            int(row["run_id"]) for row in attacks
            if math.copysign(1, row["flow_dx_image_px"]) == sx
            and math.copysign(1, row["flow_dy_image_px"]) == sy
        ]
        values = [attacked_watermarked[index] for index in ids]
        summary["quadrants"][f"({'+' if sx > 0 else '-'},{'+' if sy > 0 else '-'})"] = {
            "n": len(values),
            "tpr_at_global_after_threshold": rate_at_negative_l1_threshold(
                values, after["decision_threshold_negative_l1"]
            ),
            "score": finite_stats(values),
        }
    histograms = {
        "clean": histogram(clean),
        "watermarked": histogram(watermarked),
        "attacked_clean": histogram(attacked_clean),
        "attacked_watermarked": histogram(attacked_watermarked),
    }
    write_json(output / "metrics" / "score_histograms.json", histograms, exclusive=False)
    write_json(output / "summary.json", summary, exclusive=False)
    lines = [
        "# Tree-Ring Official RAVEN Evaluation",
        "",
        "| Metric | Before | After attack-matched |",
        "| --- | ---: | ---: |",
        f"| Threshold (-L1 decision score) | {before['decision_threshold_negative_l1']:.9f} | {after['decision_threshold_negative_l1']:.9f} |",
        f"| Actual FPR | {before['actual_fpr']:.6f} | {after['actual_fpr']:.6f} |",
        f"| TPR@1%FPR | {before['tpr_at_1pct_fpr']:.6f} | {after['tpr_at_1pct_fpr']:.6f} |",
        f"| ROC-AUC | {before['auc']:.6f} | {after['auc']:.6f} |",
        f"| False positives | {before['false_positives']} | {after['false_positives']} |",
        "",
        f"Attack success rate: `{summary['attack_success_rate']:.6f}`",
        f"Fixed-before-threshold attacked TPR: `{fixed_after:.6f}`",
        f"Post-color overlap PSNR mean: `{validation['quality']['post_color_overlap_psnr']['mean']:.4f}`",
        f"Post-color overlap SSIM mean: `{validation['quality']['post_color_overlap_ssim']['mean']:.6f}`",
        "",
        "Model provenance limitation: the requested stabilityai endpoint was unavailable; the cached Redbeard mirror was used and official weight-hash parity could not be confirmed.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    prepare.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    prepare.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    prepare.add_argument("--count", type=int, default=1000)
    prepare.add_argument("--plan-seed", type=int, default=PLAN_SEED)

    for name in ("generate", "attack", "score"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
        command.add_argument("--count", type=int, required=True)
        command.add_argument("--device", choices=("cuda",), default="cuda")

    parity = sub.add_parser("parity")
    parity.add_argument("--output-dir", type=Path, required=True)
    parity.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parity.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parity.add_argument("--device", choices=("cuda",), default="cuda")

    validate = sub.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--count", type=int, required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--expected-count", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "prepare": command_prepare,
        "generate": command_generate,
        "parity": command_parity,
        "attack": command_attack,
        "score": command_score,
        "validate": command_validate,
        "aggregate": command_aggregate,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
