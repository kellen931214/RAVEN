#!/usr/bin/env python
"""Quality decomposition experiment for DiffusionDB RAVEN outputs.

This script intentionally runs a narrow set of experiments:

1. DDIM + paper shift + fully aligned color transfer.
2. DDIM + paper shift + 0.5 blended aligned color transfer.
3. Optional DDIM + paper shift without color transfer.

It processes images incrementally to avoid CPU RAM spikes.  No DataLoader is
used and no dataset-sized tensor cache is built.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.color_transfer import (
    PAPER_EXACT_TWO_STAGE_ALIGNED,
    PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
    color_contrast_transfer,
    color_transfer_diagnostics,
)
from raven.metrics import crop_overlap_inverse_warp, roc_auc
from raven.pairing_provenance import (
    assert_attack_pair_config_match,
    audit_pairing_rows,
    build_attack_config_sha256,
    tensor_sha256,
)
from raven.pipeline_raven import RavenPipeline
from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
from scripts import raven_p1_full as p1
from scripts.raven_nfpa_tr_eval import nfpa_rate, nfpa_threshold


FORMAL_SOURCE_ROOT = Path("outputs/raven_color_alignment_ablation/diffusiondb/20260716T082019Z")
FORMAL_CONFIG = {
    "dataset": "DiffusionDB",
    "image_size": [512, 512],
    "ddim_steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "prompt": "",
    "negative_prompt": "",
    "inversion_mode": "DDIM",
    "view_guided_attention": True,
    "shift_space": "image_pixels",
    "shift_rule": "dx and dy independently sampled from [24,32] or [-32,-24]",
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "latent_sampling": "nearest",
    "padding": "reflection",
}
VARIANT_CATALOG = [
    {
        "key": "ddim_shift_no_color",
        "name": "DDIM + shift without color transfer",
        "dx_dy": "formal paper shift plan",
        "color_transfer": "none",
        "alpha": 0.0,
        "source": "paired attacked view output before color transfer",
    },
    {
        "key": "alignment_color",
        "name": "DDIM + shift + aligned color transfer",
        "dx_dy": "formal paper shift plan",
        "color_transfer": PAPER_EXACT_TWO_STAGE_ALIGNED,
        "alpha": 1.0,
        "source": "paired attacked view output plus aligned color transfer",
    },
    {
        "key": "blend_alignment_color",
        "name": "DDIM + shift + blended aligned color transfer",
        "dx_dy": "formal paper shift plan",
        "color_transfer": PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
        "alpha": 0.5,
        "source": "paired attacked view output plus blended aligned color transfer",
    },
]
DEFAULT_VARIANT_KEYS = ("alignment_color", "blend_alignment_color")
VARIANTS = [
    variant for variant in VARIANT_CATALOG if variant["key"] in DEFAULT_VARIANT_KEYS
]


def resolve_variants(keys: list[str]) -> list[dict]:
    if not keys:
        raise ValueError("at least one variant is required")
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate variants are not allowed: {keys}")
    catalog = {variant["key"]: variant for variant in VARIANT_CATALOG}
    unknown = [key for key in keys if key not in catalog]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return [catalog[key] for key in keys]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        converted.load()
    return converted


def require_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError(f"expected 512x512 image, got {image.size}: {path}")


def git_sha(ref: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return None


def quality_pair(reference_path: Path, output_path: Path, dx: float, dy: float) -> dict:
    reference = np.asarray(open_rgb(reference_path), dtype=np.float32) / 255.0
    output = np.asarray(open_rgb(output_path), dtype=np.float32) / 255.0
    ref_crop, out_crop = crop_overlap_inverse_warp(reference, output, dx, dy)
    return {
        "psnr": float(peak_signal_noise_ratio(ref_crop, out_crop, data_range=1.0)),
        "ssim": float(structural_similarity(ref_crop, out_crop, channel_axis=2, data_range=1.0)),
        "valid_overlap_width": int(ref_crop.shape[1]),
        "valid_overlap_height": int(ref_crop.shape[0]),
        "valid_overlap_area_ratio": float(ref_crop.shape[0] * ref_crop.shape[1] / (reference.shape[0] * reference.shape[1])),
    }


def clip_prompt_cosines(pairs: list[tuple[str, Path]], device: str, model_name: str, pretrained: str) -> list[float]:
    import open_clip
    import torch

    if not pairs:
        return []
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for prompt, output_path in pairs:
            out = preprocess(open_rgb(output_path)).unsqueeze(0).to(device)
            text_tokens = tokenizer([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
            out_features = model.encode_image(out)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            out_features = out_features / out_features.norm(dim=-1, keepdim=True)
            scores.append(float((text_features * out_features).sum().cpu().item()))
            del out, text_tokens, text_features, out_features
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def cleanfid_from_paths(reference_paths: list[Path], output_paths: list[Path], device: str) -> dict:
    if not reference_paths or not output_paths or len(reference_paths) != len(output_paths):
        raise ValueError("FID requires equally sized non-empty reference/output path lists")
    from cleanfid import fid

    resolved_references = [path.resolve() for path in reference_paths]
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_references)) != len(resolved_references):
        raise ValueError("FID reference paths must be unique")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("FID output paths must be unique")
    staging_manifest = []
    for index, (reference, output) in enumerate(zip(resolved_references, resolved_outputs)):
        if not reference.is_file() or not output.is_file():
            raise FileNotFoundError(f"FID source missing at index={index}: {reference} / {output}")
        staging_manifest.append({
            "index": index,
            "reference_path": str(reference),
            "reference_sha256": sha256_path(reference),
            "output_path": str(output),
            "output_sha256": sha256_path(output),
        })
    staging_manifest_sha256 = hashlib.sha256(
        json.dumps(
            staging_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    with tempfile.TemporaryDirectory(prefix="raven_fid_") as tmp:
        tmp_path = Path(tmp)
        ref_dir = tmp_path / "reference"
        out_dir = tmp_path / "output"
        ref_dir.mkdir()
        out_dir.mkdir()
        for index, (reference, output) in enumerate(zip(reference_paths, output_paths)):
            for source, target_dir, label in ((reference, ref_dir, "ref"), (output, out_dir, "out")):
                suffix = source.suffix if source.suffix else ".png"
                target = target_dir / f"{index:05d}_{label}{suffix}"
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"stale FID staging target: {target}")
                try:
                    os.symlink(source.resolve(), target)
                except OSError:
                    shutil.copy2(source, target)
                if not target.is_file():
                    raise FileNotFoundError(f"broken FID staging target: {target}")
        expected_ref = {f"{index:05d}_ref{path.suffix or '.png'}" for index, path in enumerate(reference_paths)}
        expected_out = {f"{index:05d}_out{path.suffix or '.png'}" for index, path in enumerate(output_paths)}
        actual_ref = {path.name for path in ref_dir.iterdir()}
        actual_out = {path.name for path in out_dir.iterdir()}
        if actual_ref != expected_ref or actual_out != expected_out:
            raise ValueError("FID staging file set mismatch")
        value = fid.compute_fid(str(ref_dir), str(out_dir), device=device)
    return {
        "implementation": "clean-fid",
        "feature_extractor": "clean-fid default InceptionV3",
        "reference_count": len(reference_paths),
        "output_count": len(output_paths),
        "staging_manifest_sha256": staging_manifest_sha256,
        "staging_manifest_count": len(staging_manifest),
        "fresh_temporary_staging": True,
        "stale_symlink_count": 0,
        "temporary_dirs_removed": True,
        "value": float(value),
    }


def finite_stats(values: list[float]) -> dict:
    vals = [float(value) for value in values]
    finite = [value for value in vals if math.isfinite(value)]
    if not finite:
        return {"n": len(vals), "mean": None, "median": None, "std": None}
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def render_md(aggregate: dict) -> str:
    lines = [
        "# Quality Decomposition Experiment",
        "",
        "| Variant | CLIP(prompt) ↑ | FID ↓ | PSNR ↑ | SSIM ↑ | TPR@1%FPR ↓ | Attack success ↑ | ROC-AUC ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in aggregate["variant_order"]:
        stats = aggregate["variants"][variant]
        lines.append(
            f"| {stats['name']} | {stats['clip']['mean']:.6f} | {stats['fid']['value']:.6f} | "
            f"{stats['psnr']['mean']:.3f} | {stats['ssim']['mean']:.6f} | "
            f"{stats['tpr_at_1pct_fpr']:.6f} | {stats['attack_success_rate']:.6f} | "
            f"{stats['roc_auc']:.6f} |"
        )
    lines += [
        "",
        "Quality metrics compare attacked-watermarked output against the paired original watermarked input.",
        "Shifted PSNR/SSIM use inverse-warp valid overlap; no-shift uses dx=dy=0 full valid overlap.",
        "",
        "| Variant | N | Threshold | Actual FPR | DDIM/UNet rerun |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for variant in aggregate["variant_order"]:
        stats = aggregate["variants"][variant]
        lines.append(
            f"| {stats['name']} | {stats['n']} | {stats['threshold']:.8f} | "
            f"{stats['actual_fpr']:.6f} | {stats['ddim_unet_rerun']} |"
        )
    return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_records(source_root: Path, count: int) -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, dict], dict]:
    """Load and audit only the selected sample subset.

    Audit exactly the requested subset. Formal runs request all 1001 samples;
    smoke runs remain bounded without loading image pixels into memory.
    """
    p1_dir = source_root / "p1_1001"
    nfpa_dir = source_root / "nfpa_1001"
    plan = json.loads((p1_dir / "shift_plan.json").read_text(encoding="utf-8"))
    manifest_rows = load_manifest(p1_dir / "diagnostic_manifest.csv")
    wm_rows = load_jsonl(p1_dir / "attack_records.jsonl")
    clean_rows = load_jsonl(nfpa_dir / "attacked_clean_records.jsonl")
    if min(len(plan.get("samples", [])), len(manifest_rows), len(wm_rows), len(clean_rows)) < count:
        raise ValueError(f"source has fewer than requested samples: count={count}")
    samples = plan["samples"][:count]
    run_ids = [str(sample["run_id"]) for sample in samples]
    manifest = {str(row["run_id"]): row for row in manifest_rows}
    watermarked = {str(row["run_id"]): row for row in wm_rows}
    clean = {str(row["run_id"]): row for row in clean_rows}
    pairing_audit = audit_pairing_rows(
        [manifest[run_id] for run_id in run_ids],
        expected_count=count,
        verify_files=True,
    )
    p1_provenance = json.loads((p1_dir / "provenance.json").read_text(encoding="utf-8"))
    nfpa_provenance = json.loads((nfpa_dir / "provenance.json").read_text(encoding="utf-8"))
    for label, stored in (
        ("P1", p1_provenance.get("pairing_audit")),
        ("NFPA", nfpa_provenance.get("pairing_audit")),
    ):
        if not stored or not stored.get("passed") or int(stored.get("count", 0)) < count:
            raise ValueError(f"{label} missing valid parent pairing audit")
        for field in (
            "protocol", "watermark_target_sha256", "watermark_mask_sha256",
            "generation_config_sha256", "watermark_config_sha256", "model_revision",
        ):
            if stored.get(field) != pairing_audit.get(field):
                raise ValueError(f"{label} pairing audit mismatch: {field}")
    for run_id in run_ids:
        if run_id not in manifest or run_id not in watermarked or run_id not in clean:
            raise ValueError(f"missing source row for run_id={run_id}")
        wm = watermarked[run_id]
        clean_row = clean[run_id]
        if str(clean_row.get("run_id")) != run_id:
            raise ValueError(f"clean run_id mismatch for {run_id}")
        for path in (
            Path(wm["clean_path"]),
            Path(wm["watermarked_path"]),
            Path(wm["debug_info_path"]).parent / "view_guided_output.png",
            Path(clean_row["debug_info_path"]).parent / "view_guided_output.png",
        ):
            require_image(path)
        if abs(float(wm["flow_dx_image_px"]) - float(clean_row["flow_dx_image_px"])) > 1e-6:
            raise ValueError(f"dx mismatch run_id={run_id}")
        if abs(float(wm["flow_dy_image_px"]) - float(clean_row["flow_dy_image_px"])) > 1e-6:
            raise ValueError(f"dy mismatch run_id={run_id}")
        assert_attack_pair_config_match(clean_row, wm, run_id)
    return (
        samples,
        {run_id: manifest[run_id] for run_id in run_ids},
        {run_id: watermarked[run_id] for run_id in run_ids},
        {run_id: clean[run_id] for run_id in run_ids},
        pairing_audit,
    )


def ensure_color_outputs(
    output_dir: Path,
    variant_key: str,
    color_mode: str,
    alpha: float,
    run_id: str,
    wm_pre_path: Path,
    clean_pre_path: Path,
    watermarked_path: Path,
    clean_path: Path,
    dx: float,
    dy: float,
) -> tuple[Path, Path]:
    if color_mode == "none":
        require_image(wm_pre_path)
        require_image(clean_pre_path)
        return wm_pre_path, clean_pre_path

    paths = []
    for stage, pre_path, ref_path in (
        ("attacked_watermarked", wm_pre_path, watermarked_path),
        ("attacked_clean", clean_pre_path, clean_path),
    ):
        out_path = output_dir / "variant_outputs" / variant_key / stage / f"{run_id}.png"
        if not out_path.is_file():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output = color_contrast_transfer(
                open_rgb(pre_path),
                open_rgb(ref_path),
                mode=color_mode,
                flow_dx_image_px=dx,
                flow_dy_image_px=dy,
                alpha=alpha,
            )
            Image.fromarray(output, mode="RGB").save(out_path)
            diagnostics = color_transfer_diagnostics(
                open_rgb(pre_path),
                open_rgb(ref_path),
                open_rgb(out_path),
                mode=color_mode,
                flow_dx_image_px=dx,
                flow_dy_image_px=dy,
                alpha=alpha,
            )
            write_json(out_path.with_suffix(".diagnostics.json"), diagnostics)
        require_image(out_path)
        paths.append(out_path)
    return paths[0], paths[1]


def variant_attack_config(source: dict, color_transfer_mode: str) -> dict:
    config = {
        "seed": int(source["seed"]),
        "flow_dx_image_px": float(source["flow_dx_image_px"]),
        "flow_dy_image_px": float(source["flow_dy_image_px"]),
        "exact_ddim_timestep": int(source["exact_ddim_timestep"]),
        "steps": int(source["steps"]),
        "strength": float(source["strength"]),
        "guidance_scale": float(source["guidance_scale"]),
        "inversion_mode": source["inversion_mode"],
        "inversion_prompt": source["inversion_prompt"],
        "reconstruction_prompt": source["reconstruction_prompt"],
        "warp_mode": source["warp_mode"],
        "sampling_mode": source["sampling_mode"],
        "padding_mode": source["padding_mode"],
        "normalization_formula": source["normalization_formula"],
        "color_transfer_mode": color_transfer_mode,
        "model_id": source["model_id"],
        "model_revision": source["model_revision"],
        "pairing_sha256": source["pairing_sha256"],
    }
    config["attack_config_sha256"] = build_attack_config_sha256(config)
    return config


class OfficialTreeRingL1Detector:
    """Official Tree-Ring L1 mask/key with this repo's compatible DDIM inversion."""

    def __init__(self, repo: Path, model_id: str, model_revision: str | None, device: str, dtype: str):
        repo = repo.resolve()
        if not (repo / "optim_utils.py").is_file():
            raise FileNotFoundError(f"Tree-Ring repo not found or incomplete: {repo}")
        sys.path.insert(0, str(repo))
        import torch
        from optim_utils import get_watermarking_mask, get_watermarking_pattern, transform_img

        self.torch = torch
        self.transform_img = transform_img
        self.device = device
        self.raven = RavenPipeline(model_id=model_id, revision=model_revision, device=device, dtype=dtype)
        self.raven.pipe.set_progress_bar_config(disable=True)
        self.text_embeddings = self.raven._encode_prompt(
            prompt="", negative_prompt="", guidance_scale=1.0, num_images_per_prompt=1
        )
        self.args = argparse.Namespace(
            w_seed=999999,
            w_channel=3,
            w_pattern="ring",
            w_mask_shape="circle",
            w_radius=10,
            w_measurement="l1_complex",
            w_injection="complex",
            w_pattern_const=0.0,
        )
        with torch.no_grad():
            shape = (1, int(self.raven.pipe.unet.config.in_channels), 64, 64)
            dummy = torch.zeros(shape, device=device, dtype=self.raven.dtype)
            self.watermarking_mask = get_watermarking_mask(dummy, self.args, device)
            self.gt_patch = get_watermarking_pattern(None, self.args, device, shape=shape)

    def _forward_ddim_step(self, x_t, alpha_t, alpha_tp1, eps_xt):
        return (
            alpha_tp1**0.5
            * (
                (alpha_t**-0.5 - alpha_tp1**-0.5) * x_t
                + ((1 / alpha_tp1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
        )

    def _invert_image_to_noise(self, image: Image.Image):
        torch = self.torch
        scheduler = self.raven.pipe.scheduler
        with torch.no_grad():
            image_tensor = self.transform_img(image).unsqueeze(0).to(self.raven.dtype).to(self.device)
            scaling_factor = float(getattr(self.raven.pipe.vae.config, "scaling_factor", 0.18215))
            latents = self.raven.pipe.vae.encode(image_tensor).latent_dist.mode() * scaling_factor
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(50)
            timesteps = scheduler.timesteps.to(self.device)
            for timestep in reversed(timesteps):
                latent_model_input = scheduler.scale_model_input(latents, timestep)
                noise_pred = self.raven.pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=self.text_embeddings,
                    return_dict=False,
                )[0]
                prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
                alpha_prod_t = scheduler.alphas_cumprod[timestep]
                alpha_prod_t_prev = (
                    scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else scheduler.final_alpha_cumprod
                )
                # Official forward_diffusion sets reverse_process=True, which swaps alpha_t and alpha_{t-1}.
                latents = self._forward_ddim_step(latents, alpha_prod_t_prev, alpha_prod_t, noise_pred)
        del image_tensor
        return latents

    def score(self, path: Path) -> dict:
        torch = self.torch
        from PIL import ImageOps

        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        with torch.no_grad():
            reversed_latents = self._invert_image_to_noise(image)
            recovered_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
            decoded = recovered_fft[self.watermarking_mask].flatten()
            target = self.gt_patch[self.watermarking_mask].flatten()
            score = float(torch.abs(decoded - target).mean().detach().cpu().item())
            decoded_abs_mean = float(torch.abs(decoded).mean().detach().cpu().item())
            target_abs_mean = float(torch.abs(target).mean().detach().cpu().item())
        del reversed_latents, recovered_fft, decoded, target
        if not math.isfinite(score):
            raise ValueError(f"non-finite official Tree-Ring L1 score: {path}: {score}")
        return {
            "score": score,
            "decoded_abs_mean": decoded_abs_mean,
            "target_abs_mean": target_abs_mean,
            "nan": math.isnan(score),
            "inf": math.isinf(score),
        }


def score_l1_detector(detector: OfficialTreeRingL1Detector, path: Path) -> dict:
    return detector.score(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=FORMAL_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tree-ring-repo", type=Path, default=Path("external/tree-ring-watermark"))
    parser.add_argument("--clip-model", default="ViT-bigG-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--quality-device", default="cuda")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant["key"] for variant in VARIANT_CATALOG],
        default=list(DEFAULT_VARIANT_KEYS),
    )
    args = parser.parse_args(argv)
    variants = resolve_variants(args.variants)

    limit_cpu_threads(1)
    guard = CpuMemoryGuard(min_available_gib=24.0, max_process_rss_gib=48.0, warn_available_gib=40.0)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] {utc_now()} output_dir={output_dir}", flush=True)
    guard.check("startup")

    if args.count <= 0 or args.count > 1001:
        raise ValueError("--count must be in 1..1001")
    samples, manifest, wm_records, clean_records, pairing_audit = source_records(args.source_root, args.count)
    first_row = manifest[str(samples[0]["run_id"])]
    provenance = {
        "created_at_utc": utc_now(),
        "git_sha_head": git_sha("HEAD"),
        "git_sha_origin_main": git_sha("origin/main"),
        "git_dirty": git_dirty(),
        "source_root": str(args.source_root.resolve()),
        "count_requested": args.count,
        "count_actual": len(samples),
        "formal_config": FORMAL_CONFIG,
        "pairing_audit": pairing_audit,
        "generation_watermark_target_sha256": pairing_audit["watermark_target_sha256"],
        "variants": variants,
        "quality_reference_rule": "attack output vs corresponding original watermarked input",
        "threshold_rule": "each variant calibrates an L1 threshold from its own attacked-clean scores; detection uses L1 < threshold",
        "tree_ring_detector_backend": "official YuxinWenRick/tree-ring-watermark optim_utils L1 with local DDIM inversion",
        "tree_ring_repo": str(args.tree_ring_repo.resolve()),
        "tree_ring_score_definition": "torch.abs(fft(inverted_latent)[mask] - gt_patch[mask]).mean()",
        "tree_ring_score_direction": "lower L1 means more likely watermarked",
        "clip_metric": "cosine(CLIP_text(source_prompt), CLIP_image(attacked_watermarked))",
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "memory_policy": {
            "no_dataloader": True,
            "cpu_threads": 1,
            "stream_images": True,
            "min_available_gib": guard.min_available_gib,
            "max_process_rss_gib": guard.max_process_rss_gib,
        },
    }
    write_json(output_dir / "provenance.json", provenance)

    variant_rows: dict[str, list[dict]] = {variant["key"]: [] for variant in variants}

    with (output_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as jsonl:
        for index, sample in enumerate(samples, start=1):
            run_id = str(sample["run_id"])
            row = manifest[run_id]
            wm = wm_records[run_id]
            clean = clean_records[run_id]
            watermarked_path = Path(wm["watermarked_path"])
            clean_path = Path(wm["clean_path"])
            require_image(watermarked_path)
            require_image(clean_path)
            dx = float(wm["flow_dx_image_px"])
            dy = float(wm["flow_dy_image_px"])
            seed = int(wm.get("attack_seed") or sample.get("attack_seed") or row.get("attack_seed") or row.get("watermark_seed"))
            wm_pre = Path(wm["debug_info_path"]).parent / "view_guided_output.png"
            clean_pre = Path(clean["debug_info_path"]).parent / "view_guided_output.png"
            require_image(wm_pre)
            require_image(clean_pre)

            sample_outputs = {}
            for variant in variants:
                key = variant["key"]
                color_mode = str(variant["color_transfer"])
                alpha = float(variant["alpha"])
                attacked_wm_path, attacked_clean_path = ensure_color_outputs(
                    output_dir,
                    key,
                    color_mode,
                    alpha,
                    run_id,
                    wm_pre,
                    clean_pre,
                    watermarked_path,
                    clean_path,
                    dx,
                    dy,
                )
                wm_cfg = variant_attack_config(wm, color_mode)
                clean_cfg = variant_attack_config(clean, color_mode)
                assert_attack_pair_config_match(clean_cfg, wm_cfg, run_id)
                sample_outputs[key] = (attacked_wm_path, attacked_clean_path, dx, dy, wm_cfg)

            for variant in variants:
                key = variant["key"]
                attacked_wm_path, attacked_clean_path, q_dx, q_dy, attack_cfg = sample_outputs[key]
                q = quality_pair(watermarked_path, attacked_wm_path, q_dx, q_dy)
                record = {
                    "variant": key,
                    "variant_name": variant["name"],
                    "run_id": run_id,
                    "sample_index": index,
                    "source_prompt": row["source_prompt"],
                    "source_prompt_sha256": row["prompt_sha256"],
                    "pairing_sha256": row["pairing_sha256"],
                    "base_latent_seed": int(row["base_latent_seed"]),
                    "base_latent_sha256": row["base_latent_sha256"],
                    "watermark_target_sha256": row["watermark_target_sha256"],
                    "generation_config_sha256": row["generation_config_sha256"],
                    "watermark_config_sha256": row["watermark_config_sha256"],
                    "attack_config_sha256": attack_cfg["attack_config_sha256"],
                    "watermarked_input_path": str(watermarked_path.resolve()),
                    "clean_input_path": str(clean_path.resolve()),
                    "attacked_watermarked_path": str(attacked_wm_path.resolve()),
                    "attacked_clean_path": str(attacked_clean_path.resolve()),
                    "watermarked_input_sha256": sha256_path(watermarked_path),
                    "attacked_watermarked_sha256": sha256_path(attacked_wm_path),
                    "attacked_clean_sha256": sha256_path(attacked_clean_path),
                    "flow_dx_image_px": float(q_dx),
                    "flow_dy_image_px": float(q_dy),
                    "formal_shift_flow_dx_image_px": dx,
                    "formal_shift_flow_dy_image_px": dy,
                    "seed": seed,
                    "psnr": q["psnr"],
                    "ssim": q["ssim"],
                    "valid_overlap_width": q["valid_overlap_width"],
                    "valid_overlap_height": q["valid_overlap_height"],
                    "valid_overlap_area_ratio": q["valid_overlap_area_ratio"],
                    "clip_cosine": None,
                    "tree_ring_detector_backend": "official YuxinWenRick/tree-ring-watermark optim_utils L1 with local DDIM inversion",
                    "tree_ring_score_direction": "lower L1 means more likely watermarked",
                    "tree_ring_attacked_clean_l1": None,
                    "tree_ring_attacked_clean_decoded_abs_mean": None,
                    "tree_ring_attacked_watermarked_l1": None,
                    "tree_ring_attacked_watermarked_decoded_abs_mean": None,
                    "tree_ring_target_abs_mean": None,
                    "ddim_unet_rerun": False,
                }
                append_jsonl(jsonl, record)
                variant_rows[key].append(record)
            print(f"[prepare {index}/{len(samples)}] run_id={run_id}", flush=True)
            if index == 1 or index % 10 == 0:
                guard.check(f"after prepare {index}")

    guard.check("after image preparation")

    # CLIP image-text cosine: attacked watermarked output against source prompt.
    for variant in variants:
        key = variant["key"]
        pairs = [(str(row["source_prompt"]), Path(row["attacked_watermarked_path"])) for row in variant_rows[key]]
        scores = clip_prompt_cosines(pairs, args.quality_device, args.clip_model, args.clip_pretrained)
        for row, score in zip(variant_rows[key], scores):
            row["clip_cosine"] = score
        print(f"[clip] {key} n={len(scores)} mean={statistics.fmean(scores):.6f}", flush=True)
        guard.check(f"after CLIP {key}")

    # Official Tree-Ring L1 detector scores. Each variant calibrates on its own attacked-clean scores.
    detector = OfficialTreeRingL1Detector(args.tree_ring_repo, p1.MODEL_ID, p1.MODEL_REVISION, args.device, args.dtype)
    detector_target_sha256 = tensor_sha256(detector.gt_patch)
    if detector_target_sha256 != pairing_audit["watermark_target_sha256"]:
        raise ValueError(
            "detector watermark target does not match paired generation provenance: "
            f"detector={detector_target_sha256} source={pairing_audit['watermark_target_sha256']}"
        )
    provenance["detector_watermark_target_sha256"] = detector_target_sha256
    provenance["detector_target_matches_generation"] = True
    write_json(output_dir / "provenance.json", provenance)
    torch = detector.torch
    guard.check("after loading official Tree-Ring L1 detector")
    for variant in variants:
        key = variant["key"]
        for index, row in enumerate(variant_rows[key], start=1):
            clean_score = score_l1_detector(detector, Path(row["attacked_clean_path"]))
            attacked_score = score_l1_detector(detector, Path(row["attacked_watermarked_path"]))
            row.update({
                "tree_ring_attacked_clean_l1": clean_score["score"],
                "tree_ring_attacked_clean_decoded_abs_mean": clean_score["decoded_abs_mean"],
                "tree_ring_attacked_watermarked_l1": attacked_score["score"],
                "tree_ring_attacked_watermarked_decoded_abs_mean": attacked_score["decoded_abs_mean"],
                "tree_ring_target_abs_mean": attacked_score["target_abs_mean"],
            })
            if index == 1 or index % 10 == 0:
                print(f"[tree-ring-l1 {key} {index}/{len(variant_rows[key])}]", flush=True)
                guard.check(f"tree-ring-l1 {key} {index}")
    del detector
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    aggregate: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "variant_order": [variant["key"] for variant in variants],
        "variants": {},
        "formal_config": FORMAL_CONFIG,
        "pairing_audit": pairing_audit,
        "detector_watermark_target_sha256": detector_target_sha256,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "git_sha_head": provenance["git_sha_head"],
        "git_sha_origin_main": provenance["git_sha_origin_main"],
        "git_dirty": provenance["git_dirty"],
    }
    for variant in variants:
        key = variant["key"]
        rows = variant_rows[key]
        fid = cleanfid_from_paths(
            [Path(row["watermarked_input_path"]) for row in rows],
            [Path(row["attacked_watermarked_path"]) for row in rows],
            device=args.quality_device,
        )
        clean_scores = [float(row["tree_ring_attacked_clean_l1"]) for row in rows]
        wm_scores = [float(row["tree_ring_attacked_watermarked_l1"]) for row in rows]
        numeric_values = [
            *clean_scores,
            *wm_scores,
            *[float(row["clip_cosine"]) for row in rows],
            *[float(row["psnr"]) for row in rows],
            *[float(row["ssim"]) for row in rows],
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"NaN/Inf in formal metrics for {key}")
        threshold = nfpa_threshold(clean_scores, target_fpr=0.01)
        actual_fpr = nfpa_rate(clean_scores, threshold)
        tpr = nfpa_rate(wm_scores, threshold)
        aggregate["variants"][key] = {
            "name": variant["name"],
            "n": len(rows),
            "clip": finite_stats([float(row["clip_cosine"]) for row in rows]),
            "fid": fid,
            "psnr": finite_stats([float(row["psnr"]) for row in rows]),
            "ssim": finite_stats([float(row["ssim"]) for row in rows]),
            "tree_ring_detector_backend": "official YuxinWenRick/tree-ring-watermark optim_utils L1 with local DDIM inversion",
            "tree_ring_score_direction": "lower L1 means more likely watermarked; detection uses score < threshold",
            "tree_ring_attacked_clean_l1": finite_stats(clean_scores),
            "tree_ring_attacked_watermarked_l1": finite_stats(wm_scores),
            "threshold": float(threshold),
            "target_fpr": 0.01,
            "actual_fpr": float(actual_fpr),
            "false_positives": int(np.sum(np.asarray(clean_scores, dtype=np.float64) < threshold)),
            "max_false_positives": int(math.floor(len(clean_scores) * 0.01)),
            "tpr_at_1pct_fpr": float(tpr),
            "attack_success_rate": float(1.0 - tpr),
            "roc_auc": float(roc_auc([-value for value in wm_scores], [-value for value in clean_scores])),
            "numeric_audit": {"nan_count": 0, "inf_count": 0, "passed": True},
            "detector_watermark_target_sha256": detector_target_sha256,
            "ddim_unet_rerun": False,
            "complete_settings": {**FORMAL_CONFIG, **variant},
        }
        print(f"[fid/aggregate] {key} fid={fid['value']:.6f}", flush=True)
        guard.check(f"after FID {key}")

    flat_rows: list[dict] = []
    for key in aggregate["variant_order"]:
        flat_rows.extend(variant_rows[key])
    write_csv(output_dir / "per_sample_results.csv", flat_rows)
    with (output_dir / "per_sample_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in flat_rows:
            append_jsonl(handle, row)
    write_json(output_dir / "aggregate_results.json", aggregate)
    (output_dir / "aggregate_results.md").write_text(render_md(aggregate), encoding="utf-8")
    print(render_md(aggregate), flush=True)
    print(f"[done] {utc_now()} output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
