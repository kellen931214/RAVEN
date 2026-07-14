#!/usr/bin/env python
"""Audit a partial DDIM inversion/denoising round trip on one real image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.inversion import partial_diffusion_inversion
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image, save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    return parser


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    image = load_image(args.input, size=args.size)
    save_image(image, args.output_dir / "input.png")

    pipe = RavenPipeline(
        model_id=args.model_id,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
    )
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    generator = pipe._make_generator(0)
    prompt_embeds = pipe._encode_prompt("", "", args.guidance_scale, 1)
    inversion = partial_diffusion_inversion(
        vae=pipe.pipe.vae,
        scheduler=pipe.pipe.scheduler,
        image=image,
        num_inference_steps=args.steps,
        strength=args.strength,
        generator=generator,
        device=args.device,
        dtype=pipe.dtype,
        mode="ddim",
        unet=pipe.pipe.unet,
        prompt_embeds=prompt_embeds,
        guidance_scale=args.guidance_scale,
    )

    latents = inversion.noisy_latents
    extra_step_kwargs = pipe._prepare_extra_step_kwargs(generator)
    with torch.no_grad():
        for timestep in inversion.timesteps:
            model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
            model_input = pipe.pipe.scheduler.scale_model_input(model_input, timestep)
            noise_pred = pipe.pipe.unet(
                model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if args.guidance_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
            latents = pipe.pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]

    reconstruction, clipping = pipe._decode_latents_with_diagnostics(latents)
    save_image(reconstruction, args.output_dir / "reconstruction.png")
    original_array, reconstruction_array = image_array(image), image_array(reconstruction)
    result = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "image_shape_hwc": list(original_array.shape),
        "latent_shape": list(inversion.clean_latents.shape),
        "denoise_scheduler": inversion.denoise_scheduler,
        "inverse_scheduler": inversion.inverse_scheduler,
        "prediction_type": inversion.prediction_type,
        "eta": inversion.eta,
        "exact_timestep": inversion.target_timestep,
        "inverse_timestep_sequence": [int(t) for t in inversion.inversion_timesteps.cpu().tolist()],
        "denoising_timestep_sequence": [int(t) for t in inversion.timesteps.cpu().tolist()],
        "latent_roundtrip_mae": float((latents - inversion.clean_latents).abs().float().mean().cpu().item()),
        "reconstruction_psnr": float(peak_signal_noise_ratio(original_array, reconstruction_array, data_range=1.0)),
        "reconstruction_ssim": float(
            structural_similarity(original_array, reconstruction_array, channel_axis=2, data_range=1.0)
        ),
        "lpips": None,
        "lpips_status": "skipped: LPIPS package exists but pretrained AlexNet trunk is not cached; no download allowed",
        "clipping_diagnostics": clipping,
        "input_path": str((args.output_dir / "input.png").resolve()),
        "reconstruction_path": str((args.output_dir / "reconstruction.png").resolve()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
    }
    (args.output_dir / "roundtrip.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
