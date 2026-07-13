"""Partial diffusion inversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

from .utils import image_to_tensor


@dataclass
class PartialInversionResult:
    clean_latents: Any
    noisy_latents: Any
    noise: Any
    timesteps: Any
    start_timestep: Any
    mode: str
    inversion_timesteps: Any = None


def retrieve_img2img_timesteps(scheduler, num_inference_steps: int, strength: float, device) -> Tuple[Any, int]:
    """Match Diffusers img2img timestep slicing for partial noising."""
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    scheduler.set_timesteps(num_inference_steps, device=device)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    return timesteps, num_inference_steps - t_start


def encode_image_to_latents(vae, image, device, dtype):
    tensor = image_to_tensor(image, device=device, dtype=dtype)
    posterior = vae.encode(tensor).latent_dist
    latents = posterior.mode()
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    return latents * scaling_factor


def partial_diffusion_inversion(
    vae,
    scheduler,
    image,
    num_inference_steps: int,
    strength: float,
    generator,
    device,
    dtype,
    mode: Literal["ddim", "forward_noise"] = "ddim",
    unet=None,
    prompt_embeds=None,
    guidance_scale: float = 2.5,
) -> PartialInversionResult:
    """Partially invert an image with DDIM or Equation-(4) forward noising.

    ``ddim`` follows the paper's Implementation Details. ``forward_noise``
    preserves the stochastic formulation in Equation (4) as an explicit
    reproduction ablation.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("partial_diffusion_inversion requires torch") from exc

    if mode not in {"ddim", "forward_noise"}:
        raise ValueError(f"Unsupported inversion mode: {mode}")
    clean_latents = encode_image_to_latents(vae, image, device=device, dtype=dtype)
    timesteps, _ = retrieve_img2img_timesteps(scheduler, num_inference_steps, strength, device)
    if len(timesteps) == 0:
        raise ValueError("No timesteps selected; increase strength or num_inference_steps")
    start_timestep = timesteps[:1].repeat(clean_latents.shape[0])
    inversion_timesteps = None
    if mode == "forward_noise":
        noise = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=dtype)
        noisy_latents = scheduler.add_noise(clean_latents, noise, start_timestep)
    else:
        if unet is None or prompt_embeds is None:
            raise ValueError("DDIM inversion requires unet and empty-prompt embeddings")
        try:
            from diffusers import DDIMInverseScheduler
        except ImportError as exc:
            raise ImportError("DDIM inversion requires diffusers.DDIMInverseScheduler") from exc

        inverse_scheduler = DDIMInverseScheduler.from_config(scheduler.config)
        inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        target_timestep = int(start_timestep[0].detach().cpu().item())
        selected = [t for t in inverse_scheduler.timesteps if int(t) < target_timestep]
        latents = clean_latents
        do_cfg = guidance_scale > 1.0
        with torch.no_grad():
            for timestep in selected:
                model_input = torch.cat([latents] * 2) if do_cfg else latents
                model_input = inverse_scheduler.scale_model_input(model_input, timestep)
                noise_pred = unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                latents = inverse_scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        noisy_latents = latents
        noise = None
        inversion_timesteps = inverse_scheduler.timesteps.new_tensor(
            [int(t) for t in selected]
        )
    return PartialInversionResult(
        clean_latents=clean_latents,
        noisy_latents=noisy_latents,
        noise=noise,
        timesteps=timesteps,
        start_timestep=start_timestep,
        mode=mode,
        inversion_timesteps=inversion_timesteps,
    )
