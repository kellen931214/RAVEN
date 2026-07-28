"""Inversion protocols for T2S.

The official T2S inversion and the benchmark's generic DDIM inversion are NOT
equivalent, so they are exposed as explicit modes and are never silently
substituted for one another.

``benchmark_ddim``
    ``PipeProvider.invert_images`` -> ``invert_z0``, i.e. the diffusers
    ``DDIMInverseScheduler`` driven through the full pipeline ``__call__``, with
    ``num_inference_steps`` normally matching the generation step count.

``t2s_official``
    Upstream ``InversableStableDiffusionPipeline.naive_forward_diffusion``
    (``backward_diffusion(reverse_process=True)``) for UNet models, and
    ``InversionDiffusion3Pipeline.naive_forward_diffusion`` for SD3/SD3.5. These
    walk ``reversed(scheduler.timesteps)`` with a null prompt at
    ``guidance_scale=1.0`` and default to 10 steps (upstream ``option.py``
    ``--num_inversion_steps``), which is far fewer than the generation steps.

Both modes reuse the existing pipe/scheduler primitives owned by the
``PipeProvider``; this module only adds the small adapter needed to express
upstream's update rule.
"""

from __future__ import annotations

import typing

import PIL.Image
import torch


def _backward_ddim(x_t: torch.Tensor,
                   alpha_t: torch.Tensor,
                   alpha_tm1: torch.Tensor,
                   eps_xt: torch.Tensor) -> torch.Tensor:
    """Upstream ``inverse_stable_diffusion.backward_ddim``, verbatim algebra."""
    return (
        alpha_tm1 ** 0.5
        * (
            (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


@torch.no_grad()
def official_unet_inversion(pipe,
                            z0: torch.Tensor,
                            num_inversion_steps: int = 10,
                            guidance_scale: float = 1.0) -> torch.Tensor:
    """Reproduce upstream ``naive_forward_diffusion`` for UNet pipelines (SD2.1)."""
    device = z0.device
    scheduler = pipe.scheduler

    null_embeddings = pipe.encode_prompt(
        "", device, 1, guidance_scale > 1.0, None
    )[0].to(dtype=z0.dtype)

    scheduler.set_timesteps(num_inversion_steps)
    timesteps = scheduler.timesteps.to(device)

    latents = z0 * scheduler.init_noise_sigma
    num_train_timesteps = scheduler.config.num_train_timesteps

    for t in reversed(timesteps):
        noise_pred = pipe.unet(
            scheduler.scale_model_input(latents, t),
            t,
            encoder_hidden_states=null_embeddings,
        ).sample

        prev_timestep = t - num_train_timesteps // num_inversion_steps

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        # Upstream swaps the two alphas to turn the DDIM step into inversion.
        alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        latents = _backward_ddim(latents, alpha_prod_t, alpha_prod_t_prev, noise_pred)

    return latents


@torch.no_grad()
def official_sd3_inversion(pipe,
                           z0: torch.Tensor,
                           num_inversion_steps: int = 10,
                           guidance_scale: float = 1.0) -> torch.Tensor:
    """Reproduce upstream ``InversionDiffusion3Pipeline.naive_forward_diffusion``."""
    device = z0.device
    scheduler = pipe.scheduler
    do_cfg = guidance_scale > 1.0

    prompt_embeds, _, pooled_projections, _ = pipe.encode_prompt(
        prompt="", prompt_2=None, prompt_3=None,
        device=device, do_classifier_free_guidance=do_cfg,
    )

    scheduler.set_timesteps(num_inversion_steps, device=device)
    timesteps = scheduler.timesteps

    latents = z0
    for i, t in enumerate(reversed(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = t.expand(latent_model_input.shape[0])
        index = num_inversion_steps - 1 - i

        noise_pred = pipe.transformer(
            latent_model_input,
            timestep=timestep,
            pooled_projections=pooled_projections,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = latents - (scheduler.sigmas[index + 1] - scheduler.sigmas[index]) * noise_pred

    return latents


def _is_sd3_like(pipe) -> bool:
    return hasattr(pipe, "transformer") and not hasattr(pipe, "unet")


@torch.no_grad()
def invert_image(pipe_provider,
                 image: typing.Union[PIL.Image.Image, torch.Tensor],
                 inversion_mode: str,
                 num_inversion_steps: int,
                 benchmark_num_inference_steps: typing.Optional[int] = None) -> torch.Tensor:
    """Invert one suspect image to zT under the requested protocol.

    Returns a latent with a batch dimension on the provider's device.
    """
    if inversion_mode == "benchmark_ddim":
        steps = benchmark_num_inference_steps or num_inversion_steps
        return pipe_provider.invert_images(image, num_inference_steps=steps)["zT_torch"]

    if inversion_mode != "t2s_official":
        raise ValueError(f"unknown T2S inversion mode: {inversion_mode!r}")

    # Reuse the provider's own VAE encoding (posterior mean * scaling_factor),
    # which matches upstream ``get_image_latents(..., sample=False)``.
    z0 = pipe_provider.imgs_to_latents(image)
    pipe = pipe_provider.pipe
    if pipe is None:
        raise RuntimeError("pipe_provider has no loaded pipe; generate or load one first")

    if _is_sd3_like(pipe):
        return official_sd3_inversion(pipe, z0, num_inversion_steps=num_inversion_steps)
    return official_unet_inversion(pipe, z0, num_inversion_steps=num_inversion_steps)
