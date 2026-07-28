"""Single authoritative DDIM update used by the provider-local inversion paths.

``backward_ddim`` is the update from
https://github.com/cccntu/efficient-prompt-to-prompt used verbatim by
Tree-Ring, MaXsive, ShallowDiffuse and GaussMarker. It previously existed as
byte-identical copies in ``maxsive_provider.py`` and ``shallow_provider.py``;
both now import it from here so there is exactly one implementation.
"""

from __future__ import annotations


def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """One DDIM step from noise to image (reverse direction: swap the alphas)."""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """One DDIM step from image to noise; identical formula to ``backward_ddim``."""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


def official_forward_diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    guidance_scale: float,
    num_inference_steps: int,
    device,
    callback=None,
):
    """Exact transcription of the official GaussMarker inversion loop.

    Reference: ``InversableStableDiffusionPipeline.backward_diffusion(
    reverse_process=True)`` in ``inverse_stable_diffusion.py`` of
    https://github.com/SunnierLee/GaussMarker at commit
    ``4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155``.

    Notes on why this is *not* interchangeable with
    ``DPMSolverMultistepInverseScheduler``: the official code keeps the DPM
    scheduler only for its timestep grid, ``init_noise_sigma``,
    ``scale_model_input`` and ``alphas_cumprod``; the state update itself is the
    plain DDIM equation above and ``scheduler.step`` is never called. Using the
    generic inverse scheduler changes the update rule.
    """
    do_classifier_free_guidance = guidance_scale > 1.0
    scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = scheduler.timesteps.to(device)
    latents = latents * scheduler.init_noise_sigma

    import torch  # local import keeps this module importable without torch at parse time

    for i, t in enumerate(reversed(timesteps_tensor)):
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        prev_timestep = (
            t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
        if callback is not None:
            callback(i, t, latents)

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        # reverse_process=True in the official implementation
        alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        latents = backward_ddim(
            x_t=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )
    return latents
