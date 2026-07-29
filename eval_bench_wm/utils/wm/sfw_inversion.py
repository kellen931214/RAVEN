"""SFWMark (HSQR / HSTR) detection front-end: preprocessing, VAE and DDIM inversion.

Single authoritative implementation of the official SFWMark ``src/detect.py``
inversion protocol, shared by every SFWMark method so HSQR (Issue #5) and HSTR
(Issue #4) cannot drift apart:

1. ``Resize((512, 512))`` — a *fixed* target size, not the aspect-preserving
   ``Resize(512)`` + ``CenterCrop`` used by the Tree-Ring/GaussMarker family;
2. ``ToTensor()`` then ``2 * x - 1`` into ``[-1, 1]``;
3. VAE posterior **mode** (deterministic — no sampling, hence no RNG state and
   no per-image seed), scaled by the VAE scaling factor;
4. ``DDIMInverseScheduler.from_config(<current scheduler>.config)``;
5. empty inversion prompt, guidance scale 0 (no classifier-free guidance),
   50 inversion steps.

Official reference:
    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb (``src/detect.py``)

This is deliberately not the generic ``PipeProvider.invert_images`` path, which
resizes differently (``PIL_to_torch`` keeps the source resolution), calls the
full diffusers pipeline with ``guidance_scale=1.0`` and re-enters the
pipeline's own preprocessing.

Parity status
-------------
Verified **bitwise** against the frozen official implementation. The official
``transform_img`` / ``pil2latent`` / ``ddim_invert`` were executed from the
hash-pinned official ``src/utils.py`` and compared against this module on the
same loaded pipeline; the preprocessed input tensor, VAE latent, all 50 inverse
scheduler timesteps, every sampled intermediate latent, the final recovered
latent, the HSQR L1 distance and the canonical score all agreed exactly
(``max_abs_diff == 0``). Evidence:
``eval_bench_wm/tests/fixtures/hsqr_inversion_parity_evidence.json``; regenerate
with ``eval_bench_wm/tools/hsqr_inversion_parity.py``.

What that does **not** establish is numerical reproduction of the published
SFWMark numbers: that needs the official ``stabilityai/stable-diffusion-2-1-base``
weights, and the whole ``stabilityai`` SD-2 family is currently delisted from the
Hugging Face Hub (HTTP 404 with a valid token). The parity run therefore used
mirror weights, which is recorded in the evidence file via ``official_model_used``
and in :data:`SFW_INVERSION_WEIGHTS_PARITY`. The two claims are kept separate on
purpose: the *code* is proven equivalent, the *weights* are not proven identical.
"""

from __future__ import annotations

import typing

import torch


#: Recorded in every inversion/detection record. Upgraded from
#: ``documented_protocol_transcription_not_fixture_verified`` once the
#: element-wise comparison against the frozen official code came back bitwise
#: identical on every compared artifact.
SFW_INVERSION_PARITY_STATUS = "official_code_parity_verified_bitwise"

#: The separate, still-unproven half of the claim: the official weights are
#: delisted from the Hub, so the parity run used mirror weights.
SFW_INVERSION_WEIGHTS_PARITY = "official_weights_unavailable_not_verified"

#: Evidence backing :data:`SFW_INVERSION_PARITY_STATUS`, relative to the repo root.
SFW_INVERSION_PARITY_EVIDENCE = (
    "eval_bench_wm/tests/fixtures/hsqr_inversion_parity_evidence.json"
)

SFW_INVERSION_IMPL_VERSION = "sfw_inversion_v1"


def transform_img(image, target_size: int = 512) -> torch.Tensor:
    """Official ``detect.py`` preprocessing: fixed resize, to tensor, to [-1, 1]."""
    from torchvision import transforms

    tform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ]
    )
    return 2.0 * tform(image) - 1.0


@torch.no_grad()
def encode_image_latents(pipe, image_tensor: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """Official VAE front-end: posterior **mode** (deterministic), then scaling."""
    posterior = pipe.vae.encode(image_tensor).latent_dist
    return posterior.mode() * scaling_factor


def build_inverse_scheduler(scheduler):
    """``DDIMInverseScheduler.from_config(<current scheduler>.config)``.

    The repository's vendored ``DDIMInverseScheduler`` is used so the inversion
    update rule is the one the rest of ``eval_bench_wm`` is validated against.
    """
    from utils.pipe.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

    return DDIMInverseScheduler.from_config(scheduler.config)


@torch.no_grad()
def text_embedding(pipe, prompt: str, device) -> torch.Tensor:
    input_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    return pipe.text_encoder(input_ids.to(device))[0]


@torch.no_grad()
def ddim_inverse_loop(
    unet,
    inverse_scheduler,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    device,
) -> torch.Tensor:
    """Official DDIM inversion loop driven by ``DDIMInverseScheduler``.

    Unlike the GaussMarker path (``utils/wm/ddim_inversion.official_forward_diffusion``),
    which keeps the scheduler only for its timestep grid and applies the plain
    DDIM equation itself, SFWMark ``detect.py`` calls the inverse scheduler's own
    ``step``. The two are *not* interchangeable, which is exactly why this lives
    in its own module instead of reusing the GaussMarker loop.
    """
    do_cfg = guidance_scale > 1.0
    inverse_scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents.to(device)

    embeddings = torch.cat([text_embeddings] * 2) if do_cfg else text_embeddings

    for timestep in inverse_scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = unet(
            latent_model_input, timestep, encoder_hidden_states=embeddings
        ).sample
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = inverse_scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents


@torch.no_grad()
def invert_pil_image(
    image,
    pipe_provider_target,
    resolution: int = 512,
    inversion_prompt: str = "",
    guidance_scale: float = 0.0,
    num_inference_steps: int = 50,
    vae_scaling_factor: typing.Optional[float] = None,
) -> typing.Dict[str, typing.Any]:
    """Run the complete official SFWMark detection front-end on one PIL image."""
    pipe = getattr(pipe_provider_target, "pipe", None)
    if pipe is None:
        # ``PipeProvider`` loads lazily; a scheduler access is enough to force it.
        pipe_provider_target.get_latent_shape()
        pipe = getattr(pipe_provider_target, "pipe", None)
    if pipe is None:
        raise RuntimeError("pipe provider has no loaded pipeline")

    device = pipe_provider_target.device
    if vae_scaling_factor is None:
        vae_scaling_factor = float(pipe.vae.config.scaling_factor)

    embeddings = text_embedding(pipe, inversion_prompt, device)

    image_tensor = transform_img(image, target_size=resolution)
    image_tensor = image_tensor.unsqueeze(0).to(embeddings.dtype).to(device)

    z0 = encode_image_latents(pipe, image_tensor, vae_scaling_factor)
    zT = ddim_inverse_loop(
        unet=pipe.unet,
        inverse_scheduler=build_inverse_scheduler(pipe_provider_target.scheduler),
        latents=z0,
        text_embeddings=embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    return {
        "z0_torch": z0,
        "zT_torch": zT,
        "inversion_steps": int(num_inference_steps),
        "inversion_impl_version": SFW_INVERSION_IMPL_VERSION,
        "inversion_parity_status": SFW_INVERSION_PARITY_STATUS,
        "inversion_weights_parity": SFW_INVERSION_WEIGHTS_PARITY,
    }
