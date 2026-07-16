"""Tree-Ring generation, inversion, scoring, and ROC with official semantics.

The formulas in this module are a compatibility port of commit
3015283d9cf82e90b628f02ad2121bd37408ca9a from the Tree-Ring repository.
They intentionally preserve the original coordinate, FFT, seed, and ROC
semantics while running on the repository's current PyTorch/Diffusers stack.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


OFFICIAL_TREE_RING_COMMIT = "3015283d9cf82e90b628f02ad2121bd37408ca9a"


@dataclass(frozen=True)
class TreeRingSettings:
    model_id: str = "stabilityai/stable-diffusion-2-1-base"
    generation_scheduler: str = "DPMSolverMultistepScheduler"
    generation_guidance_scale: float = 7.5
    generation_steps: int = 50
    image_size: int = 512
    generation_seed: int = 0
    watermark_seed: int = 999999
    watermark_channel: int = 0
    watermark_pattern: str = "rand"
    watermark_mask_shape: str = "circle"
    watermark_radius: int = 10
    watermark_measurement: str = "l1_complex"
    watermark_injection: str = "complex"
    detector_prompt: str = ""
    detector_guidance_scale: float = 1.0
    detector_steps: int = 50


def set_official_random_seed(seed: int) -> None:
    """Match ``optim_utils.set_random_seed`` exactly."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def stable_tensor_hash(tensor: Any, storage_dtype: str = "float32") -> str:
    import torch

    dtype = torch.complex64 if tensor.is_complex() else getattr(torch, storage_dtype)
    value = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    header = f"shape={tuple(value.shape)};dtype={value.dtype};".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def circle_mask(size: int = 64, radius: int = 10) -> np.ndarray:
    x0 = y0 = size // 2
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= radius**2


def make_watermark_mask(latents: Any, channel: int = 0, radius: int = 10):
    import torch

    mask = torch.zeros(latents.shape, dtype=torch.bool, device=latents.device)
    spatial_mask = torch.as_tensor(circle_mask(latents.shape[-1], radius), device=latents.device)
    if channel == -1:
        mask[:, :] = spatial_mask
    else:
        mask[:, channel] = spatial_mask
    return mask


def make_rand_watermark_target(gt_init: Any):
    import torch

    target = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    target[:] = target[0]
    return target


def inject_complex_watermark(latents: Any, mask: Any, target: Any):
    import torch

    latent_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
    latent_fft[mask] = target[mask].clone()
    return torch.fft.ifft2(torch.fft.ifftshift(latent_fft, dim=(-1, -2))).real


def official_complex_l1(recovered_latents: Any, mask: Any, target: Any) -> float:
    import torch

    recovered_fft = torch.fft.fftshift(torch.fft.fft2(recovered_latents), dim=(-1, -2))
    score = torch.abs(recovered_fft[mask] - target[mask]).mean().item()
    if not math.isfinite(score):
        raise ValueError(f"non-finite Tree-Ring official complex-L1 score: {score}")
    return float(score)


def backward_ddim(x_t: Any, alpha_t: Any, alpha_tm1: Any, eps_xt: Any):
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def get_empty_text_embedding(pipe: Any):
    text_input_ids = pipe.tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    return pipe.text_encoder(text_input_ids.to(pipe.device))[0]


def image_to_official_latents(pipe: Any, image: Any):
    from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

    transform = Compose([Resize(512), CenterCrop(512), ToTensor()])
    image_tensor = (2.0 * transform(image.convert("RGB")) - 1.0).unsqueeze(0)
    image_tensor = image_tensor.to(device=pipe.device, dtype=pipe.text_encoder.dtype)
    posterior = pipe.vae.encode(image_tensor).latent_dist
    scaling_factor = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    return posterior.mode() * scaling_factor


def official_forward_diffusion(
    pipe: Any,
    latents: Any,
    text_embeddings: Any,
    num_inference_steps: int = 50,
) -> tuple[Any, dict[str, Any]]:
    """Port the official hand-written forward DDIM loop without substitutions."""
    import torch

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    descending = scheduler.timesteps.to(pipe.device)
    current = latents * scheduler.init_noise_sigma
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    inverse_timesteps: list[int] = []
    recovered_trace: list[dict[str, float | int]] = []

    with torch.inference_mode():
        for timestep in reversed(descending):
            t = int(timestep.item())
            inverse_timesteps.append(t)
            model_input = scheduler.scale_model_input(current, timestep)
            noise_pred = pipe.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
            previous_timestep = t - step_size
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_previous = (
                scheduler.alphas_cumprod[previous_timestep]
                if previous_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            current = backward_ddim(current, alpha_previous, alpha_t, noise_pred)
            recovered_trace.append({
                "timestep": t,
                "previous_timestep": previous_timestep,
                "latent_mean": float(current.float().mean().detach().cpu().item()),
                "latent_std": float(current.float().std().detach().cpu().item()),
            })

    return current, {
        "scheduler": scheduler.__class__.__name__,
        "scheduler_timesteps_descending": [int(t) for t in descending.detach().cpu().tolist()],
        "official_inverse_timesteps": inverse_timesteps,
        "step_size": int(step_size),
        "final_recovered_latent_mean": float(current.float().mean().detach().cpu().item()),
        "final_recovered_latent_std": float(current.float().std().detach().cpu().item()),
        "trace": recovered_trace,
    }


def score_image(pipe: Any, image: Any, mask: Any, target: Any, steps: int = 50) -> tuple[float, dict[str, Any]]:
    import torch

    latents = image_to_official_latents(pipe, image)
    text_embeddings = get_empty_text_embedding(pipe)
    recovered, diagnostics = official_forward_diffusion(pipe, latents, text_embeddings, steps)
    recovered_fft = torch.fft.fftshift(torch.fft.fft2(recovered), dim=(-1, -2))
    diagnostics.update({
        "vae_posterior": "mode",
        "vae_sample": False,
        "vae_scaling_factor": float(getattr(pipe.vae.config, "scaling_factor", 0.18215)),
        "detector_prompt": "",
        "detector_guidance_scale": 1.0,
        "masked_fft_mean_magnitude": float(recovered_fft[mask].abs().float().mean().detach().cpu().item()),
    })
    return official_complex_l1(recovered, mask, target), diagnostics


def fail_on_nonfinite(values: Iterable[float], names: Iterable[str] | None = None) -> None:
    labels = list(names) if names is not None else []
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            label = labels[index] if index < len(labels) else str(index)
            raise ValueError(f"non-finite Tree-Ring score for {label}: {value}")


def official_roc(clean_l1: Iterable[float], watermarked_l1: Iterable[float], target_fpr: float = 0.01) -> dict[str, Any]:
    """Use the official negative-distance ROC and strict ``fpr < 0.01`` lookup."""
    from sklearn import metrics

    clean = np.asarray(list(clean_l1), dtype=np.float64)
    watermarked = np.asarray(list(watermarked_l1), dtype=np.float64)
    if clean.size == 0 or watermarked.size == 0:
        raise ValueError("official ROC requires non-empty clean and watermarked scores")
    fail_on_nonfinite(clean, (f"clean[{i}]" for i in range(clean.size)))
    fail_on_nonfinite(watermarked, (f"watermarked[{i}]" for i in range(watermarked.size)))
    predictions = np.concatenate([-clean, -watermarked])
    labels = np.concatenate([
        np.zeros(clean.size, dtype=np.int64),
        np.ones(watermarked.size, dtype=np.int64),
    ])
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    eligible = np.where(fpr < target_fpr)[0]
    if eligible.size == 0:
        raise RuntimeError(f"official ROC has no point with FPR < {target_fpr}")
    index = int(eligible[-1])
    return {
        "target_fpr": float(target_fpr),
        "actual_fpr": float(fpr[index]),
        "tpr_at_1pct_fpr": float(tpr[index]),
        "decision_threshold_negative_l1": float(thresholds[index]),
        "complex_l1_threshold_equivalent": float(-thresholds[index]),
        "auc": float(metrics.auc(fpr, tpr)),
        "roc_index": index,
        "false_positives": int(round(float(fpr[index]) * clean.size)),
        "n_clean": int(clean.size),
        "n_watermarked": int(watermarked.size),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds_negative_l1": thresholds.tolist(),
        "selection_rule": "tpr[np.where(fpr < 0.01)[0][-1]]",
    }


def rate_at_negative_l1_threshold(l1_scores: Iterable[float], threshold: float) -> float:
    values = np.asarray(list(l1_scores), dtype=np.float64)
    fail_on_nonfinite(values)
    return float(np.mean(-values >= float(threshold)))
