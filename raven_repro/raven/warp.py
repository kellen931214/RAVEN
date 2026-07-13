"""Latent-space viewpoint modulation utilities."""

from __future__ import annotations

import random
from typing import Literal, Tuple


ShiftSign = Literal["positive", "negative", "random"]
ShiftSampling = Literal["independent_axes", "coupled_diagonal"]
ShiftSpace = Literal["image_pixels", "latent_pixels"]
PaddingMode = Literal["reflection", "border", "zeros"]


def sample_translation(
    shift_min: int,
    shift_max: int,
    shift_sign: ShiftSign = "random",
    seed: int | None = None,
    sampling: ShiftSampling = "independent_axes",
) -> Tuple[int, int]:
    """Sample per-axis shifts from the paper's positive/negative intervals.

    ``coupled_diagonal`` preserves the original reproduction behavior for
    ablations. The paper-facing default samples each axis independently.
    """
    if shift_min < 0 or shift_max < 0:
        raise ValueError("shift_min and shift_max must be non-negative")
    if shift_min > shift_max:
        raise ValueError(f"shift_min must be <= shift_max, got {shift_min} > {shift_max}")
    if shift_sign not in {"positive", "negative", "random"}:
        raise ValueError(f"Unsupported shift_sign: {shift_sign}")
    if sampling not in {"independent_axes", "coupled_diagonal"}:
        raise ValueError(f"Unsupported shift sampling: {sampling}")

    rng = random.Random(seed)
    def sample_axis() -> int:
        magnitude = rng.randint(shift_min, shift_max)
        sign = rng.choice([-1, 1]) if shift_sign == "random" else (1 if shift_sign == "positive" else -1)
        return sign * magnitude

    dx = sample_axis()
    return (dx, dx) if sampling == "coupled_diagonal" else (dx, sample_axis())


def translate_latent(
    latents,
    dx: float,
    dy: float,
    shift_space: ShiftSpace = "image_pixels",
    vae_scale_factor: int = 8,
    padding_mode: PaddingMode = "reflection",
):
    """Translate BCHW latents with torch.nn.functional.grid_sample.

    Positive dx samples from the right and positive dy samples from lower rows,
    matching the flow convention used by NFPA's latent warping helper.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("translate_latent requires torch") from exc

    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape (B, C, H, W), got {tuple(latents.shape)}")
    if shift_space not in {"image_pixels", "latent_pixels"}:
        raise ValueError(f"Unsupported shift_space: {shift_space}")
    if padding_mode not in {"reflection", "border", "zeros"}:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")

    shift_x = float(dx) / vae_scale_factor if shift_space == "image_pixels" else float(dx)
    shift_y = float(dy) / vae_scale_factor if shift_space == "image_pixels" else float(dy)

    batch, _, height, width = latents.shape
    ys = torch.linspace(-1.0, 1.0, height, device=latents.device, dtype=latents.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=latents.device, dtype=latents.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

    norm_x = 0.0 if width <= 1 else 2.0 * shift_x / (width - 1)
    norm_y = 0.0 if height <= 1 else 2.0 * shift_y / (height - 1)
    grid[..., 0] = grid[..., 0] + norm_x
    grid[..., 1] = grid[..., 1] + norm_y

    return F.grid_sample(
        latents,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
