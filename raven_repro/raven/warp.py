"""Latent-space viewpoint modulation utilities."""

from __future__ import annotations

import random
from typing import Any, Literal, Tuple


ShiftSign = Literal["positive", "negative", "random"]
ShiftSampling = Literal["independent_axes", "coupled_diagonal"]
ShiftSpace = Literal["image_pixels", "latent_pixels"]
PaddingMode = Literal["reflection", "border", "zeros"]
LatentSamplingMode = Literal["nearest", "bilinear"]
WarpMode = Literal[
    "integer", "grid_sample", "nfpa_exact", "nfpa_pixel_center",
    "latent_grid_nearest_reflection", "latent_grid",
    "raven_paper_nfpa_gap_fill",
]

RAVEN_PAPER_NFPA_GAP_FILL = "raven_paper_nfpa_gap_fill"
RAVEN_PAPER_NFPA_GAP_FILL_CLASSIFICATION = (
    "RAVEN paper-faithful settings with NFPA-based gap filling "
    "for underspecified warp implementation details."
)
NFPA_IMAGE_GRID_IMPLEMENTATION_VERSION = "nfpa_image_grid_w_h_norm_v1"



def coords_grid(batch: int, ht: int, wd: int, device, dtype=None):
    """NFPA/RAFT-style pixel coordinate grid with channels ordered as x, y."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("coords_grid requires torch") from exc

    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    if dtype is not None:
        coords = coords.to(dtype)
    return coords[None].repeat(batch, 1, 1, 1)


def create_nfpa_translation_flow(
    dx_image_px: float,
    dy_image_px: float,
    batch: int = 1,
    height: int = 512,
    width: int = 512,
    device=None,
    dtype=None,
):
    """Create NFPA-compatible global translation flow in image pixels."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("create_nfpa_translation_flow requires torch") from exc

    reference_flow = torch.zeros((batch, 2, height, width), device=device, dtype=dtype)
    reference_flow[:, 0, :, :] = dx_image_px
    reference_flow[:, 1, :, :] = dy_image_px
    return reference_flow


def nfpa_warp_single_latent(
    latent,
    reference_flow,
    return_metadata: bool = False,
    pixel_center_offset: float = 0.0,
    sampling_mode: LatentSamplingMode = "nearest",
):
    """Warp a latent following NFPA utils.py coordinate and sampling convention."""
    try:
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("nfpa_warp_single_latent requires torch") from exc

    if latent.ndim != 4:
        raise ValueError(f"Expected latent BCHW, got {tuple(latent.shape)}")
    if sampling_mode not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
    if reference_flow.ndim != 4 or reference_flow.shape[1] != 2:
        raise ValueError(f"Expected reference_flow B2HW, got {tuple(reference_flow.shape)}")
    if reference_flow.shape[0] != latent.shape[0]:
        if reference_flow.shape[0] == 1:
            reference_flow = reference_flow.repeat(latent.shape[0], 1, 1, 1)
        else:
            raise ValueError("reference_flow batch must match latent batch or be 1")

    batch, _, latent_h, latent_w = latent.shape
    _, _, flow_h, flow_w = reference_flow.shape
    coords0 = coords_grid(batch, flow_h, flow_w, device=latent.device, dtype=latent.dtype)
    coords_t0 = coords0 + reference_flow.to(device=latent.device, dtype=latent.dtype)
    if pixel_center_offset:
        coords_t0 = coords_t0 + float(pixel_center_offset)
    coords_minmax_before_norm = {
        "x_min": float(coords_t0[:, 0].detach().float().min().cpu().item()),
        "x_max": float(coords_t0[:, 0].detach().float().max().cpu().item()),
        "y_min": float(coords_t0[:, 1].detach().float().min().cpu().item()),
        "y_max": float(coords_t0[:, 1].detach().float().max().cpu().item()),
    }
    coords_t0[:, 0] /= flow_w
    coords_t0[:, 1] /= flow_h
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_minmax_normalized = {
        "x_min": float(coords_t0[:, 0].detach().float().min().cpu().item()),
        "x_max": float(coords_t0[:, 0].detach().float().max().cpu().item()),
        "y_min": float(coords_t0[:, 1].detach().float().min().cpu().item()),
        "y_max": float(coords_t0[:, 1].detach().float().max().cpu().item()),
    }
    coords_resized = F.interpolate(coords_t0, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    grid = coords_resized.permute(0, 2, 3, 1)
    warped = F.grid_sample(
        latent,
        grid,
        mode=sampling_mode,
        padding_mode="reflection",
        align_corners=False,
    )
    metadata: dict[str, Any] = {
        "coordinate_space": "image_pixels",
        "image_coordinate_grid_shape": [batch, 2, flow_h, flow_w],
        "latent_grid_shape": [batch, 2, latent_h, latent_w],
        "normalization_formula": (
            "x_norm = 2*(x_pixel+0.5)/W - 1; y_norm = 2*(y_pixel+0.5)/H - 1"
            if pixel_center_offset == 0.5
            else "x_norm = 2*x_pixel/W - 1; y_norm = 2*y_pixel/H - 1"
        ),
        "pixel_center_offset_image_px": float(pixel_center_offset),
        "normalized_flow_dx": float((reference_flow[:, 0].detach().float().mean() * 2.0 / flow_w).cpu().item()),
        "normalized_flow_dy": float((reference_flow[:, 1].detach().float().mean() * 2.0 / flow_h).cpu().item()),
        "coordinate_grid_minmax_before_norm": coords_minmax_before_norm,
        "coordinate_grid_minmax_normalized": coords_minmax_normalized,
        "coordinates_exceed_unit_range": bool(
            (coords_t0.detach().float().min() < -1.0).cpu().item()
            or (coords_t0.detach().float().max() > 1.0).cpu().item()
        ),
        "coordinate_resize_mode": "bilinear",
        "nfpa_source_omitted_interpolate_align_corners": True,
        "effective_interpolate_align_corners": False,
        "grid_implementation_version": NFPA_IMAGE_GRID_IMPLEMENTATION_VERSION,
        "latent_sampling_mode": sampling_mode,
        "padding_mode": "reflection",
        "nfpa_source_omitted_grid_sample_align_corners": True,
        "effective_grid_sample_align_corners": False,
        "grid_sample_inverse_sampling": True,
    }
    if return_metadata:
        return warped, metadata
    return warped


def raven_paper_nfpa_gap_fill_warp(
    latent,
    dx_image_px: float,
    dy_image_px: float,
    vae_scale_factor: int = 8,
    sampling_mode: LatentSamplingMode = "nearest",
    return_metadata: bool = False,
):
    """RAVEN paper shift plan with NFPA coordinate/sampling gap filling.

    ``dx_image_px`` and ``dy_image_px`` are RAVEN paper image-pixel flow values.
    NFPA supplies only the coordinate-grid construction, normalization, resize,
    inverse sampling, nearest/bilinear value sampling, and reflection padding.
    """
    if latent.ndim != 4:
        raise ValueError(f"Expected latent BCHW, got {tuple(latent.shape)}")
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")
    if sampling_mode not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
    batch, _, latent_h, latent_w = latent.shape
    flow = create_nfpa_translation_flow(
        dx_image_px=float(dx_image_px),
        dy_image_px=float(dy_image_px),
        batch=batch,
        height=latent_h * vae_scale_factor,
        width=latent_w * vae_scale_factor,
        device=latent.device,
        dtype=latent.dtype,
    )
    warped, metadata = nfpa_warp_single_latent(
        latent,
        flow,
        return_metadata=True,
        pixel_center_offset=0.0,
        sampling_mode=sampling_mode,
    )
    metadata.update({
        "transform_setting_name": RAVEN_PAPER_NFPA_GAP_FILL,
        "implementation_classification": RAVEN_PAPER_NFPA_GAP_FILL_CLASSIFICATION,
        "raven_shift_unit": "image_pixels",
        "raven_shift_rule": (
            "dx and dy sampled independently from [24,32] or [-32,-24] image pixels"
        ),
        "nfpa_gap_fill_scope": (
            "coordinate grid, constant motion field, W/H normalization, coordinate-grid "
            "resize, inverse grid_sample convention, reflection padding"
        ),
        "excluded_nfpa_behaviors": [
            "max_warp_latents",
            "adaptive_xy_search",
            "NFP_XY_40",
            "NFPA checkpoint or inference hyperparameters",
        ],
    })
    if return_metadata:
        return warped, metadata
    return warped


def latent_grid_warp(
    latent,
    dx_image_px: float,
    dy_image_px: float,
    vae_scale_factor: int = 8,
    sampling_mode: str = "nearest",
    padding_mode: PaddingMode = "reflection",
    return_metadata: bool = False,
):
    """Warp via a direct latent grid using align_corners=False coordinates."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("latent_grid_warp requires torch") from exc

    if latent.ndim != 4:
        raise ValueError(f"Expected latent BCHW, got {tuple(latent.shape)}")
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")
    if sampling_mode not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
    if padding_mode not in {"reflection", "border", "zeros"}:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    batch, _, height, width = latent.shape
    dx_latent = float(dx_image_px) / vae_scale_factor
    dy_latent = float(dy_image_px) / vae_scale_factor
    x = (torch.arange(width, device=latent.device, dtype=latent.dtype) + 0.5) / width
    y = (torch.arange(height, device=latent.device, dtype=latent.dtype) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx * 2.0 - 1.0, yy * 2.0 - 1.0), dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
    normalized_dx = 2.0 * dx_latent / width
    normalized_dy = 2.0 * dy_latent / height
    grid[..., 0] += normalized_dx
    grid[..., 1] += normalized_dy
    warped = F.grid_sample(
        latent, grid, mode=sampling_mode, padding_mode=padding_mode, align_corners=False
    )
    metadata: dict[str, Any] = {
        "coordinate_space": "latent_pixels",
        "latent_grid_shape": [batch, 2, height, width],
        "normalization_formula": (
            "identity = 2*(latent_index+0.5)/latent_size - 1; "
            "delta_norm = 2*(image_shift/vae_scale_factor)/latent_size"
        ),
        "dx_latent_cells": dx_latent,
        "dy_latent_cells": dy_latent,
        "normalized_flow_dx": normalized_dx,
        "normalized_flow_dy": normalized_dy,
        "coordinate_resize_mode": "none",
        "latent_sampling_mode": sampling_mode,
        "padding_mode": padding_mode,
        "effective_grid_sample_align_corners": False,
        "grid_sample_inverse_sampling": True,
        "pixel_center_offset_latent_cells": 0.5,
    }
    if return_metadata:
        return warped, metadata
    return warped


def latent_grid_warp_nearest_reflection(
    latent,
    dx_image_px: float,
    dy_image_px: float,
    vae_scale_factor: int = 8,
    return_metadata: bool = False,
):
    return latent_grid_warp(
        latent,
        dx_image_px,
        dy_image_px,
        vae_scale_factor=vae_scale_factor,
        sampling_mode="nearest",
        padding_mode="reflection",
        return_metadata=return_metadata,
    )


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
    padding_mode: PaddingMode = "zeros",
    warp_mode: WarpMode = "integer",
):
    """Translate BCHW latents using a right/down-positive convention.

    The primary ``integer`` mode uses slicing and explicit zero padding:
    positive ``dx`` moves content right and positive ``dy`` moves content
    down. The interpolating implementation remains available as the explicit
    ``grid_sample`` ablation.
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
    if warp_mode not in {
        "integer", "grid_sample", "nfpa_exact", "nfpa_pixel_center",
        "latent_grid_nearest_reflection", "latent_grid",
        "raven_paper_nfpa_gap_fill",
    }:
        raise ValueError(f"Unsupported warp_mode: {warp_mode}")
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")

    batch, _, height, width = latents.shape
    if warp_mode == "raven_paper_nfpa_gap_fill":
        if shift_space != "image_pixels":
            raise ValueError("raven_paper_nfpa_gap_fill requires image-pixel flow")
        return raven_paper_nfpa_gap_fill_warp(
            latents,
            dx,
            dy,
            vae_scale_factor=vae_scale_factor,
            sampling_mode="nearest",
        )
    if warp_mode in {"nfpa_exact", "nfpa_pixel_center"}:
        if shift_space != "image_pixels":
            raise ValueError(f"{warp_mode} requires image-pixel flow")
        reference_flow = create_nfpa_translation_flow(
            dx_image_px=float(dx),
            dy_image_px=float(dy),
            batch=batch,
            height=height * vae_scale_factor,
            width=width * vae_scale_factor,
            device=latents.device,
            dtype=latents.dtype,
        )
        return nfpa_warp_single_latent(
            latents, reference_flow,
            pixel_center_offset=0.5 if warp_mode == "nfpa_pixel_center" else 0.0,
        )
    if warp_mode == "latent_grid_nearest_reflection":
        if shift_space != "image_pixels":
            raise ValueError("latent_grid_nearest_reflection requires image-pixel shifts")
        return latent_grid_warp_nearest_reflection(
            latents, dx, dy, vae_scale_factor=vae_scale_factor
        )
    if warp_mode == "latent_grid":
        if shift_space != "image_pixels":
            raise ValueError("latent_grid requires image-pixel shifts")
        return latent_grid_warp(
            latents, dx, dy, vae_scale_factor=vae_scale_factor, padding_mode=padding_mode
        )

    shift_x = float(dx) / vae_scale_factor if shift_space == "image_pixels" else float(dx)
    shift_y = float(dy) / vae_scale_factor if shift_space == "image_pixels" else float(dy)

    if warp_mode == "integer":
        if padding_mode != "zeros":
            raise ValueError("integer warp_mode requires padding_mode='zeros'")
        rounded_x, rounded_y = round(shift_x), round(shift_y)
        if abs(shift_x - rounded_x) > 1e-6 or abs(shift_y - rounded_y) > 1e-6:
            raise ValueError(
                "integer warp_mode requires shifts divisible by vae_scale_factor; "
                f"got latent shift ({shift_x}, {shift_y})"
            )
        shift_x, shift_y = int(rounded_x), int(rounded_y)
        if abs(shift_x) >= width or abs(shift_y) >= height:
            raise ValueError(
                f"latent shift ({shift_x}, {shift_y}) leaves no valid content for {width}x{height}"
            )
        output = torch.zeros_like(latents)
        source_x0, source_x1 = max(0, -shift_x), width - max(0, shift_x)
        source_y0, source_y1 = max(0, -shift_y), height - max(0, shift_y)
        target_x0, target_x1 = max(0, shift_x), width - max(0, -shift_x)
        target_y0, target_y1 = max(0, shift_y), height - max(0, -shift_y)
        output[:, :, target_y0:target_y1, target_x0:target_x1] = latents[
            :, :, source_y0:source_y1, source_x0:source_x1
        ]
        return output

    ys = torch.linspace(-1.0, 1.0, height, device=latents.device, dtype=latents.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=latents.device, dtype=latents.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

    norm_x = 0.0 if width <= 1 else 2.0 * shift_x / (width - 1)
    norm_y = 0.0 if height <= 1 else 2.0 * shift_y / (height - 1)
    # grid_sample specifies source coordinates, hence subtract to move output
    # content right/down for positive shifts.
    grid[..., 0] = grid[..., 0] - norm_x
    grid[..., 1] = grid[..., 1] - norm_y

    return F.grid_sample(
        latents,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
