import math

import numpy as np
import pytest
import torch

from raven.metrics import pair_quality_metrics
from raven.warp import raven_paper_nfpa_gap_fill_warp


def translated(reference, dx, dy):
    attacked = np.zeros_like(reference)
    x0, x1 = max(0, -dx), min(reference.shape[1], reference.shape[1] - dx)
    y0, y1 = max(0, -dy), min(reference.shape[0], reference.shape[0] - dy)
    attacked[y0:y1, x0:x1] = reference[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return attacked


@pytest.mark.parametrize("magnitude", [24, 27, 28, 29, 32, -24, -27, -28, -29, -32])
@pytest.mark.parametrize("axis,other_sign", [("x", 1), ("x", -1), ("y", 1), ("y", -1)])
def test_effective_flow_comes_from_actual_grid_and_drives_overlap(magnitude, axis, other_sign):
    dx = magnitude if axis == "x" else other_sign * 27
    dy = other_sign * 27 if axis == "x" else magnitude
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    latent = torch.stack((xx, yy), dim=0).float().unsqueeze(0)
    warped, metadata = raven_paper_nfpa_gap_fill_warp(
        latent, dx, dy, vae_scale_factor=8, sampling_mode="nearest", return_metadata=True
    )
    center = 32
    actual_dx = float(warped[0, 0, center, center] - center)
    actual_dy = float(warped[0, 1, center, center] - center)
    assert metadata["effective_source_dx_latent"] == actual_dx
    assert metadata["effective_source_dy_latent"] == actual_dy
    assert metadata["planned_flow_dx_image_px"] == dx
    assert metadata["planned_flow_dy_image_px"] == dy
    effective_dx = int(metadata["effective_source_flow_dx_image_px"])
    effective_dy = int(metadata["effective_source_flow_dy_image_px"])

    height = width = 128
    grid_y, grid_x = np.mgrid[:height, :width]
    reference = np.stack((grid_x, grid_y, grid_x + grid_y), axis=-1).astype(np.uint8)
    attacked = translated(reference, effective_dx, effective_dy)
    metrics = pair_quality_metrics(reference, attacked, effective_dx, effective_dy)
    assert math.isinf(metrics["overlap_psnr"])
    assert metrics["overlap_ssim"] == pytest.approx(1.0)
    assert metrics["valid_overlap_width"] == width - abs(effective_dx)
    assert metrics["valid_overlap_height"] == height - abs(effective_dy)
    if (dx, dy) != (effective_dx, effective_dy) and float(dx).is_integer() and float(dy).is_integer():
        wrong = pair_quality_metrics(reference, attacked, int(dx), int(dy))
        assert not math.isinf(wrong["overlap_psnr"])
