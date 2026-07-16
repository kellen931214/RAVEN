import pytest

np = pytest.importorskip("numpy")

from raven.metrics import (
    align_fractional_overlap_inverse_warp,
    crop_overlap,
    crop_overlap_inverse_warp,
    pair_quality_metrics,
    psnr,
)


@pytest.mark.parametrize("dx,dy", [(24, 24), (-24, -24), (32, -32)])
def test_known_integer_translation_has_perfect_overlap(dx, dy):
    height = width = 512
    yy, xx = np.mgrid[:height, :width]
    original = np.stack((xx, yy, xx + yy), axis=2).astype(np.float64)
    translated = np.full_like(original, -999.0)

    output_x0, output_x1 = max(0, dx), width - max(0, -dx)
    output_y0, output_y1 = max(0, dy), height - max(0, -dy)
    input_x0, input_x1 = max(0, -dx), width - max(0, dx)
    input_y0, input_y1 = max(0, -dy), height - max(0, dy)
    translated[output_y0:output_y1, output_x0:output_x1] = original[
        input_y0:input_y1, input_x0:input_x1
    ]

    first, second = crop_overlap(original, translated, dx, dy)
    assert np.array_equal(first, second)
    assert math_is_inf(psnr(first, second, data_range=float(original.max())))


def math_is_inf(value):
    return value == float("inf")


def test_shift_larger_than_image_is_rejected():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        crop_overlap(image, image, 8, 0)


@pytest.mark.parametrize("dx,dy", [(24, 24), (24, -24), (-32, 24), (-32, -32)])
def test_inverse_warp_overlap_four_quadrants(dx, dy):
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    reference = np.stack((xx, yy, xx + 2 * yy), axis=2).astype(np.float64)
    attacked = np.full_like(reference, -999.0)

    attacked_x0 = max(0, -dx)
    attacked_x1 = min(width, width - dx)
    attacked_y0 = max(0, -dy)
    attacked_y1 = min(height, height - dy)
    for y in range(attacked_y0, attacked_y1):
        for x in range(attacked_x0, attacked_x1):
            attacked[y, x] = reference[y + dy, x + dx]

    reference_crop, attacked_crop = crop_overlap_inverse_warp(reference, attacked, dx, dy)
    assert reference_crop.shape == attacked_crop.shape
    assert np.array_equal(reference_crop, attacked_crop)
    assert math_is_inf(psnr(reference_crop, attacked_crop, data_range=float(reference.max())))
    assert not np.any(attacked_crop == -999.0)


def test_inverse_warp_overlap_quality_synthetic_perfect_translation():
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    reference = np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8)
    dx, dy = 8, -8
    attacked = np.zeros_like(reference)
    attacked_x0 = max(0, -dx)
    attacked_x1 = min(width, width - dx)
    attacked_y0 = max(0, -dy)
    attacked_y1 = min(height, height - dy)
    attacked[attacked_y0:attacked_y1, attacked_x0:attacked_x1] = reference[
        attacked_y0 + dy:attacked_y1 + dy, attacked_x0 + dx:attacked_x1 + dx
    ]
    metrics = pair_quality_metrics(reference, attacked, dx, dy)
    assert metrics["overlap_psnr"] == float("inf")
    assert metrics["overlap_ssim"] == pytest.approx(1.0)
    assert metrics["valid_overlap_width"] == width - abs(dx)
    assert metrics["valid_overlap_height"] == height - abs(dy)


def test_inverse_warp_rejects_non_integer_flow_for_metrics():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        crop_overlap_inverse_warp(image, image, 3.5, 0)


def _inverse_warp_array(reference, dx, dy):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    height, width = reference.shape[:2]
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((
        2.0 * (xx + dx + 0.5) / width - 1.0,
        2.0 * (yy + dy + 0.5) / height - 1.0,
    ), dim=-1).unsqueeze(0)
    tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)
    return F.grid_sample(
        tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0].permute(1, 2, 0).numpy()


def test_fractional_overlap_aligns_reference_without_rounding():
    rng = np.random.default_rng(7)
    reference = rng.random((64, 64, 3), dtype=np.float32)
    dx, dy = 3.375, -3.625
    attacked = _inverse_warp_array(reference, dx, dy)
    aligned, attacked_crop = align_fractional_overlap_inverse_warp(
        reference, attacked, dx, dy
    )
    assert aligned.shape == attacked_crop.shape
    assert np.allclose(aligned, attacked_crop, atol=1e-6)


def test_effective_integer_flow_outperforms_legacy_requested_flow_on_translation():
    rng = np.random.default_rng(11)
    reference = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    attacked = np.zeros_like(reference)
    effective_dx, effective_dy = 24, -24
    x0, x1 = max(0, -effective_dx), min(96, 96 - effective_dx)
    y0, y1 = max(0, -effective_dy), min(96, 96 - effective_dy)
    attacked[y0:y1, x0:x1] = reference[
        y0 + effective_dy:y1 + effective_dy,
        x0 + effective_dx:x1 + effective_dx,
    ]
    corrected = pair_quality_metrics(reference, attacked, effective_dx, effective_dy)
    legacy = pair_quality_metrics(reference, attacked, 27, -25)
    assert corrected["overlap_psnr"] == float("inf")
    assert corrected["overlap_ssim"] == pytest.approx(1.0)
    assert legacy["overlap_psnr"] < 20
    assert legacy["overlap_ssim"] < 0.5


def test_fractional_pair_quality_protocol_is_explicit():
    rng = np.random.default_rng(13)
    reference = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    dx, dy = 3.25, -3.75
    attacked = _inverse_warp_array(reference.astype(np.float32), dx, dy)
    attacked = np.clip(np.rint(attacked), 0, 255).astype(np.uint8)
    result = pair_quality_metrics(
        reference, attacked, dx, dy, alignment_mode="fractional_grid_sample"
    )
    assert result["overlap_protocol"] == (
        "effective_fractional_inverse_warp_bilinear_reference_alignment"
    )
    assert result["flow_dx_px"] == dx
    assert result["flow_dy_px"] == dy


def test_formal_tree_ring_quality_uses_effective_flow_and_marks_requested_legacy():
    from PIL import Image
    from scripts.tree_ring_official_raven_eval import image_quality

    rng = np.random.default_rng(17)
    reference = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    attacked = np.zeros_like(reference)
    effective_dx, effective_dy = 24, -24
    x0, x1 = max(0, -effective_dx), min(96, 96 - effective_dx)
    y0, y1 = max(0, -effective_dy), min(96, 96 - effective_dy)
    attacked[y0:y1, x0:x1] = reference[
        y0 + effective_dy:y1 + effective_dy,
        x0 + effective_dx:x1 + effective_dx,
    ]
    result = image_quality(
        Image.fromarray(reference), Image.fromarray(attacked),
        requested_dx=27, requested_dy=-25,
        effective_dx=effective_dx, effective_dy=effective_dy,
        alignment_mode="integer_crop",
    )
    assert result["post_color_overlap_psnr"] == float("inf")
    assert result["post_color_overlap_ssim"] == pytest.approx(1.0)
    assert result["formal_quality_protocol"].startswith("effective_integer")
    assert result["legacy_requested_flow_overlap_psnr"] < 20
    assert result["legacy_quality_protocol"] == "requested_flow_integer_crop_diagnostic_only"
