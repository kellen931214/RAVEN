import pytest

np = pytest.importorskip("numpy")

from raven.metrics import crop_overlap, crop_overlap_inverse_warp, pair_quality_metrics, psnr


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


def test_inverse_warp_fractional_effective_flow_uses_bilinear_source_samples():
    height = width = 16
    yy, xx = np.mgrid[:height, :width]
    reference = np.stack((xx, yy, xx + 2 * yy), axis=2).astype(np.float32)
    dx, dy = 3.5, -2.5
    from raven.metrics import sample_inverse_warp_reference

    sampled, (y0, y1, x0, x1) = sample_inverse_warp_reference(reference, dx, dy)
    attacked = np.full_like(reference, -999.0)
    attacked[y0:y1, x0:x1] = sampled
    reference_crop, attacked_crop = crop_overlap_inverse_warp(reference, attacked, dx, dy)
    assert np.array_equal(reference_crop, attacked_crop)
    assert reference_crop.shape[:2] == (height - 3, width - 4)
    assert math_is_inf(psnr(reference_crop, attacked_crop, data_range=float(reference.max())))
    metrics = pair_quality_metrics(reference, attacked, dx, dy)
    # pair_quality_metrics normalizes RGB inputs before resampling; this only
    # changes floating-point operation order, not the effective-flow pairing.
    assert metrics["overlap_psnr"] > 100.0
    assert metrics["overlap_ssim"] == pytest.approx(1.0)
    assert metrics["valid_overlap_width"] == width - 4
    assert metrics["valid_overlap_height"] == height - 3
