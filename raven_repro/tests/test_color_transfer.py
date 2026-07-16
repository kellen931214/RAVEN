import hashlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")

from raven.color_transfer import (
    align_original_chroma_to_generated,
    color_contrast_transfer,
    color_transfer_diagnostics,
)


def test_color_transfer_shape_dtype_and_range():
    original = np.zeros((32, 32, 3), dtype=np.uint8)
    original[..., 0] = 180
    original[..., 1] = 80
    original[..., 2] = 40
    generated = np.full((32, 32, 3), 120, dtype=np.uint8)
    out = color_contrast_transfer(generated, original)
    assert out.shape == original.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_color_transfer_is_deterministic_for_fixed_input():
    rng = np.random.default_rng(7)
    original = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    generated = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    first = color_contrast_transfer(generated, original)
    second = color_contrast_transfer(generated, original)
    assert np.array_equal(first, second)


def test_constant_luminance_has_no_nan_or_inf():
    original = np.full((16, 16, 3), (180, 80, 40), dtype=np.uint8)
    generated = np.full((16, 16, 3), 120, dtype=np.uint8)
    output = color_contrast_transfer(generated, original)
    diagnostics = color_transfer_diagnostics(generated, original, output)
    numeric = [value for value in diagnostics.values() if isinstance(value, float)]
    assert all(np.isfinite(value) for value in numeric)
    assert 0.0 <= diagnostics["output_saturated_pixel_ratio"] <= 1.0


def test_paper_exact_luminance_mean_std_close_to_original():
    rng = np.random.default_rng(123)
    original = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    generated = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    output = color_contrast_transfer(generated, original, mode="paper_exact_two_stage")
    diagnostics = color_transfer_diagnostics(generated, original, output, mode="paper_exact_two_stage")
    assert diagnostics["final_output_L_mean_abs_error_vs_original"] < 0.75
    assert diagnostics["final_output_L_std_abs_error_vs_original"] < 1.25
    assert diagnostics["L_c_mean"] is not None
    assert diagnostics["L_c_std"] is not None


def test_two_stage_and_direct_stats_differ_under_gamut_clipping():
    rng = np.random.default_rng(0)
    original = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    generated = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    direct = color_contrast_transfer(generated, original, mode="direct_stats")
    paper = color_contrast_transfer(generated, original, mode="paper_exact_two_stage")
    assert not np.array_equal(direct, paper)
    assert np.abs(direct.astype(np.int16) - paper.astype(np.int16)).max() > 0


def test_color_transfer_diagnostics_include_required_fields():
    original = np.full((16, 16, 3), (180, 80, 40), dtype=np.uint8)
    generated = np.full((16, 16, 3), 120, dtype=np.uint8)
    output = color_contrast_transfer(generated, original)
    diagnostics = color_transfer_diagnostics(generated, original, output)
    for key in (
        "L_opt_mean",
        "L_opt_std",
        "L_c_mean",
        "L_c_std",
        "L_w_mean",
        "L_w_std",
        "L_final_before_clip_min",
        "L_final_before_clip_max",
        "L_final_after_clip_min",
        "L_final_after_clip_max",
        "final_output_L_mean",
        "final_output_L_std",
        "output_saturated_pixel_ratio",
    ):
        assert key in diagnostics
    assert diagnostics["color_transfer_mode"] == "paper_exact_two_stage"



@pytest.mark.parametrize(
    "dx,dy",
    [
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (2, 2), (2, -2), (-2, 2), (-2, -2),
    ],
)
def test_aligned_chroma_uses_inverse_warp_correspondence(dx, dy):
    height, width = 7, 9
    original = np.zeros((height, width, 2), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    original[..., 0] = yy * 100 + xx
    original[..., 1] = -(yy * 100 + xx)
    generated = np.full_like(original, -999.0)
    aligned, valid = align_original_chroma_to_generated(original, generated, dx, dy)
    for y in range(height):
        for x in range(width):
            source_y, source_x = y + dy, x + dx
            if 0 <= source_y < height and 0 <= source_x < width:
                assert valid[y, x]
                assert np.array_equal(aligned[y, x], original[source_y, source_x])
            else:
                assert not valid[y, x]
                assert np.array_equal(aligned[y, x], generated[y, x])


def test_aligned_chroma_has_no_circular_wrap_and_preserves_non_overlap():
    original = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    generated = np.full_like(original, 777.0)
    aligned, valid = align_original_chroma_to_generated(original, generated, 2, -1)
    assert np.all(aligned[~valid] == 777.0)
    assert np.all(aligned[:, -2:] == 777.0)
    assert np.all(aligned[0] == 777.0)


def test_aligned_chroma_alpha_endpoints():
    rng = np.random.default_rng(10)
    original = rng.normal(size=(8, 9, 2)).astype(np.float32)
    generated = rng.normal(size=(8, 9, 2)).astype(np.float32)
    zero, valid = align_original_chroma_to_generated(original, generated, 2, -1, alpha=0.0)
    one, valid_one = align_original_chroma_to_generated(original, generated, 2, -1, alpha=1.0)
    assert np.array_equal(zero, generated)
    assert np.array_equal(valid, valid_one)
    yy, xx = np.nonzero(valid)
    assert np.allclose(one[yy, xx], original[yy - 1, xx + 2])
    assert np.array_equal(one[~valid], generated[~valid])


@pytest.mark.parametrize(
    "mode,alpha",
    [
        ("paper_exact_two_stage_aligned", 1.0),
        ("paper_exact_two_stage_aligned_blend", 0.5),
    ],
)
def test_aligned_modes_shape_dtype_range_and_finite(mode, alpha):
    rng = np.random.default_rng(11)
    original = rng.integers(0, 256, (24, 25, 3), dtype=np.uint8)
    generated = rng.integers(0, 256, (24, 25, 3), dtype=np.uint8)
    output = color_contrast_transfer(
        generated, original, mode=mode,
        flow_dx_image_px=3, flow_dy_image_px=-2, alpha=alpha,
    )
    diagnostics = color_transfer_diagnostics(
        generated, original, output, mode=mode,
        flow_dx_image_px=3, flow_dy_image_px=-2, alpha=alpha,
    )
    assert output.shape == generated.shape
    assert output.dtype == np.uint8
    assert output.min() >= 0 and output.max() <= 255
    assert all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, float))
    assert diagnostics["alignment_formula"] == "generated[y,x] <- original[y+flow_dy,x+flow_dx]"
    assert 0.0 <= diagnostics["output_rgb_out_of_gamut_ratio_before_clip"] <= 1.0


def test_paper_exact_two_stage_baseline_regression_hash_unchanged():
    rng = np.random.default_rng(20260716)
    generated = rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)
    original = rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)
    output = color_contrast_transfer(generated, original, mode="paper_exact_two_stage")
    assert hashlib.sha256(output.tobytes()).hexdigest() == (
        "865e1f6332c95bae6e7bcdb066b345a03f04b7cd6d232805af9e12434127321a"
    )
