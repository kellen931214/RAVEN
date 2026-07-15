import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")

from raven.color_transfer import color_contrast_transfer, color_transfer_diagnostics


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
