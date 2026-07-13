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


def test_color_transfer_diagnostics_are_bounded():
    original = np.full((16, 16, 3), (180, 80, 40), dtype=np.uint8)
    generated = np.full((16, 16, 3), 120, dtype=np.uint8)
    output = color_contrast_transfer(generated, original)
    diagnostics = color_transfer_diagnostics(generated, original, output)
    assert 0.0 <= diagnostics["output_saturated_pixel_ratio"] <= 1.0
    assert diagnostics["original_l_std"] >= 0.0
