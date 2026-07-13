import pytest

np = pytest.importorskip("numpy")

from raven.metrics import crop_overlap, psnr


@pytest.mark.parametrize("dx,dy", [(24, 24), (-24, -24), (32, -32)])
def test_known_integer_translation_has_perfect_overlap(dx, dy):
    height = width = 512
    yy, xx = np.mgrid[:height, :width]
    original = np.stack((xx, yy, xx + yy), axis=2).astype(np.float64)
    translated = np.full_like(original, -999.0)

    output_x0, output_x1 = max(0, -dx), width - max(0, dx)
    output_y0, output_y1 = max(0, -dy), height - max(0, dy)
    input_x0, input_x1 = max(0, dx), width + min(0, dx)
    input_y0, input_y1 = max(0, dy), height + min(0, dy)
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
