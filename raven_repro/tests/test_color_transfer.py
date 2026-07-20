import inspect

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")

from raven.pipeline_raven import require_effective_source_flow

from raven.color_transfer import (
    PAPER_EXACT_TWO_STAGE_ALIGNED,
    align_original_chroma_to_generated,
    color_contrast_transfer,
    color_transfer_diagnostics,
)


def _images(seed=11, shape=(32, 33, 3)):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, 256, shape, dtype=np.uint8),
        rng.integers(0, 256, shape, dtype=np.uint8),
    )


def test_aligned_mode_is_the_only_public_mode():
    assert PAPER_EXACT_TWO_STAGE_ALIGNED == "paper_exact_two_stage_aligned"
    signature = inspect.signature(color_contrast_transfer)
    assert "effective_source_flow_dx_image_px" in signature.parameters
    assert "effective_source_flow_dy_image_px" in signature.parameters
    assert "flow_dx_image_px" not in signature.parameters
    assert "flow_dy_image_px" not in signature.parameters
    assert "alpha" not in signature.parameters


@pytest.mark.parametrize(
    "legacy_mode",
    ["paper_exact_two_stage", "paper_exact_two_stage_aligned_blend", "direct_stats"],
)
def test_legacy_color_transfer_modes_are_rejected(legacy_mode):
    generated, original = _images()
    with pytest.raises(ValueError, match="Unsupported color transfer mode"):
        color_contrast_transfer(
            generated,
            original,
            mode=legacy_mode,
            effective_source_flow_dx_image_px=3,
            effective_source_flow_dy_image_px=-2,
        )


def test_aligned_color_transfer_requires_effective_flow():
    generated, original = _images()
    with pytest.raises(ValueError, match="requires effective source flow"):
        color_contrast_transfer(generated, original)


def test_aligned_color_transfer_is_deterministic_and_finite():
    generated, original = _images()
    kwargs = {
        "effective_source_flow_dx_image_px": 3,
        "effective_source_flow_dy_image_px": -2,
    }
    first = color_contrast_transfer(generated, original, **kwargs)
    second = color_contrast_transfer(generated, original, **kwargs)
    diagnostics = color_transfer_diagnostics(generated, original, first, **kwargs)
    assert np.array_equal(first, second)
    assert first.shape == generated.shape
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert all(
        np.isfinite(value)
        for value in diagnostics.values()
        if isinstance(value, float)
    )
    assert diagnostics["color_transfer_mode"] == PAPER_EXACT_TWO_STAGE_ALIGNED
    assert diagnostics["alignment_flow_source"] == (
        "effective source flow from actual warp grid"
    )
    assert diagnostics["effective_source_flow_dx_image_px"] == 3.0
    assert diagnostics["effective_source_flow_dy_image_px"] == -2.0


@pytest.mark.parametrize(
    "dx,dy",
    [
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (2, 2), (2, -2), (-2, 2), (-2, -2),
    ],
)
def test_aligned_chroma_uses_effective_inverse_warp_correspondence(dx, dy):
    height, width = 7, 9
    original = np.zeros((height, width, 2), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    original[..., 0] = yy * 100 + xx
    original[..., 1] = -(yy * 100 + xx)
    generated = np.full_like(original, -999.0)
    aligned, valid = align_original_chroma_to_generated(
        original,
        generated,
        effective_source_flow_dx_image_px=dx,
        effective_source_flow_dy_image_px=dy,
    )
    for y in range(height):
        for x in range(width):
            source_y, source_x = y + dy, x + dx
            if 0 <= source_y < height and 0 <= source_x < width:
                assert valid[y, x]
                assert np.array_equal(aligned[y, x], original[source_y, source_x])
            else:
                assert not valid[y, x]
                assert np.array_equal(aligned[y, x], generated[y, x])


def test_effective_flow_changes_alignment_and_preserves_non_overlap():
    original = np.arange(8 * 9 * 2, dtype=np.float32).reshape(8, 9, 2)
    generated = np.full_like(original, 777.0)
    flow_3, valid_3 = align_original_chroma_to_generated(
        original,
        generated,
        effective_source_flow_dx_image_px=3,
        effective_source_flow_dy_image_px=-2,
    )
    flow_4, valid_4 = align_original_chroma_to_generated(
        original,
        generated,
        effective_source_flow_dx_image_px=4,
        effective_source_flow_dy_image_px=-2,
    )
    assert not np.array_equal(flow_3, flow_4)
    assert np.all(flow_3[~valid_3] == 777.0)
    assert np.all(flow_4[~valid_4] == 777.0)


def test_pipeline_requires_actual_grid_effective_flow():
    metadata = {
        "planned_flow_dx_image_px": 27.0,
        "planned_flow_dy_image_px": -29.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
    }
    assert require_effective_source_flow(metadata) == (24.0, -32.0)
    with pytest.raises(RuntimeError, match="planned-flow fallback is forbidden"):
        require_effective_source_flow({
            "planned_flow_dx_image_px": 27.0,
            "planned_flow_dy_image_px": -29.0,
        })
