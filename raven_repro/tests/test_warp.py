import pytest

torch = pytest.importorskip("torch")

from raven.warp import (
    RAVEN_PAPER_NFPA_GAP_FILL,
    create_nfpa_translation_flow,
    nfpa_warp_single_latent,
    raven_paper_nfpa_gap_fill_warp,
    sample_translation,
    translate_latent,
)


def test_sample_translation_diagonal_and_seeded():
    assert sample_translation(24, 32, "positive", seed=1)[0] > 0
    dx, dy = sample_translation(24, 32, "negative", seed=1, sampling="coupled_diagonal")
    assert dx == dy
    assert dx < 0


def test_independent_axis_sampling_is_seeded_and_bounded():
    first = sample_translation(24, 32, "random", seed=7, sampling="independent_axes")
    second = sample_translation(24, 32, "random", seed=7, sampling="independent_axes")
    assert first == second
    assert all(24 <= abs(value) <= 32 for value in first)


def test_translate_latent_preserves_shape_and_finite():
    latents = torch.randn(2, 4, 64, 64)
    shifted = translate_latent(latents, dx=24, dy=24)
    assert shifted.shape == latents.shape
    assert torch.isfinite(shifted).all()

@pytest.mark.parametrize("padding_mode", ["reflection", "border", "zeros"])
def test_grid_sample_ablation_preserves_shape_and_finite(padding_mode):
    latents = torch.randn(2, 4, 64, 64)
    shifted = translate_latent(
        latents, dx=24, dy=24, padding_mode=padding_mode, warp_mode="grid_sample"
    )
    assert shifted.shape == latents.shape
    assert torch.isfinite(shifted).all()



def test_translate_latent_latent_pixels():
    latents = torch.randn(1, 4, 16, 16)
    shifted = translate_latent(latents, dx=2, dy=-2, shift_space="latent_pixels")
    assert shifted.shape == latents.shape


def test_image_pixel_translation_scales_to_latent_cells():
    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 30, 30] = 1.0
    shifted_24 = translate_latent(
        latent, dx=24, dy=24, shift_space="image_pixels", vae_scale_factor=8, padding_mode="zeros"
    )
    shifted_32 = translate_latent(
        latent, dx=32, dy=32, shift_space="image_pixels", vae_scale_factor=8, padding_mode="zeros"
    )
    assert shifted_24[0, 0, 33, 33].item() == pytest.approx(1.0)
    assert shifted_32[0, 0, 34, 34].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "dx,dy,expected_y,expected_x",
    [(8, 0, 4, 5), (-8, 0, 4, 3), (0, 8, 5, 4), (0, -8, 3, 4)],
)
def test_positive_means_right_and_down(dx, dy, expected_y, expected_x):
    latent = torch.zeros(1, 1, 9, 9)
    latent[0, 0, 4, 4] = 1.0
    shifted = translate_latent(latent, dx=dx, dy=dy, vae_scale_factor=8)
    location = torch.nonzero(shifted[0, 0] == 1.0, as_tuple=False)
    assert location.tolist() == [[expected_y, expected_x]]
    assert shifted.sum().item() == pytest.approx(1.0)


def test_integer_translation_has_no_wraparound():
    latent = torch.zeros(1, 1, 8, 8)
    latent[0, 0, 4, 7] = 1.0
    shifted = translate_latent(latent, dx=8, dy=0, vae_scale_factor=8)
    assert shifted.sum().item() == 0.0


@pytest.mark.parametrize(
    "dx,dy,expected_x_direction,expected_y_direction",
    [
        (12.0, 20.0, 1, 1),
        (12.0, -20.0, 1, -1),
        (-12.0, 20.0, -1, 1),
        (-12.0, -20.0, -1, -1),
    ],
)
def test_fractional_grid_sample_direction(dx, dy, expected_x_direction, expected_y_direction):
    latent = torch.zeros(1, 1, 17, 17)
    latent[0, 0, 8, 8] = 1.0
    shifted = translate_latent(
        latent,
        dx=dx,
        dy=dy,
        shift_space="image_pixels",
        vae_scale_factor=8,
        padding_mode="zeros",
        warp_mode="grid_sample",
    )
    weights = shifted[0, 0]
    yy, xx = torch.meshgrid(torch.arange(17), torch.arange(17), indexing="ij")
    center_x = float((weights * xx).sum() / weights.sum())
    center_y = float((weights * yy).sum() / weights.sum())
    assert (center_x - 8.0) * expected_x_direction > 0
    assert (center_y - 8.0) * expected_y_direction > 0
    assert center_x == pytest.approx(8.0 + dx / 8.0, abs=1e-5)
    assert center_y == pytest.approx(8.0 + dy / 8.0, abs=1e-5)



def test_nfpa_coords_grid_shape_and_order():
    import torch
    from raven.warp import coords_grid

    grid = coords_grid(2, 3, 4, torch.device("cpu"), dtype=torch.float32)
    assert tuple(grid.shape) == (2, 2, 3, 4)
    assert grid[0, 0, 0, 3].item() == 3
    assert grid[0, 1, 2, 0].item() == 2


def test_nfpa_normalization_uses_width_height_not_minus_one():
    import torch
    from raven.warp import create_nfpa_translation_flow, nfpa_warp_single_latent

    latent = torch.zeros(1, 1, 8, 8)
    flow = create_nfpa_translation_flow(24, -32, height=64, width=64, device=latent.device, dtype=latent.dtype)
    _, meta = nfpa_warp_single_latent(latent, flow, return_metadata=True)
    assert meta["normalization_formula"] == "x_norm = 2*x_pixel/W - 1; y_norm = 2*y_pixel/H - 1"
    assert meta["normalized_flow_dx"] == 2 * 24 / 64
    assert meta["normalized_flow_dy"] == 2 * -32 / 64
    assert meta["effective_interpolate_align_corners"] is False
    assert meta["effective_grid_sample_align_corners"] is False


def test_nfpa_four_direction_impulse_inverse_sampling():
    import torch
    from raven.warp import create_nfpa_translation_flow, nfpa_warp_single_latent

    cases = [
        (24, 24, -3, -3),
        (24, -24, -3, 3),
        (-24, 24, 3, -3),
        (-24, -24, 3, 3),
    ]
    for dx, dy, visual_x, visual_y in cases:
        latent = torch.zeros(1, 1, 64, 64)
        latent[0, 0, 32, 32] = 1
        flow = create_nfpa_translation_flow(dx, dy, height=512, width=512, device=latent.device, dtype=latent.dtype)
        shifted = nfpa_warp_single_latent(latent, flow)
        y, x = torch.nonzero(shifted[0, 0] == shifted.max(), as_tuple=True)
        assert int(x[0]) == 32 + visual_x
        assert int(y[0]) == 32 + visual_y


def test_nfpa_reflection_padding_and_nearest_sampling():
    import torch
    from raven.warp import create_nfpa_translation_flow, nfpa_warp_single_latent

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 0, 0] = 7
    flow = create_nfpa_translation_flow(-32, -32, height=512, width=512, device=latent.device, dtype=latent.dtype)
    shifted = nfpa_warp_single_latent(latent, flow)
    assert shifted[0, 0, 4, 4].item() == 7
    assert torch.isfinite(shifted).all()


def test_nfpa_effective_displacement_quantization():
    import torch
    from raven.warp import create_nfpa_translation_flow, nfpa_warp_single_latent

    expected = {24: -3, 28: -3, 32: -4}
    for dx, observed in expected.items():
        latent = torch.zeros(1, 1, 64, 64)
        latent[0, 0, 32, 32] = 1
        flow = create_nfpa_translation_flow(dx, 0, height=512, width=512, device=latent.device, dtype=latent.dtype)
        shifted = nfpa_warp_single_latent(latent, flow)
        y, x = torch.nonzero(shifted[0, 0] == shifted.max(), as_tuple=True)
        assert int(x[0]) - 32 == observed
        assert int(y[0]) == 32


def test_nfpa_cpu_gpu_consistency_if_available():
    import torch
    from raven.warp import create_nfpa_translation_flow, nfpa_warp_single_latent

    if not torch.cuda.is_available():
        return
    latent_cpu = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    flow_cpu = create_nfpa_translation_flow(28, -24, height=512, width=512, device=latent_cpu.device, dtype=latent_cpu.dtype)
    out_cpu = nfpa_warp_single_latent(latent_cpu, flow_cpu)
    latent_gpu = latent_cpu.cuda()
    flow_gpu = flow_cpu.cuda()
    out_gpu = nfpa_warp_single_latent(latent_gpu, flow_gpu).cpu()
    assert torch.equal(out_cpu, out_gpu)


@pytest.mark.parametrize("dx,dy", [(24, 24), (28, -29), (-31, 27), (-32, -32)])
def test_pixel_center_and_direct_latent_grid_match(dx, dy):
    import torch

    from raven.warp import translate_latent

    torch.manual_seed(7)
    latent = torch.randn(1, 4, 64, 64)
    pixel_center = translate_latent(
        latent, dx, dy, shift_space="image_pixels", vae_scale_factor=8,
        padding_mode="reflection", warp_mode="nfpa_pixel_center",
    )
    direct_latent = translate_latent(
        latent, dx, dy, shift_space="image_pixels", vae_scale_factor=8,
        padding_mode="reflection", warp_mode="latent_grid_nearest_reflection",
    )
    assert torch.equal(pixel_center, direct_latent)


def test_normalization_modes_share_sampling_and_expose_formula():
    import torch

    from raven.warp import (
        create_nfpa_translation_flow,
        latent_grid_warp_nearest_reflection,
        nfpa_warp_single_latent,
    )

    latent = torch.zeros(1, 1, 64, 64)
    flow = create_nfpa_translation_flow(28, -30, device=latent.device, dtype=latent.dtype)
    _, nfpa = nfpa_warp_single_latent(latent, flow, return_metadata=True)
    _, centered = nfpa_warp_single_latent(
        latent, flow, return_metadata=True, pixel_center_offset=0.5
    )
    _, direct = latent_grid_warp_nearest_reflection(
        latent, 28, -30, return_metadata=True
    )
    for metadata in (nfpa, centered, direct):
        assert metadata["latent_sampling_mode"] == "nearest"
        assert metadata["padding_mode"] == "reflection"
        assert metadata["effective_grid_sample_align_corners"] is False
    assert nfpa["pixel_center_offset_image_px"] == 0.0
    assert centered["pixel_center_offset_image_px"] == 0.5
    assert "image_shift/vae_scale_factor" in direct["normalization_formula"]


def test_half_pixel_offset_changes_nearest_quantization_at_28_pixels():
    import torch

    from raven.warp import translate_latent

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    nfpa = translate_latent(
        latent, 28, 0, shift_space="image_pixels", warp_mode="nfpa_exact",
        padding_mode="reflection",
    )
    centered = translate_latent(
        latent, 28, 0, shift_space="image_pixels", warp_mode="nfpa_pixel_center",
        padding_mode="reflection",
    )
    nfpa_x = int(torch.argmax(nfpa[0, 0]).item()) % 64
    centered_x = int(torch.argmax(centered[0, 0]).item()) % 64
    assert nfpa_x == 29
    assert centered_x == 28



def test_latent_grid_sampling_padding_ablation_metadata():
    import torch
    from raven.warp import latent_grid_warp

    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    for sampling_mode in ("nearest", "bilinear"):
        for padding_mode in ("reflection", "zeros"):
            warped, metadata = latent_grid_warp(
                latent,
                24,
                -32,
                vae_scale_factor=8,
                sampling_mode=sampling_mode,
                padding_mode=padding_mode,
                return_metadata=True,
            )
            assert warped.shape == latent.shape
            assert torch.isfinite(warped).all()
            assert metadata["latent_sampling_mode"] == sampling_mode
            assert metadata["padding_mode"] == padding_mode
            assert metadata["effective_grid_sample_align_corners"] is False
            assert metadata["grid_sample_inverse_sampling"] is True
            assert "image_shift/vae_scale_factor" in metadata["normalization_formula"]


def test_translate_latent_generic_latent_grid_uses_requested_padding():
    import torch
    from raven.warp import translate_latent

    latent = torch.randn(1, 2, 64, 64)
    reflected = translate_latent(
        latent,
        28,
        28,
        shift_space="image_pixels",
        vae_scale_factor=8,
        padding_mode="reflection",
        warp_mode="latent_grid",
    )
    zeroed = translate_latent(
        latent,
        28,
        28,
        shift_space="image_pixels",
        vae_scale_factor=8,
        padding_mode="zeros",
        warp_mode="latent_grid",
    )
    assert reflected.shape == zeroed.shape == latent.shape
    assert not torch.equal(reflected, zeroed)


def test_raven_shift_sampler_signs_bounds_and_reproducibility():
    samples = [sample_translation(24, 32, "random", seed=i, sampling="independent_axes") for i in range(64)]
    assert samples == [sample_translation(24, 32, "random", seed=i, sampling="independent_axes") for i in range(64)]
    assert all(24 <= abs(value) <= 32 for pair in samples for value in pair)
    assert any(dx > 0 for dx, _ in samples)
    assert any(dx < 0 for dx, _ in samples)
    assert any(dy > 0 for _, dy in samples)
    assert any(dy < 0 for _, dy in samples)


def _nfpa_reference_warp(latent, flow, mode="nearest"):
    import torch.nn.functional as F

    _, _, H, W = flow.size()
    _, _, h, w = latent.size()
    coords = torch.meshgrid(torch.arange(H, device=latent.device), torch.arange(W, device=latent.device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float().to(latent.dtype)[None].repeat(latent.shape[0], 1, 1, 1)
    coords_t0 = coords + flow.to(latent.device, latent.dtype)
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear", align_corners=False)
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))
    return F.grid_sample(latent, coords_t0, mode=mode, padding_mode="reflection", align_corners=False)


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_raven_paper_nfpa_gap_fill_matches_reference_helper(mode):
    torch.manual_seed(11)
    latent = torch.randn(1, 3, 16, 16)
    flow = create_nfpa_translation_flow(29, -31, height=128, width=128, device=latent.device, dtype=latent.dtype)
    expected = _nfpa_reference_warp(latent, flow, mode=mode)
    actual, metadata = raven_paper_nfpa_gap_fill_warp(
        latent, 29, -31, vae_scale_factor=8, sampling_mode=mode, return_metadata=True
    )
    assert torch.equal(actual, expected)
    assert metadata["transform_setting_name"] == RAVEN_PAPER_NFPA_GAP_FILL
    assert metadata["latent_sampling_mode"] == mode
    assert metadata["padding_mode"] == "reflection"
    assert metadata["normalization_formula"] == "x_norm = 2*x_pixel/W - 1; y_norm = 2*y_pixel/H - 1"


def test_raven_paper_nfpa_gap_fill_inverse_warp_direction():
    latent = torch.zeros(1, 1, 64, 64)
    latent[0, 0, 32, 32] = 1.0
    shifted = translate_latent(
        latent, 24, 24, shift_space="image_pixels", warp_mode="raven_paper_nfpa_gap_fill", padding_mode="reflection"
    )
    y, x = torch.nonzero(shifted[0, 0] == shifted.max(), as_tuple=True)
    assert int(x[0]) == 29
    assert int(y[0]) == 29


def test_raven_gap_fill_nearest_and_bilinear_only_change_sampling_mode():
    latent = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    nearest, nearest_meta = raven_paper_nfpa_gap_fill_warp(
        latent, 27, -30, sampling_mode="nearest", return_metadata=True
    )
    bilinear, bilinear_meta = raven_paper_nfpa_gap_fill_warp(
        latent, 27, -30, sampling_mode="bilinear", return_metadata=True
    )
    assert nearest.shape == bilinear.shape == latent.shape
    assert not torch.equal(nearest, bilinear)
    comparable = set(nearest_meta) - {"latent_sampling_mode"}
    for key in comparable:
        assert nearest_meta[key] == bilinear_meta[key]
