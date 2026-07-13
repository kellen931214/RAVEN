import pytest

torch = pytest.importorskip("torch")

from raven.warp import sample_translation, translate_latent


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


@pytest.mark.parametrize("padding_mode", ["reflection", "border", "zeros"])
def test_translate_latent_preserves_shape_and_finite(padding_mode):
    latents = torch.randn(2, 4, 64, 64)
    shifted = translate_latent(latents, dx=24, dy=24, padding_mode=padding_mode)
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
    assert shifted_24[0, 0, 27, 27].item() == pytest.approx(1.0)
    assert shifted_32[0, 0, 26, 26].item() == pytest.approx(1.0)
