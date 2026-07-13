"""LAB color and contrast transfer for RAVEN outputs."""

from __future__ import annotations

from PIL import Image


def _as_uint8_rgb(image):
    import numpy as np

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).round().astype(np.uint8)
    return arr


def color_contrast_transfer(generated_rgb, original_rgb):
    """Transfer original chroma and luminance statistics to a generated image.

    The generated image contributes luminance detail. The original image
    contributes LAB a/b chroma and target L mean/std.
    """
    try:
        import numpy as np
        from skimage import color
    except ImportError as exc:
        raise ImportError("color_contrast_transfer requires numpy and scikit-image") from exc

    generated = _as_uint8_rgb(generated_rgb).astype(np.float32) / 255.0
    original = _as_uint8_rgb(original_rgb).astype(np.float32) / 255.0
    if generated.shape != original.shape:
        raise ValueError(f"Generated and original shapes must match, got {generated.shape} and {original.shape}")

    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)

    gen_l = gen_lab[..., 0]
    orig_l = orig_lab[..., 0]
    gen_std = float(gen_l.std())
    orig_std = float(orig_l.std())
    if gen_std < 1e-6:
        matched_l = orig_l.mean() + (gen_l - gen_l.mean())
    else:
        matched_l = (gen_l - gen_l.mean()) * (orig_std / (gen_std + 1e-6)) + orig_l.mean()

    out_lab = gen_lab.copy()
    out_lab[..., 0] = np.clip(matched_l, 0.0, 100.0)
    out_lab[..., 1] = orig_lab[..., 1]
    out_lab[..., 2] = orig_lab[..., 2]

    out_rgb = color.lab2rgb(out_lab)
    out_uint8 = np.clip(out_rgb * 255.0, 0, 255).round().astype(np.uint8)
    return out_uint8


def color_contrast_transfer_pil(generated: Image.Image, original: Image.Image) -> Image.Image:
    return Image.fromarray(color_contrast_transfer(generated, original), mode="RGB")


def color_transfer_diagnostics(generated_rgb, original_rgb, output_rgb) -> dict:
    """Return low-cost numeric diagnostics without changing transfer behavior."""
    import numpy as np
    from skimage import color

    generated = _as_uint8_rgb(generated_rgb).astype(np.float32) / 255.0
    original = _as_uint8_rgb(original_rgb).astype(np.float32) / 255.0
    output = _as_uint8_rgb(output_rgb)
    generated_l = color.rgb2lab(generated)[..., 0]
    original_l = color.rgb2lab(original)[..., 0]
    saturated = np.any((output == 0) | (output == 255), axis=2)
    return {
        "generated_l_mean": float(generated_l.mean()),
        "generated_l_std": float(generated_l.std()),
        "original_l_mean": float(original_l.mean()),
        "original_l_std": float(original_l.std()),
        "output_saturated_pixel_ratio": float(saturated.mean()),
    }
