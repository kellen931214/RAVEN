"""LAB color and contrast transfer for RAVEN outputs."""

from __future__ import annotations

from typing import Literal

from PIL import Image

ColorTransferMode = Literal["paper_exact_two_stage", "direct_stats"]


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


def _as_float_rgb(image):
    return _as_uint8_rgb(image).astype("float32") / 255.0


def _to_uint8_rgb(rgb_float):
    import numpy as np

    return np.clip(rgb_float * 255.0, 0, 255).round().astype(np.uint8)


def _saturated_pixel_ratio(output_uint8) -> float:
    import numpy as np

    saturated = np.any((output_uint8 == 0) | (output_uint8 == 255), axis=2)
    return float(saturated.mean())


def _lab_stats(channel) -> dict:
    import numpy as np

    return {
        "mean": float(np.mean(channel)),
        "std": float(np.std(channel)),
        "min": float(np.min(channel)),
        "max": float(np.max(channel)),
    }


def _direct_stats_transfer(gen_lab, orig_lab, eps: float):
    import numpy as np

    gen_l = gen_lab[..., 0]
    orig_l = orig_lab[..., 0]
    gen_std = float(gen_l.std())
    orig_std = float(orig_l.std())
    if gen_std < eps:
        matched_l = orig_l.mean() + (gen_l - gen_l.mean())
    else:
        matched_l = (gen_l - gen_l.mean()) * (orig_std / (gen_std + eps)) + orig_l.mean()
    out_lab = gen_lab.copy()
    out_lab[..., 0] = np.clip(matched_l, 0.0, 100.0)
    out_lab[..., 1] = orig_lab[..., 1]
    out_lab[..., 2] = orig_lab[..., 2]
    return out_lab, {
        "method": "direct_stats",
        "L_final_before_clip_min": float(np.min(matched_l)),
        "L_final_before_clip_max": float(np.max(matched_l)),
        "L_final_after_clip_min": float(np.min(out_lab[..., 0])),
        "L_final_after_clip_max": float(np.max(out_lab[..., 0])),
        "L_c_mean": None,
        "L_c_std": None,
    }


def _paper_exact_two_stage_transfer(generated, gen_lab, orig_lab, eps: float):
    import numpy as np
    from skimage import color

    l_opt = gen_lab[..., 0]
    l_w = orig_lab[..., 0]

    intermediate_lab = np.empty_like(gen_lab, dtype=np.float32)
    intermediate_lab[..., 0] = l_opt
    intermediate_lab[..., 1] = orig_lab[..., 1]
    intermediate_lab[..., 2] = orig_lab[..., 2]

    # Keep the paper's two-stage behavior in float: LAB -> RGB -> LAB before
    # computing contrast statistics. Quantization is intentionally delayed.
    x_c_rgb = color.lab2rgb(intermediate_lab)
    x_c_lab = color.rgb2lab(x_c_rgb.astype(np.float32))
    l_c = x_c_lab[..., 0]

    mu_c = float(l_c.mean())
    sigma_c = float(l_c.std())
    mu_w = float(l_w.mean())
    sigma_w = float(l_w.std())
    l_final = (sigma_w / (sigma_c + eps)) * (l_c - mu_c) + mu_w
    l_final_clipped = np.clip(l_final, 0.0, 100.0)

    final_lab = np.empty_like(gen_lab, dtype=np.float32)
    final_lab[..., 0] = l_final_clipped
    final_lab[..., 1] = orig_lab[..., 1]
    final_lab[..., 2] = orig_lab[..., 2]
    return final_lab, {
        "method": "paper_exact_two_stage",
        "L_c_mean": mu_c,
        "L_c_std": sigma_c,
        "L_final_before_clip_min": float(np.min(l_final)),
        "L_final_before_clip_max": float(np.max(l_final)),
        "L_final_after_clip_min": float(np.min(l_final_clipped)),
        "L_final_after_clip_max": float(np.max(l_final_clipped)),
        "intermediate_rgb_min": float(np.min(x_c_rgb)),
        "intermediate_rgb_max": float(np.max(x_c_rgb)),
    }


def color_contrast_transfer(
    generated_rgb,
    original_rgb,
    mode: ColorTransferMode = "paper_exact_two_stage",
    eps: float = 1e-6,
):
    """Transfer original chroma and luminance statistics to a generated image.

    ``paper_exact_two_stage`` follows the RAVEN formula: use generated
    luminance with original a/b, convert LAB->RGB->LAB to get actual ``L_c``,
    then match ``L_c`` mean/std to the original watermarked luminance.

    ``direct_stats`` is the previous local implementation kept as an explicit
    ablation.
    """
    try:
        import numpy as np
        from skimage import color
    except ImportError as exc:
        raise ImportError("color_contrast_transfer requires numpy and scikit-image") from exc

    if mode not in {"paper_exact_two_stage", "direct_stats"}:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    if generated.shape != original.shape:
        raise ValueError(f"Generated and original shapes must match, got {generated.shape} and {original.shape}")

    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)

    if mode == "direct_stats":
        out_lab, _ = _direct_stats_transfer(gen_lab, orig_lab, eps)
    else:
        out_lab, _ = _paper_exact_two_stage_transfer(generated, gen_lab, orig_lab, eps)

    out_rgb = color.lab2rgb(out_lab)
    return _to_uint8_rgb(out_rgb)


def color_contrast_transfer_pil(
    generated: Image.Image,
    original: Image.Image,
    mode: ColorTransferMode = "paper_exact_two_stage",
    eps: float = 1e-6,
) -> Image.Image:
    return Image.fromarray(color_contrast_transfer(generated, original, mode=mode, eps=eps), mode="RGB")


def color_transfer_diagnostics(
    generated_rgb,
    original_rgb,
    output_rgb=None,
    mode: ColorTransferMode = "paper_exact_two_stage",
    eps: float = 1e-6,
) -> dict:
    """Return numeric diagnostics for the selected color-transfer formula."""
    import numpy as np
    from skimage import color

    if mode not in {"paper_exact_two_stage", "direct_stats"}:
        raise ValueError(f"Unsupported color transfer mode: {mode}")

    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)

    if mode == "direct_stats":
        out_lab, transfer_diag = _direct_stats_transfer(gen_lab, orig_lab, eps)
    else:
        out_lab, transfer_diag = _paper_exact_two_stage_transfer(generated, gen_lab, orig_lab, eps)

    computed_output = _to_uint8_rgb(color.lab2rgb(out_lab))
    output = _as_uint8_rgb(output_rgb) if output_rgb is not None else computed_output
    output_lab = color.rgb2lab(output.astype(np.float32) / 255.0)
    l_opt = gen_lab[..., 0]
    l_w = orig_lab[..., 0]
    output_l = output_lab[..., 0]

    final_l_stats = _lab_stats(output_l)
    original_l_stats = _lab_stats(l_w)
    return {
        "color_transfer_mode": mode,
        "eps": float(eps),
        "L_opt_mean": float(l_opt.mean()),
        "L_opt_std": float(l_opt.std()),
        "L_c_mean": transfer_diag["L_c_mean"],
        "L_c_std": transfer_diag["L_c_std"],
        "L_w_mean": float(l_w.mean()),
        "L_w_std": float(l_w.std()),
        "L_final_before_clip_min": transfer_diag["L_final_before_clip_min"],
        "L_final_before_clip_max": transfer_diag["L_final_before_clip_max"],
        "L_final_after_clip_min": transfer_diag["L_final_after_clip_min"],
        "L_final_after_clip_max": transfer_diag["L_final_after_clip_max"],
        "final_output_L_mean": final_l_stats["mean"],
        "final_output_L_std": final_l_stats["std"],
        "final_output_L_mean_abs_error_vs_original": abs(final_l_stats["mean"] - original_l_stats["mean"]),
        "final_output_L_std_abs_error_vs_original": abs(final_l_stats["std"] - original_l_stats["std"]),
        "output_saturated_pixel_ratio": _saturated_pixel_ratio(output),
        "output_matches_computed_transfer": bool(np.array_equal(output, computed_output)),
        # Backward-compatible names used by older diagnostics.
        "generated_l_mean": float(l_opt.mean()),
        "generated_l_std": float(l_opt.std()),
        "original_l_mean": float(l_w.mean()),
        "original_l_std": float(l_w.std()),
    }
