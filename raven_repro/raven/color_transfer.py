"""LAB color and contrast transfer for RAVEN outputs."""

from __future__ import annotations

from typing import Literal

from PIL import Image

ColorTransferMode = Literal[
    "paper_exact_two_stage",
    "paper_exact_two_stage_aligned",
    "paper_exact_two_stage_aligned_blend",
    "direct_stats",
]

PAPER_EXACT_TWO_STAGE = "paper_exact_two_stage"
PAPER_EXACT_TWO_STAGE_ALIGNED = "paper_exact_two_stage_aligned"
PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND = "paper_exact_two_stage_aligned_blend"


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


def _integer_flow(value: float, name: str) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1e-6:
        raise ValueError(f"{name} must be an integer image-pixel flow, got {value}")
    return rounded


def align_original_chroma_to_generated(
    original_chroma,
    generated_chroma,
    flow_dx_image_px: float,
    flow_dy_image_px: float,
    alpha: float = 1.0,
):
    """Align original LAB a/b to an inverse-warped generated view.

    The formal warp convention is generated[y, x] corresponds to
    original[y + flow_dy, x + flow_dx]. Only valid source coordinates are
    used. Non-overlap pixels retain generated chroma, so reflected padding is
    never treated as a real correspondence and no circular wrap can occur.
    """
    import numpy as np

    original = np.asarray(original_chroma, dtype=np.float32)
    generated = np.asarray(generated_chroma, dtype=np.float32)
    if original.shape != generated.shape or original.ndim != 3 or original.shape[2] != 2:
        raise ValueError(
            "original_chroma and generated_chroma must have matching HxWx2 shapes, "
            f"got {original.shape} and {generated.shape}"
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    dx = _integer_flow(flow_dx_image_px, "flow_dx_image_px")
    dy = _integer_flow(flow_dy_image_px, "flow_dy_image_px")
    height, width, _ = generated.shape
    target_x0 = max(0, -dx)
    target_x1 = min(width, width - dx)
    target_y0 = max(0, -dy)
    target_y1 = min(height, height - dy)
    if target_x0 >= target_x1 or target_y0 >= target_y1:
        raise ValueError(f"flow ({dx}, {dy}) leaves no overlap for {width}x{height}")
    source_x0, source_x1 = target_x0 + dx, target_x1 + dx
    source_y0, source_y1 = target_y0 + dy, target_y1 + dy

    result = generated.copy()
    valid = np.zeros((height, width), dtype=bool)
    valid[target_y0:target_y1, target_x0:target_x1] = True
    source = original[source_y0:source_y1, source_x0:source_x1]
    target = generated[target_y0:target_y1, target_x0:target_x1]
    result[target_y0:target_y1, target_x0:target_x1] = (
        (1.0 - float(alpha)) * target + float(alpha) * source
    )
    return result, valid


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



def _paper_aligned_two_stage_transfer(
    gen_lab,
    orig_lab,
    eps: float,
    flow_dx_image_px: float,
    flow_dy_image_px: float,
    alpha: float,
    method: str,
):
    import numpy as np
    from skimage import color

    selected_chroma, valid = align_original_chroma_to_generated(
        orig_lab[..., 1:3], gen_lab[..., 1:3],
        flow_dx_image_px, flow_dy_image_px, alpha=alpha,
    )
    l_opt = gen_lab[..., 0]
    l_w = orig_lab[..., 0]
    intermediate_lab = np.empty_like(gen_lab, dtype=np.float32)
    intermediate_lab[..., 0] = l_opt
    intermediate_lab[..., 1:3] = selected_chroma
    x_c_rgb = color.lab2rgb(intermediate_lab)
    x_c_lab = color.rgb2lab(x_c_rgb.astype(np.float32))
    l_c = x_c_lab[..., 0]
    mu_c, sigma_c = float(l_c.mean()), float(l_c.std())
    mu_w, sigma_w = float(l_w.mean()), float(l_w.std())
    l_final = (sigma_w / (sigma_c + eps)) * (l_c - mu_c) + mu_w
    l_final_clipped = np.clip(l_final, 0.0, 100.0)
    final_lab = np.empty_like(gen_lab, dtype=np.float32)
    final_lab[..., 0] = l_final_clipped
    final_lab[..., 1:3] = selected_chroma
    chroma_delta = selected_chroma - gen_lab[..., 1:3]
    return final_lab, {
        "method": method,
        "L_c_mean": mu_c,
        "L_c_std": sigma_c,
        "L_final_before_clip_min": float(np.min(l_final)),
        "L_final_before_clip_max": float(np.max(l_final)),
        "L_final_after_clip_min": float(np.min(l_final_clipped)),
        "L_final_after_clip_max": float(np.max(l_final_clipped)),
        "intermediate_rgb_min": float(np.min(x_c_rgb)),
        "intermediate_rgb_max": float(np.max(x_c_rgb)),
        "flow_dx_image_px": float(flow_dx_image_px),
        "flow_dy_image_px": float(flow_dy_image_px),
        "visual_shift_dx_image_px": -float(flow_dx_image_px),
        "visual_shift_dy_image_px": -float(flow_dy_image_px),
        "alignment_formula": "generated[y,x] <- original[y+flow_dy,x+flow_dx]",
        "alignment_alpha": float(alpha),
        "valid_overlap_ratio": float(valid.mean()),
        "mean_abs_a_difference_vs_generated": float(np.abs(chroma_delta[..., 0]).mean()),
        "mean_abs_b_difference_vs_generated": float(np.abs(chroma_delta[..., 1]).mean()),
        "mean_chroma_delta_e76_vs_generated": float(np.linalg.norm(chroma_delta, axis=2).mean()),
    }


def _rgb_out_of_gamut_ratio(lab) -> float:
    """Measure pre-clip sRGB gamut violations using skimage conversion."""
    import warnings
    import numpy as np
    from skimage import color
    from skimage.color.colorconv import rgb_from_xyz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xyz = color.lab2xyz(np.asarray(lab, dtype=np.float32))
    linear = xyz @ rgb_from_xyz.T.astype(xyz.dtype)
    encoded = linear.copy()
    mask = encoded > 0.0031308
    encoded[mask] = 1.055 * np.power(encoded[mask], 1.0 / 2.4) - 0.055
    encoded[~mask] *= 12.92
    return float(np.any((encoded < 0.0) | (encoded > 1.0), axis=2).mean())


def _compute_transfer(
    generated, gen_lab, orig_lab, mode: ColorTransferMode, eps: float,
    flow_dx_image_px: float | None, flow_dy_image_px: float | None, alpha: float,
):
    if mode == "direct_stats":
        return _direct_stats_transfer(gen_lab, orig_lab, eps)
    if mode == PAPER_EXACT_TWO_STAGE:
        # Preserve the formal baseline helper and arithmetic path exactly.
        return _paper_exact_two_stage_transfer(generated, gen_lab, orig_lab, eps)
    if flow_dx_image_px is None or flow_dy_image_px is None:
        raise ValueError(f"{mode} requires flow_dx_image_px and flow_dy_image_px")
    selected_alpha = 1.0 if mode == PAPER_EXACT_TWO_STAGE_ALIGNED else float(alpha)
    return _paper_aligned_two_stage_transfer(
        gen_lab, orig_lab, eps, flow_dx_image_px, flow_dy_image_px,
        selected_alpha, mode,
    )


def color_contrast_transfer(
    generated_rgb,
    original_rgb,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE,
    eps: float = 1e-6,
    *,
    flow_dx_image_px: float | None = None,
    flow_dy_image_px: float | None = None,
    alpha: float = 0.5,
):
    """Transfer chroma and paper two-stage luminance statistics.

    Baseline paper_exact_two_stage behavior is unchanged. Aligned modes use the
    formal inverse-warp correspondence and retain generated chroma outside
    valid overlap.
    """
    try:
        from skimage import color
    except ImportError as exc:
        raise ImportError("color_contrast_transfer requires numpy and scikit-image") from exc

    valid_modes = {
        PAPER_EXACT_TWO_STAGE,
        PAPER_EXACT_TWO_STAGE_ALIGNED,
        PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
        "direct_stats",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    if generated.shape != original.shape:
        raise ValueError(f"Generated and original shapes must match, got {generated.shape} and {original.shape}")
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)
    out_lab, _ = _compute_transfer(
        generated, gen_lab, orig_lab, mode, eps,
        flow_dx_image_px, flow_dy_image_px, alpha,
    )
    return _to_uint8_rgb(color.lab2rgb(out_lab))


def color_contrast_transfer_pil(
    generated: Image.Image,
    original: Image.Image,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE,
    eps: float = 1e-6,
    *,
    flow_dx_image_px: float | None = None,
    flow_dy_image_px: float | None = None,
    alpha: float = 0.5,
) -> Image.Image:
    return Image.fromarray(
        color_contrast_transfer(
            generated, original, mode=mode, eps=eps,
            flow_dx_image_px=flow_dx_image_px,
            flow_dy_image_px=flow_dy_image_px,
            alpha=alpha,
        ),
        mode="RGB",
    )


def color_transfer_diagnostics(
    generated_rgb,
    original_rgb,
    output_rgb=None,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE,
    eps: float = 1e-6,
    *,
    flow_dx_image_px: float | None = None,
    flow_dy_image_px: float | None = None,
    alpha: float = 0.5,
) -> dict:
    """Return numeric diagnostics for the selected color-transfer formula."""
    import numpy as np
    from skimage import color

    valid_modes = {
        PAPER_EXACT_TWO_STAGE,
        PAPER_EXACT_TWO_STAGE_ALIGNED,
        PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
        "direct_stats",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)
    out_lab, transfer_diag = _compute_transfer(
        generated, gen_lab, orig_lab, mode, eps,
        flow_dx_image_px, flow_dy_image_px, alpha,
    )
    computed_output = _to_uint8_rgb(color.lab2rgb(out_lab))
    output = _as_uint8_rgb(output_rgb) if output_rgb is not None else computed_output
    output_lab = color.rgb2lab(output.astype(np.float32) / 255.0)
    l_opt, l_w, output_l = gen_lab[..., 0], orig_lab[..., 0], output_lab[..., 0]
    final_l_stats = _lab_stats(output_l)
    original_l_stats = _lab_stats(l_w)
    result = {
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
        "output_rgb_out_of_gamut_ratio_before_clip": _rgb_out_of_gamut_ratio(out_lab),
        "output_matches_computed_transfer": bool(np.array_equal(output, computed_output)),
        "generated_l_mean": float(l_opt.mean()),
        "generated_l_std": float(l_opt.std()),
        "original_l_mean": float(l_w.mean()),
        "original_l_std": float(l_w.std()),
    }
    for key in (
        "flow_dx_image_px", "flow_dy_image_px",
        "visual_shift_dx_image_px", "visual_shift_dy_image_px",
        "alignment_formula", "alignment_alpha", "valid_overlap_ratio",
        "mean_abs_a_difference_vs_generated",
        "mean_abs_b_difference_vs_generated",
        "mean_chroma_delta_e76_vs_generated",
    ):
        if key in transfer_diag:
            result[key] = transfer_diag[key]
    return result
