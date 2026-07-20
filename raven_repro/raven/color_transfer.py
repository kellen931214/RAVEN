"""LAB color and contrast transfer for RAVEN outputs."""

from __future__ import annotations

from typing import Literal

from PIL import Image

ColorTransferMode = Literal["paper_exact_two_stage_aligned"]

PAPER_EXACT_TWO_STAGE_ALIGNED = "paper_exact_two_stage_aligned"


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
    effective_source_flow_dx_image_px: float,
    effective_source_flow_dy_image_px: float,
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
    dx = _integer_flow(
        effective_source_flow_dx_image_px, "effective_source_flow_dx_image_px"
    )
    dy = _integer_flow(
        effective_source_flow_dy_image_px, "effective_source_flow_dy_image_px"
    )
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
    result[target_y0:target_y1, target_x0:target_x1] = source
    return result, valid



def _paper_aligned_two_stage_transfer(
    gen_lab,
    orig_lab,
    eps: float,
    effective_source_flow_dx_image_px: float,
    effective_source_flow_dy_image_px: float,
):
    import numpy as np
    from skimage import color

    selected_chroma, valid = align_original_chroma_to_generated(
        orig_lab[..., 1:3], gen_lab[..., 1:3],
        effective_source_flow_dx_image_px, effective_source_flow_dy_image_px,
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
        "method": PAPER_EXACT_TWO_STAGE_ALIGNED,
        "L_c_mean": mu_c,
        "L_c_std": sigma_c,
        "L_final_before_clip_min": float(np.min(l_final)),
        "L_final_before_clip_max": float(np.max(l_final)),
        "L_final_after_clip_min": float(np.min(l_final_clipped)),
        "L_final_after_clip_max": float(np.max(l_final_clipped)),
        "intermediate_rgb_min": float(np.min(x_c_rgb)),
        "intermediate_rgb_max": float(np.max(x_c_rgb)),
        "effective_source_flow_dx_image_px": float(effective_source_flow_dx_image_px),
        "effective_source_flow_dy_image_px": float(effective_source_flow_dy_image_px),
        "effective_visual_shift_dx_image_px": -float(
            effective_source_flow_dx_image_px
        ),
        "effective_visual_shift_dy_image_px": -float(
            effective_source_flow_dy_image_px
        ),
        "alignment_formula": "generated[y,x] <- original[y+flow_dy,x+flow_dx]",
        "alignment_flow_source": "effective source flow from actual warp grid",
        "alignment_alpha": 1.0,
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
    gen_lab,
    orig_lab,
    eps: float,
    effective_source_flow_dx_image_px: float | None,
    effective_source_flow_dy_image_px: float | None,
):
    if (
        effective_source_flow_dx_image_px is None
        or effective_source_flow_dy_image_px is None
    ):
        raise ValueError(
            "paper_exact_two_stage_aligned requires effective source flow"
        )
    return _paper_aligned_two_stage_transfer(
        gen_lab,
        orig_lab,
        eps,
        effective_source_flow_dx_image_px,
        effective_source_flow_dy_image_px,
    )


def color_contrast_transfer(
    generated_rgb,
    original_rgb,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE_ALIGNED,
    eps: float = 1e-6,
    *,
    effective_source_flow_dx_image_px: float | None = None,
    effective_source_flow_dy_image_px: float | None = None,
):
    """Apply the sole supported aligned two-stage color transfer."""
    try:
        from skimage import color
    except ImportError as exc:
        raise ImportError(
            "color_contrast_transfer requires numpy and scikit-image"
        ) from exc
    if mode != PAPER_EXACT_TWO_STAGE_ALIGNED:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    if generated.shape != original.shape:
        raise ValueError(
            f"Generated and original shapes must match, got "
            f"{generated.shape} and {original.shape}"
        )
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)
    out_lab, _ = _compute_transfer(
        gen_lab,
        orig_lab,
        eps,
        effective_source_flow_dx_image_px,
        effective_source_flow_dy_image_px,
    )
    return _to_uint8_rgb(color.lab2rgb(out_lab))


def color_contrast_transfer_pil(
    generated: Image.Image,
    original: Image.Image,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE_ALIGNED,
    eps: float = 1e-6,
    *,
    effective_source_flow_dx_image_px: float | None = None,
    effective_source_flow_dy_image_px: float | None = None,
) -> Image.Image:
    return Image.fromarray(
        color_contrast_transfer(
            generated,
            original,
            mode=mode,
            eps=eps,
            effective_source_flow_dx_image_px=effective_source_flow_dx_image_px,
            effective_source_flow_dy_image_px=effective_source_flow_dy_image_px,
        ),
        mode="RGB",
    )


def color_transfer_diagnostics(
    generated_rgb,
    original_rgb,
    output_rgb=None,
    mode: ColorTransferMode = PAPER_EXACT_TWO_STAGE_ALIGNED,
    eps: float = 1e-6,
    *,
    effective_source_flow_dx_image_px: float | None = None,
    effective_source_flow_dy_image_px: float | None = None,
) -> dict:
    """Return diagnostics for effective-flow aligned color transfer."""
    import numpy as np
    from skimage import color
    if mode != PAPER_EXACT_TWO_STAGE_ALIGNED:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)
    out_lab, transfer_diag = _compute_transfer(
        gen_lab,
        orig_lab,
        eps,
        effective_source_flow_dx_image_px,
        effective_source_flow_dy_image_px,
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
        "final_output_L_mean_abs_error_vs_original": abs(
            final_l_stats["mean"] - original_l_stats["mean"]
        ),
        "final_output_L_std_abs_error_vs_original": abs(
            final_l_stats["std"] - original_l_stats["std"]
        ),
        "output_saturated_pixel_ratio": _saturated_pixel_ratio(output),
        "output_rgb_out_of_gamut_ratio_before_clip": _rgb_out_of_gamut_ratio(out_lab),
        "output_matches_computed_transfer": bool(np.array_equal(output, computed_output)),
        "generated_l_mean": float(l_opt.mean()),
        "generated_l_std": float(l_opt.std()),
        "original_l_mean": float(l_w.mean()),
        "original_l_std": float(l_w.std()),
    }
    for key in (
        "effective_source_flow_dx_image_px",
        "effective_source_flow_dy_image_px",
        "effective_visual_shift_dx_image_px",
        "effective_visual_shift_dy_image_px",
        "alignment_formula",
        "alignment_flow_source",
        "alignment_alpha",
        "valid_overlap_ratio",
        "mean_abs_a_difference_vs_generated",
        "mean_abs_b_difference_vs_generated",
        "mean_chroma_delta_e76_vs_generated",
    ):
        if key in transfer_diag:
            result[key] = transfer_diag[key]
    return result
