"""LAB color and contrast transfer for RAVEN outputs."""

from __future__ import annotations

from typing import Literal

from PIL import Image

from .metrics import inverse_warp_valid_bounds, sample_inverse_warp_reference

ColorTransferMode = Literal[
    "paper_exact_two_stage",
    "paper_exact_two_stage_aligned",
]

PAPER_EXACT_TWO_STAGE = "paper_exact_two_stage"
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


def align_original_chroma_to_generated(
    original_chroma,
    generated_chroma,
    effective_source_flow_dx_image_px: float,
    effective_source_flow_dy_image_px: float,
):
    """Align original LAB a/b with actual inverse-warp effective flow.

    Fractional effective flow is sampled bilinearly from real source pixels;
    invalid targets retain generated chroma so reflected padding never enters
    color transfer.
    """
    import numpy as np

    original = np.asarray(original_chroma, dtype=np.float32)
    generated = np.asarray(generated_chroma, dtype=np.float32)
    if original.shape != generated.shape or original.ndim != 3 or original.shape[2] != 2:
        raise ValueError(
            "original_chroma and generated_chroma must have matching HxWx2 shapes, "
            f"got {original.shape} and {generated.shape}"
        )
    sampled, (y0, y1, x0, x1) = sample_inverse_warp_reference(
        original,
        effective_source_flow_dx_image_px,
        effective_source_flow_dy_image_px,
    )
    result = generated.copy()
    valid = np.zeros(generated.shape[:2], dtype=bool)
    valid[y0:y1, x0:x1] = True
    result[y0:y1, x0:x1] = sampled
    return result, valid



def _paper_two_stage_transfer(
    gen_lab,
    orig_lab,
    selected_chroma,
    eps: float,
    *,
    method: ColorTransferMode,
    metadata: dict,
):
    """Apply the two CIELAB stages stated in RAVEN Sec. 4.2.4.

    Paper stage 1 (color): x_c = F_RGB(L_a, a_o, b_o), where L_a is
    the attacked/view-guided luminance and a_o,b_o are original chroma.
    Paper stage 2 (contrast): after converting x_c back to LAB,
    L' = sigma(L_o) / sigma(L_c) * (L_c - mu(L_c)) + mu(L_o),
    and the final RGB image is F_RGB(L', a_o, b_o).

    ``selected_chroma`` is exactly original a_o,b_o for paper-faithful mode.
    The aligned ablation supplies effective-flow-aligned original chroma.
    """
    import numpy as np
    from skimage import color

    l_a = gen_lab[..., 0]
    l_o = orig_lab[..., 0]
    intermediate_lab = np.empty_like(gen_lab, dtype=np.float32)
    intermediate_lab[..., 0] = l_a
    intermediate_lab[..., 1:3] = selected_chroma
    x_c_rgb = color.lab2rgb(intermediate_lab)
    x_c_lab = color.rgb2lab(x_c_rgb.astype(np.float32))
    l_c = x_c_lab[..., 0]
    mu_c, sigma_c = float(l_c.mean()), float(l_c.std())
    mu_o, sigma_o = float(l_o.mean()), float(l_o.std())
    l_final = (sigma_o / (sigma_c + eps)) * (l_c - mu_c) + mu_o
    l_final_clipped = np.clip(l_final, 0.0, 100.0)
    final_lab = np.empty_like(gen_lab, dtype=np.float32)
    final_lab[..., 0] = l_final_clipped
    final_lab[..., 1:3] = selected_chroma
    return final_lab, {
        "method": method,
        "protocol_classification": (
            "paper-faithful unaligned paper-exact color transfer"
            if method == PAPER_EXACT_TWO_STAGE
            else "effective-flow aligned color-transfer ablation"
        ),
        "paper_formula_stage_1": "x_c = F_RGB(L_a, a_o, b_o)",
        "paper_formula_stage_2": (
            "L_prime = sigma(L_o)/sigma(L_c) * "
            "(L_c - mu(L_c)) + mu(L_o)"
        ),
        "L_c_mean": mu_c,
        "L_c_std": sigma_c,
        "L_final_before_clip_min": float(np.min(l_final)),
        "L_final_before_clip_max": float(np.max(l_final)),
        "L_final_after_clip_min": float(np.min(l_final_clipped)),
        "L_final_after_clip_max": float(np.max(l_final_clipped)),
        "intermediate_rgb_min": float(np.min(x_c_rgb)),
        "intermediate_rgb_max": float(np.max(x_c_rgb)),
        **metadata,
    }


def _paper_exact_two_stage_transfer(gen_lab, orig_lab, eps: float):
    """Paper-faithful, unaligned transfer using original chroma pixel-for-pixel."""
    import numpy as np

    selected_chroma = orig_lab[..., 1:3]
    chroma_delta = selected_chroma - gen_lab[..., 1:3]
    return _paper_two_stage_transfer(
        gen_lab,
        orig_lab,
        selected_chroma,
        eps,
        method=PAPER_EXACT_TWO_STAGE,
        metadata={
            "alignment_formula": "none; a_o,b_o copied at identical pixel coordinates",
            "alignment_flow_source": "none",
            "alignment_interpolation": "none",
            "alignment_alpha": 0.0,
            "valid_overlap_ratio": 1.0,
            "mean_abs_a_difference_vs_generated": float(
                np.abs(chroma_delta[..., 0]).mean()
            ),
            "mean_abs_b_difference_vs_generated": float(
                np.abs(chroma_delta[..., 1]).mean()
            ),
            "mean_chroma_delta_e76_vs_generated": float(
                np.linalg.norm(chroma_delta, axis=2).mean()
            ),
        },
    )


def _paper_aligned_two_stage_transfer(
    gen_lab,
    orig_lab,
    eps: float,
    effective_source_flow_dx_image_px: float,
    effective_source_flow_dy_image_px: float,
):
    import numpy as np

    selected_chroma, valid = align_original_chroma_to_generated(
        orig_lab[..., 1:3], gen_lab[..., 1:3],
        effective_source_flow_dx_image_px, effective_source_flow_dy_image_px,
    )
    chroma_delta = selected_chroma - gen_lab[..., 1:3]
    return _paper_two_stage_transfer(
        gen_lab,
        orig_lab,
        selected_chroma,
        eps,
        method=PAPER_EXACT_TWO_STAGE_ALIGNED,
        metadata={
            "effective_source_flow_dx_image_px": float(
                effective_source_flow_dx_image_px
            ),
            "effective_source_flow_dy_image_px": float(
                effective_source_flow_dy_image_px
            ),
            "effective_visual_shift_dx_image_px": -float(
                effective_source_flow_dx_image_px
            ),
            "effective_visual_shift_dy_image_px": -float(
                effective_source_flow_dy_image_px
            ),
            "alignment_formula": (
                "generated[y,x] <- original[y+flow_dy,x+flow_dx]"
            ),
            "alignment_flow_source": "effective source flow from actual warp grid",
            "alignment_interpolation": (
                "direct_integer_effective_flow"
                if float(effective_source_flow_dx_image_px).is_integer()
                and float(effective_source_flow_dy_image_px).is_integer()
                else "bilinear_continuous_effective_flow"
            ),
            "alignment_alpha": 1.0,
            "valid_overlap_ratio": float(valid.mean()),
            "mean_abs_a_difference_vs_generated": float(
                np.abs(chroma_delta[..., 0]).mean()
            ),
            "mean_abs_b_difference_vs_generated": float(
                np.abs(chroma_delta[..., 1]).mean()
            ),
            "mean_chroma_delta_e76_vs_generated": float(
                np.linalg.norm(chroma_delta, axis=2).mean()
            ),
        },
    )

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
    mode: ColorTransferMode,
    eps: float,
    effective_source_flow_dx_image_px: float | None,
    effective_source_flow_dy_image_px: float | None,
):
    if mode == PAPER_EXACT_TWO_STAGE:
        if (
            effective_source_flow_dx_image_px is not None
            or effective_source_flow_dy_image_px is not None
        ):
            raise ValueError(
                "paper_exact_two_stage is unaligned and must not receive effective flow"
            )
        return _paper_exact_two_stage_transfer(gen_lab, orig_lab, eps)
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
    """Apply paper-faithful unaligned or effective-flow aligned transfer."""
    try:
        from skimage import color
    except ImportError as exc:
        raise ImportError(
            "color_contrast_transfer requires numpy and scikit-image"
        ) from exc
    if mode not in {PAPER_EXACT_TWO_STAGE, PAPER_EXACT_TWO_STAGE_ALIGNED}:
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
        mode,
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
    """Return protocol and numeric diagnostics for the selected transfer."""
    import numpy as np
    from skimage import color
    if mode not in {PAPER_EXACT_TWO_STAGE, PAPER_EXACT_TWO_STAGE_ALIGNED}:
        raise ValueError(f"Unsupported color transfer mode: {mode}")
    generated = _as_float_rgb(generated_rgb)
    original = _as_float_rgb(original_rgb)
    gen_lab = color.rgb2lab(generated)
    orig_lab = color.rgb2lab(original)
    out_lab, transfer_diag = _compute_transfer(
        gen_lab,
        orig_lab,
        mode,
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
        "protocol_classification": transfer_diag["protocol_classification"],
        "paper_formula_stage_1": transfer_diag["paper_formula_stage_1"],
        "paper_formula_stage_2": transfer_diag["paper_formula_stage_2"],
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
