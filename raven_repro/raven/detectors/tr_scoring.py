"""Package-local Tree-Ring scoring.

Default protocol (issue #28):

    score_definition       = complex_l1_mean
    raw_score              = torch.abs(decoded_watermark - target_watermark).mean()
    raw_score_direction    = lower_is_watermarked
    canonical_score        = -raw_score
    canonical_score_direction = higher_is_watermarked
    comparison_operator    = >=

The complex L1 mean is computed in FFT space over the masked watermark
positions, exactly as the legacy ``raven_nfpa_tr_eval.complex_l1_score`` did:

- image decode (EXIF transpose, RGB)
- DDIM inversion to the latent ``zT``
- ``fftshift(fft2(zT))`` with the canonical dim convention (-1, -2)
- ``decoded = latent_fft[watermarking_mask[0]].flatten()``
- ``target  = gt_patch[0][watermarking_mask[0]].flatten()``
- distance = ``torch.abs(decoded - target)``, score = mean

A p-value protocol (``-log10(p)``) is retained ONLY as an explicitly named
optional mode (``score_mode="log10p"``) with separate provenance fields.  It
is never the default; production scoring must call ``score_complex_l1`` unless
metadata explicitly names the p-value mode.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Protocol constants — the single source of truth for TR score semantics
# ---------------------------------------------------------------------------
SCORE_DEFINITION = "complex_l1_mean"
RAW_SCORE_DIRECTION = "lower_is_watermarked"
CANONICAL_SCORE_DIRECTION = "higher_is_watermarked"
COMPARISON_OPERATOR = ">="

# Explicitly named optional p-value protocol with separate provenance.
LOG10P_MODE = "log10p"
SUPPORTED_SCORE_MODES: frozenset[str] = frozenset({SCORE_DEFINITION, LOG10P_MODE})


def canonical_score(raw_score: float) -> float:
    """Complex-L1 canonical score: higher-is-watermarked means -raw."""
    return -float(raw_score)


def log10p_canonical(raw_p_value: float) -> float:
    """-log10(p) canonical score for the explicitly named p-value mode."""
    raw = float(raw_p_value)
    return -math.log10(max(raw, sys.float_info.min))


def decode_image(path: Path) -> Image.Image:
    """Decode an image the canonical way (EXIF transpose + RGB)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def invert_image(torch, provider, pipe, image: Image.Image, steps: int):
    """Invert *image* to its latent with the provider/pipe canonical path."""
    if hasattr(provider, "invert_images"):
        inversion = provider.invert_images(
            image, pipe_provider_target=pipe, num_inference_steps=steps)
    else:
        inversion = pipe.invert_images(image, num_inference_steps=steps)
    return inversion["zT_torch"]


def watermark_vectors(provider):
    """Return (mask, target) in FFT space, complex, no real/imag split.

    ``mask`` is ``provider.watermarking_mask[0]`` and ``target`` is
    ``provider.gt_patch[0][mask].flatten()`` — exactly the vectors the
    canonical complex-L1 formula compares against.
    """
    mask = provider.watermarking_mask[0]
    target = provider.gt_patch[0][mask].flatten()
    return mask, target


def complex_l1_score(torch, provider, pipe, path: Path, steps: int) -> dict:
    """Score one image with the default complex-L1 protocol.

    Returns a dict with ``score`` (raw L1 mean) plus diagnostic magnitudes
    (``decoded_abs_mean``, ``target_abs_mean``) and finite flags.  Raises
    ``FileNotFoundError`` when the image is missing.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    image = decode_image(path)
    with torch.no_grad():
        recovered = invert_image(torch, provider, pipe, image, steps)
        recovered_fft = torch.fft.fftshift(
            torch.fft.fft2(recovered), dim=(-1, -2))
        mask, target = watermark_vectors(provider)
        scores = []
        decoded_norms = []
        target_norms = []
        for latent_fft in recovered_fft:
            decoded = latent_fft[mask].flatten()
            distance = torch.abs(decoded - target)
            scores.append(float(distance.mean().detach().cpu().item()))
            decoded_norms.append(
                float(torch.abs(decoded).mean().detach().cpu().item()))
            target_norms.append(
                float(torch.abs(target).mean().detach().cpu().item()))
    return {
        "score": scores[0],
        "decoded_abs_mean": decoded_norms[0],
        "target_abs_mean": target_norms[0],
        "nan": not math.isfinite(scores[0]) or math.isnan(scores[0]),
        "inf": math.isinf(scores[0]),
    }


def evaluate_log10p(torch, provider, pipe, path: Path, steps: int) -> dict:
    """Explicitly named optional p-value protocol (never the default).

    Runs the canonical inversion, then the provider's non-central chi-square
    p-value test (``get_accuracies``) and attaches the same diagnostics the
    legacy extraction path produced.  Score semantics are identical to the
    legacy path: ``raw_score = p_values[0]``, canonical ``-log10(p)``.
    """
    import scipy.stats

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    image = decode_image(path)
    with torch.no_grad():
        recovered = invert_image(torch, provider, pipe, image, steps)
        result = provider.get_accuracies(recovered)
        if provider.get_wm_type() == "TR":
            diagnostics = []
            recovered_fft = torch.fft.fftshift(
                torch.fft.fft2(recovered), dim=(-1, -2))
            mask = provider.watermarking_mask[0]
            target = provider.gt_patch[0][mask].flatten()
            target = torch.concatenate([target.real, target.imag])
            for latent_fft in recovered_fft:
                observed = latent_fft[mask].flatten()
                observed = torch.concatenate([observed.real, observed.imag])
                sigma = observed.std()
                noncentrality = (
                    target.square() / sigma.square()).sum().item()
                statistic = (
                    ((observed - target) / sigma).square()).sum().item()
                raw_log_p = float(scipy.stats.ncx2.logcdf(
                    statistic, df=len(target), nc=noncentrality))
                p_value = float(result["p_values"][len(diagnostics)])
                p_underflow = (
                    p_value == 0.0 or not math.isfinite(raw_log_p))
                log_p = (
                    raw_log_p if math.isfinite(raw_log_p)
                    else math.log(sys.float_info.min))
                diagnostics.append({
                    "log_p": log_p,
                    "sigma": float(sigma.item()),
                    "lambda": noncentrality,
                    "statistic": statistic,
                    "df": len(target),
                    "p_underflow": p_underflow,
                })
            result["p_value_diagnostics"] = diagnostics
    return result


def raw_log10p_score(result: dict) -> float:
    """Raw p-value for the optional log10p mode."""
    return float(result["p_values"][0])
