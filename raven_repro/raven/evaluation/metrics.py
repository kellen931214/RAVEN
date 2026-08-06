"""Protocol-correct metrics for RAVEN evaluation.

All watermark scores exposed here follow one convention: larger values mean
"more likely watermarked".  This keeps threshold calibration independent of
provider-specific score direction.

FID and CLIP are lazy-imported so ``import raven.evaluation.metrics`` does not
pull in diffusers, torch, open_clip, or cleanfid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
# Detection calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    target_fpr: float
    actual_fpr: float
    max_false_positives: int
    false_positives: int
    num_clean: int


@dataclass(frozen=True)
class DetectionSummary:
    calibration: ThresholdCalibration
    watermarked_tpr: float
    attacked_tpr: float
    num_watermarked: int
    num_attacked: int
    watermarked_auc: float
    attacked_auc: float

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_threshold(clean_scores: Sequence[float], target_fpr: float = 0.01) -> ThresholdCalibration:
    """Select the closest empirical FPR not exceeding ``target_fpr``.

    Detection uses ``score >= threshold``.  Tied clean scores are kept as one
    group; if including a tied group would exceed the false-positive budget,
    the group is excluded.  This makes the achieved FPR explicit and avoids a
    hidden tie-breaking dependency.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    values = [float(value) for value in clean_scores]
    if not values:
        raise ValueError("at least one clean score is required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("clean scores must all be finite")

    max_fp = math.floor(len(values) * target_fpr)
    descending = sorted(values, reverse=True)
    included = 0
    threshold = math.nextafter(descending[0], math.inf)
    index = 0
    while index < len(descending):
        value = descending[index]
        end = index + 1
        while end < len(descending) and descending[end] == value:
            end += 1
        if end > max_fp:
            break
        included = end
        threshold = value
        index = end

    actual = included / len(values)
    return ThresholdCalibration(
        threshold=threshold,
        target_fpr=target_fpr,
        actual_fpr=actual,
        max_false_positives=max_fp,
        false_positives=included,
        num_clean=len(values),
    )


def detection_rate(scores: Sequence[float], threshold: float) -> float:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("at least one positive score is required")
    return sum(value >= threshold for value in values) / len(values)


def roc_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute AUC as pairwise ranking probability, with ties worth one half."""
    positives = [float(value) for value in positive_scores]
    negatives = [float(value) for value in negative_scores]
    if not positives or not negatives:
        raise ValueError("AUC requires positive and negative scores")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def summarize_detection(
    clean_scores: Sequence[float],
    watermarked_scores: Sequence[float],
    attacked_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> DetectionSummary:
    calibration = calibrate_threshold(clean_scores, target_fpr=target_fpr)
    return DetectionSummary(
        calibration=calibration,
        watermarked_tpr=detection_rate(watermarked_scores, calibration.threshold),
        attacked_tpr=detection_rate(attacked_scores, calibration.threshold),
        num_watermarked=len(watermarked_scores),
        num_attacked=len(attacked_scores),
        watermarked_auc=roc_auc(watermarked_scores, clean_scores),
        attacked_auc=roc_auc(attacked_scores, clean_scores),
    )


# --------------------------------------------------------------------------- #
# Inverse-warp pair quality
# --------------------------------------------------------------------------- #
def inverse_warp_valid_bounds(
    height: int, width: int, flow_dx_px: float, flow_dy_px: float
) -> tuple[int, int, int, int]:
    """Return target bounds whose inverse-warp source coordinates are real."""
    dx, dy = float(flow_dx_px), float(flow_dy_px)
    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError(f"non-finite inverse-warp flow: ({dx}, {dy})")
    target_x0 = max(0, math.ceil(-dx))
    target_x1 = min(width, math.floor((width - 1) - dx) + 1)
    target_y0 = max(0, math.ceil(-dy))
    target_y1 = min(height, math.floor((height - 1) - dy) + 1)
    if target_x0 >= target_x1 or target_y0 >= target_y1:
        raise ValueError(
            f"flow ({dx}, {dy}) leaves no overlapping region for {width}x{height}"
        )
    return target_y0, target_y1, target_x0, target_x1


def sample_inverse_warp_reference(reference, flow_dx_px: float, flow_dy_px: float):
    """Sample reference at actual inverse-warp coordinates without padding."""
    import numpy as np

    source = np.asarray(reference)
    if source.ndim < 2:
        raise ValueError(f"reference must have at least two dimensions, got {source.shape}")
    height, width = source.shape[:2]
    y0, y1, x0, x1 = inverse_warp_valid_bounds(
        height, width, flow_dx_px, flow_dy_px
    )
    yy = np.arange(y0, y1, dtype=np.float64) + float(flow_dy_px)
    xx = np.arange(x0, x1, dtype=np.float64) + float(flow_dx_px)
    y_floor = np.floor(yy).astype(np.intp)
    x_floor = np.floor(xx).astype(np.intp)
    y_ceil = np.minimum(y_floor + 1, height - 1)
    x_ceil = np.minimum(x_floor + 1, width - 1)
    wy = yy - y_floor
    wx = xx - x_floor
    tail = (1,) * max(0, source.ndim - 2)
    wy = wy.reshape((-1, 1) + tail)
    wx = wx.reshape((1, -1) + tail)
    source = source.astype(np.float32, copy=False)
    top_left = source[y_floor[:, None], x_floor[None, :]]
    top_right = source[y_floor[:, None], x_ceil[None, :]]
    bottom_left = source[y_ceil[:, None], x_floor[None, :]]
    bottom_right = source[y_ceil[:, None], x_ceil[None, :]]
    sampled = (
        top_left * (1.0 - wy) * (1.0 - wx)
        + top_right * (1.0 - wy) * wx
        + bottom_left * wy * (1.0 - wx)
        + bottom_right * wy * wx
    )
    return sampled, (y0, y1, x0, x1)


def crop_overlap_inverse_warp(reference, attacked, flow_dx_px: float, flow_dy_px: float):
    """Crop valid inverse-warp correspondence using the effective source flow."""
    height = min(reference.shape[0], attacked.shape[0])
    width = min(reference.shape[1], attacked.shape[1])
    reference = reference[:height, :width]
    attacked = attacked[:height, :width]
    sampled_reference, (y0, y1, x0, x1) = sample_inverse_warp_reference(
        reference, flow_dx_px, flow_dy_px
    )
    return sampled_reference, attacked[y0:y1, x0:x1]


def rgb_float_array(image):
    """Return RGB float32 image data in [0, 1]."""
    import numpy as np

    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def pair_quality_metrics(reference, attacked, flow_dx_px: float | None = None, flow_dy_px: float | None = None) -> dict:
    """Compute raw full-image and optional inverse-warp overlap PSNR/SSIM."""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    reference_array = rgb_float_array(reference)
    attacked_array = rgb_float_array(attacked)
    height = min(reference_array.shape[0], attacked_array.shape[0])
    width = min(reference_array.shape[1], attacked_array.shape[1])
    reference_array = reference_array[:height, :width]
    attacked_array = attacked_array[:height, :width]
    result = {
        "raw_full_psnr": float(peak_signal_noise_ratio(reference_array, attacked_array, data_range=1.0)),
        "raw_full_ssim": float(structural_similarity(reference_array, attacked_array, data_range=1.0, channel_axis=-1)),
    }
    if flow_dx_px is not None and flow_dy_px is not None:
        reference_crop, attacked_crop = crop_overlap_inverse_warp(
            reference_array, attacked_array, flow_dx_px, flow_dy_px
        )
        result.update({
            "overlap_psnr": float(peak_signal_noise_ratio(reference_crop, attacked_crop, data_range=1.0)),
            "overlap_ssim": float(structural_similarity(reference_crop, attacked_crop, data_range=1.0, channel_axis=-1)),
            "valid_overlap_width": int(reference_crop.shape[1]),
            "valid_overlap_height": int(reference_crop.shape[0]),
            "valid_overlap_area_ratio": float(reference_crop.shape[0] * reference_crop.shape[1] / (height * width)),
            "overlap_protocol": "inverse_warp_valid_correspondence",
            "reference_sampling": (
                "direct_integer_effective_flow"
                if float(flow_dx_px).is_integer() and float(flow_dy_px).is_integer()
                else "bilinear_continuous_effective_flow"
            ),
            "flow_dx_px": float(flow_dx_px),
            "flow_dy_px": float(flow_dy_px),
        })
    return result


# --------------------------------------------------------------------------- #
# FID (lazy import — requires clean-fid)
# --------------------------------------------------------------------------- #
FID_PRIMARY_MODE = "legacy_tensorflow"
FID_SECONDARY_MODES: tuple[str, ...] = ("clean",)
FID_MODES: dict[str, str] = {
    "legacy_tensorflow": (
        "TF Inception-2015-12-05 pool3 features with TensorFlow-compatible "
        "bilinear resizing (original TensorFlow FID protocol)"
    ),
    "legacy_pytorch": (
        "pytorch-fid ported Inception-2015-12-05 weights with PIL bilinear resizing"
    ),
    "clean": (
        "clean-fid default: Inception-2015-12-05 features with clean-fid "
        "anti-aliased bicubic resizing"
    ),
}


def require_fid_mode(mode: str) -> str:
    """Fail closed on an unregistered FID mode instead of silently using another."""
    if mode not in FID_MODES:
        raise ValueError(f"unknown FID mode {mode!r}; known modes: {sorted(FID_MODES)}")
    return mode


def fid_protocol_descriptor(
    mode: str = FID_PRIMARY_MODE,
    secondary_modes: Sequence[str] = FID_SECONDARY_MODES,
) -> str:
    """Stable provenance string for the FID protocol actually used."""
    require_fid_mode(mode)
    secondary = [require_fid_mode(name) for name in secondary_modes if name != mode]
    text = f"clean-fid {mode} watermarked-vs-raven"
    if secondary:
        text += " (also recorded: " + ", ".join(sorted(secondary)) + ")"
    return text


def clean_fid(
    reference_dir: str | Path,
    attacked_dir: str | Path,
    device: str = "cuda",
    mode: str = FID_PRIMARY_MODE,
    secondary_modes: Sequence[str] = FID_SECONDARY_MODES,
) -> dict:
    """FID between two staged folders, primary value under the TF FID protocol."""
    import importlib.metadata
    from cleanfid import fid

    require_fid_mode(mode)
    modes = [mode, *[name for name in secondary_modes if name != mode]]
    values: dict[str, float] = {}
    for name in modes:
        values[require_fid_mode(name)] = float(
            fid.compute_fid(str(reference_dir), str(attacked_dir), device=device, mode=name)
        )
    return {
        "implementation": "clean-fid",
        "clean_fid_version": importlib.metadata.version("clean-fid"),
        "mode": mode,
        "primary_mode": mode,
        "secondary_modes": [name for name in modes if name != mode],
        "protocol": fid_protocol_descriptor(mode, secondary_modes),
        "feature_extractor": FID_MODES[mode],
        "mode_values": values,
        "mode_feature_extractors": {name: FID_MODES[name] for name in modes},
        "reference_dir": str(Path(reference_dir).resolve()),
        "attacked_dir": str(Path(attacked_dir).resolve()),
        "value": values[mode],
    }


# --------------------------------------------------------------------------- #
# CLIP (lazy import — requires open_clip_torch)
# --------------------------------------------------------------------------- #
def openclip_text_image_scores(
    image_paths: Sequence[str | Path],
    prompts: Sequence[str],
    device: str = "cuda",
    model_name: str = "ViT-bigG-14",
    pretrained: str = "laion2b_s39b_b160k",
) -> dict:
    if len(image_paths) != len(prompts) or not image_paths:
        raise ValueError("CLIP requires equally sized, non-empty image and prompt lists")
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    scores = []
    model.eval()
    with torch.no_grad():
        for path, prompt in zip(image_paths, prompts):
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores.append(float((image_features * text_features).sum().cpu().item()))
    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "metric": "prompt-image cosine similarity",
        "scores": scores,
        "mean": sum(scores) / len(scores),
    }
