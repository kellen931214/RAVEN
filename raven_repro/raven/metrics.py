"""Protocol-correct metrics shared by RAVEN diagnostics and evaluation.

All watermark scores exposed here follow one convention: larger values mean
"more likely watermarked".  This keeps threshold calibration independent of
provider-specific score direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


SEMANTIC_METHODS = {"TR", "RID", "HSTR", "HSQR"}


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
        payload = asdict(self)
        return payload


def canonical_watermark_score(method: str, raw_score: float) -> float:
    """Convert a provider score to the common higher-means-watermark direction."""
    method = method.upper()
    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError(f"score must be finite, got {raw_score!r}")
    if method in SEMANTIC_METHODS:
        return -value
    if method == "GS":
        return value
    raise ValueError(f"Unsupported watermark method: {method}")


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


def normalize_bit_string(value: str, expected_length: int | None = None) -> str:
    bits = "".join(str(value).split())
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("bit strings must contain only 0 and 1")
    if expected_length is not None and len(bits) != expected_length:
        raise ValueError(f"expected {expected_length} bits, got {len(bits)}")
    return bits


def bit_accuracy(ground_truth: str, prediction: str, expected_length: int | None = None) -> dict:
    gt = normalize_bit_string(ground_truth, expected_length=expected_length)
    pred = normalize_bit_string(prediction, expected_length=expected_length)
    if len(gt) != len(pred):
        raise ValueError(f"bit-string length mismatch: {len(gt)} != {len(pred)}")
    errors = [index for index, (left, right) in enumerate(zip(gt, pred)) if left != right]
    return {
        "num_bits": len(gt),
        "num_errors": len(errors),
        "accuracy": 1.0 - len(errors) / len(gt),
        "error_indices": errors,
    }


def crop_overlap(first, second, dx: int, dy: int):
    """Crop arrays to correspondence under RAVEN's right/down convention.

    Positive ``dx`` moves content right and positive ``dy`` moves content down.
    """
    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    first = first[:height, :width]
    second = second[:height, :width]
    first_x0, first_x1 = max(0, -dx), width - max(0, dx)
    first_y0, first_y1 = max(0, -dy), height - max(0, dy)
    second_x0, second_x1 = max(0, dx), width - max(0, -dx)
    second_y0, second_y1 = max(0, dy), height - max(0, -dy)
    if first_x0 >= first_x1 or first_y0 >= first_y1:
        raise ValueError(f"shift ({dx}, {dy}) leaves no overlapping region for {width}x{height}")
    return (
        first[first_y0:first_y1, first_x0:first_x1],
        second[second_y0:second_y1, second_x0:second_x1],
    )


def _require_integer_flow(value: float, name: str) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1e-6:
        raise ValueError(f"{name} must be an integer image-pixel flow for overlap metrics, got {value}")
    return rounded


def crop_overlap_inverse_warp(reference, attacked, flow_dx_px: float, flow_dy_px: float):
    """Crop arrays to valid correspondence under inverse-warp flow.

    The convention is ``attacked[y, x]`` corresponds to
    ``reference[y + flow_dy_px, x + flow_dx_px]``. Only pixels with a real
    reference/attacked pair are retained; padding or reflected boundary values
    are excluded from quality metrics.
    """
    dx = _require_integer_flow(flow_dx_px, "flow_dx_px")
    dy = _require_integer_flow(flow_dy_px, "flow_dy_px")
    height = min(reference.shape[0], attacked.shape[0])
    width = min(reference.shape[1], attacked.shape[1])
    reference = reference[:height, :width]
    attacked = attacked[:height, :width]

    attacked_x0 = max(0, -dx)
    attacked_x1 = min(width, width - dx)
    attacked_y0 = max(0, -dy)
    attacked_y1 = min(height, height - dy)
    if attacked_x0 >= attacked_x1 or attacked_y0 >= attacked_y1:
        raise ValueError(f"flow ({dx}, {dy}) leaves no overlapping region for {width}x{height}")

    reference_x0 = attacked_x0 + dx
    reference_x1 = attacked_x1 + dx
    reference_y0 = attacked_y0 + dy
    reference_y1 = attacked_y1 + dy
    return (
        reference[reference_y0:reference_y1, reference_x0:reference_x1],
        attacked[attacked_y0:attacked_y1, attacked_x0:attacked_x1],
    )


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
            "flow_dx_px": float(flow_dx_px),
            "flow_dy_px": float(flow_dy_px),
        })
    return result


def summarize_numeric(values: Iterable[float]) -> dict:
    import numpy as np

    items = [float(value) for value in values]
    finite = [value for value in items if math.isfinite(value)]
    if not finite:
        return {
            "n": len(items), "mean": None, "median": None, "std": None,
            "min": None, "max": None, "nan_count": sum(math.isnan(value) for value in items),
            "inf_count": sum(math.isinf(value) for value in items),
        }
    return {
        "n": len(items),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(min(finite)),
        "max": float(max(finite)),
        "nan_count": sum(math.isnan(value) for value in items),
        "inf_count": sum(math.isinf(value) for value in items),
    }


def psnr(first, second, data_range: float = 1.0) -> float:
    import numpy as np

    mse = float(np.mean((first - second) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def mean_finite(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    if not items:
        raise ValueError("no finite values")
    return sum(items) / len(items)
