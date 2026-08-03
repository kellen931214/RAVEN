"""Gaussian Shading detector adapter.

Delegates to the canonical GS provider and scoring in
``extract_verification_scores.py``.  GS is per-sample — each row has its
own secret index.  Provider is constructed per-image from the record's
metadata fields.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "gs_secret_index",
    "gs_secret_bundle_sha256",
    "gs_protocol_mode",
})


def describe_required_artifacts() -> list[str]:
    return [
        "GS secret bundle directory",
        "gs_secret_index per row in source metadata",
        "gs_message_sha256, gs_key_sha256, gs_nonce_sha256",
        "gs_secret_bundle_sha256",
        "gs_protocol_mode",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GS pipe and provider class.  Per-row construction in score_image."""
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gs_provider import GsProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"GS dependencies not available: {exc}"
        ) from exc

    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path="RedbeardNZ/stable-diffusion-2-1-base",
            resolution=512,
            device=device_obj,
            eager_loading=False,
            schedulers_name="DDIM",
            disable_tqdm=True,
        )
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"GS pipe init failed: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "pipe": pipe,
        "provider_class": GsProvider,
        "device_obj": device_obj,
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one GS image.  Constructs provider per-row from the record's secret.

    ``record`` MUST contain gs_secret_index.  Without real secret bundles,
    this raises ``DetectorMissingStateError``.
    """
    import torch
    from PIL import Image, ImageOps
    from raven.eval_protocol import canonical_json_hash

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    if record is None:
        raise DetectorMissingStateError(
            "GS requires per-row record with gs_secret_index"
        )

    secret_index = record.get("gs_secret_index")
    if secret_index is None or not str(secret_index).strip():
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: missing gs_secret_index"
        )

    pipe = provider_info["pipe"]
    GsProvider = provider_info["provider_class"]
    device_obj = provider_info["device_obj"]

    try:
        provider = GsProvider(
            latent_shape=pipe.get_latent_shape(),
            dtype=pipe.get_dtype(),
            device=device_obj,
            offset=int(secret_index),
            gs_secret_index=int(secret_index),
        )
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"GS provider construction failed: {exc}"
        ) from exc

    # Validate secret provenance
    for field, src_field in (
        ("gs_message_sha256", "message_sha256"),
        ("gs_key_sha256", "key_sha256"),
        ("gs_nonce_sha256", "nonce_sha256"),
        ("gs_secret_bundle_sha256", "secret_bundle_sha256"),
    ):
        recorded = str(record.get(field, ""))
        if hasattr(provider, "secret_provenance"):
            secret = provider.secret_provenance()
            actual = str(secret.get(src_field, ""))
            if recorded and recorded != actual:
                raise DetectorStateValidationError(
                    f"run_id={record.get('run_id')}: GS {field} mismatch: "
                    f"recorded={recorded!r} actual={actual!r}"
                )

    # Validate target/mask SHA
    source_target = str(record.get("watermark_target_sha256", ""))
    source_mask = str(record.get("watermark_mask_sha256", ""))

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    try:
        with torch.no_grad():
            inversion = provider.invert_images(
                image, pipe_provider_target=pipe, num_inference_steps=steps,
            )
            result = provider.get_accuracies(inversion["zT_torch"])
    except Exception as exc:
        raise DetectorScoringError(
            f"GS scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    bit_accuracy = float(result.get("bit_accuracies", [0])[0])
    decoded_str = result.get("message_bits_str_list", [""])[0]
    decoded_sha = __import__("hashlib").sha256(
        decoded_str.encode("ascii")
    ).hexdigest()

    thresholds = provider.official_thresholds()
    return {
        "raw_score": bit_accuracy,
        "canonical_score": bit_accuracy,
        "bit_accuracy": bit_accuracy,
        "decoded_bits_sha256": decoded_sha,
        "gs_secret_index": int(secret_index),
        "gs_secret_bundle_sha256": record.get("gs_secret_bundle_sha256", ""),
        "gs_official_tau_onebit": thresholds["tau_onebit"],
        "gs_official_tau_bits": thresholds["tau_bits"],
        "score_direction": "higher_is_watermarked",
    }


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate GS detector rows."""
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != ROW_STATUS_SCORED:
            continue
        cohort = row.get("evaluation_cohort", "")
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored

    required = {"original_watermarked", "attacked_watermarked"}
    missing = sorted(required - set(cohorts))

    result: dict[str, Any] = {
        "method": "GS",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_direction": "higher_is_watermarked",
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
