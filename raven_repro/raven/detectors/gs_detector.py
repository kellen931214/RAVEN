"""Gaussian Shading detector adapter.

Delegates to the canonical GS provider and scoring path from
``extract_verification_scores.py``.  GS is per-sample — each source sample
has its own canonical ``GsProvider`` constructed via the official
``provider_kwargs`` helper.  Provider is never shared between rows; every
``score_image`` call binds a fresh provider to the resolved metadata fields
for that specific (run_id, role) pair.

Target/mask identity, secret provenance, protocol mode, and official
thresholds are all validated against the resolved source row before scoring.
No GS algorithm is reimplemented here — inversion and scoring delegate
entirely to the provider.
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
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_protocol_mode",
    "watermark_target_sha256",
    "watermark_mask_sha256",
})


def describe_required_artifacts() -> list[str]:
    return [
        "GS secret bundle directory",
        "gs_secret_index per row in source metadata",
        "gs_message_sha256, gs_key_sha256, gs_nonce_sha256",
        "gs_secret_bundle_sha256",
        "gs_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GS pipe and provider class.  Per-row provider construction happens
    in ``score_image`` via the canonical ``provider_kwargs`` helper."""
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
    """Score one GS image.  Constructs a canonical per-sample ``GsProvider``
    from the resolved metadata row, validates every provenance field against
    the provider's own identity, then delegates inversion and scoring to the
    provider.

    Failure classification (matching the canonical extraction path):

    * missing ``gs_secret_index`` or any provenance field in the resolved
      record → ``DetectorMissingStateError`` → ``failed_missing_required_state``
    * SHA / target / mask / protocol mismatch between the resolved record and
      the provider's own identity → ``DetectorStateValidationError`` →
      ``failed_state_validation``
    * runtime inversion or decoding failure → ``DetectorScoringError`` →
      ``failed_scoring``
    """
    import torch
    from PIL import Image, ImageOps
    from raven.eval_protocol import canonical_json_hash
    from raven.pairing_provenance import tensor_sha256

    _ensure_paths()
    from extract_verification_scores import provider_kwargs as _canonical_gs_kwargs

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    if record is None:
        raise DetectorMissingStateError(
            "GS requires per-row record with gs_secret_index"
        )

    # ---- canonical provider kwargs from the resolved source row ----
    gs_kwargs = _canonical_gs_kwargs("GS", record)
    secret_index = gs_kwargs.get("gs_secret_index")
    if secret_index is None:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: missing gs_secret_index"
        )

    pipe = provider_info["pipe"]
    GsProvider = provider_info["provider_class"]
    device_obj = provider_info["device_obj"]

    # ---- per-sample provider construction ----
    try:
        provider = GsProvider(
            latent_shape=pipe.get_latent_shape(),
            dtype=pipe.get_dtype(),
            device=device_obj,
            **gs_kwargs,
        )
    except (TypeError, ValueError) as exc:
        raise DetectorProviderInitializationError(
            f"GS provider construction failed: {type(exc).__name__}: {exc}"
        ) from exc

    # ---- secret provenance (matching extract_verification_scores.py) ----
    secret = provider.secret_provenance()

    # secret index identity
    recorded_index = int(secret_index)
    actual_index = int(secret.get("secret_index", -1))
    if recorded_index != actual_index:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GS secret_index mismatch: "
            f"recorded={recorded_index!r} actual={actual_index!r}"
        )

    # message, key, nonce, secret-bundle SHA identity
    for row_field, secret_field in (
        ("gs_message_sha256", "message_sha256"),
        ("gs_key_sha256", "key_sha256"),
        ("gs_nonce_sha256", "nonce_sha256"),
        ("gs_secret_bundle_sha256", "secret_bundle_sha256"),
    ):
        recorded = str(record.get(row_field, ""))
        actual = str(secret.get(secret_field, ""))
        if not recorded:
            raise DetectorMissingStateError(
                f"run_id={record.get('run_id')}: missing {row_field} in "
                "resolved metadata"
            )
        if recorded != actual:
            raise DetectorStateValidationError(
                f"run_id={record.get('run_id')}: GS {row_field} mismatch: "
                f"recorded={recorded!r} actual={actual!r}"
            )

    # protocol mode provenance
    recorded_protocol = str(record.get("gs_protocol_mode", ""))
    actual_protocol = str(getattr(provider, "gs_protocol_mode", ""))
    if not recorded_protocol:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: missing gs_protocol_mode in "
            "resolved metadata"
        )
    if recorded_protocol != actual_protocol:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GS protocol_mode mismatch: "
            f"recorded={recorded_protocol!r} actual={actual_protocol!r}"
        )

    # ---- watermark target identity (per-sample target tensor) ----
    source_target = str(record.get("watermark_target_sha256", ""))
    detector_target = tensor_sha256(provider.watermark_target_tensor())
    if not source_target:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: missing watermark_target_sha256 "
            "in resolved metadata"
        )
    if source_target != detector_target:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GS target SHA mismatch: "
            f"recorded={source_target!r} detector={detector_target!r}"
        )

    # ---- watermark mask identity (GS has no mask — canonical sentinel) ----
    source_mask = str(record.get("watermark_mask_sha256", ""))
    detector_mask = canonical_json_hash(
        {"method": "GS", "mask": "not_applicable", "version": 1}
    )
    if not source_mask:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: missing watermark_mask_sha256 "
            "in resolved metadata"
        )
    if source_mask != detector_mask:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GS mask SHA mismatch: "
            f"recorded={source_mask!r} detector={detector_mask!r}"
        )

    # ---- canonical inversion + scoring (no GS math reimplemented here) ----
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

    # ---- official outputs preserved verbatim ----
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
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_protocol_mode": actual_protocol,
        "watermark_target_sha256": detector_target,
        "watermark_mask_sha256": detector_mask,
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
