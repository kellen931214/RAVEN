"""Tree-Ring detector adapter.

Delegates to the canonical TR scoring in ``extract_verification_scores.py``.
Does NOT reimplement FFT / non-central chi-square / -log10(p) math.

All TR provider parameters MUST come from metadata.  Silent fallback to
defaults is forbidden — missing fields cause ``DetectorMissingStateError``.
Mixed provider configurations across records are rejected before scoring with
``DetectorStateValidationError``.  A uniform cohort constructs exactly one
provider.
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
    DetectorScoringError,
    DetectorStateValidationError,
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "w_seed",
    "w_channel",
    "w_radius",
    "w_pattern",
    "w_mask_shape",
    "w_measurement",
    "w_injection",
})

# Fields that must be uniform across all evaluated records when present.
# Mixed values cause ``DetectorStateValidationError``.
UNIFORM_IDENTITY_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "scheduler",
    "inverse_scheduler",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "provider_config_hash",
)

_extract_module = None


def _get_extract_module():
    global _extract_module
    if _extract_module is not None:
        return _extract_module
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    eb = str(repo / "eval_bench_wm")
    if eb not in sys.path:
        sys.path.insert(0, eb)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _extract_module = mod
    return mod


def describe_required_artifacts() -> list[str]:
    return [
        "TR provider parameters in source metadata: "
        "w_seed, w_channel, w_radius, w_pattern, w_mask_shape, w_measurement, w_injection",
        "Stable Diffusion pipe provider (pipe_utils.get_pipe_provider)",
    ]


def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load TR provider and pipe.  Raises on missing/bad state, never swallows.

    Validates every record — not just the first — so a mixed-key or
    mixed-profile cohort is rejected before any scoring happens.  A uniform
    cohort constructs exactly one provider.
    """
    import torch

    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.tr_provider import TrProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"TR dependencies not available: {exc}"
        ) from exc

    if not records:
        raise DetectorMissingStateError(
            "TR provider requires at least one record with metadata. "
            "All required fields: " + ", ".join(sorted(REQUIRED_METADATA_FIELDS))
        )

    # ---- validate every record has all required metadata fields ----
    for idx, record in enumerate(records):
        missing = sorted(f for f in REQUIRED_METADATA_FIELDS
                         if not str(record.get(f, "")).strip())
        if missing:
            raise DetectorMissingStateError(
                f"TR provider fields missing from metadata at record index "
                f"{idx} (run_id={record.get('run_id', '?')}): {missing}. "
                "All required: " + ", ".join(sorted(REQUIRED_METADATA_FIELDS))
            )

    # ---- validate uniform TR provider config via canonical helpers ----
    from raven.eval_protocol import provider_config_hash, canonical_json_hash

    provider_hashes: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records):
        try:
            h = provider_config_hash("TR", record)
            # Also compute the raw config dict for later provider construction
            cfg = _tr_provider_config(record)
            provider_hashes.setdefault(h, cfg)
        except (ValueError, TypeError) as exc:
            raise DetectorMissingStateError(
                f"TR provider config invalid at record index {idx} "
                f"(run_id={record.get('run_id', '?')}): {exc}"
            ) from exc

    if len(provider_hashes) != 1:
        raise DetectorStateValidationError(
            f"Mixed TR provider configurations in cohort: "
            f"{len(provider_hashes)} distinct configs across "
            f"{len(records)} records. All records must share the same "
            f"w_seed, w_channel, w_radius, w_pattern, w_mask_shape, "
            f"w_measurement, and w_injection."
        )

    config_hash, uniform_cfg = next(iter(provider_hashes.items()))

    # ---- validate uniform identity fields across all records ----
    for field in UNIFORM_IDENTITY_FIELDS:
        values: set[str] = set()
        for record in records:
            val = str(record.get(field, "")).strip()
            if val:
                values.add(val)
        if len(values) > 1:
            raise DetectorStateValidationError(
                f"Mixed {field} across TR cohort: {sorted(values)}. "
                f"All {len(records)} records must agree on detector identity "
                f"fields."
            )

    # ---- validate provider_config_hash recorded vs computed ----
    recorded_hashes: set[str] = set()
    for record in records:
        rh = str(record.get("provider_config_hash", "")).strip()
        if rh:
            recorded_hashes.add(rh)
    if recorded_hashes and recorded_hashes != {config_hash}:
        raise DetectorStateValidationError(
            f"Recorded provider_config_hash {sorted(recorded_hashes)} does not "
            f"match canonical computed hash {config_hash}"
        )

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
        latent_shape = pipe.get_latent_shape()

        provider = TrProvider(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device_obj,
            **uniform_cfg,
        )
    except DetectorMissingStateError:
        raise
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"TR provider construction failed: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"TR initialization error: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "provider_kwargs": uniform_cfg,
        "device_obj": device_obj,
        "provider_config_hash": config_hash,
    }


def _tr_provider_config(record: dict[str, Any]) -> dict[str, Any]:
    """Extract TR provider kwargs from one record with no implicit defaults.

    Every required field must be present and non-empty; raises ValueError
    otherwise.  This is the fail-closed counterpart of
    ``eval_protocol.provider_config`` for the TR-specific adapter.
    """
    return {
        "w_seed": int(record["w_seed"]),
        "w_channel": int(record["w_channel"]),
        "w_radius": int(record["w_radius"]),
        "w_pattern": str(record["w_pattern"]),
        "w_mask_shape": str(record["w_mask_shape"]),
        "w_measurement": str(record["w_measurement"]),
        "w_injection": str(record["w_injection"]),
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using the canonical TR detection path.

    Delegates to ``extract_verification_scores.evaluate_image``.
    """
    import torch
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.is_file():
        raise DetectorMissingStateError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    pipe = provider_info["pipe"]
    mod = provider_info["extract_module"]

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    try:
        result = mod.evaluate_image(torch, provider, pipe, path, steps)
    except Exception as exc:
        raise DetectorScoringError(
            f"TR scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    raw = mod.raw_score("TR", result)
    canonical = mod.canonical_score("TR", raw, result)

    score: dict[str, Any] = {
        "raw_score": raw,
        "canonical_score": canonical,
    }
    diagnostics = result.get("p_value_diagnostics") or []
    if diagnostics:
        d = diagnostics[0]
        score["tr_log_p"] = d.get("log_p")
        score["tr_sigma"] = d.get("sigma")
        score["tr_lambda"] = d.get("lambda")
        score["tr_statistic"] = d.get("statistic")
        score["tr_df"] = d.get("df")
        score["tr_p_underflow"] = d.get("p_underflow", False)
    return score


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate TR detector rows across cohorts."""
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

    result: dict[str, Any] = {
        "method": "TR",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": [],
    }

    # Check required cohorts
    required = {"original_watermarked", "attacked_watermarked"}
    missing_cohorts = sorted(required - set(cohorts))
    result["missing_cohorts"] = missing_cohorts

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_comparison_operator": ">=",
            "original_clean_threshold": summary.calibration.threshold,
            "original_clean_target_fpr": summary.calibration.target_fpr,
            "original_clean_actual_fpr": summary.calibration.actual_fpr,
            "original_clean_false_positives": summary.calibration.false_positives,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr_at_original_threshold": summary.attacked_tpr,
            "watermarked_roc_auc": summary.watermarked_auc,
            "attacked_roc_auc": summary.attacked_auc,
            "attack_success_at_original_threshold": 1.0 - summary.attacked_tpr,
        }

    attacked_clean = cohorts.get("attacked_clean", [])
    if attacked_clean and clean:
        try:
            recal = summarize_detection(
                attacked_clean, watermarked, attacked, target_fpr=0.01)
            result["tr_recalibrated"] = {
                "recalibrated_metrics_available": True,
                "attacked_clean_count": len(attacked_clean),
                "attacked_clean_recalibrated_threshold": recal.calibration.threshold,
                "attacked_clean_target_fpr": recal.calibration.target_fpr,
                "attacked_clean_actual_fpr": recal.calibration.actual_fpr,
                "attacked_clean_false_positives": recal.calibration.false_positives,
                "attacked_watermarked_tpr_at_recalibrated_threshold": recal.attacked_tpr,
                "attack_success_at_recalibrated_threshold": 1.0 - recal.attacked_tpr,
                "recalibrated_roc_auc": recal.attacked_auc,
            }
        except Exception:
            result["tr_recalibrated"] = {
                "recalibrated_metrics_available": False,
            }
    else:
        result["tr_recalibrated"] = {
            "recalibrated_metrics_available": False,
        }

    return result
