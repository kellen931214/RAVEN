"""GaussMarker detector adapter.

Bind GM evaluation to the canonical persisted bundle and provenance.
Every row is validated through ``gm_bundle_manifest`` and
``gm_provider_kwargs`` from ``extract_verification_scores.py`` before
any provider is constructed.  Mixed bundles, missing provenance, and
protocol/profile mismatches all fail closed.

``gm_protocol_mode`` and ``gm_profile`` are distinct concepts:
*protocol* names the shared-clean evaluation protocol (e.g.
``GM_SHARED_TR_CLEAN_MODE``); *profile* names the GmProvider bundle
configuration (e.g. ``legacy``).  They are validated independently
and never compared to each other.
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

# ---------------------------------------------------------------------------
# Required metadata — every row MUST carry these (non-empty, non-whitespace).
# ---------------------------------------------------------------------------
_GM_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_m_sha256",
    "gm_watermark_sha256",
    "gm_target_sha256",
    "gm_protocol_mode",
    "watermark_target_sha256",
    "watermark_mask_sha256",
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(_GM_REQUIRED_METADATA_FIELDS)

# ---------------------------------------------------------------------------
# Canonical provider-kwargs identity — every row must produce the same set.
# ---------------------------------------------------------------------------
_CANONICAL_KWARGS_FIELDS: tuple[str, ...] = (
    "gm_profile",
    "gm_bundle_dir",
    "gm_torch_dtype",
    "gm_channel_copy",
    "gm_w_copy",
    "gm_h_copy",
    "gm_watermark_bits_seed",
    "gm_use_gnr",
    "gm_gnr_path",
    "gm_use_classifier",
    "gm_classifier_path",
    "modelid_target",
    "model_revision",
    "scheduler_target",
    "resolution",
    "w_seed",
    "w_channel",
    "w_pattern",
    "w_mask_shape",
    "w_radius",
    "w_measurement",
    "w_injection",
)

# ---------------------------------------------------------------------------
# Required scorer-output field names.
# ---------------------------------------------------------------------------
_REQUIRED_SCORER_OUTPUTS: tuple[str, ...] = (
    "gm_raw_bit_accuracy",
    "gm_raw_ring_l1",
    "gm_report_label",
    "gm_score_definition",
    "gm_threshold_source",
    "gm_comparison_operator",
)


def describe_required_artifacts() -> list[str]:
    return [
        "gm_bundle_dir (directory with manifest.json, w1.pth, w2.pth)",
        "gm_bundle_config_sha256, gm_w1_file_sha256, gm_w2_file_sha256",
        "gm_watermark_sha256, gm_m_sha256, gm_target_sha256, gm_mask_sha256",
        "gm_protocol_mode",
        "Stable Diffusion inversion pipe",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_extract_module():
    """Import ``extract_verification_scores.py`` as a module (canonical helpers)."""
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores_gm",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate_required_gm_metadata(record: dict[str, Any]) -> None:
    """Every required field must be present, non-None, and non-whitespace.

    Raises ``DetectorMissingStateError`` on the first missing field.
    """
    run_id = str(record.get("run_id", "?"))
    for field in _GM_REQUIRED_METADATA_FIELDS:
        value = record.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise DetectorMissingStateError(
                f"run_id={run_id}: missing required GM metadata field "
                f"{field!r}"
            )


def _canonical_provider_identity(kwargs: dict[str, Any]) -> str:
    """Deterministic hash of the canonical provider configuration."""
    from raven.eval_protocol import canonical_json_hash

    payload: dict[str, Any] = {}
    for field in _CANONICAL_KWARGS_FIELDS:
        value = kwargs.get(field)
        if field in ("gm_bundle_dir", "gm_gnr_path", "gm_classifier_path"):
            if isinstance(value, str) and value:
                value = str(Path(value).resolve())
        payload[field] = value
    return canonical_json_hash(payload)


# ---------------------------------------------------------------------------
# Protocol validation (gm_protocol_mode ≠ gm_profile)
# ---------------------------------------------------------------------------

def _validate_gm_protocol_mode(record: dict[str, Any]) -> None:
    """Validate the row's ``gm_protocol_mode`` against the known shared-clean
    evaluation protocol.

    Protocol names the generation/evaluation contract the cohort follows;
    it is NOT the bundle profile name.
    """
    from raven.pairing_provenance import GM_SHARED_TR_CLEAN_MODE

    actual = str(record.get("gm_protocol_mode", ""))
    if actual != GM_SHARED_TR_CLEAN_MODE:
        raise DetectorStateValidationError(
            f"GM protocol mode {actual!r} does not match expected "
            f"{GM_SHARED_TR_CLEAN_MODE!r}"
        )


def _validate_gm_provider_profile(
    manifest: dict[str, Any],
    provider_kwargs: dict[str, Any],
    provider: Any | None = None,
) -> None:
    """Validate that manifest profile, provider kwargs profile, and actual
    provider profile identity are consistent.

    Profile names the GmProvider bundle configuration (e.g. ``legacy``);
    it is NOT the evaluation protocol.
    """
    manifest_profile = str(manifest.get("profile", ""))
    kwargs_profile = str(provider_kwargs.get("gm_profile", ""))

    if manifest_profile != kwargs_profile:
        raise DetectorStateValidationError(
            f"GM provider profile mismatch: manifest profile="
            f"{manifest_profile!r} but provider kwargs gm_profile="
            f"{kwargs_profile!r}"
        )

    if provider is not None:
        # Resolve the actual provider's profile from whichever attribute it exposes.
        actual_provider_profile = ""
        for attr in ("profile", "gm_profile", "profile_name"):
            val = getattr(provider, attr, None)
            if val is not None and str(val).strip():
                actual_provider_profile = str(val)
                break
        # Fall back to bundle manifest.
        if not actual_provider_profile and getattr(provider, "bundle", None) is not None:
            bundle_manifest = getattr(provider.bundle, "manifest", {})
            actual_provider_profile = str(bundle_manifest.get("profile", ""))

        if actual_provider_profile and actual_provider_profile != kwargs_profile:
            raise DetectorStateValidationError(
                f"GM provider profile mismatch: provider reports "
                f"{actual_provider_profile!r} but kwargs gm_profile="
                f"{kwargs_profile!r}"
            )


def _validate_bundle_files_exist(bundle_dir: str) -> Path:
    """Check the three bundle artifacts exist on disk.

    Returns the resolved ``Path``.  Missing files → ``DetectorMissingStateError``.
    """
    resolved = Path(bundle_dir).resolve()
    if not resolved.is_dir():
        raise DetectorMissingStateError(
            f"GM bundle directory not found: {resolved}"
        )
    for name in ("manifest.json", "w1.pth", "w2.pth"):
        if not (resolved / name).is_file():
            raise DetectorMissingStateError(
                f"GM bundle artifact missing: {resolved / name}"
            )
    return resolved


# ---------------------------------------------------------------------------
# Public detector contract
# ---------------------------------------------------------------------------

def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GM provider via canonical extract-script helpers.

    Every row is validated:
    1. Required metadata preflight (missing → ``DetectorMissingStateError``).
    2. Protocol mode validated independently.
    3. ``gm_bundle_manifest(row, run_id)`` — any exception after preflight
       → ``DetectorStateValidationError``.
    4. ``gm_provider_kwargs(row, run_id)`` — any exception
       → ``DetectorStateValidationError``.
    5. Provider profile validated across manifest / kwargs / actual provider.
    6. All rows must produce the same canonical provider identity.
    """
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gm_provider import GmProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"GM dependencies not available: {exc}"
        ) from exc

    if not records:
        raise DetectorMissingStateError("GM detector requires at least one record")

    # 1. Required metadata preflight on EVERY row.
    for row in records:
        _validate_required_gm_metadata(row)

    # 2. Validate protocol mode (independent of profile).
    for row in records:
        _validate_gm_protocol_mode(row)

    # 3. Load the canonical extraction module.
    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    # 4. Per-row canonical binding: gm_bundle_manifest + gm_provider_kwargs.
    row_bindings: list[dict[str, Any]] = []
    for row in records:
        run_id = str(row.get("run_id", "0"))

        # Bundle files must exist on disk.
        _validate_bundle_files_exist(str(row["gm_bundle_dir"]))

        # Any exception (not just RuntimeError) → StateValidationError
        # after the file-existence preflight has passed.
        try:
            bundle_dir_path, manifest = mod.gm_bundle_manifest(row, run_id)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"run_id={run_id}: GM bundle manifest validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            kwargs = mod.gm_provider_kwargs(row, run_id)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"run_id={run_id}: GM provider kwargs validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # 5. Validate provider profile (manifest ↔ kwargs, provider checked later).
        _validate_gm_provider_profile(manifest, kwargs)

        row_bindings.append({
            "run_id": run_id,
            "bundle_dir": str(bundle_dir_path),
            "manifest": manifest,
            "kwargs": kwargs,
        })

    # 6. All rows must share the same canonical provider identity.
    identities: set[str] = set()
    for binding in row_bindings:
        identities.add(_canonical_provider_identity(binding["kwargs"]))
    if len(identities) != 1:
        raise DetectorStateValidationError(
            f"GM cohort has mixed canonical provider identities: "
            f"{sorted(identities)}"
        )

    # 7. Construct pipe and provider from the validated canonical kwargs.
    first_kwargs = row_bindings[0]["kwargs"]
    first_bundle_dir = Path(row_bindings[0]["bundle_dir"])

    model_id = str(first_kwargs["modelid_target"])
    model_revision = str(first_kwargs["model_revision"])
    scheduler = str(first_kwargs["scheduler_target"])
    resolution = int(first_kwargs["resolution"])

    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            revision=model_revision,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name=scheduler,
            disable_tqdm=True,
        )
        latent_shape = pipe.get_latent_shape()

        provider = GmProvider(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device_obj,
            **first_kwargs,
        )
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"GM provider construction failed: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"GM provider/pipe construction failed: {type(exc).__name__}: {exc}"
        ) from exc

    # 8. Final profile validation against the actual constructed provider.
    _validate_gm_provider_profile(
        row_bindings[0]["manifest"], first_kwargs, provider=provider,
    )

    # 9. Require persisted bundle as the state source.
    if provider.bundle is None or getattr(provider, "state_source", "") != "bundle":
        raise DetectorStateValidationError(
            "GM provider requires persisted bundle; "
            f"state_source={getattr(provider, 'state_source', 'unknown')}"
        )

    # 10. Derive provider-side target and mask hashes.
    from raven.pairing_provenance import tensor_sha256

    if provider.gt_patch is None:
        raise DetectorStateValidationError(
            "GM provider has no gt_patch — cannot derive detector target hash"
        )
    provider_target_hash = tensor_sha256(provider.gt_patch.real.contiguous())

    if getattr(provider, "watermarking_mask", None) is None:
        raise DetectorStateValidationError(
            "GM provider has no watermarking_mask — cannot derive detector mask hash"
        )
    provider_mask_hash = tensor_sha256(provider.watermarking_mask.contiguous())

    # 11. Build verified provenance.
    first_manifest = row_bindings[0]["manifest"]
    from raven.pairing_provenance import GM_SHARED_TR_CLEAN_MODE

    verified_provenance: dict[str, Any] = {
        "gm_bundle_dir": str(first_bundle_dir),
        "gm_bundle_config_sha256": str(first_manifest.get("bundle_config_sha256", "")),
        "gm_w1_file_sha256": str(first_manifest.get("w1_file_sha256", "")),
        "gm_w2_file_sha256": str(first_manifest.get("w2_file_sha256", "")),
        "gm_m_sha256": str(first_manifest.get("m_sha256", "")),
        "gm_watermark_sha256": str(first_manifest.get("watermark_sha256", "")),
        "gm_target_sha256": str(first_manifest.get("w2_tensor_sha256", "")),
        "gm_protocol_mode": GM_SHARED_TR_CLEAN_MODE,
        "gm_profile": str(first_kwargs.get("gm_profile", "")),
        "gm_state_source": str(getattr(provider, "state_source", "bundle")),
    }

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "provider_target_hash": provider_target_hash,
        "provider_mask_hash": provider_mask_hash,
        "bundle_dir": str(first_bundle_dir),
        "verified_provenance": verified_provenance,
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using GaussMarker provider via canonical ``evaluate_image``.

    Requires a valid *record* with resolved source metadata.  Validates
    source target/mask against provider-derived values in four explicit
    stages; ``record=None`` raises ``DetectorMissingStateError``.
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(image_path)

    if record is None:
        raise DetectorMissingStateError(
            "GM scoring requires resolved source metadata (record=…)"
        )

    provider = provider_info["provider"]
    mod = provider_info["extract_module"]

    # ── Four-stage target/mask validation ──
    source_target = str(record.get("watermark_target_sha256", ""))
    source_mask = str(record.get("watermark_mask_sha256", ""))

    # Stage 1: source must declare target and mask.
    if not source_target:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: source watermark_target_sha256 "
            f"is missing or empty"
        )
    if not source_mask:
        raise DetectorMissingStateError(
            f"run_id={record.get('run_id')}: source watermark_mask_sha256 "
            f"is missing or empty"
        )

    # Stage 2: provider must have produced target and mask hashes.
    detector_target = provider_info.get("provider_target_hash", "")
    detector_mask = provider_info.get("provider_mask_hash", "")

    if not detector_target:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GM provider did not produce "
            f"a detector target hash"
        )
    if not detector_mask:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GM provider did not produce "
            f"a detector mask hash"
        )

    # Stage 3: source must equal detector.
    if source_target != detector_target:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GM detector/source "
            f"target SHA mismatch: source={source_target!r} "
            f"detector={detector_target!r}"
        )

    if source_mask != detector_mask:
        raise DetectorStateValidationError(
            f"run_id={record.get('run_id')}: GM detector/source "
            f"mask SHA mismatch: source={source_mask!r} "
            f"detector={detector_mask!r}"
        )

    # ── Canonical scoring path (single try boundary) ──
    try:
        result = mod.evaluate_image(torch, provider, provider_info["pipe"],
                                     path, steps)
        raw = float(mod.raw_score("GM", result))
        canonical = float(mod.canonical_score("GM", raw, result))

        if not math.isfinite(raw):
            raise ValueError("non-finite GM raw score")
        if not math.isfinite(canonical):
            raise ValueError("non-finite GM canonical score")

        # ── Validate required scorer outputs ──
        _validate_scorer_outputs(result)

    except Exception as exc:
        raise DetectorScoringError(
            f"GM scoring failed for {image_path}: {type(exc).__name__}: {exc}"
        ) from exc

    # ── Resolve GNR / classifier usage ──
    gnr_used = _resolve_gnr_classifier_usage(
        result, provider_info, kind="gnr",
    )
    classifier_used = _resolve_gnr_classifier_usage(
        result, provider_info, kind="classifier",
    )

    # ── Build score record with verified provenance ──
    score: dict[str, Any] = {
        "raw_score": raw,
        "canonical_score": canonical,
        "gm_raw_bit_accuracy": float(result["gm_raw_bit_accuracy"]),
        "gm_raw_ring_l1": float(result["gm_raw_ring_l1"]),
        "gm_restored_bit_accuracy": result.get("gm_restored_bit_accuracy"),
        "gm_classifier_probability": result.get("gm_classifier_probability"),
        "gm_report_label": str(result["gm_report_label"]),
        "gm_score_definition": str(result["gm_score_definition"]),
        "gm_threshold_source": str(result["gm_threshold_source"]),
        "gm_comparison_operator": str(result["gm_comparison_operator"]),
        "gm_gnr_used": gnr_used,
        "gm_classifier_used": classifier_used,
        "source_watermark_target_sha256": source_target,
        "detector_watermark_target_sha256": detector_target,
        "source_watermark_mask_sha256": source_mask,
        "detector_watermark_mask_sha256": detector_mask,
        "gm_target_verified": True,
        "gm_mask_verified": True,
    }

    score.update(provider_info.get("verified_provenance", {}))
    return score


# ---------------------------------------------------------------------------
# Scorer-output validation
# ---------------------------------------------------------------------------

def _validate_scorer_outputs(result: dict[str, Any]) -> None:
    """Every required scorer output must be present, correct type, and finite.

    Raises ``ValueError`` (→ ``DetectorScoringError``) on first violation.
    """
    for field in _REQUIRED_SCORER_OUTPUTS:
        value = result.get(field)
        if value is None:
            raise ValueError(
                f"required GM scorer output {field!r} is None"
            )
        if field in ("gm_raw_bit_accuracy", "gm_raw_ring_l1"):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"GM scorer output {field!r} has wrong type "
                    f"{type(value).__name__!r}, expected numeric"
                )
            fval = float(value)
            if not math.isfinite(fval):
                raise ValueError(
                    f"GM scorer output {field!r} is non-finite: {value!r}"
                )
            if field == "gm_raw_bit_accuracy" and not (0.0 <= fval <= 1.0):
                raise ValueError(
                    f"GM scorer output {field!r} out of [0,1]: {fval!r}"
                )
        elif field in ("gm_report_label", "gm_score_definition",
                        "gm_threshold_source", "gm_comparison_operator"):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(
                    f"GM scorer output {field!r} is empty or wrong type: "
                    f"{type(value).__name__!r}"
                )

    # Optional fields: if present, must be finite and (for accuracy/probability)
    # in [0, 1].
    for opt_field in ("gm_restored_bit_accuracy", "gm_classifier_probability"):
        opt_val = result.get(opt_field)
        if opt_val is not None:
            if not isinstance(opt_val, (int, float)):
                raise ValueError(
                    f"GM scorer output {opt_field!r} has wrong type "
                    f"{type(opt_val).__name__!r}, expected numeric or None"
                )
            fopt = float(opt_val)
            if not math.isfinite(fopt):
                raise ValueError(
                    f"GM scorer output {opt_field!r} is non-finite: {opt_val!r}"
                )
            if not (0.0 <= fopt <= 1.0):
                raise ValueError(
                    f"GM scorer output {opt_field!r} out of [0,1]: {fopt!r}"
                )


def _resolve_gnr_classifier_usage(
    result: dict[str, Any],
    provider_info: dict[str, Any],
    *,
    kind: str,
) -> bool:
    """Resolve whether GNR or classifier was actually used.

    1. Scorer-reported flags (``gm_used_gnr`` / ``gm_gnr_used`` etc.) take
       priority.
    2. If scorer reports nothing, fall back to the canonical provider kwargs
       (``gm_use_gnr`` / ``gm_use_classifier``).
    3. If scorer and provider config contradict, raise
       ``DetectorScoringError``.
    """
    if kind == "gnr":
        scorer_keys = ("gm_used_gnr", "gm_gnr_used")
        provider_key = "gm_use_gnr"
    else:
        scorer_keys = ("gm_used_classifier", "gm_classifier_used")
        provider_key = "gm_use_classifier"

    scorer_values = []
    for key in scorer_keys:
        val = result.get(key)
        if val is not None:
            scorer_values.append(bool(val))

    # Resolve provider-config value.
    verified = provider_info.get("verified_provenance", {})
    provider_value = verified.get(provider_key)
    if provider_value is None:
        # Fall back to the first row's kwargs stored in provider_info.
        provider_value = provider_info.get("_canonical_kwargs", {}).get(provider_key)

    if len(scorer_values) >= 1:
        declared = scorer_values[0]
        # Check all scorer-reported values agree.
        if len(set(scorer_values)) != 1:
            raise DetectorScoringError(
                f"GM scorer reported conflicting {kind} values: "
                f"{scorer_values}"
            )
        if provider_value is not None and declared != bool(provider_value):
            raise DetectorScoringError(
                f"GM {kind} usage contradiction: scorer reports "
                f"{declared} but provider config has {provider_value}"
            )
        return declared

    # No scorer output — fall back to provider config.
    if provider_value is not None:
        return bool(provider_value)
    return False


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate GM detector rows."""
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
        "method": "GM",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_type": "gm_raw_bit_accuracy",
        "score_direction": "higher_is_watermarked",
        "official_ensemble_threshold_available": False,
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_type": "empirical_clean_1pct_fpr",
            "threshold_comparison_operator": ">=",
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
