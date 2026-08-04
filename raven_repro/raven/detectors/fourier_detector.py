"""RID / HSTR / HSQR (Fourier) detector adapter.

Delegates to the canonical bundle loading and scoring in
``extract_verification_scores.py``.  Uses method-specific provider kwargs
helpers (rid_provider_kwargs_from_bundle, hstr_provider_kwargs_from_bundle,
hsqr_provider_from_bundle).

Method-specific state gates (Issue #24):
  - RID/HSTR: require persisted bundle AND ``state_source == "bundle"``.
  - HSQR: require valid persisted bundle WITHOUT imposing an unsupported
    ``state_source`` contract.

Canonical score = ``-raw_l1`` (higher is watermarked).

All failure classification is structural — never derived from exception
message substrings.
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

FOURIER_METHODS = frozenset({"RID", "HSTR", "HSQR"})

# ---------------------------------------------------------------------------
# Method-specific score definition labels (exact strings from the canonical
# extract_verification_scores.py, lines 891-895).
# ---------------------------------------------------------------------------
_METHOD_SCORE_DEFINITIONS: dict[str, str] = {
    "RID": "rid_neg_channel_min_complex_l1",
    "HSTR": "hstr_score=-min(channel_0_l1,channel_3_l1)",
    "HSQR": "hsqr_negative_mean_complex_l1_distance",
}

# ---------------------------------------------------------------------------
# Method-specific required metadata contract.
# Every field must exist AND be non-empty (after stripping) for every row.
# Missing → DetectorMissingStateError.
# ---------------------------------------------------------------------------
_METHOD_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "RID": (
        "method",
        "rid_bundle_dir",
        "rid_bundle_config_sha256",
        "rid_selected_pattern_sha256",
        "rid_mask_sha256",
        "rid_key_index",
        "rid_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
    ),
    "HSTR": (
        "method",
        "hstr_bundle_dir",
        "hstr_bundle_config_sha256",
        "hstr_selected_pattern_sha256",
        "hstr_mask_sha256",
        "hstr_key_index",
        "hstr_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
    ),
    "HSQR": (
        "method",
        "hsqr_bundle_dir",
        "hsqr_bundle_config_sha256",
        "hsqr_selected_pattern_sha256",
        "hsqr_mask_sha256",
        "hsqr_key_index",
        "hsqr_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
    ),
}

# Exported for introspection by the unified pipeline contract.
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(
    _METHOD_REQUIRED_FIELDS.get("RID", ())
)


def describe_required_artifacts() -> list[str]:
    return [
        "<prefix>_bundle_dir (directory with manifest.json)",
        "<prefix>_bundle_config_sha256, <prefix>_selected_pattern_sha256",
        "<prefix>_mask_sha256, <prefix>_key_index",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_extract_module():
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = repo / "raven_repro" / "scripts"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_verification_scores_fourier",
        scripts_dir / "extract_verification_scores.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Structured validation helpers — fail closed, never parse error messages
# ===========================================================================

def _validate_required_row_metadata(
    record: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Preflight: every required metadata field must be present and non-empty.

    Called BEFORE any canonical helper so missing fields are always
    ``DetectorMissingStateError``, never misclassified as state validation.
    """
    required = _METHOD_REQUIRED_FIELDS.get(method, ())
    for field in required:
        value = record.get(field)
        if value is None:
            raise DetectorMissingStateError(
                f"{row_label}: required metadata field {field!r} is None"
            )
        if not isinstance(value, str):
            value = str(value)
        if not value.strip():
            raise DetectorMissingStateError(
                f"{row_label}: required metadata field {field!r} is missing or empty"
            )


def _validate_method_tag(
    record: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Verify record method tag matches the evaluation method."""
    record_method = str(record.get("method", "")).upper()
    if record_method and record_method != method:
        raise DetectorStateValidationError(
            f"{row_label}: record method tag {record_method!r} does not match "
            f"evaluation method {method!r}"
        )


def _validate_manifest_method_identity(
    manifest: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Verify bundle manifest method identity matches evaluation method.

    Only checks when the manifest carries an explicit method/schema tag.
    Does not fabricate fields for schemas that lack them.
    """
    # RID bundles use rid_bundle schema
    schema = str(manifest.get("schema", manifest.get("bundle_schema", "")))
    if schema:
        rid_schemas = {"rid_bundle_v1", "rid_bundle_v2"}
        sfw_schemas = {"sfw_bundle_v1", "sfw_bundle_v2"}
        if method in {"RID"} and schema in sfw_schemas:
            raise DetectorStateValidationError(
                f"{row_label}: manifest schema {schema!r} is not a RID bundle"
            )
        if method in {"HSTR", "HSQR"} and schema in rid_schemas:
            raise DetectorStateValidationError(
                f"{row_label}: manifest schema {schema!r} is not an SFW bundle"
            )

    # SFW bundles carry a methods array; RID bundles do not
    manifest_methods = manifest.get("methods", manifest.get("watermark_methods"))
    if manifest_methods is not None:
        if isinstance(manifest_methods, list):
            if method not in manifest_methods:
                raise DetectorStateValidationError(
                    f"{row_label}: manifest methods {manifest_methods} do not "
                    f"include {method!r}"
                )
        elif isinstance(manifest_methods, str):
            if method != manifest_methods:
                raise DetectorStateValidationError(
                    f"{row_label}: manifest method {manifest_methods!r} != "
                    f"{method!r}"
                )


def _validate_key_index(
    record: dict[str, Any],
    method: str,
    manifest: dict[str, Any],
    kwargs: dict[str, Any],
    row_label: str,
) -> None:
    """Verify row key_index matches bundle manifest and provider kwargs."""
    prefix = method.lower()
    row_key = str(record.get(f"{prefix}_key_index", ""))

    # Canonical key identity from the manifest
    if method == "RID":
        manifest_key = str(manifest.get("selected_key_index", ""))
        kwargs_key = str(kwargs.get("rid_key_index", ""))
    elif method == "HSTR":
        manifest_key = str(manifest.get("selected_key_index", ""))
        kwargs_key = str(kwargs.get("hstr_key_index", ""))
    else:  # HSQR
        manifest_key = str(manifest.get("selected_key_index", ""))
        kwargs_key = ""  # HSQR doesn't use kwargs for key

    if manifest_key and row_key != manifest_key:
        raise DetectorStateValidationError(
            f"{row_label}: row {prefix}_key_index={row_key!r} does not match "
            f"manifest selected_key_index={manifest_key!r}"
        )
    if kwargs_key and row_key != kwargs_key:
        raise DetectorStateValidationError(
            f"{row_label}: row {prefix}_key_index={row_key!r} does not match "
            f"provider kwargs {prefix}_key_index={kwargs_key!r}"
        )


def _validate_protocol_profile(
    record: dict[str, Any],
    method: str,
    manifest: dict[str, Any],
    row_label: str,
) -> None:
    """Verify row protocol_mode matches the bundle's canonical profile identity."""
    prefix = method.lower()
    row_protocol = str(record.get(f"{prefix}_protocol_mode", ""))

    # Canonical profile identity from manifest
    profile_name = str(manifest.get("profile_name", ""))
    if profile_name and row_protocol != profile_name:
        raise DetectorStateValidationError(
            f"{row_label}: row {prefix}_protocol_mode={row_protocol!r} does not "
            f"match manifest profile_name={profile_name!r}"
        )


# ---------------------------------------------------------------------------
# Detector-side target / mask identity (never sourced from the record)
# ---------------------------------------------------------------------------
def _compute_detector_target_hash(provider, method: str, manifest: dict) -> str:
    """Compute the detector-side target hash — never from source metadata."""
    from raven.pairing_provenance import tensor_sha256

    if method in {"RID", "HSTR"}:
        if hasattr(provider, "selected_pattern_sha256") and provider.selected_pattern_sha256:
            return str(provider.selected_pattern_sha256)
        if manifest.get("selected_pattern_sha256"):
            return str(manifest["selected_pattern_sha256"])
        return tensor_sha256(provider.gt_patch)

    # HSQR: from bundle, never from record
    bundle = getattr(provider, "bundle", None)
    if bundle is not None and hasattr(bundle, "manifest"):
        return str(bundle.manifest.get("selected_pattern_sha256", ""))
    return ""


def _compute_detector_mask_hash(provider, method: str, manifest: dict) -> str:
    """Compute the detector-side mask hash — never from source metadata."""
    from raven.pairing_provenance import tensor_sha256

    if method in {"RID", "HSTR"}:
        if hasattr(provider, "watermark_mask_sha256") and provider.watermark_mask_sha256:
            return str(provider.watermark_mask_sha256)
        if manifest.get("mask_sha256"):
            return str(manifest["mask_sha256"])
        if hasattr(provider, "watermarking_mask"):
            return tensor_sha256(provider.watermarking_mask)
        if hasattr(provider, "watermark_region_mask_hstr"):
            return tensor_sha256(provider.watermark_region_mask_hstr)
        return ""

    # HSQR: canonical sources only — provider attribute, manifest, or
    # hsqr_center_slice_mask_sha256 helper.  NEVER fall back to the
    # source record's hsqr_mask_sha256 (that would be self-validation).
    if hasattr(provider, "watermark_mask_sha256") and provider.watermark_mask_sha256:
        return str(provider.watermark_mask_sha256)
    if manifest.get("mask_sha256"):
        return str(manifest["mask_sha256"])
    # Use canonical center-slice mask identity if available
    if hasattr(provider, "watermark_channels") and hasattr(provider, "start") and hasattr(provider, "end"):
        try:
            from raven.eval_protocol import canonical_json_hash
            return canonical_json_hash({
                "method": "HSQR",
                "mask_identity": "center_slice_protocol",
                "center_slice": [int(provider.start), int(provider.end)],
                "watermark_channels": [int(ch) for ch in provider.watermark_channels],
                "latent_shape": [int(dim) for dim in provider.latent_shape],
                "version": 1,
            })
        except Exception:
            pass
    # Fallback: hash the actual mask tensor from provider
    if hasattr(provider, "watermarking_mask"):
        return tensor_sha256(provider.watermarking_mask)
    return ""


def _validate_row_target_mask(
    provider,
    method: str,
    record: dict[str, Any],
    manifest: dict,
) -> None:
    """Validate source target/mask SHA against detector-computed values.

    Fail-closed: missing/empty source provenance → DetectorMissingStateError.
    Detector cannot derive identity → DetectorStateValidationError.
    Mismatch → DetectorStateValidationError.
    """
    row_id = str(record.get("run_id", "unknown"))
    source_target = str(record.get("watermark_target_sha256", "")).strip()
    source_mask = str(record.get("watermark_mask_sha256", "")).strip()

    # Source provenance is mandatory
    if not source_target:
        raise DetectorMissingStateError(
            f"{method}: missing watermark_target_sha256 for run_id={row_id}"
        )
    if not source_mask:
        raise DetectorMissingStateError(
            f"{method}: missing watermark_mask_sha256 for run_id={row_id}"
        )

    detector_target = _compute_detector_target_hash(provider, method, manifest)
    detector_mask = _compute_detector_mask_hash(provider, method, manifest)

    # Detector must be able to derive identities
    if not detector_target:
        raise DetectorStateValidationError(
            f"{method}: detector could not derive target identity for run_id={row_id}"
        )
    if not detector_mask:
        raise DetectorStateValidationError(
            f"{method}: detector could not derive mask identity for run_id={row_id}"
        )

    if source_target != detector_target:
        raise DetectorStateValidationError(
            f"{method}: detector/source target SHA mismatch for run_id={row_id}: "
            f"source={source_target!r} detector={detector_target!r}"
        )
    if source_mask != detector_mask:
        raise DetectorStateValidationError(
            f"{method}: detector/source mask SHA mismatch for run_id={row_id}: "
            f"source={source_mask!r} detector={detector_mask!r}"
        )


# ===========================================================================
# Public API
# ===========================================================================

def load_state(records: list[dict[str, Any]], device: str,
               method: str = "RID", **extra) -> dict[str, Any]:
    """Load Fourier provider via canonical bundle helpers.

    Validation order (structural — never derived from error messages):

    1. Method tag verification
    2. Required metadata preflight (every row)
    3. Bundle directory existence (path preflight)
    4. Canonical bundle manifest loading
    5. Manifest method identity check
    6. Provider construction via method-specific kwargs helper
    7. Method-specific state gates (state_source for RID/HSTR)
    8. Key index match against manifest/provider
    9. Protocol/profile match against manifest
    10. Cohort consistency (same bundle_dir/key_index/protocol across rows)
    """
    import torch

    _ensure_paths()
    method = method.upper()
    if method not in FOURIER_METHODS:
        raise ValueError(f"Unknown Fourier method: {method}")
    prefix = method.lower()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
    except ImportError as exc:
        raise DetectorDependencyError(
            f"Fourier dependencies not available: {exc}"
        ) from exc

    if not records:
        raise DetectorMissingStateError(
            f"{method}: no records provided to load_state"
        )

    # --- Step 1: method tag verification on every row ------------------------
    for i, record in enumerate(records):
        row_id = str(record.get("run_id", i))
        _validate_method_tag(record, method, f"{method} run_id={row_id}")

    # --- Step 2: required metadata preflight on every row -------------------
    for i, record in enumerate(records):
        row_id = str(record.get("run_id", i))
        _validate_required_row_metadata(record, method,
                                         f"{method} run_id={row_id}")

    # --- Step 3: path preflight — bundle directory must exist ---------------
    first = records[0]
    identifier = str(first.get("run_id", "0"))
    bundle_dir_str = str(first.get(f"{prefix}_bundle_dir", ""))
    bundle_dir = Path(bundle_dir_str)
    if not bundle_dir.is_dir():
        raise DetectorMissingStateError(
            f"{method}: {prefix}_bundle_dir does not exist or is not a "
            f"directory: {bundle_dir_str}"
        )
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DetectorMissingStateError(
            f"{method}: bundle manifest.json not found at {manifest_path}"
        )

    # --- Step 4: load extract module ----------------------------------------
    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    # --- Step 5: canonical bundle manifest loading --------------------------
    # At this point path preflight already passed, so any RuntimeError from
    # the canonical helper is a state validation failure (SHA/config mismatch).
    try:
        bundle_dir_path, manifest = mod.fourier_bundle_manifest(
            first, str(identifier), method)
    except Exception as exc:
        raise DetectorStateValidationError(
            f"{method} bundle validation failed for run_id={identifier}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # --- Step 6: manifest method identity check -----------------------------
    _validate_manifest_method_identity(manifest, method,
                                        f"{method} run_id={identifier}")

    # --- Step 7: provider construction --------------------------------------
    try:
        if method == "RID":
            kwargs = mod.rid_provider_kwargs_from_bundle(first, str(identifier))
        elif method == "HSTR":
            kwargs = mod.hstr_provider_kwargs_from_bundle(first, str(identifier))
        else:
            kwargs = {}

        device_obj = torch.device(device)

        # Pipe configuration from validated manifest / canonical kwargs.
        # Never hardcode fallback scheduler or model — the bundle is the
        # source of truth for the profile that embedded this cohort.
        if method in {"RID", "HSTR"}:
            model_id = str(kwargs["modelid_target"])
            model_revision = str(kwargs.get("model_revision", ""))
            resolution = int(kwargs["resolution"])
            if method == "RID":
                scheduler = str(kwargs["scheduler_target"])
            else:
                scheduler = str(kwargs["scheduler_target"])
        else:  # HSQR — extract from manifest directly
            model_id = str(manifest.get("model_id", ""))
            model_revision = str(manifest.get("model_revision", ""))
            resolution = int(manifest.get("resolution", 0))
            scheduler = str(manifest.get("scheduler_type",
                            manifest.get("scheduler", "DDIM")))

        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name=scheduler,
            disable_tqdm=True,
        )
        latent_shape = pipe.get_latent_shape()

        if method == "RID":
            from eval_bench_wm.utils.wm.ringid_provider import RingIDProvider
            provider = RingIDProvider(
                latent_shape=latent_shape,
                dtype=pipe.get_dtype(),
                device=device_obj,
                **kwargs,
            )
        elif method == "HSTR":
            from eval_bench_wm.utils.wm.hstr_provider import HSTRProvider
            provider = HSTRProvider(
                latent_shape=latent_shape,
                dtype=pipe.get_dtype(),
                device=device_obj,
                **kwargs,
            )
        elif method == "HSQR":
            from eval_bench_wm.utils.wm.hsqr_provider import HSQRProvider
            provider = mod.hsqr_provider_from_bundle(
                first, str(identifier), latent_shape, device_obj)
        else:
            raise DetectorProviderInitializationError(
                f"Unknown Fourier method: {method}"
            )

        # --- Step 8: method-specific state gates ---------------------------
        if method in {"RID", "HSTR"}:
            if getattr(provider, "bundle", None) is None:
                raise DetectorStateValidationError(
                    f"{method} provider has no persisted bundle"
                )
            if getattr(provider, "state_source", "") != "bundle":
                raise DetectorStateValidationError(
                    f"{method}: state_source is not 'bundle': "
                    f"{getattr(provider, 'state_source', 'unknown')}"
                )
        elif method == "HSQR":
            if getattr(provider, "bundle", None) is None:
                raise DetectorStateValidationError(
                    f"{method} provider has no persisted bundle"
                )
    except (DetectorStateValidationError, DetectorMissingStateError):
        raise
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"{method} provider construction failed: {exc}"
        ) from exc

    # --- Step 9: key index validation against manifest/provider ------------
    _validate_key_index(first, method, manifest, kwargs,
                         f"{method} run_id={identifier}")

    # --- Step 10: protocol/profile validation against manifest -------------
    _validate_protocol_profile(first, method, manifest,
                                f"{method} run_id={identifier}")

    # -------------------------------------------------------------------
    # Per-row cohort consistency + cross-validation.
    # Every row must share the same canonical bundle identity AND each
    # row's own metadata must be validated against it.
    # -------------------------------------------------------------------
    cohort_bundle_dir = str(first.get(f"{prefix}_bundle_dir", ""))
    cohort_key_index = str(first.get(f"{prefix}_key_index", ""))
    cohort_protocol = str(first.get(f"{prefix}_protocol_mode", ""))

    for i, record in enumerate(records):
        row_id = str(record.get("run_id", i))
        row_label = f"{method} run_id={row_id}"

        # Per-row canonical bundle validation
        try:
            mod.fourier_bundle_manifest(record, row_id, method)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"{row_label}: bundle validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Cross-validate key index and protocol against canonical values
        _validate_key_index(record, method, manifest, kwargs, row_label)
        _validate_protocol_profile(record, method, manifest, row_label)

        # Cohort consistency
        row_bundle_dir = str(record.get(f"{prefix}_bundle_dir", ""))
        row_key_index = str(record.get(f"{prefix}_key_index", ""))
        row_protocol = str(record.get(f"{prefix}_protocol_mode", ""))

        if row_bundle_dir != cohort_bundle_dir:
            raise DetectorStateValidationError(
                f"{row_label}: mixed bundle_dir in cohort: "
                f"expected={cohort_bundle_dir!r} got={row_bundle_dir!r}"
            )
        if row_key_index != cohort_key_index:
            raise DetectorStateValidationError(
                f"{row_label}: mixed key_index in cohort: "
                f"expected={cohort_key_index!r} got={row_key_index!r}"
            )
        if row_protocol != cohort_protocol:
            raise DetectorStateValidationError(
                f"{row_label}: mixed protocol_mode in cohort: "
                f"expected={cohort_protocol!r} got={row_protocol!r}"
            )

    score_definition = _METHOD_SCORE_DEFINITIONS.get(
        method, f"{prefix}_score = -raw_l1")

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "method": method,
        "score_definition": score_definition,
        "_cohort_bundle_dir": cohort_bundle_dir,
        "_cohort_key_index": cohort_key_index,
        "_cohort_protocol": cohort_protocol,
        "_cohort_bundle_config": str(first.get(f"{prefix}_bundle_config_sha256", "")),
        "_cohort_selected_pattern": str(first.get(f"{prefix}_selected_pattern_sha256", "")),
        "_cohort_mask": str(first.get(f"{prefix}_mask_sha256", "")),
        "_manifest": manifest,
        "_kwargs": kwargs,
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one image using the Fourier provider via canonical evaluate_image.

    Per-row target/mask SHA validation (Issue #24): when ``record`` is
    provided, validates source watermarked target/mask identity against the
    detector's computed values before scoring.  Missing provenance is a
    hard failure (fail-closed).

    Missing image raises ``FileNotFoundError`` (image absence is not a
    detector state issue — Issue #25 taxonomy).

    Canonical score = ``-raw_l1`` (delegates to extract module's
    canonical_score).
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    method = provider_info["method"]
    mod = provider_info["extract_module"]
    manifest = provider_info.get("_manifest", {})

    # Per-row target/mask identity validation (fail-closed)
    if record is not None:
        _validate_row_target_mask(provider, method, record, manifest)

    # Delegate scoring to canonical helpers — no watermark maths rewritten
    try:
        result = mod.evaluate_image(
            torch, provider, provider_info["pipe"], path, steps)
    except Exception as exc:
        raise DetectorScoringError(
            f"{method} scoring failed for {image_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raw = mod.raw_score(method, result)
    canonical = mod.canonical_score(method, raw, result)

    score_definition = provider_info.get(
        "score_definition",
        _METHOD_SCORE_DEFINITIONS.get(method, f"{method.lower()}_score = -raw_l1"),
    )

    return {
        "raw_score": raw,
        "canonical_score": canonical,
        "raw_l1": raw,
        "score_definition": score_definition,
        "score_direction": "higher_is_watermarked",
    }


def aggregate(detector_rows: list[dict[str, Any]],
              method: str = "RID", **extra) -> dict[str, Any]:
    """Aggregate Fourier detector rows with method-specific score labels."""
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    method = method.upper()
    score_definition = _METHOD_SCORE_DEFINITIONS.get(
        method, f"{method.lower()}_score = -raw_l1")

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
        "method": method,
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_type": f"{method.lower()}_score",
        "score_direction": "higher_is_watermarked",
        "score_definition": score_definition,
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "threshold_type": "empirical_clean_1pct_fpr",
            "threshold_score_space": "canonical_score",
            "threshold_comparison_operator": ">=",
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
