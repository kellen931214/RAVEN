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

Protocol mode and provider profile are DISTINCT identities:
  - ``<prefix>_protocol_mode`` is the generation/evaluation protocol,
    validated against the method's shared-tr-clean constant
    (``raven.pairing_provenance``).
  - ``manifest.profile_name`` is the provider profile, validated against
    the canonical kwargs profile and ``provider.profile``.

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

# Bundle schema families.  RID uses RidBundle; HSTR/HSQR share SfwBundle.
_RID_BUNDLE_SCHEMAS = frozenset({"rid_bundle_v1"})
_SFW_BUNDLE_SCHEMAS = frozenset({"sfw_bundle_v1"})


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


def _protocol_mode_for_method(method: str) -> str:
    """Canonical protocol mode constant for a Fourier method."""
    from raven.pairing_provenance import METHOD_PROTOCOL_MODES
    field, mode = METHOD_PROTOCOL_MODES[method]
    return mode


def _profile_for_kwargs(method: str, kwargs: dict[str, Any]) -> str:
    """Provider profile carried by the canonical kwargs helper."""
    if method == "RID":
        return str(kwargs.get("rid_profile", ""))
    if method == "HSTR":
        return str(kwargs.get("hstr_profile", ""))
    return ""


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


def _validate_manifest_schema(
    manifest: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Fail closed on bundle schema identity.

    RID must be a repository-supported RidBundle schema; HSTR/HSQR must be a
    repository-supported SfwBundle schema.  Missing and unknown schemas are
    both rejected here — the adapter never delegates schema classification
    to a later provider constructor.
    """
    schema = str(manifest.get("schema", manifest.get("bundle_schema", ""))).strip()
    if not schema:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no schema"
        )
    if method == "RID" and schema not in _RID_BUNDLE_SCHEMAS:
        raise DetectorStateValidationError(
            f"{row_label}: unsupported RID bundle schema {schema!r}; "
            f"supported: {sorted(_RID_BUNDLE_SCHEMAS)}"
        )
    if method in {"HSTR", "HSQR"} and schema not in _SFW_BUNDLE_SCHEMAS:
        raise DetectorStateValidationError(
            f"{row_label}: unsupported SFW bundle schema {schema!r}; "
            f"supported: {sorted(_SFW_BUNDLE_SCHEMAS)}"
        )


def _validate_manifest_method_identity(
    manifest: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Verify the manifest's canonical method tag matches the requested method.

    Both canonical bundle formats (RidBundle and SfwBundle) write a ``method``
    field into their manifests.  A missing or mismatched tag is a state
    validation failure and must be caught before any pipe/provider
    construction — never classified as provider initialization.
    """
    manifest_method = str(manifest.get("method", "")).strip()
    if not manifest_method:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no method tag"
        )
    if manifest_method != method:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest method={manifest_method!r} does not "
            f"match requested method {method!r}"
        )


def _validate_protocol_mode(
    record: dict[str, Any],
    method: str,
    row_label: str,
) -> None:
    """Verify the row protocol mode against the method's canonical constant.

    Protocol mode is the generation/evaluation protocol — a different
    identity from the provider profile.
    """
    prefix = method.lower()
    row_protocol = str(record.get(f"{prefix}_protocol_mode", ""))
    expected = _protocol_mode_for_method(method)
    if row_protocol != expected:
        raise DetectorStateValidationError(
            f"{row_label}: row {prefix}_protocol_mode={row_protocol!r} does not "
            f"match canonical protocol mode {expected!r}"
        )


def _validate_provider_profile(
    method: str,
    manifest: dict[str, Any],
    kwargs: dict[str, Any],
    provider,
    row_label: str,
) -> None:
    """Verify manifest profile == kwargs profile == provider profile."""
    prefix = method.lower()
    manifest_profile = str(manifest.get("profile_name", "")).strip()
    kwargs_profile = _profile_for_kwargs(method, kwargs)
    provider_profile = str(getattr(provider, "profile", "")).strip()

    if not manifest_profile:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no profile_name"
        )
    if kwargs_profile and manifest_profile != kwargs_profile:
        raise DetectorStateValidationError(
            f"{row_label}: manifest profile_name={manifest_profile!r} does not "
            f"match provider kwargs {prefix}_profile={kwargs_profile!r}"
        )
    if provider_profile and manifest_profile != provider_profile:
        raise DetectorStateValidationError(
            f"{row_label}: manifest profile_name={manifest_profile!r} does not "
            f"match provider profile={provider_profile!r}"
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


def _validate_pipe_profile_fields(
    method: str,
    manifest: dict[str, Any],
    row_label: str,
) -> tuple[str, str, str, int]:
    """Extract and validate pipe profile fields from the manifest.

    All fields are mandatory — no fallback to default model/scheduler.
    Missing → DetectorStateValidationError (bundle exists but incomplete).
    Returns (model_id, model_revision, scheduler, resolution).
    """
    model_id = str(manifest.get("model_id", "")).strip()
    model_revision = str(manifest.get("model_revision", "")).strip()
    resolution_raw = manifest.get("resolution")
    scheduler = str(
        manifest.get("scheduler_type", manifest.get("scheduler", ""))
    ).strip()

    if not model_id:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no model_id"
        )
    if not model_revision:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no model_revision"
        )
    if not scheduler:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no scheduler/scheduler_type"
        )
    if resolution_raw is None or not str(resolution_raw).strip():
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest has no resolution"
        )
    try:
        resolution = int(resolution_raw)
    except (ValueError, TypeError) as exc:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest resolution is not an integer: "
            f"{resolution_raw!r}"
        ) from exc
    if resolution <= 0:
        raise DetectorStateValidationError(
            f"{row_label}: bundle manifest resolution must be positive: "
            f"{resolution}"
        )
    return model_id, model_revision, scheduler, resolution


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

    Validation order — every step structural, never message-derived:

    1. Method tag verification (all rows)
    2. Required metadata preflight (all rows)
    3. Artifact path preflight — bundle dir + manifest.json (all rows)
    4. Canonical manifest loading (all rows)
    5. Manifest schema / method identity (all rows)
    6. Protocol mode (all rows)
    7. Mixed-cohort identity comparison (all rows) — BEFORE any pipe/provider
    8. One pipe construction from validated profile (no fallback)
    9. One provider construction via method-specific kwargs helper
    10. State gates (state_source for RID/HSTR)
    11. Key index + provider profile validation
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

    # --- Step 3: artifact path preflight on every row -----------------------
    # All rows must point at an existing bundle dir with manifest.json —
    # not just the first row.
    for i, record in enumerate(records):
        row_id = str(record.get("run_id", i))
        row_label = f"{method} run_id={row_id}"
        bundle_dir = Path(str(record.get(f"{prefix}_bundle_dir", "")))
        if not bundle_dir.is_dir():
            raise DetectorMissingStateError(
                f"{row_label}: {prefix}_bundle_dir does not exist or is not "
                f"a directory: {bundle_dir}"
            )
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise DetectorMissingStateError(
                f"{row_label}: bundle manifest.json not found at {manifest_path}"
            )

    # --- Step 4: load extract module ----------------------------------------
    try:
        mod = _get_extract_module()
    except Exception as exc:
        raise DetectorDependencyError(
            f"Cannot load extract_verification_scores: {exc}"
        ) from exc

    # --- Step 5: canonical manifest loading + validation on every row -------
    first = records[0]
    identifier = str(first.get("run_id", "0"))
    row_manifests: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        row_id = str(record.get("run_id", i))
        row_label = f"{method} run_id={row_id}"
        try:
            _bundle_path, manifest = mod.fourier_bundle_manifest(
                record, str(row_id), method)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"{row_label}: bundle validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        row_manifests.append(manifest)
        _validate_manifest_schema(manifest, method, row_label)
        _validate_manifest_method_identity(manifest, method, row_label)
        _validate_protocol_mode(record, method, row_label)

    manifest = row_manifests[0]

    # --- Step 6: mixed-cohort identity comparison BEFORE provider -----------
    # If any row disagrees on resolved bundle identity, fail before any
    # pipe/provider construction.
    cohort_identity_fields = (
        f"{prefix}_bundle_dir",
        f"{prefix}_bundle_config_sha256",
        f"{prefix}_selected_pattern_sha256",
        f"{prefix}_mask_sha256",
        f"{prefix}_key_index",
        f"{prefix}_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
    )
    for field in cohort_identity_fields:
        expected = str(first.get(field, ""))
        for i, record in enumerate(records):
            row_id = str(record.get("run_id", i))
            actual = str(record.get(field, ""))
            if actual != expected:
                raise DetectorStateValidationError(
                    f"{method} run_id={row_id}: mixed cohort field {field}: "
                    f"expected={expected!r} got={actual!r}"
                )

    # Manifest-level identity must be uniform across rows too, including the
    # method tag — a mixed manifest method cohort must fail before pipe.
    for manifest_field in ("method", "bundle_config_sha256",
                           "selected_pattern_sha256",
                           "mask_sha256", "selected_key_index",
                           "profile_name", "model_id", "model_revision",
                           "scheduler", "scheduler_type", "resolution"):
        values = {
            str(m.get(manifest_field, ""))
            for m in row_manifests
        }
        if len(values) > 1:
            raise DetectorStateValidationError(
                f"{method}: mixed manifest {manifest_field} across rows: "
                f"{sorted(values)}"
            )

    # --- Step 7: pipe profile fields (no fallback) --------------------------
    row_label = f"{method} run_id={identifier}"
    model_id, model_revision, scheduler, resolution = \
        _validate_pipe_profile_fields(method, manifest, row_label)

    # --- Step 8: one pipe construction --------------------------------------
    try:
        device_obj = torch.device(device)
        load_options = (
            {"revision": model_revision} if model_revision else {}
        )
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=model_id,
            resolution=resolution,
            device=device_obj,
            eager_loading=False,
            schedulers_name=scheduler,
            disable_tqdm=True,
            **load_options,
        )
        latent_shape = pipe.get_latent_shape()
    except (DetectorMissingStateError, DetectorStateValidationError):
        raise
    except ImportError as exc:
        raise DetectorDependencyError(
            f"{method} pipe dependencies not available: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"{method} pipe construction failed: {type(exc).__name__}: {exc}"
        ) from exc

    # --- Step 9: one provider construction ----------------------------------
    try:
        if method == "RID":
            kwargs = mod.rid_provider_kwargs_from_bundle(first, str(identifier))
        elif method == "HSTR":
            kwargs = mod.hstr_provider_kwargs_from_bundle(first, str(identifier))
        else:
            kwargs = {}

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
            provider = mod.hsqr_provider_from_bundle(
                first, str(identifier), latent_shape, device_obj)
        else:
            raise DetectorProviderInitializationError(
                f"Unknown Fourier method: {method}"
            )

        # --- Step 10: method-specific state gates ---------------------------
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

        # --- Step 11: key index + provider profile --------------------------
        _validate_key_index(first, method, manifest, kwargs,
                             f"{method} run_id={identifier}")
        _validate_provider_profile(method, manifest, kwargs, provider,
                                    f"{method} run_id={identifier}")
    except (DetectorMissingStateError, DetectorStateValidationError):
        raise
    except ImportError as exc:
        raise DetectorDependencyError(
            f"{method} provider dependencies not available: {exc}"
        ) from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"{method} provider construction failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    score_definition = _METHOD_SCORE_DEFINITIONS.get(
        method, f"{prefix}_score = -raw_l1")

    return {
        "provider": provider,
        "pipe": pipe,
        "extract_module": mod,
        "device_obj": device_obj,
        "method": method,
        "score_definition": score_definition,
        "_cohort_bundle_dir": str(first.get(f"{prefix}_bundle_dir", "")),
        "_cohort_key_index": str(first.get(f"{prefix}_key_index", "")),
        "_cohort_protocol": str(first.get(f"{prefix}_protocol_mode", "")),
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

    Per-row target/mask SHA validation (Issue #24): the resolved source
    record is mandatory.  ``record=None`` fails closed — provenance
    validation is never optional.

    Missing image raises ``FileNotFoundError`` (image absence is not a
    detector state issue — Issue #25 taxonomy).

    Canonical score = ``-raw_l1`` (delegates to extract module's
    canonical_score).  The whole scoring boundary is wrapped:
    evaluate_image / raw_score / canonical_score / finiteness all map to
    ``DetectorScoringError``.
    """
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    provider = provider_info["provider"]
    method = provider_info["method"]
    mod = provider_info["extract_module"]
    manifest = provider_info.get("_manifest", {})

    # Per-row target/mask identity validation — mandatory
    if record is None:
        raise DetectorMissingStateError(
            f"{method} scoring requires resolved source metadata"
        )
    _validate_row_target_mask(provider, method, record, manifest)

    # Delegate scoring to canonical helpers — no watermark maths rewritten.
    # Entire scoring boundary in one try → DetectorScoringError.
    try:
        result = mod.evaluate_image(
            torch, provider, provider_info["pipe"], path, steps)
        raw = float(mod.raw_score(method, result))
        canonical = float(mod.canonical_score(method, raw, result))
        if not math.isfinite(raw) or not math.isfinite(canonical):
            raise ValueError("non-finite score")
    except Exception as exc:
        raise DetectorScoringError(
            f"{method} scoring failed for {image_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

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
