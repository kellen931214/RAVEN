"""GaussMarker detector adapter — strict fail-closed canonical validation."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
)

_GM_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "gm_bundle_dir", "gm_bundle_config_sha256",
    "gm_w1_file_sha256", "gm_w2_file_sha256",
    "gm_m_sha256", "gm_watermark_sha256", "gm_target_sha256",
    "gm_protocol_mode", "watermark_target_sha256", "watermark_mask_sha256",
)
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(_GM_REQUIRED_METADATA_FIELDS)

_CANONICAL_KWARGS_FIELDS: tuple[str, ...] = (
    "gm_profile", "gm_bundle_dir", "gm_create_bundle", "gm_allow_in_memory_state",
    "gm_torch_dtype", "gm_channel_copy", "gm_w_copy", "gm_h_copy",
    "gm_watermark_bits_seed", "gm_use_gnr", "gm_gnr_path",
    "gm_model_nf", "gm_classifier_type", "gm_use_classifier", "gm_classifier_path",
    "modelid_target", "model_revision", "scheduler_target", "resolution",
    "gm_inversion_guidance", "gm_inversion_steps", "gm_inversion_seed",
    "gm_inversion_prompt", "gm_vae_sample", "gm_vae_scaling_factor",
    "gm_profile_is_official",
    "w_seed", "w_channel", "w_pattern", "w_mask_shape", "w_radius",
    "w_measurement", "w_injection",
)

_REQUIRED_SCORER_OUTPUTS: tuple[str, ...] = (
    "gm_raw_bit_accuracy", "gm_raw_ring_l1",
    "gm_report_label", "gm_score_definition",
    "gm_threshold_source", "gm_comparison_operator",
)
_NUMERIC_SCORER_FIELDS: frozenset[str] = frozenset({
    "gm_raw_bit_accuracy", "gm_raw_ring_l1",
    "gm_restored_bit_accuracy", "gm_classifier_probability",
})
_PROBABILITY_SCORER_FIELDS: frozenset[str] = frozenset({
    "gm_raw_bit_accuracy", "gm_restored_bit_accuracy", "gm_classifier_probability",
})
_VALID_GNR_CLASSIFIER_KINDS: frozenset[str] = frozenset({"gnr", "classifier"})
_STRICT_BOOL_KWARGS: frozenset[str] = frozenset({
    "gm_use_gnr", "gm_use_classifier", "gm_create_bundle",
    "gm_allow_in_memory_state", "gm_vae_sample", "gm_profile_is_official",
})
_PERSISTED_BUNDLE_FALSE_KWARGS: frozenset[str] = frozenset({
    "gm_create_bundle", "gm_allow_in_memory_state",
})
_NULLABLE_CANONICAL_FIELDS: frozenset[str] = frozenset({
    "gm_gnr_path", "gm_classifier_path", "gm_watermark_bits_seed",
})

_MISSING = object()

# Provider attr → canonical key → expected type tuple
_PROVIDER_ATTR_BINDINGS: tuple[tuple[str, str, Any], ...] = (
    ("profile", "gm_profile", str),
    ("profile_is_official", "gm_profile_is_official", bool),
    ("gm_torch_dtype", "gm_torch_dtype", str),
    ("ch", "gm_channel_copy", int),
    ("w", "gm_w_copy", int),
    ("h", "gm_h_copy", int),
    ("watermark_bits_seed", "gm_watermark_bits_seed", (int, type(None))),
    ("model_nf", "gm_model_nf", int),
    ("classifier_type", "gm_classifier_type", int),
    ("use_gnr", "gm_use_gnr", bool),
    ("use_classifier", "gm_use_classifier", bool),
    ("model_id", "modelid_target", str),
    ("model_revision", "model_revision", str),
    ("scheduler_name", "scheduler_target", str),
    ("resolution", "resolution", int),
    ("inversion_guidance", "gm_inversion_guidance", numbers.Real),
    ("inversion_steps", "gm_inversion_steps", int),
    ("inversion_seed", "gm_inversion_seed", int),
    ("inversion_prompt", "gm_inversion_prompt", str),
    ("vae_sample", "gm_vae_sample", bool),
    ("vae_scaling_factor", "gm_vae_scaling_factor", numbers.Real),
    ("w_seed", "w_seed", int),
    ("w_channel", "w_channel", int),
    ("w_pattern", "w_pattern", str),
    ("w_mask_shape", "w_mask_shape", str),
    ("w_radius", "w_radius", int),
    ("w_measurement", "w_measurement", str),
    ("w_injection", "w_injection", str),
)


def describe_required_artifacts() -> list[str]:
    return [
        "gm_bundle_dir (directory with manifest.json, w1.pth, w2.pth)",
        "gm_bundle_config_sha256, gm_w1_file_sha256, gm_w2_file_sha256",
        "gm_watermark_sha256, gm_m_sha256, gm_target_sha256, watermark_mask_sha256",
        "gm_protocol_mode",
        "Stable Diffusion inversion pipe",
    ]


def _get_scoring_module():
    from raven.evaluation import scoring
    return scoring


def _strict_type_check(value: Any, expected: Any, *, label: str) -> None:
    """Verify *value* matches *expected* type tuple strictly.  Bool never matches int."""
    types_tuple = expected if isinstance(expected, tuple) else (expected,)
    if bool in types_tuple:
        if type(value) is not bool:
            raise DetectorStateValidationError(
                f"{label} must be bool, got {type(value).__name__}: {value!r}")
        return
    if type(value) is bool:
        raise DetectorStateValidationError(
            f"{label} expected {types_tuple}, got bool: {value!r}")
    if not isinstance(value, types_tuple):
        raise DetectorStateValidationError(
            f"{label} must be one of {types_tuple}, got {type(value).__name__}: {value!r}")
    if numbers.Real in types_tuple:
        fv = float(value)
        if not math.isfinite(fv):
            raise DetectorStateValidationError(f"{label} must be finite, got {value!r}")


def _validate_required_gm_metadata(record: dict[str, Any]) -> None:
    run_id = str(record.get("run_id", "?"))
    for field in _GM_REQUIRED_METADATA_FIELDS:
        value = record.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise DetectorMissingStateError(
                f"run_id={run_id}: missing required GM metadata field {field!r}")


def _canonical_provider_identity(kwargs: dict[str, Any]) -> str:
    from raven.protocol import canonical_json_hash
    payload: dict[str, Any] = {}
    for field in _CANONICAL_KWARGS_FIELDS:
        value = kwargs.get(field)
        if field in ("gm_bundle_dir", "gm_gnr_path", "gm_classifier_path"):
            if isinstance(value, str) and value:
                value = str(Path(value).resolve())
        payload[field] = value
    return canonical_json_hash(payload)


def _validate_gm_protocol_mode(record: dict[str, Any]) -> None:
    from raven.detectors.protocols import GM_SHARED_TR_CLEAN_MODE
    actual = str(record.get("gm_protocol_mode", ""))
    if actual != GM_SHARED_TR_CLEAN_MODE:
        raise DetectorStateValidationError(
            f"GM protocol mode {actual!r} != {GM_SHARED_TR_CLEAN_MODE!r}")


def _validate_gm_provider_profile(manifest, kwargs, provider=None):
    mp = str(manifest.get("profile", ""))
    kp = str(kwargs.get("gm_profile", ""))
    if mp != kp:
        raise DetectorStateValidationError(f"GM profile mismatch: manifest={mp!r} kwargs={kp!r}")
    if provider is not None:
        actual = ""
        for attr in ("profile", "gm_profile", "profile_name"):
            val = getattr(provider, attr, None)
            if val is not None and str(val).strip():
                actual = str(val); break
        if not actual and getattr(provider, "bundle", None) is not None:
            actual = str(getattr(provider.bundle, "manifest", {}).get("profile", ""))
        if actual and actual != kp:
            raise DetectorStateValidationError(
                f"GM profile mismatch: provider={actual!r} kwargs={kp!r}")


def _validate_canonical_provider_kwargs(kwargs: Any, *, run_id: str) -> None:
    if not isinstance(kwargs, dict):
        raise DetectorStateValidationError(
            f"run_id={run_id}: canonical kwargs must be dict, got {type(kwargs).__name__}")
    extra = sorted(set(kwargs) - set(_CANONICAL_KWARGS_FIELDS))
    if extra:
        raise DetectorStateValidationError(
            f"run_id={run_id}: unexpected canonical keys: {extra}")
    for field in _STRICT_BOOL_KWARGS:
        if field not in kwargs or kwargs[field] is None:
            raise DetectorStateValidationError(
                f"run_id={run_id}: canonical {field!r} missing — must be present and bool")
        if not isinstance(kwargs[field], bool):
            raise DetectorStateValidationError(
                f"run_id={run_id}: canonical {field!r} must be bool, "
                f"got {type(kwargs[field]).__name__}: {kwargs[field]!r}")
    for field in _PERSISTED_BUNDLE_FALSE_KWARGS:
        if kwargs[field] is not False:
            raise DetectorStateValidationError(
                f"run_id={run_id}: unified detector requires {field}=False, got {kwargs[field]!r}")
    missing = sorted(f for f in _CANONICAL_KWARGS_FIELDS if f not in kwargs)
    if missing:
        raise DetectorStateValidationError(
            f"run_id={run_id}: canonical kwargs missing keys: {missing}")
    nulls = sorted(f for f in _CANONICAL_KWARGS_FIELDS
                   if kwargs[f] is None and f not in _NULLABLE_CANONICAL_FIELDS)
    if nulls:
        raise DetectorStateValidationError(
            f"run_id={run_id}: canonical fields must not be None: {nulls}")


def _validate_bundle_files_exist(bundle_dir: str) -> Path:
    resolved = Path(bundle_dir).resolve()
    if not resolved.is_dir():
        raise DetectorMissingStateError(f"GM bundle directory not found: {resolved}")
    for name in ("manifest.json", "w1.pth", "w2.pth"):
        if not (resolved / name).is_file():
            raise DetectorMissingStateError(f"GM bundle artifact missing: {resolved / name}")
    return resolved


def _verify_provider_attrs_match_kwargs(provider: Any, kwargs: dict[str, Any]) -> None:
    """Verify every provider attribute matches its canonical kwarg."""
    for attr, key, expected_type in _PROVIDER_ATTR_BINDINGS:
        if not hasattr(provider, attr):
            raise DetectorStateValidationError(
                f"GM provider missing required attribute {attr!r}")
        pval = getattr(provider, attr)
        kval = kwargs.get(key)
        label = f"GM provider.{attr}"
        _strict_type_check(pval, expected_type, label=label)

        # nullables: None in both is OK
        if key in _NULLABLE_CANONICAL_FIELDS and pval is None and kval is None:
            continue

        if pval is None:
            raise DetectorStateValidationError(
                f"GM provider.{attr} is None but canonical {key} is non-nullable")
        if kval is None:
            raise DetectorStateValidationError(
                f"GM canonical {key} is None but provider.{attr} is non-nullable")

        # Bool uses identity (True != 1)
        if expected_type is bool:
            if pval is not kval:
                raise DetectorStateValidationError(
                    f"GM provider.{attr}={pval!r} != canonical {key}={kval!r}")
        elif type(pval) is not type(kval):
            raise DetectorStateValidationError(
                f"GM provider.{attr} type {type(pval).__name__} != canonical {key} "
                f"type {type(kval).__name__}")
        elif pval != kval:
            raise DetectorStateValidationError(
                f"GM provider.{attr}={pval!r} != canonical {key}={kval!r}")


def _strict_provider_bool(provider: Any, attr: str) -> bool:
    if not hasattr(provider, attr):
        raise DetectorStateValidationError(f"GM provider missing required attribute {attr!r}")
    value = getattr(provider, attr)
    if type(value) is not bool:
        raise DetectorStateValidationError(
            f"GM provider.{attr} must be bool, got {type(value).__name__}: {value!r}")
    return value


def load_state(records: list[dict[str, Any]], device: str, **extra) -> dict[str, Any]:
    import torch
    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gm_provider import GmProvider
    except ImportError as exc:
        raise DetectorDependencyError(f"GM dependencies not available: {exc}") from exc
    if not records:
        raise DetectorMissingStateError("GM detector requires at least one record")

    for row in records:
        _validate_required_gm_metadata(row)
    for row in records:
        _validate_gm_protocol_mode(row)

    try:
        mod = _get_scoring_module()
    except Exception as exc:
        raise DetectorDependencyError(f"Cannot load raven.evaluation.scoring: {exc}") from exc

    row_bindings: list[dict[str, Any]] = []
    for row in records:
        run_id = str(row.get("run_id", "0"))
        _validate_bundle_files_exist(str(row["gm_bundle_dir"]))
        try:
            bdp, manifest = mod.gm_bundle_manifest(row, run_id)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"run_id={run_id}: bundle manifest failed: {type(exc).__name__}: {exc}") from exc
        try:
            kwargs = mod.gm_provider_kwargs(row, run_id)
        except Exception as exc:
            raise DetectorStateValidationError(
                f"run_id={run_id}: provider kwargs failed: {type(exc).__name__}: {exc}") from exc
        _validate_canonical_provider_kwargs(kwargs, run_id=run_id)
        _validate_gm_provider_profile(manifest, kwargs)
        row_bindings.append({"run_id": run_id, "bundle_dir": str(bdp),
                              "manifest": manifest, "kwargs": kwargs})

    identities = {_canonical_provider_identity(b["kwargs"]) for b in row_bindings}
    if len(identities) != 1:
        raise DetectorStateValidationError(f"mixed canonical identities: {sorted(identities)}")

    first_kwargs = row_bindings[0]["kwargs"]
    first_bundle_dir = Path(row_bindings[0]["bundle_dir"])

    try:
        device_obj = torch.device(device)
        pipe = pipe_utils.get_pipe_provider(
            pretrained_model_name_or_path=first_kwargs["modelid_target"],
            revision=first_kwargs["model_revision"],
            resolution=first_kwargs["resolution"],
            device=device_obj, eager_loading=False,
            schedulers_name=first_kwargs["scheduler_target"], disable_tqdm=True)
        latent_shape = pipe.get_latent_shape()
        provider = GmProvider(latent_shape=latent_shape, dtype=pipe.get_dtype(),
                               device=device_obj, **first_kwargs)
    except TypeError as exc:
        raise DetectorProviderInitializationError(f"GM provider construction failed: {exc}") from exc
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"GM provider/pipe construction failed: {type(exc).__name__}: {exc}") from exc

    _validate_gm_provider_profile(row_bindings[0]["manifest"], first_kwargs, provider=provider)

    if provider.bundle is None or getattr(provider, "state_source", "") != "bundle":
        raise DetectorStateValidationError(
            f"GM provider requires persisted bundle; state_source="
            f"{getattr(provider, 'state_source', 'unknown')!r}")

    _verify_provider_attrs_match_kwargs(provider, first_kwargs)

    # ── Three-way profile_is_official validation ──
    manifest_pio = row_bindings[0]["manifest"].get("profile_is_official", _MISSING)
    canonical_pio = first_kwargs.get("gm_profile_is_official", _MISSING)
    provider_pio = _strict_provider_bool(provider, "profile_is_official")
    bundle_pio = _MISSING
    if hasattr(provider, "bundle") and provider.bundle is not None:
        bundle_pio = provider.bundle.manifest.get("profile_is_official", _MISSING)

    if manifest_pio is _MISSING:
        raise DetectorStateValidationError("GM bundle manifest missing profile_is_official")
    if canonical_pio is _MISSING:
        raise DetectorStateValidationError("GM canonical kwargs missing gm_profile_is_official")
    if bundle_pio is _MISSING:
        raise DetectorStateValidationError("GM provider.bundle.manifest missing profile_is_official")
    if type(manifest_pio) is not bool:
        raise DetectorStateValidationError(
            f"GM bundle manifest profile_is_official must be bool, got {type(manifest_pio).__name__}")
    if type(canonical_pio) is not bool:
        raise DetectorStateValidationError(
            f"GM canonical gm_profile_is_official must be bool, got {type(canonical_pio).__name__}")
    if type(bundle_pio) is not bool:
        raise DetectorStateValidationError(
            f"GM provider.bundle.manifest profile_is_official must be bool, got {type(bundle_pio).__name__}")
    if not (manifest_pio == canonical_pio == provider_pio == bundle_pio):
        raise DetectorStateValidationError(
            f"GM profile_is_official mismatch: manifest={manifest_pio!r} "
            f"canonical={canonical_pio!r} provider={provider_pio!r} "
            f"bundle.manifest={bundle_pio!r}")

    from raven.detectors.protocols import tensor_sha256
    if provider.gt_patch is None:
        raise DetectorStateValidationError("GM provider has no gt_patch")
    provider_target_hash = tensor_sha256(provider.gt_patch.real.contiguous())
    if getattr(provider, "watermarking_mask", None) is None:
        raise DetectorStateValidationError("GM provider has no watermarking_mask")
    provider_mask_hash = tensor_sha256(provider.watermarking_mask.contiguous())

    for row in records:
        rt, rm = str(row.get("watermark_target_sha256", "")), str(row.get("watermark_mask_sha256", ""))
        rid = str(row.get("run_id", "?"))
        if rt != provider_target_hash:
            raise DetectorStateValidationError(f"run_id={rid}: cohort target SHA mismatch")
        if rm != provider_mask_hash:
            raise DetectorStateValidationError(f"run_id={rid}: cohort mask SHA mismatch")

    first_manifest = row_bindings[0]["manifest"]
    from raven.detectors.protocols import GM_SHARED_TR_CLEAN_MODE
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
        "gm_profile_is_official": provider_pio,
    }

    return {
        "provider": provider, "pipe": pipe, "scoring_module": mod,
        "device_obj": device_obj,
        "provider_target_hash": provider_target_hash,
        "provider_mask_hash": provider_mask_hash,
        "bundle_dir": str(first_bundle_dir),
        "verified_provenance": verified_provenance,
        "_canonical_kwargs": dict(first_kwargs),
    }


# ── Score-time helpers ──

def _require_score_time_canonical_kwargs(provider_info: dict[str, Any]) -> dict[str, Any]:
    if "_canonical_kwargs" not in provider_info:
        raise DetectorStateValidationError("GM provider_info missing _canonical_kwargs")
    kwargs = provider_info["_canonical_kwargs"]
    if not isinstance(kwargs, dict):
        raise DetectorStateValidationError(
            f"_canonical_kwargs must be dict, got {type(kwargs).__name__}")
    return kwargs


def _canonical_bool_config(provider_info: dict[str, Any], key: str) -> bool:
    """Score-time: return strict bool canonical config.  Never returns None."""
    kwargs = _require_score_time_canonical_kwargs(provider_info)
    if key not in kwargs:
        raise DetectorStateValidationError(
            f"canonical provider config missing required key {key!r}")
    value = kwargs[key]
    if type(value) is not bool:
        raise DetectorStateValidationError(
            f"canonical provider config {key!r} must be bool, "
            f"got {type(value).__name__}: {value!r}")
    return value


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    import torch

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(image_path)
    if record is None:
        raise DetectorMissingStateError("GM scoring requires resolved source metadata (record=…)")

    provider = provider_info["provider"]
    mod = provider_info["scoring_module"]

    # ── All state validation BEFORE evaluate_image ──
    source_target = str(record.get("watermark_target_sha256", ""))
    source_mask = str(record.get("watermark_mask_sha256", ""))
    if not source_target:
        raise DetectorMissingStateError(f"run_id={record.get('run_id')}: source target missing")
    if not source_mask:
        raise DetectorMissingStateError(f"run_id={record.get('run_id')}: source mask missing")
    dt = provider_info.get("provider_target_hash", "")
    dm = provider_info.get("provider_mask_hash", "")
    if not dt:
        raise DetectorStateValidationError(f"run_id={record.get('run_id')}: detector target missing")
    if not dm:
        raise DetectorStateValidationError(f"run_id={record.get('run_id')}: detector mask missing")
    if source_target != dt:
        raise DetectorStateValidationError(f"run_id={record.get('run_id')}: target SHA mismatch")
    if source_mask != dm:
        raise DetectorStateValidationError(f"run_id={record.get('run_id')}: mask SHA mismatch")

    # Canonical state validation.
    canonical_kwargs = _require_score_time_canonical_kwargs(provider_info)

    if "gm_inversion_steps" not in canonical_kwargs:
        raise DetectorStateValidationError("canonical kwargs missing gm_inversion_steps")
    can_steps = canonical_kwargs["gm_inversion_steps"]
    if isinstance(can_steps, bool) or not isinstance(can_steps, int):
        raise DetectorStateValidationError(
            f"canonical gm_inversion_steps must be int, got {type(can_steps).__name__}: {can_steps!r}")
    if can_steps <= 0:
        raise DetectorStateValidationError(f"canonical gm_inversion_steps must be >0: {can_steps}")

    if not hasattr(provider, "inversion_steps"):
        raise DetectorStateValidationError("GM provider missing inversion_steps")
    prov_steps = provider.inversion_steps
    if isinstance(prov_steps, bool) or not isinstance(prov_steps, int):
        raise DetectorStateValidationError(
            f"provider.inversion_steps must be int, got {type(prov_steps).__name__}: {prov_steps!r}")
    if prov_steps <= 0:
        raise DetectorStateValidationError(f"provider.inversion_steps must be >0: {prov_steps}")
    if prov_steps != can_steps:
        raise DetectorStateValidationError(
            f"provider.inversion_steps={prov_steps} != canonical gm_inversion_steps={can_steps}")

    # GNR/classifier canonical bools — must exist, be valid, before scoring.
    gnr_canonical = _canonical_bool_config(provider_info, "gm_use_gnr")
    clf_canonical = _canonical_bool_config(provider_info, "gm_use_classifier")

    steps = prov_steps

    # ── Scoring ──
    try:
        result = mod.evaluate_image(torch, provider, provider_info["pipe"], path, steps)
        raw_val = mod.raw_score("GM", result)
        canonical_val = mod.canonical_score("GM", raw_val, result)
        if not isinstance(raw_val, numbers.Real) or isinstance(raw_val, bool):
            raise ValueError(f"raw_score must be Real, got {type(raw_val).__name__}: {raw_val!r}")
        if not isinstance(canonical_val, numbers.Real) or isinstance(canonical_val, bool):
            raise ValueError(f"canonical_score must be Real, got {type(canonical_val).__name__}: {canonical_val!r}")
        raw = float(raw_val)
        canonical = float(canonical_val)
        if not math.isfinite(raw):
            raise ValueError("non-finite raw score")
        if not math.isfinite(canonical):
            raise ValueError("non-finite canonical score")
        _validate_scorer_outputs(result)
    except Exception as exc:
        raise DetectorScoringError(
            f"GM scoring failed for {image_path}: {type(exc).__name__}: {exc}") from exc

    gnr_used = _resolve_gnr_classifier_usage(result, provider_info, kind="gnr")
    classifier_used = _resolve_gnr_classifier_usage(result, provider_info, kind="classifier")

    restored = result.get("gm_restored_bit_accuracy")
    cls_prob = result.get("gm_classifier_probability")
    if not gnr_used and restored is not None:
        raise DetectorScoringError("GNR not used but restored_bit_accuracy present")
    if gnr_used and restored is None:
        raise DetectorScoringError("GNR used but restored_bit_accuracy missing")
    if not classifier_used and cls_prob is not None:
        raise DetectorScoringError("classifier not used but probability present")
    if classifier_used and cls_prob is None:
        raise DetectorScoringError("classifier used but probability missing")
    if classifier_used and not gnr_used:
        raise DetectorScoringError("classifier used but GNR not enabled")
    if classifier_used and restored is None:
        raise DetectorScoringError("classifier used but restored_bit_accuracy missing")

    score: dict[str, Any] = {
        "raw_score": raw, "canonical_score": canonical,
        "gm_raw_bit_accuracy": float(result["gm_raw_bit_accuracy"]),
        "gm_raw_ring_l1": float(result["gm_raw_ring_l1"]),
        "gm_restored_bit_accuracy": result.get("gm_restored_bit_accuracy"),
        "gm_classifier_probability": result.get("gm_classifier_probability"),
        "gm_report_label": str(result["gm_report_label"]),
        "gm_score_definition": str(result["gm_score_definition"]),
        "gm_threshold_source": str(result["gm_threshold_source"]),
        "gm_comparison_operator": str(result["gm_comparison_operator"]),
        "gm_gnr_used": gnr_used, "gm_classifier_used": classifier_used,
        "source_watermark_target_sha256": source_target,
        "detector_watermark_target_sha256": dt,
        "source_watermark_mask_sha256": source_mask,
        "detector_watermark_mask_sha256": dm,
        "gm_target_verified": True, "gm_mask_verified": True,
    }
    score.update(provider_info.get("verified_provenance", {}))
    return score


def _validate_scorer_outputs(result: dict[str, Any]) -> None:
    for field in _REQUIRED_SCORER_OUTPUTS:
        value = result.get(field)
        if value is None:
            raise ValueError(f"required GM scorer output {field!r} is None")
        if field in _NUMERIC_SCORER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field!r} wrong type {type(value).__name__!r}")
            fv = float(value)
            if not math.isfinite(fv):
                raise ValueError(f"{field!r} non-finite: {value!r}")
            if field == "gm_raw_ring_l1" and fv < 0.0:
                raise ValueError(f"gm_raw_ring_l1 must be >= 0: {fv!r}")
            if field in _PROBABILITY_SCORER_FIELDS and not (0.0 <= fv <= 1.0):
                raise ValueError(f"{field!r} out of [0,1]: {fv!r}")
        elif field in ("gm_report_label", "gm_score_definition",
                        "gm_threshold_source", "gm_comparison_operator"):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{field!r} empty/wrong type {type(value).__name__!r}")
    for opt in ("gm_restored_bit_accuracy", "gm_classifier_probability"):
        ov = result.get(opt)
        if ov is not None:
            if isinstance(ov, bool) or not isinstance(ov, (int, float)):
                raise ValueError(f"{opt!r} wrong type {type(ov).__name__!r}")
            fo = float(ov)
            if not math.isfinite(fo):
                raise ValueError(f"{opt!r} non-finite: {ov!r}")
            if not (0.0 <= fo <= 1.0):
                raise ValueError(f"{opt!r} out of [0,1]: {fo!r}")


def _resolve_gnr_classifier_usage(result, provider_info, *, kind) -> bool:
    if kind not in _VALID_GNR_CLASSIFIER_KINDS:
        raise DetectorScoringError(f"invalid kind {kind!r}")
    if kind == "gnr":
        scorer_keys = ("gm_used_gnr", "gm_gnr_used")
        provider_key = "gm_use_gnr"
    else:
        scorer_keys = ("gm_used_classifier", "gm_classifier_used")
        provider_key = "gm_use_classifier"

    scorer_values: list[bool] = []
    for key in scorer_keys:
        val = result.get(key)
        if val is None:
            continue
        if not isinstance(val, bool):
            raise DetectorScoringError(
                f"scorer output {key!r} must be bool, got {type(val).__name__}: {val!r}")
        scorer_values.append(val)

    provider_value = _canonical_bool_config(provider_info, provider_key)

    if len(scorer_values) >= 1:
        declared = scorer_values[0]
        if any(v != declared for v in scorer_values):
            raise DetectorScoringError(f"conflicting {kind} values: {scorer_values}")
        if declared != provider_value:
            raise DetectorScoringError(
                f"{kind} usage contradiction: scorer={declared} canonical={provider_value}")
        return declared
    return provider_value


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    from raven.evaluation.metrics import summarize_detection
    from . import ROW_STATUS_SCORED
    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != ROW_STATUS_SCORED:
            continue
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(row.get("evaluation_cohort", ""), []).append(float(cs))
    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored
    result: dict[str, Any] = {
        "method": "GM", "requested_count": len(detector_rows),
        "scored_count": scored, "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": sorted({"original_watermarked", "attacked_watermarked"} - set(cohorts)),
        "score_type": "gm_raw_bit_accuracy", "score_direction": "higher_is_watermarked",
        "official_ensemble_threshold_available": False,
    }
    clean, wm, atk = cohorts.get("original_clean", []), cohorts.get("original_watermarked", []), cohorts.get("attacked_watermarked", [])
    if clean and wm and atk:
        s = summarize_detection(clean, wm, atk, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01, "threshold_type": "empirical_clean_1pct_fpr",
            "threshold_comparison_operator": ">=",
            "clean_calibrated_threshold": s.calibration.threshold,
            "clean_calibrated_actual_fpr": s.calibration.actual_fpr,
            "original_watermarked_tpr": s.watermarked_tpr,
            "attacked_watermarked_tpr": s.attacked_tpr,
            "attack_success": 1.0 - s.attacked_tpr,
        }
    return result
