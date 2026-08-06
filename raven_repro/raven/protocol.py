"""Canonical protocol helpers for RAVEN attack and evaluation runtime.

Provider config hashing, canonical JSON serialization, scheduler config
normalization, and transform payload construction.  No generation layout,
no data-root paths, no formal-run orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


# --------------------------------------------------------------------------- #
# Finite-value guard
# --------------------------------------------------------------------------- #
def _reject_non_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains non-finite float: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


# --------------------------------------------------------------------------- #
# Canonical JSON hashing
# --------------------------------------------------------------------------- #
def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    _reject_non_finite(payload)
    text = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Scheduler config — deterministic, no private metadata
# --------------------------------------------------------------------------- #
def canonical_scheduler_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only deterministic scheduler parameters that affect inference.

    Diffusers materializes ``_use_default_values`` from a set, so its list order
    can differ between otherwise identical scheduler instances. Package and
    class provenance are recorded separately and must not make paired attack
    transform hashes depend on non-semantic private metadata.
    """
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }


# --------------------------------------------------------------------------- #
# File SHA-256
# --------------------------------------------------------------------------- #
def sha256_path(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Transform config payload
# --------------------------------------------------------------------------- #
def transform_config_payload(debug_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": debug_info["model_id"],
        "model_revision": debug_info["model_revision"],
        "steps": int(debug_info["steps"]),
        "strength": float(debug_info["strength"]),
        "guidance_scale": float(debug_info["guidance_scale"]),
        "inversion_mode": debug_info["inversion_mode"],
        "exact_timestep": int(debug_info["exact_timestep"]),
        "inversion_prompt": debug_info["inversion_prompt"],
        "reconstruction_prompt": debug_info["reconstruction_prompt"],
        "negative_prompt": debug_info["negative_prompt"],
        "warp_mode": debug_info["warp_mode"],
        "latent_sampling_mode": debug_info["interpolation_mode"],
        "padding_mode": debug_info["padding_mode"],
        "align_corners": debug_info["align_corners"],
        "normalized_coordinate_formula": debug_info["normalized_coordinate_formula"],
        "pixel_center_offset_image_px": (
            None
            if debug_info["pixel_center_offset_image_px"] is None
            else float(debug_info["pixel_center_offset_image_px"])
        ),
        "warp_coordinate_convention": debug_info["warp_coordinate_convention"],
        "warp_implementation_version": debug_info["warp_implementation_version"],
        "planned_flow_dx_image_px": float(debug_info["planned_flow_dx_image_px"]),
        "planned_flow_dy_image_px": float(debug_info["planned_flow_dy_image_px"]),
        "effective_source_flow_dx_image_px": float(
            debug_info["effective_source_flow_dx_image_px"]
        ),
        "effective_source_flow_dy_image_px": float(
            debug_info["effective_source_flow_dy_image_px"]
        ),
        "view_guided_attention": bool(debug_info["view_guided_attention"]),
        "color_transfer": bool(debug_info["color_transfer"]),
        "color_transfer_mode": debug_info["color_transfer_mode"],
        "attack_device_class": debug_info["attack_device_class"],
        "attack_dtype": debug_info["attack_dtype"],
        "scheduler_class": debug_info["scheduler_class"],
        "scheduler_config": debug_info["scheduler_config"],
        "scheduler_config_hash": debug_info["scheduler_config_hash"],
        "torch_version": debug_info["torch_version"],
        "diffusers_version": debug_info["diffusers_version"],
    }


# --------------------------------------------------------------------------- #
# Provider config fields — canonical per-method schema
# --------------------------------------------------------------------------- #
TR_PROVIDER_FIELDS = (
    "w_seed",
    "w_channel",
    "w_radius",
    "w_pattern",
    "w_mask_shape",
    "w_measurement",
    "w_injection",
    "w_pattern_const",
)

PROVIDER_FIELDS_BY_METHOD = {
    "TR": TR_PROVIDER_FIELDS,
    "RID": (
        "rid_protocol_mode",
        "rid_bundle_config_sha256",
        "rid_selected_pattern_sha256",
        "rid_mask_sha256",
        "rid_key_index",
    ),
    "HSTR": (
        "hstr_protocol_mode",
        "hstr_bundle_config_sha256",
        "hstr_selected_pattern_sha256",
        "hstr_mask_sha256",
        "hstr_key_index",
    ),
    "HSQR": (
        "hsqr_protocol_mode",
        "hsqr_bundle_config_sha256",
        "hsqr_selected_pattern_sha256",
        "hsqr_mask_sha256",
        "hsqr_key_index",
    ),
    "GS": (
        "gs_protocol_mode",
        "message_width_in_bytes",
        "l",
        "num_replications",
        "gs_channel_copy",
        "gs_hw_copy",
        "gs_fpr",
        "gs_user_number",
    ),
    "GM": (
        "gm_protocol_mode",
        "gm_bundle_config_sha256",
        "gm_w1_file_sha256",
        "gm_w2_file_sha256",
        "gm_watermark_sha256",
        "gm_m_sha256",
        "gm_target_sha256",
        "gm_mask_sha256",
    ),
    "T2S": (
        "t2s_protocol_mode",
        "t2s_rng_mode",
        "t2s_inversion_mode",
        "t2s_num_inversion_steps",
        "t2s_provider_config_sha256",
    ),
}

PROVIDER_REQUIRED_NONEMPTY_FIELDS: dict[str, frozenset[str]] = {
    "GM": frozenset(PROVIDER_FIELDS_BY_METHOD["GM"]),
    "T2S": frozenset(PROVIDER_FIELDS_BY_METHOD["T2S"]),
    "RID": frozenset(PROVIDER_FIELDS_BY_METHOD["RID"]),
    "HSTR": frozenset(PROVIDER_FIELDS_BY_METHOD["HSTR"]),
    "HSQR": frozenset(PROVIDER_FIELDS_BY_METHOD["HSQR"]),
}

PROVIDER_DEFAULTS = {
    "TR": {
        "w_seed": 999999,
        "w_channel": 3,
        "w_radius": 10,
        "w_pattern": "ring",
        "w_mask_shape": "circle",
        "w_measurement": "l1_complex",
        "w_injection": "complex",
        "w_pattern_const": 0.0,
    },
    "RID": {
        "rid_protocol_mode": "",
        "rid_bundle_config_sha256": "",
        "rid_selected_pattern_sha256": "",
        "rid_mask_sha256": "",
        "rid_key_index": 0,
    },
    "HSTR": {
        "hstr_protocol_mode": "",
        "hstr_bundle_config_sha256": "",
        "hstr_selected_pattern_sha256": "",
        "hstr_mask_sha256": "",
        "hstr_key_index": 0,
    },
    "HSQR": {
        "hsqr_protocol_mode": "",
        "hsqr_bundle_config_sha256": "",
        "hsqr_selected_pattern_sha256": "",
        "hsqr_mask_sha256": "",
        "hsqr_key_index": 0,
    },
    "GS": {
        "gs_protocol_mode": "official_compatible",
        "message_width_in_bytes": 32,
        "l": 1,
        "num_replications": 64,
        "gs_channel_copy": 1,
        "gs_hw_copy": 8,
        "gs_fpr": 1e-6,
        "gs_user_number": 1000000,
    },
    "GM": {
        "gm_protocol_mode": "",
        "gm_bundle_config_sha256": "",
        "gm_w1_file_sha256": "",
        "gm_w2_file_sha256": "",
        "gm_watermark_sha256": "",
        "gm_m_sha256": "",
        "gm_target_sha256": "",
        "gm_mask_sha256": "",
    },
    "T2S": {
        "t2s_protocol_mode": "",
        "t2s_rng_mode": "",
        "t2s_inversion_mode": "",
        "t2s_num_inversion_steps": 0,
        "t2s_provider_config_sha256": "",
    },
}


# --------------------------------------------------------------------------- #
# Provider config normalization
# --------------------------------------------------------------------------- #
def _normalized_scalar(value: Any, default: Any) -> Any:
    if value in (None, ""):
        value = default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite provider value: {value!r}")
        return value
    return str(value)


def provider_config(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    method = method.upper()
    if method not in PROVIDER_FIELDS_BY_METHOD:
        raise ValueError(f"unsupported watermark method: {method}")
    nested = row.get("provider_config")
    if isinstance(nested, str) and nested:
        nested = json.loads(nested)
    source = nested if isinstance(nested, Mapping) else row
    defaults = PROVIDER_DEFAULTS[method]
    required = PROVIDER_REQUIRED_NONEMPTY_FIELDS.get(method, frozenset())
    missing = sorted(
        field
        for field in PROVIDER_FIELDS_BY_METHOD[method]
        if field in required and str(source.get(field, "")).strip() == ""
    )
    if missing:
        raise ValueError(
            f"{method} provider config is missing required fields: {missing}"
        )
    return {
        field: _normalized_scalar(source.get(field), defaults[field])
        for field in PROVIDER_FIELDS_BY_METHOD[method]
    }


def provider_config_hash(method: str, row: Mapping[str, Any]) -> str:
    return canonical_json_hash(provider_config(method, row))


def require_uniform_provider_config(
    method: str, rows: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    configs: dict[str, dict[str, Any]] = {}
    count = 0
    for row in rows:
        count += 1
        config = provider_config(method, row)
        configs[canonical_json_hash(config)] = config
    if count == 0:
        raise ValueError("provider cohort is empty")
    if len(configs) != 1:
        raise ValueError(
            "mixed provider configs are forbidden in one formal cohort: "
            f"{sorted(configs)}"
        )
    config_hash, config = next(iter(configs.items()))
    return config, config_hash
