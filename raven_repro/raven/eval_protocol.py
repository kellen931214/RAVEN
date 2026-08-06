"""Authoritative protocol and provenance helpers for formal RAVEN evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
CLEAN_DATA_ROOT = DATA_ROOT / "clean"

# method -> (watermarked data root, run output root)
METHOD_DATA_ROOTS: dict[str, Path] = {
    "TR": DATA_ROOT / "tr",
    "GS": DATA_ROOT / "gs",
    # shared_tr_clean_v2 cohorts (Issue #9). These roots hold only the
    # method-specific watermarked images; the clean images stay in data/clean/
    # and are never duplicated per method.
    "GM": DATA_ROOT / "gm",
    "T2S": DATA_ROOT / "t2s",
    "RID": DATA_ROOT / "rid",
    "HSTR": DATA_ROOT / "hstr",
    "HSQR": DATA_ROOT / "hsqr",
}
METHOD_OUTPUT_ROOTS: dict[str, Path] = {
    "TR": OUTPUTS_ROOT / "tr",
    "GS": OUTPUTS_ROOT / "gs",
    "GM": OUTPUTS_ROOT / "gm",
    "T2S": OUTPUTS_ROOT / "t2s",
    "RID": OUTPUTS_ROOT / "rid",
    "HSTR": OUTPUTS_ROOT / "hstr",
    "HSQR": OUTPUTS_ROOT / "hsqr",
}

# Methods whose formal protocol includes the attacked-clean recalibration branch.
# Everything else must never produce attacked-clean artifacts.
def method_data_root(method: str) -> Path:
    """Canonical watermarked-data root for ``method`` (fail closed on unknown)."""
    key = str(method).upper()
    try:
        return METHOD_DATA_ROOTS[key]
    except KeyError:
        raise ValueError(
            f"no canonical data root for method {method!r}; "
            f"known methods: {sorted(METHOD_DATA_ROOTS)}"
        ) from None


def source_metadata_path(method: str, dataset: str) -> Path:
    """Canonical source metadata CSV for a generated cohort."""
    return cohort_dir(method, dataset) / "metadata.csv"


def _reject_non_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains non-finite float: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


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


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


