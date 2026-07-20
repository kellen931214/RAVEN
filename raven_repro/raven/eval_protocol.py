"""Authoritative protocol and provenance helpers for formal RAVEN evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_PROTOCOL_VERSION = "raven_formal_eval_v1"

FORMAL_ATTACK_CONFIG: dict[str, Any] = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "image_size": [512, 512],
    "steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "shift_space": "image_pixels",
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "latent_sampling_mode": "nearest",
    "padding_mode": "reflection",
    "view_guided_attention": True,
    "color_transfer": True,
    "color_transfer_mode": "paper_exact_two_stage",
    "inversion_mode": "ddim",
    "prompt": "",
    "negative_prompt": "",
    "shift_magnitudes_image_px": [24, 27, 28, 29, 32],
    "shift_sign_policy": "deterministic_four_quadrant_schedule",
    "seed_policy": "base_seed_plus_numeric_run_id",
    "base_seed": 42,
}

CLIP_CONFIG: dict[str, Any] = {
    "clip_metric_type": "prompt-image cosine similarity",
    "clip_model_name": "ViT-bigG-14",
    "clip_pretrained": "laion2b_s39b_b160k",
    "preprocess_identifier": "open_clip.create_model_and_transforms",
    "text_source": "original generation prompt",
    "normalization": "L2-normalized image/text embeddings",
    "paper_name_interpretation": (
        "ViT-bigG-14 is an implementation interpretation of the paper's "
        "\"OpenCLIP-ViT/G\" naming, not a fully specified model/tag pair "
        "quoted verbatim from the paper."
    ),
}

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
    "RID": ("rid_seed", "fix_gt", "time_shift", "time_shift_factor"),
    "HSTR": ("hstr_seed", "fix_gt"),
    "HSQR": ("hsqr_seed", "fix_gt", "delta"),
    "GS": ("offset", "message_width_in_bytes", "num_replications"),
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
    "RID": {"rid_seed": 999999, "fix_gt": 1, "time_shift": 1, "time_shift_factor": 1},
    "HSTR": {"hstr_seed": 999999, "fix_gt": 1},
    "HSQR": {"hsqr_seed": 999999, "fix_gt": 1, "delta": 0},
    "GS": {"offset": 0, "message_width_in_bytes": 32, "num_replications": 1},
}

FORMAL_DEBUG_FIELDS = (
    "model_id",
    "model_revision",
    "steps",
    "strength",
    "guidance_scale",
    "inversion_mode",
    "exact_timestep",
    "warp_mode",
    "interpolation_mode",
    "padding_mode",
    "align_corners",
    "normalized_coordinate_formula",
    "color_transfer_mode",
    "inversion_prompt",
    "reconstruction_prompt",
    "transform_config_hash",
)


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


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_source_manifest(
    path: str | Path, *, repo_root: str | Path
) -> tuple[dict[str, Any], str]:
    """Validate the immutable formal source manifest against the live files."""
    path = Path(path).resolve()
    companion = path.with_suffix(".sha256")
    if not path.is_file() or not companion.is_file():
        raise FileNotFoundError(f"source manifest or companion SHA is missing: {path}")
    actual_manifest_sha = sha256_path(path)
    recorded_manifest_sha = companion.read_text(encoding="utf-8").strip().split()[0]
    if actual_manifest_sha != recorded_manifest_sha:
        raise RuntimeError(
            "source manifest SHA mismatch: "
            f"recorded={recorded_manifest_sha} actual={actual_manifest_sha}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(repo_root).resolve()
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("formal source manifest has no files")
    seen: set[str] = set()
    for record in files:
        relative = str(record.get("relative_path", ""))
        if not relative or relative in seen:
            raise RuntimeError(f"invalid or duplicate source manifest path: {relative!r}")
        seen.add(relative)
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"source manifest path escapes repository: {relative}") from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"source size drift: {relative}")
        actual_sha = sha256_path(source)
        if actual_sha != record.get("sha256"):
            raise RuntimeError(
                f"source SHA drift: {relative} stored={record.get('sha256')} actual={actual_sha}"
            )
    return payload, actual_manifest_sha


def formal_attack_config_hash() -> str:
    return canonical_json_hash(FORMAL_ATTACK_CONFIG)


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
    }


def assert_formal_debug_info(
    debug_info: Mapping[str, Any],
    *,
    planned_flow_dx_image_px: float | None = None,
    planned_flow_dy_image_px: float | None = None,
) -> str:
    missing = [field for field in FORMAL_DEBUG_FIELDS if field not in debug_info]
    if missing:
        raise RuntimeError(f"formal debug info missing fields: {missing}")
    expected = {
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "steps": FORMAL_ATTACK_CONFIG["steps"],
        "strength": FORMAL_ATTACK_CONFIG["strength"],
        "guidance_scale": FORMAL_ATTACK_CONFIG["guidance_scale"],
        "inversion_mode": FORMAL_ATTACK_CONFIG["inversion_mode"],
        "warp_mode": FORMAL_ATTACK_CONFIG["warp_mode"],
        "interpolation_mode": FORMAL_ATTACK_CONFIG["latent_sampling_mode"],
        "padding_mode": FORMAL_ATTACK_CONFIG["padding_mode"],
        "align_corners": False,
        "color_transfer_mode": FORMAL_ATTACK_CONFIG["color_transfer_mode"],
        "inversion_prompt": "",
        "reconstruction_prompt": "",
    }
    for field, expected_value in expected.items():
        actual = debug_info[field]
        if isinstance(expected_value, float):
            if not math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"formal debug mismatch: {field}={actual!r}, expected {expected_value!r}"
                )
        elif actual != expected_value:
            raise RuntimeError(
                f"formal debug mismatch: {field}={actual!r}, expected {expected_value!r}"
            )
    if not isinstance(debug_info["exact_timestep"], int):
        raise RuntimeError("formal debug exact_timestep must be an integer")
    if not debug_info["normalized_coordinate_formula"]:
        raise RuntimeError("formal debug normalized_coordinate_formula is empty")
    for field in (
        "effective_source_dx_latent",
        "effective_source_dy_latent",
        "effective_source_flow_dx_image_px",
        "effective_source_flow_dy_image_px",
        "effective_visual_shift_dx_image_px",
        "effective_visual_shift_dy_image_px",
    ):
        value = float(debug_info[field])
        if not math.isfinite(value):
            raise RuntimeError(f"formal debug {field} is non-finite")
    if planned_flow_dx_image_px is not None and not math.isclose(
        float(debug_info["planned_flow_dx_image_px"]),
        float(planned_flow_dx_image_px),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("formal debug planned x flow does not match shift plan")
    if planned_flow_dy_image_px is not None and not math.isclose(
        float(debug_info["planned_flow_dy_image_px"]),
        float(planned_flow_dy_image_px),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("formal debug planned y flow does not match shift plan")
    computed = canonical_json_hash(transform_config_payload(debug_info))
    if debug_info["transform_config_hash"] != computed:
        raise RuntimeError(
            "formal debug transform_config_hash mismatch: "
            f"stored={debug_info['transform_config_hash']} computed={computed}"
        )
    return computed


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


def require_uniform_clip_provenance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "clip_metric_type",
        "clip_model_name",
        "clip_pretrained",
        "open_clip_version",
        "preprocess_identifier",
        "text_source",
        "normalization",
        "paper_name_interpretation",
    )
    values = {
        canonical_json_hash({key: record.get(key) for key in keys}): {
            key: record.get(key) for key in keys
        }
        for record in records
    }
    if len(values) != 1:
        raise ValueError(f"mixed CLIP provenance: {sorted(values)}")
    return next(iter(values.values()))


def current_clip_provenance() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("open_clip_torch")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {**CLIP_CONFIG, "open_clip_version": version}


def validate_resume_record(
    record: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise RuntimeError(
                f"resume mismatch for {field}: stored={record.get(field)!r} "
                f"expected={expected_value!r}"
            )
    run_id = str(expected["run_id"])
    for path_field, hash_field in (
        ("attacked_path", "attacked_sha256"),
        ("debug_info_path", "debug_info_sha256"),
    ):
        path = Path(str(record.get(path_field, "")))
        if not path.is_file():
            raise RuntimeError(f"resume missing {path_field} run_id={run_id}: {path}")
        actual = sha256_path(path)
        if actual != record.get(hash_field):
            raise RuntimeError(
                f"resume {hash_field} mismatch run_id={run_id}: "
                f"stored={record.get(hash_field)!r} actual={actual}"
            )
    debug_path = Path(str(record["debug_info_path"]))
    debug_info = json.loads(debug_path.read_text(encoding="utf-8"))
    transform_hash = assert_formal_debug_info(
        debug_info,
        planned_flow_dx_image_px=float(expected["planned_flow_dx_image_px"]),
        planned_flow_dy_image_px=float(expected["planned_flow_dy_image_px"]),
    )
    if record.get("transform_config_hash") != transform_hash:
        raise RuntimeError(f"resume debug transform hash mismatch run_id={run_id}")


def stage_fid_records(
    records: Sequence[Mapping[str, Any]],
    *,
    formal_output: str | Path,
    quality_config_hash: str,
    expected_count: int,
) -> tuple[Path, dict[str, Any]]:
    fid_root = Path(formal_output) / "metrics" / "fid" / quality_config_hash
    if fid_root.exists():
        raise FileExistsError(f"refusing to reuse FID staging root: {fid_root}")
    run_ids = [str(record["run_id"]) for record in records]
    if len(run_ids) != expected_count or len(set(run_ids)) != expected_count:
        raise ValueError(
            f"FID requires {expected_count} unique completed run IDs, got {len(run_ids)}"
        )
    fid_root.mkdir(parents=True)
    reference_dir = fid_root / "reference_watermarked"
    attacked_dir = fid_root / "attacked"
    reference_dir.mkdir()
    attacked_dir.mkdir()
    manifest_rows = []
    width = max(6, max(len(run_id) for run_id in run_ids))
    for record in sorted(records, key=lambda item: int(item["run_id"])):
        run_id = str(record["run_id"])
        name = f"{int(run_id):0{width}d}.png"
        reference_source = Path(str(record["watermarked_path"])).resolve()
        attacked_source = Path(str(record["attacked_path"])).resolve()
        expected_reference_sha = str(record["watermarked_sha256"])
        expected_attacked_sha = str(record["attacked_sha256"])
        for source, expected_sha, destination in (
            (reference_source, expected_reference_sha, reference_dir / name),
            (attacked_source, expected_attacked_sha, attacked_dir / name),
        ):
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_sha = sha256_path(source)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"FID source SHA mismatch run_id={run_id}: "
                    f"expected={expected_sha} actual={actual_sha}"
                )
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            try:
                os.symlink(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            if not destination.is_file():
                raise FileNotFoundError(f"broken FID staged file: {destination}")
        manifest_rows.append(
            {
                "run_id": run_id,
                "staged_name": name,
                "reference_source_path": str(reference_source),
                "reference_source_sha256": expected_reference_sha,
                "attacked_source_path": str(attacked_source),
                "attacked_source_sha256": expected_attacked_sha,
            }
        )
    expected_names = {row["staged_name"] for row in manifest_rows}
    reference_names = {path.name for path in reference_dir.iterdir()}
    attacked_names = {path.name for path in attacked_dir.iterdir()}
    if reference_names != expected_names or attacked_names != expected_names:
        raise RuntimeError("FID staged run-ID sets do not match")
    manifest_payload = {
        "metric_name": "per_method_fid_watermarked_vs_raven",
        "image_count": expected_count,
        "reference_definition": "original watermarked images from immutable formal snapshots",
        "attacked_definition": "formal RAVEN final post-color-transfer attacked images",
        "quality_config_hash": quality_config_hash,
        "records": manifest_rows,
    }
    manifest_payload["manifest_hash"] = canonical_json_hash(manifest_payload)
    manifest_path = fid_root / "fid_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_file_sha = sha256_path(manifest_path)
    with (fid_root / "fid_manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{manifest_file_sha}  fid_manifest.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_payload["manifest_file_sha256"] = manifest_file_sha
    return fid_root, manifest_payload
