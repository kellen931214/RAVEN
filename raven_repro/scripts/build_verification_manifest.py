#!/usr/bin/env python
"""Build a strict clean/watermarked/attacked verification pairing manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROVIDER_FIELDS = (
    "w_seed", "w_channel", "w_pattern", "w_mask_shape", "w_radius",
    "w_measurement", "w_injection", "w_pattern_const", "rid_seed",
    "hstr_seed", "hsqr_seed", "fix_gt", "time_shift", "time_shift_factor",
    "ring_width", "quantization_levels", "ring_value_range", "channel_min",
    "assigned_keys", "offset", "message_width_in_bytes", "num_replications",
    "l", "qr_version", "box_size", "delta",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["diffusiondb", "mscoco"])
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raven-results", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--watermark-config", type=Path, required=True)
    parser.add_argument("--clean-config", type=Path, required=True)
    parser.add_argument("--raven-config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def index_by_run_id(rows: list[dict[str, str]], source: Path) -> dict[int, dict[str, str]]:
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        run_id = int(row["run_id"])
        if run_id in indexed:
            raise ValueError(f"Duplicate run_id={run_id} in {source}")
        indexed[run_id] = row
    return indexed


def load_args(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"Missing args object in {path}")
    return args


def resolve_recorded_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    candidates = [path]
    if path.is_absolute() and len(path.parts) >= 3 and path.parts[1] == "workspace":
        candidates.append(workspace.joinpath(*path.parts[2:]))
    elif not path.is_absolute():
        candidates.append(workspace / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Recorded image does not exist: {value}; tried {candidates}")


def normalized_prompt(value: str | None) -> str:
    return " ".join((value or "").split())


def provider_defaults(method: str) -> dict:
    defaults = {
        "GS": {"offset": 0},
        "TR": {"w_seed": 999999, "w_channel": 3, "w_pattern": "ring", "w_mask_shape": "circle", "w_radius": 10},
        "RID": {"rid_seed": 999999, "fix_gt": 1, "time_shift": 1},
        "HSTR": {"hstr_seed": 999999, "fix_gt": 1},
        "HSQR": {"hsqr_seed": 999999, "fix_gt": 1, "delta": 0},
    }
    return defaults[method].copy()


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output}")

    workspace = args.workspace_root.resolve()
    metadata_rows = load_rows(args.metadata)
    raven_rows = index_by_run_id(load_rows(args.raven_results), args.raven_results)
    watermark_args = load_args(args.watermark_config)
    clean_args = load_args(args.clean_config)
    raven_args = load_args(args.raven_config)
    provider = provider_defaults(args.method)
    provider.update({key: watermark_args[key] for key in PROVIDER_FIELDS if key in watermark_args})

    output_rows = []
    for metadata in metadata_rows:
        run_id = int(metadata["run_id"])
        if args.limit is not None and len(output_rows) >= args.limit:
            break
        if metadata.get("dataset_name") != args.dataset:
            raise ValueError(f"run_id={run_id}: dataset mismatch")
        if metadata.get("wm_type") != args.method:
            raise ValueError(f"run_id={run_id}: method mismatch")
        if run_id not in raven_rows:
            raise ValueError(f"run_id={run_id}: missing RAVEN result")
        raven = raven_rows[run_id]
        if raven.get("error"):
            raise ValueError(f"run_id={run_id}: RAVEN row has error: {raven['error']}")
        if normalized_prompt(metadata.get("prompt")) != normalized_prompt(raven.get("prompt")):
            raise ValueError(f"run_id={run_id}: prompt mismatch between metadata and RAVEN results")
        if str(metadata.get("prompt_id", "")) != str(raven.get("prompt_id", "")):
            raise ValueError(f"run_id={run_id}: prompt_id mismatch")

        clean = (args.clean_dir / f"{run_id:06d}.png").resolve()
        if not clean.is_file():
            raise FileNotFoundError(f"run_id={run_id}: missing clean image {clean}")
        watermarked = resolve_recorded_path(metadata["watermarked_image_path"], workspace)
        attacked = resolve_recorded_path(raven["raven_output_path"], workspace)
        debug_info = resolve_recorded_path(raven["debug_info_path"], workspace)
        debug = json.loads(debug_info.read_text(encoding="utf-8"))

        row = {
            "dataset": args.dataset,
            "method": args.method,
            "run_id": run_id,
            "prompt_id": metadata.get("prompt_id", ""),
            "prompt": metadata.get("prompt", ""),
            "source": metadata.get("source", ""),
            "clean_path": str(clean),
            "watermarked_path": str(watermarked),
            "attacked_path": str(attacked),
            "debug_info_path": str(debug_info),
            "clean_sha256": sha256(clean),
            "watermarked_sha256": sha256(watermarked),
            "attacked_sha256": sha256(attacked),
            "generation_seed": clean_args.get("seed", ""),
            "attack_seed": int(raven_args.get("seed", 42)) + run_id,
            "model_id": metadata.get("target_model") or watermark_args.get("modelid_target", ""),
            "model_revision": watermark_args.get("model_revision", "unspecified"),
            "vae_id": watermark_args.get("vae_id") or "checkpoint-default",
            "scheduler": metadata.get("scheduler_target", ""),
            "inversion_steps": metadata.get("num_inference_steps_target", ""),
            "resolution": metadata.get("resolution", ""),
            "generation_dtype_clean": clean_args.get("dtype", "unspecified"),
            "detector_dtype": "float32",
            "legacy_threshold": metadata.get("detection_threshold", ""),
            "legacy_before_score": metadata.get("before_detection_metric_value", ""),
            "legacy_attacked_score": raven.get("after_detection_metric_value", ""),
            "raven_model_id": debug.get("model_id", raven.get("raven_model_id", "")),
            "raven_dtype": debug.get("dtype", raven_args.get("raven_dtype", "")),
            "raven_steps": raven.get("raven_steps", ""),
            "raven_strength": raven.get("raven_strength", ""),
            "view_guided_attention": raven.get("view_guided_attention", ""),
            "color_transfer": raven.get("color_transfer", ""),
            "dx": debug.get("dx", raven.get("dx", "")),
            "dy": debug.get("dy", raven.get("dy", "")),
        }
        row.update({key: provider.get(key, "") for key in PROVIDER_FIELDS})
        output_rows.append(row)

    if not output_rows:
        raise ValueError("No paired rows selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} rows={len(output_rows)} dataset={args.dataset} method={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
