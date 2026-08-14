#!/usr/bin/env python3
"""Create inverse-shifted RGB copies of a completed RAVEN attack cohort.

The RAVEN NFPA warp stores its actual *source* displacement in
``effective_source_flow_{x,y}_image_px``.  ``grid_sample`` uses source
coordinates, so its visible content displacement is the negative of that
value.  The operation here applies content displacement equal to the stored
source flow, i.e. it negates the stored visible displacement.

This is deliberately an RGB post-processing experiment.  It does not, and
cannot, invert DDIM inversion/regeneration or reconstruct pixels replaced by
reflection padding during the original latent-space warp.
"""

from __future__ import annotations

import argparse

import gc

import hashlib

import json

import logging

import math

from pathlib import Path

import shutil

import time

from typing import Any, Iterator

import numpy as np

from PIL import Image, ImageOps


LOG = logging.getLogger("raven.inverse_shift")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reflection_indices(length: int, translation: int) -> np.ndarray:
    """Return PyTorch reflection-padding source indices for an integer shift.

    For output pixel ``x``, an ordinary content translation by ``translation``
    samples source coordinate ``x - translation``.  PyTorch's ``reflection``
    padding reflects around the exterior of the boundary pixels (the boundary
    pixel itself is not repeated), with period ``2 * (length - 1)``.
    """
    if length <= 0:
        raise ValueError(f"invalid image dimension: {length}")
    if length == 1:
        return np.zeros(1, dtype=np.intp)
    coordinate = np.arange(length, dtype=np.int64) - int(translation)
    period = 2 * (length - 1)
    folded = np.mod(coordinate, period)
    reflected = np.where(folded < length, folded, period - folded)
    return reflected.astype(np.intp, copy=False)


def inverse_shift_reflection_nearest(
    image: Image.Image, content_dx: int, content_dy: int
) -> Image.Image:
    """Apply an integer RGB content translation with nearest/reflection rules."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x_indices = _reflection_indices(width, content_dx)
    y_indices = _reflection_indices(height, content_dy)
    shifted = rgb[y_indices[:, None], x_indices[None, :], :]
    return Image.fromarray(shifted, mode="RGB")


def _memory_snapshot() -> str:
    """Small best-effort RAM log; never loads the cohort into memory."""
    try:
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size
        total = os.sysconf("SC_PHYS_PAGES") * page_size
        return f"available_ram_gib={available / 2**30:.1f}/{total / 2**30:.1f}"
    except (AttributeError, OSError, ValueError):
        return "available_ram_gib=unavailable"


def _resolve_retained_path(recorded_path: str, source_root: Path) -> Path:
    """Resolve a formal-record path after the historic absolute root moved."""
    path = Path(recorded_path)
    if path.is_file():
        return path.resolve()
    try:
        tail_index = path.parts.index("attack_cache")
    except ValueError as exc:
        raise FileNotFoundError(f"recorded source path is missing: {path}") from exc
    remapped = source_root.joinpath(*path.parts[tail_index:])
    if not remapped.is_file():
        raise FileNotFoundError(
            f"recorded source path is missing and remapped path is absent: {remapped}"
        )
    return remapped


def _validate_record(record: dict[str, Any], source_root: Path) -> tuple[Path, int, int, Path]:
    run_id = str(record.get("run_id", ""))
    if not run_id:
        raise ValueError("record is missing run_id")
    attack_path = _resolve_retained_path(str(record.get("attacked_path", "")), source_root)
    debug_path = _resolve_retained_path(str(record.get("debug_info_path", "")), source_root)
    if source_root not in attack_path.parents:
        raise ValueError(f"run_id={run_id}: attacked image is outside source root")

    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("padding_mode", "reflection"),
        ("interpolation_mode", "nearest"),
        ("shift_space", "image_pixels"),
        ("warp_mode", "raven_paper_nfpa_gap_fill"),
    ):
        actual = debug.get(key)
        if actual != expected:
            raise ValueError(
                f"run_id={run_id}: {key}={actual!r}, expected {expected!r}; "
                "refusing to apply this protocol-specific inverse shift"
            )

    dx = debug.get("effective_source_flow_dx_image_px")
    dy = debug.get("effective_source_flow_dy_image_px")
    if dx is None or dy is None:
        raise ValueError(f"run_id={run_id}: missing effective source flow")
    if not (math.isfinite(float(dx)) and math.isfinite(float(dy))):
        raise ValueError(f"run_id={run_id}: non-finite effective source flow")
    if not (float(dx).is_integer() and float(dy).is_integer()):
        raise ValueError(
            f"run_id={run_id}: non-integer effective source flow ({dx}, {dy}); "
            "nearest RGB inverse shift requires integer flow"
        )

    # The formal record is duplicated in debug_info.  Refuse a mismatch rather
    # than silently using a stale record.
    for axis, record_value, debug_value in (
        ("x", record.get("effective_source_flow_dx_image_px"), dx),
        ("y", record.get("effective_source_flow_dy_image_px"), dy),
    ):
        if record_value is not None and float(record_value) != float(debug_value):
            raise ValueError(
                f"run_id={run_id}: record/debug effective source flow mismatch "
                f"on {axis}: {record_value} != {debug_value}"
            )
    recorded_sha = record.get("attacked_sha256") or record.get("output_sha256")
    if recorded_sha and _sha256(attack_path) != str(recorded_sha):
        raise ValueError(f"run_id={run_id}: retained attacked image SHA256 disagrees with manifest")
    recorded_debug_sha = record.get("debug_info_sha256") or record.get("debug_sha256")
    if recorded_debug_sha and _sha256(debug_path) != str(recorded_debug_sha):
        raise ValueError(f"run_id={run_id}: retained debug metadata SHA256 disagrees with manifest")
    return attack_path, int(dx), int(dy), debug_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = args.source_manifest.resolve()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest missing: {source_manifest}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root missing: {source_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is non-empty: {output_dir}; use --resume to continue"
        )
    output_images = output_dir / "inverse_shifted_attacked_images"
    output_images.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "inverse_shift_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"records already exist: {records_path}; use --resume")

    completed_ids: set[str] = set()
    if args.resume and records_path.is_file():
        completed_ids = {str(record["run_id"]) for record in _read_jsonl(records_path)}

    copied_manifest = output_dir / "source_attack_records_watermarked.jsonl"
    if not copied_manifest.exists():
        shutil.copy2(source_manifest, copied_manifest)

    processed = 0
    skipped = 0
    source_count = 0
    with records_path.open("a", encoding="utf-8") as records_handle:
        for record in _read_jsonl(source_manifest):
            if args.limit is not None and source_count >= args.limit:
                break
            source_count += 1
            run_id = str(record.get("run_id", ""))
            if run_id in completed_ids:
                skipped += 1
                continue
            attacked_path, effective_source_dx, effective_source_dy, debug_path = _validate_record(
                record, source_root
            )
            def fixed_direction(value: int) -> int:
                if value == 0:
                    raise ValueError(f"run_id={run_id}: zero effective source flow has no attack direction")
                return 1 if value > 0 else -1

            if args.fixed_inverse_magnitude is None:
                inverse_content_dx = effective_source_dx if "x" in args.axes else 0
                inverse_content_dy = effective_source_dy if "y" in args.axes else 0
                inverse_mode = "exact_effective_source_flow"
            else:
                # visible attack displacement is -effective_source_flow.  Thus
                # -sign(visible_attack)*magnitude equals sign(source_flow)*magnitude.
                inverse_content_dx = (fixed_direction(effective_source_dx) * args.fixed_inverse_magnitude) if "x" in args.axes else 0
                inverse_content_dy = (fixed_direction(effective_source_dy) * args.fixed_inverse_magnitude) if "y" in args.axes else 0
                inverse_mode = "fixed_magnitude_from_effective_attack_direction"
            output_path = output_images / f"{int(run_id):06d}.png"
            with Image.open(attacked_path) as opened:
                source_image = ImageOps.exif_transpose(opened).convert("RGB")
                source_image.load()
            transformed = inverse_shift_reflection_nearest(
                source_image, inverse_content_dx, inverse_content_dy
            )
            transformed.save(output_path)
            output_record = {
                "run_id": run_id,
                "source_attacked_path": str(attacked_path),
                "source_attacked_sha256": _sha256(attacked_path),
                "inverse_shifted_attacked_path": str(output_path),
                "inverse_shifted_attacked_sha256": _sha256(output_path),
                "source_debug_info_path": str(debug_path),
                "planned_flow_dx_image_px": record.get("planned_flow_dx_image_px"),
                "planned_flow_dy_image_px": record.get("planned_flow_dy_image_px"),
                "effective_source_flow_dx_image_px": effective_source_dx,
                "effective_source_flow_dy_image_px": effective_source_dy,
                "effective_visual_shift_dx_image_px": -effective_source_dx,
                "effective_visual_shift_dy_image_px": -effective_source_dy,
                "attack_direction_source_flow_dx": 1 if effective_source_dx > 0 else -1,
                "attack_direction_source_flow_dy": 1 if effective_source_dy > 0 else -1,
                "attack_direction_visible_dx": -1 if effective_source_dx > 0 else 1,
                "attack_direction_visible_dy": -1 if effective_source_dy > 0 else 1,
                "inverse_axes": args.axes,
                "inverse_mode": inverse_mode,
                "fixed_inverse_magnitude_px": args.fixed_inverse_magnitude,
                "inverse_rgb_content_dx_image_px": inverse_content_dx,
                "inverse_rgb_content_dy_image_px": inverse_content_dy,
                "inverse_operation": (
                    "output[y,x]=input[reflect(y-inverse_content_dy),"
                    "reflect(x-inverse_content_dx)]; nearest integer RGB sampling"
                ),
                "padding_mode": "reflection",
                "interpolation_mode": "nearest",
                "note": (
                    "Inverse-shifted attacked image only; not an inversion of "
                    "DDIM/regeneration or reflection-padded latent warp."
                ),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            records_handle.write(json.dumps(output_record, sort_keys=True) + "\n")
            records_handle.flush()
            processed += 1
            del source_image, transformed
            if processed % args.log_every == 0:
                gc.collect()
                LOG.info(
                    "processed=%d source_seen=%d %s", processed, source_count, _memory_snapshot()
                )

    summary = {
        "status": "completed",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "source_records_seen": source_count,
        "processed_count": processed,
        "resume_skipped_count": skipped,
        "limit": args.limit,
        "inverse_axes": args.axes,
        "fixed_inverse_magnitude_px": args.fixed_inverse_magnitude,
        "coordinate_interpretation": (
            "RAVEN effective_source_flow is a grid_sample source displacement; "
            "visible shift is its negative. RGB inverse content shift equals "
            "effective_source_flow and equals -effective_visual_shift."
        ),
        "rgb_padding_mode": "reflection",
        "rgb_interpolation_mode": "nearest",
        "memory_policy": "one 512x512 RGB image at a time; no DataLoader or cohort cache",
    }
    (output_dir / "inverse_shift_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--axes", choices=("xy", "x", "y"), default="xy",
        help="Axes whose effective source flow is inverse-applied: xy (both), x (horizontal only), y (vertical only).",
    )
    parser.add_argument(
        "--fixed-inverse-magnitude", type=int, default=None,
        help="Use only effective attack directions and this fixed per-axis inverse magnitude; omit for exact effective-flow inverse.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.fixed_inverse_magnitude is not None and args.fixed_inverse_magnitude <= 0:
        raise ValueError("--fixed-inverse-magnitude must be positive")
    summary = run(args)
    LOG.info("completed: %s", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
