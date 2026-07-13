#!/usr/bin/env python
"""Audit RAVEN dataset/result pairing without running a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_COLUMNS = {
    "clean": ("clean_path", "clean_image", "original_path"),
    "watermarked": ("watermarked_path", "image_path", "before_path", "path"),
    "attacked": ("attacked_path", "after_path", "raven_path", "output_path"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_value(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_path(value: str | None, metadata: Path, workspace: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    candidates = [candidate] if candidate.is_absolute() else [metadata.parent / candidate, workspace / candidate]
    for item in candidates:
        if item.is_file():
            return item.resolve()
    return candidates[0].resolve()


def inspect_image(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with Image.open(path) as opened:
        source_mode = opened.mode
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return {
            "path": str(path),
            "exists": True,
            "sha256": sha256(path),
            "source_mode": source_mode,
            "decoded_mode": image.mode,
            "width": image.width,
            "height": image.height,
            "rgb_512": image.mode == "RGB" and image.size == (512, 512),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output; stdout is always written")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = args.metadata.resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {metadata}")

    identifiers = []
    path_counts: Counter[str] = Counter()
    samples = []
    missing = []
    invalid_images = []
    for index, row in enumerate(rows):
        identifier = first_value(row, ("run_id", "sample_id", "id", "index")) or str(index)
        identifiers.append(identifier)
        images = {}
        for stage, columns in IMAGE_COLUMNS.items():
            path = resolve_path(first_value(row, columns), metadata, args.workspace_root.resolve())
            info = inspect_image(path)
            images[stage] = info
            if info:
                path_counts[info["path"]] += 1
                if not info["exists"]:
                    missing.append({"run_id": identifier, "stage": stage, "path": info["path"]})
                elif not info["rgb_512"]:
                    invalid_images.append({"run_id": identifier, "stage": stage, **info})
        if index < args.sample_limit:
            samples.append({
                "run_id": identifier,
                "prompt": first_value(row, ("prompt", "caption", "text")),
                "seed": first_value(row, ("seed", "generation_seed")),
                "attack_seed": first_value(row, ("attack_seed", "raven_seed")),
                "key_index": first_value(row, ("offset", "key_index", "sample_index", "fix_gt")),
                "dx": first_value(row, ("dx", "shift_x")),
                "dy": first_value(row, ("dy", "shift_y")),
                "images": images,
            })

    duplicate_ids = sorted(value for value, count in Counter(identifiers).items() if count > 1)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    report = {
        "metadata": str(metadata),
        "num_rows": len(rows),
        "duplicate_run_ids": duplicate_ids,
        "duplicate_paths": duplicate_paths,
        "missing_images": missing,
        "invalid_images": invalid_images,
        "samples": samples,
        "ok": not duplicate_ids and not duplicate_paths and not missing and not invalid_images,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
