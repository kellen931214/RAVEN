#!/usr/bin/env python3
"""Repoint Tree-Ring metadata at the flat cohort layout.

The Tree-Ring cohort moved from::

    data/tr/<dataset>/TR/<run_id>/watermarked.png
    data/tr/<dataset>/TR/metadata.csv

to the flat canonical layout::

    data/tr/<dataset>/<run_id>/watermarked.png
    data/tr/<dataset>/metadata.csv

Only ``watermarked_path`` and ``watermarked_image_path`` are rewritten. Every
other cell — including every content SHA-256 and ``pairing_sha256`` — is
preserved byte-for-byte, which is sound because paths are deliberately not part
of ``PAIRING_HASH_FIELDS``.

The images themselves are never regenerated, moved, renamed or re-encoded: this
tool only reads them to verify that each recorded ``watermarked_sha256`` still
matches the file at its new location. Any mismatch, missing file, or unexpected
pairing-hash drift fails closed and nothing is written.

Usage::

    python experiments/migrate_tr_flat_layout.py --dry-run <metadata.csv> ...
    python experiments/migrate_tr_flat_layout.py <metadata.csv> ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
RAVEN_ROOT = WORKSPACE / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.pairing_provenance import (  # noqa: E402
    audit_pairing_rows,
    build_pairing_sha256,
    sha256_path,
)

PATH_FIELDS = ("watermarked_path", "watermarked_image_path")

# Everything below must survive the migration bit-for-bit.
IMMUTABLE_FIELDS = (
    "run_id",
    "clean_path",
    "clean_sha256",
    "watermarked_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
    "watermarked_latent_sha256",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
    "pairing_sha256",
)


class TrMigrationError(RuntimeError):
    """Fail-closed migration error: the metadata file is left untouched."""


def flat_watermarked_path(old: str, csv_path: Path) -> Path:
    """Map a recorded watermarked path onto the flat cohort layout.

    The cohort root is taken from the metadata file's own directory, so the
    mapping never depends on a hard-coded dataset name.
    """
    old_path = Path(str(old))
    if old_path.name != "watermarked.png":
        raise TrMigrationError(f"unexpected watermarked filename: {old_path}")
    run_dir = old_path.parent.name
    if not run_dir or not run_dir.isdigit():
        raise TrMigrationError(f"cannot resolve run directory from path: {old_path}")
    return csv_path.parent / run_dir / "watermarked.png"


def migrate_rows(rows: list[dict[str, str]], csv_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    migrated: list[dict[str, str]] = []
    rewritten = 0
    already_flat = 0
    for row in rows:
        run_id = str(row.get("run_id", "?"))
        new_row = dict(row)
        recorded = str(row.get("watermarked_path", ""))
        if not recorded:
            raise TrMigrationError(f"row is missing watermarked_path: run_id={run_id}")
        target = flat_watermarked_path(recorded, csv_path)
        if not target.is_file():
            raise TrMigrationError(
                f"watermarked image missing at the flat layout run_id={run_id}: {target}"
            )
        expected_sha = str(row.get("watermarked_sha256", ""))
        if not expected_sha:
            raise TrMigrationError(f"row is missing watermarked_sha256: run_id={run_id}")
        actual_sha = sha256_path(target)
        if actual_sha != expected_sha:
            raise TrMigrationError(
                f"watermarked SHA mismatch run_id={run_id}: "
                f"recorded={expected_sha} actual={actual_sha} at {target}"
            )
        resolved = str(target.resolve())
        for field in PATH_FIELDS:
            if field not in row:
                raise TrMigrationError(f"row is missing {field}: run_id={run_id}")
            if str(row[field]) != resolved:
                new_row[field] = resolved
        if any(new_row[field] != row[field] for field in PATH_FIELDS):
            rewritten += 1
        else:
            already_flat += 1

        for field in IMMUTABLE_FIELDS:
            if field in row and new_row.get(field) != row.get(field):
                raise TrMigrationError(
                    f"migration would change immutable field {field} run_id={run_id}"
                )
        # Paths are not hashed, so the pairing hash must be unchanged and must
        # still recompute from the migrated row.
        recomputed = build_pairing_sha256(new_row)
        if recomputed != str(row["pairing_sha256"]):
            raise TrMigrationError(
                f"pairing hash drifted during migration run_id={run_id}: "
                f"stored={row['pairing_sha256']} recomputed={recomputed}"
            )
        migrated.append(new_row)

    return migrated, {
        "rows": len(migrated),
        "paths_rewritten": rewritten,
        "already_flat": already_flat,
    }


def write_atomic(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)


def migrate_metadata_file(csv_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise TrMigrationError(f"metadata file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise TrMigrationError(f"metadata file has no rows: {csv_path}")
    for field in PATH_FIELDS:
        if field not in fieldnames:
            raise TrMigrationError(f"metadata is missing {field}: {csv_path}")

    migrated, stats = migrate_rows(rows, csv_path)
    result = {
        "metadata_csv": str(csv_path),
        "dry_run": bool(dry_run),
        "sha256_before": sha256_path(csv_path),
        **stats,
    }
    if dry_run:
        result["sha256_after"] = None
        return result

    write_atomic(csv_path, migrated, fieldnames)

    # Re-read from disk and run the real audit: the migration is only complete
    # when the file on disk passes the authoritative pairing audit.
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reloaded = list(csv.DictReader(handle))
    audit = audit_pairing_rows(reloaded, expected_count=len(rows), verify_files=True)
    result["sha256_after"] = sha256_path(csv_path)
    result["pairing_audit"] = audit
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results = []
    for path in args.metadata:
        try:
            results.append(migrate_metadata_file(path, dry_run=args.dry_run))
        except Exception as exc:
            print(f"FAILED: {path}: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
