#!/usr/bin/env python3
"""Repoint GS shared-clean V2 metadata at a moved Tree-Ring source cohort.

When the Tree-Ring cohort moves, its metadata file's path and SHA-256 change.
Every GS shared-clean V2 row records both, and ``shared_clean_source_metadata_sha256``
is part of the V2 pairing hash, so the rows must be updated together:

- ``shared_clean_source_metadata_path``  -> the new TR metadata path
- ``shared_clean_source_metadata_sha256`` -> the new TR metadata SHA-256
- ``pairing_sha256``                     -> recomputed with the authoritative
  :func:`build_pairing_sha256`

Nothing else changes. Image bytes are never touched, and every content SHA in
the row — GS watermarked, clean, latent, target and secret hashes — is asserted
byte-identical before and after. The new TR cohort must itself pass the pairing
audit, and each row's shared-clean identity must still match the TR row it
claims, or the migration fails closed and writes nothing.

Usage::

    python experiments/migrate_gs_shared_clean_source.py --dry-run \\
        --tr-metadata data/tr/diffusiondb/metadata.csv <gs-metadata.csv> ...
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

from raven.eval_protocol import source_metadata_path  # noqa: E402
from raven.pairing_provenance import (  # noqa: E402
    GS_SHARED_TR_CLEAN_PROTOCOL,
    audit_pairing_rows,
    audit_tr_gs_shared_clean,
    build_pairing_sha256,
    sha256_path,
)

MUTABLE_FIELDS = (
    "shared_clean_source_metadata_path",
    "shared_clean_source_metadata_sha256",
    "pairing_sha256",
)

# Every content hash and every image path must survive untouched.
IMMUTABLE_FIELDS = (
    "run_id",
    "clean_path",
    "clean_sha256",
    "watermarked_path",
    "watermarked_image_path",
    "watermarked_sha256",
    "watermarked_latent_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_sampling_uniform_sha256",
    "gs_secret_index",
    "shared_clean_sample_sha256",
    "tr_base_latent_sha256",
    "tr_clean_path",
    "tr_clean_sha256",
)


class GsSourceMigrationError(RuntimeError):
    """Fail-closed migration error: the metadata file is left untouched."""


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return list(reader), fieldnames


def migrate_rows(
    rows: list[dict[str, str]],
    *,
    tr_rows: list[dict[str, str]],
    tr_metadata: Path,
    tr_metadata_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    tr_by_id = {str(row["run_id"]): row for row in tr_rows}
    migrated: list[dict[str, str]] = []
    changed = 0
    for row in rows:
        run_id = str(row.get("run_id", "?"))
        protocol = str(row.get("protocol", ""))
        if protocol != GS_SHARED_TR_CLEAN_PROTOCOL:
            raise GsSourceMigrationError(
                f"row is not a shared-clean V2 row run_id={run_id}: protocol={protocol!r}"
            )
        for field in MUTABLE_FIELDS:
            if field not in row:
                raise GsSourceMigrationError(f"row is missing {field}: run_id={run_id}")

        tr_row = tr_by_id.get(run_id)
        if tr_row is None:
            raise GsSourceMigrationError(
                f"GS run_id={run_id} has no matching row in {tr_metadata}"
            )
        # The row must still describe the same shared clean sample; only the
        # location and digest of the source metadata file may change.
        for gs_field, tr_field in (
            ("tr_base_latent_sha256", "base_latent_sha256"),
            ("tr_clean_sha256", "clean_sha256"),
            ("tr_clean_path", "clean_path"),
            ("base_latent_seed", "base_latent_seed"),
            ("prompt_sha256", "prompt_sha256"),
        ):
            if str(row[gs_field]) != str(tr_row[tr_field]):
                raise GsSourceMigrationError(
                    f"shared-clean identity drift run_id={run_id}: "
                    f"{gs_field}={row[gs_field]!r} vs TR {tr_field}={tr_row[tr_field]!r}"
                )

        new_row = dict(row)
        new_row["shared_clean_source_metadata_path"] = str(tr_metadata)
        new_row["shared_clean_source_metadata_sha256"] = tr_metadata_sha256
        new_row["pairing_sha256"] = build_pairing_sha256(new_row)

        for field in IMMUTABLE_FIELDS:
            if field in row and new_row.get(field) != row.get(field):
                raise GsSourceMigrationError(
                    f"migration would change immutable field {field} run_id={run_id}"
                )
        if any(new_row[field] != row[field] for field in MUTABLE_FIELDS):
            changed += 1
        migrated.append(new_row)

    return migrated, {"rows": len(migrated), "rows_changed": changed}


def write_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def migrate_metadata_file(
    gs_metadata: Path, *, tr_metadata: Path, dry_run: bool = False
) -> dict[str, Any]:
    gs_metadata = Path(gs_metadata).resolve()
    tr_metadata = Path(tr_metadata).resolve()
    if not gs_metadata.is_file():
        raise GsSourceMigrationError(f"GS metadata not found: {gs_metadata}")
    if not tr_metadata.is_file():
        raise GsSourceMigrationError(f"TR metadata not found: {tr_metadata}")

    tr_rows, _ = load_rows(tr_metadata)
    # The new source cohort must be valid before anything is repointed at it.
    audit_pairing_rows(tr_rows, expected_count=len(tr_rows), verify_files=True)
    tr_metadata_sha256 = sha256_path(tr_metadata)

    rows, fieldnames = load_rows(gs_metadata)
    if not rows:
        raise GsSourceMigrationError(f"GS metadata has no rows: {gs_metadata}")

    migrated, stats = migrate_rows(
        rows,
        tr_rows=tr_rows,
        tr_metadata=tr_metadata,
        tr_metadata_sha256=tr_metadata_sha256,
    )
    result = {
        "gs_metadata": str(gs_metadata),
        "tr_metadata": str(tr_metadata),
        "tr_metadata_sha256": tr_metadata_sha256,
        "dry_run": bool(dry_run),
        "sha256_before": sha256_path(gs_metadata),
        **stats,
    }
    if dry_run:
        result["sha256_after"] = None
        return result

    write_atomic(gs_metadata, migrated, fieldnames)

    reloaded, _ = load_rows(gs_metadata)
    result["sha256_after"] = sha256_path(gs_metadata)
    result["pairing_audit"] = audit_pairing_rows(
        reloaded, expected_count=len(reloaded), verify_files=True
    )
    cross = audit_tr_gs_shared_clean(tr_rows, reloaded, verify_files=True)
    result["cross_method_shared_clean_audit"] = {
        key: value for key, value in cross.items() if key != "rows"
    }
    audit_path = gs_metadata.parent / "cross_method_shared_clean_audit.json"
    audit_path.write_text(
        json.dumps(cross, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["cross_method_shared_clean_audit_path"] = str(audit_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gs_metadata", nargs="+", type=Path)
    parser.add_argument(
        "--tr-metadata",
        type=Path,
        default=source_metadata_path("TR", "diffusiondb"),
        help="canonical TR source metadata (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results = []
    for path in args.gs_metadata:
        try:
            results.append(
                migrate_metadata_file(
                    path, tr_metadata=args.tr_metadata, dry_run=args.dry_run
                )
            )
        except Exception as exc:
            print(f"FAILED: {path}: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
