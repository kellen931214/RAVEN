#!/usr/bin/env python
"""Migrate existing official-compatible Gaussian Shading metadata to the official
detection-threshold schema WITHOUT regenerating any images.

Background
----------
GS images generated before commit 02b669c recorded the legacy detection threshold
(``detection_threshold=0.70703125``, ``detection_threshold_type=legacy_default_threshold``)
and computed ``before_detection_successful`` against that legacy threshold. The GS
default is now the official beta-tail ``tau_onebit`` (``official_onebit`` detection
mode). Those images do NOT need to be regenerated — only the detection metadata is
stale. This script upgrades the metadata in place.

Hard guarantees (fail closed on any inconsistency):
- PNG files are never read for pixels nor rewritten; their recorded SHA-256 is only
  *verified* against the file on disk.
- Watermark latent construction, message/key/nonce mapping, gs_sampling_seed and
  sampling uniforms are never changed — they are re-derived and *verified*.
- ``watermarked_latent_sha256``, ``generation_config_sha256`` and
  ``watermark_config_sha256`` are never modified.
- Detection fields are NOT part of ``PAIRING_HASH_FIELDS``/``GS_REQUIRED_FIELDS``,
  so ``pairing_sha256`` must stay identical; the script recomputes it and fails
  closed if it would change.
- A timestamped backup is written and the file is replaced atomically.
- The migration is idempotent: re-running on already-migrated metadata makes no
  change (no backup, no write).
- After migration the full pairing audit is re-run and must pass.

``before_detection_successful`` is recomputed as ``bit_accuracy >= tau_onebit``. The
bit accuracy is read from an existing metadata column (never re-inverted). If no
usable bit-accuracy column exists, the row is rejected (re-run inversion / score
extraction; do NOT regenerate the PNG).

Usage
-----
    python experiments/migrate_gs_detection_metadata.py <metadata.csv> [<metadata.csv> ...]
    python experiments/migrate_gs_detection_metadata.py --dry-run <metadata.csv>
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "eval_bench_wm"))
sys.path.insert(0, str(WORKSPACE / "raven_repro"))

import torch  # noqa: E402

from raven.pairing_provenance import (  # noqa: E402
    audit_pairing_rows,
    build_pairing_sha256,
    sha256_path,
    tensor_sha256,
)
from utils.wm.gs_provider import GsProvider  # noqa: E402

# --- Official detection schema (kept in lockstep with gs_provider defaults) ---
TAU_ONEBIT = 0.6484375
TAU_BITS = 0.71484375
DETECTION_MODE = "official_onebit"
DETECTION_THRESHOLD = TAU_ONEBIT
DETECTION_THRESHOLD_TYPE = "official_beta_tail_tau_onebit"
DETECTION_COMPARISON = ">="
WATERMARK_IMPLEMENTATION_PROTOCOL = "official_compatible"
GENERATION_BENCHMARK_PROTOCOL = "shared_formal_cohort_redbeardnz_ddim"
UPSTREAM_OFFICIAL_REPRODUCTION_RUNNER = (
    "stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16"
)

# Columns that may legitimately carry the pre-inversion GS bit accuracy. The value
# is only read, never recomputed. Multiple present values must agree.
ALLOWED_BIT_ACCURACY_FIELDS = ("before_bit_accuracy", "before_detection_metric_value")

# Detection/protocol columns this migration sets. NONE of these are part of
# PAIRING_HASH_FIELDS / GS_REQUIRED_FIELDS, so pairing_sha256 must not change.
DETECTION_FIELDS = (
    "gs_detection_mode",
    "gs_official_tau_onebit",
    "gs_official_tau_bits",
    "detection_threshold",
    "detection_threshold_type",
    "detection_threshold_comparison_operator",
    "threshold_calibrated_from_current_clean_negatives",
    "watermark_implementation_protocol",
    "generation_benchmark_protocol",
    "upstream_official_reproduction_runner",
    "before_detection_successful",
)

# Provenance columns that must be present and non-empty on every row.
REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "gs_protocol_mode",
    "clean_path",
    "watermarked_path",
    "clean_sha256",
    "watermarked_sha256",
    "watermarked_latent_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
    "gs_secret_index",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_sampling_seed",
    "gs_sampling_uniform_sha256",
    "watermark_target_sha256",
    "pairing_sha256",
)


class MigrationError(RuntimeError):
    """Raised on any inconsistency; migration fails closed."""


def _require(row: Mapping[str, Any], field: str, run_id: str) -> str:
    value = row.get(field)
    if value is None or str(value) == "":
        raise MigrationError(f"run_id={run_id}: missing required field {field!r}")
    return str(value)


def resolve_bit_accuracy(row: Mapping[str, Any], run_id: str) -> float:
    """Return the recorded pre-inversion GS bit accuracy, or fail closed.

    Reads from ALLOWED_BIT_ACCURACY_FIELDS only. If several are present they must
    agree; if none is present/parseable the row is rejected (the caller must re-run
    inversion / score extraction — never regenerate the PNG).
    """
    values: Dict[str, float] = {}
    for field in ALLOWED_BIT_ACCURACY_FIELDS:
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            values[field] = float(raw)
        except (TypeError, ValueError):
            raise MigrationError(
                f"run_id={run_id}: bit-accuracy field {field!r}={raw!r} is not numeric"
            )
    if not values:
        raise MigrationError(
            f"run_id={run_id}: no usable bit-accuracy column "
            f"({', '.join(ALLOWED_BIT_ACCURACY_FIELDS)}); re-run inversion/score "
            f"extraction (do NOT regenerate the PNG)"
        )
    distinct = list(values.values())
    if max(distinct) - min(distinct) > 1e-9:
        raise MigrationError(
            f"run_id={run_id}: inconsistent bit-accuracy columns {values!r}"
        )
    acc = distinct[0]
    if not math.isfinite(acc) or not (0.0 <= acc <= 1.0):
        raise MigrationError(f"run_id={run_id}: bit accuracy out of range: {acc!r}")
    return acc


def _build_provider(watermark_config: Mapping[str, Any], resolution: int) -> GsProvider:
    channel_copy = int(watermark_config.get("channel_copy", 1))
    hw_copy = int(watermark_config.get("hw_copy", 8))
    message_width_in_bytes = int(watermark_config.get("message_width_in_bytes", 32))
    latent_res = int(resolution) // 8
    return GsProvider(
        latent_shape=(1, 4, latent_res, latent_res),
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=1,
        gs_protocol_mode="official_compatible",
        gs_channel_copy=channel_copy,
        gs_hw_copy=hw_copy,
        message_width_in_bytes=message_width_in_bytes,
    )


def verify_row_provenance(
    row: Mapping[str, Any], provider: GsProvider, *, verify_files: bool
) -> None:
    """Re-derive and verify GS provenance for a single row. Fail closed."""
    run_id = str(row.get("run_id", "unknown"))
    for field in REQUIRED_PROVENANCE_FIELDS:
        _require(row, field, run_id)

    if str(row["gs_protocol_mode"]) != "official_compatible":
        raise MigrationError(
            f"run_id={run_id}: gs_protocol_mode={row['gs_protocol_mode']!r}, "
            f"expected official_compatible"
        )

    secret_index = int(row["gs_secret_index"])
    sp = provider.secret_provenance(secret_index)
    for label, stored_field in (
        ("message", "gs_message_sha256"),
        ("key", "gs_key_sha256"),
        ("nonce", "gs_nonce_sha256"),
    ):
        recomputed = sp[f"{label}_sha256"]
        if str(row[stored_field]) != recomputed:
            raise MigrationError(
                f"run_id={run_id}: {stored_field} mismatch (secret mapping drift)"
            )
    if str(row["gs_secret_bundle_sha256"]) != sp["secret_bundle_sha256"]:
        raise MigrationError(f"run_id={run_id}: gs_secret_bundle_sha256 mismatch")

    # Watermark target (payload) is fully re-derivable from the secret index.
    recomputed_target = tensor_sha256(provider.watermark_target_tensor(secret_index))
    if str(row["watermark_target_sha256"]) != recomputed_target:
        raise MigrationError(
            f"run_id={run_id}: watermark_target_sha256 mismatch (latent/payload drift)"
        )

    # pairing_sha256 must already be consistent before we touch anything.
    if str(row["pairing_sha256"]) != build_pairing_sha256(row):
        raise MigrationError(f"run_id={run_id}: pairing_sha256 mismatch before migration")

    if verify_files:
        for label, path_field, sha_field in (
            ("clean", "clean_path", "clean_sha256"),
            ("watermarked", "watermarked_path", "watermarked_sha256"),
        ):
            path = Path(str(row[path_field]))
            if not path.is_file():
                raise MigrationError(f"run_id={run_id}: missing {label} image {path}")
            actual = sha256_path(path)
            if actual != str(row[sha_field]):
                raise MigrationError(
                    f"run_id={run_id}: {label} image SHA mismatch "
                    f"({actual} != {row[sha_field]})"
                )


def migrate_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new row dict with official detection fields, PNG/latent untouched."""
    run_id = str(row.get("run_id", "unknown"))
    bit_accuracy = resolve_bit_accuracy(row, run_id)
    migrated = dict(row)
    migrated["gs_detection_mode"] = DETECTION_MODE
    migrated["gs_official_tau_onebit"] = repr(TAU_ONEBIT)
    migrated["gs_official_tau_bits"] = repr(TAU_BITS)
    migrated["detection_threshold"] = repr(DETECTION_THRESHOLD)
    migrated["detection_threshold_type"] = DETECTION_THRESHOLD_TYPE
    migrated["detection_threshold_comparison_operator"] = DETECTION_COMPARISON
    migrated["threshold_calibrated_from_current_clean_negatives"] = "False"
    migrated["watermark_implementation_protocol"] = WATERMARK_IMPLEMENTATION_PROTOCOL
    migrated["generation_benchmark_protocol"] = GENERATION_BENCHMARK_PROTOCOL
    migrated["upstream_official_reproduction_runner"] = UPSTREAM_OFFICIAL_REPRODUCTION_RUNNER
    migrated["before_detection_successful"] = str(bit_accuracy >= TAU_ONEBIT)

    # Detection fields must not perturb the pairing hash.
    if build_pairing_sha256(migrated) != build_pairing_sha256(row):
        raise MigrationError(
            f"run_id={run_id}: pairing_sha256 would change after migration (aborting)"
        )
    # Immutable provenance fields must be byte-identical.
    for immutable in (
        "watermarked_latent_sha256",
        "generation_config_sha256",
        "watermark_config_sha256",
        "clean_sha256",
        "watermarked_sha256",
        "watermark_target_sha256",
        "gs_sampling_uniform_sha256",
    ):
        if str(migrated.get(immutable)) != str(row.get(immutable)):
            raise MigrationError(f"run_id={run_id}: immutable field {immutable!r} changed")
    return migrated


def _unified_fieldnames(original: List[str]) -> List[str]:
    fieldnames = list(original)
    for field in DETECTION_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def _rows_to_csv_text(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue()


def migrate_metadata_file(
    csv_path: Path, *, dry_run: bool = False, verify_files: bool = True
) -> Dict[str, Any]:
    """Migrate one GS metadata.csv. Returns a summary dict. Fail closed."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise MigrationError(f"metadata file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        original_fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    if not rows:
        raise MigrationError(f"no rows in {csv_path}")

    # Only GS cohorts are in scope.
    methods = {str(r.get("wm_type", "")).upper() for r in rows}
    if methods != {"GS"}:
        raise MigrationError(
            f"{csv_path}: expected a pure GS cohort, found wm_type set {methods!r}"
        )

    watermark_config = _load_watermark_config(csv_path)
    resolution = int(rows[0].get("resolution") or 512)
    provider = _build_provider(watermark_config, resolution)

    migrated_rows: List[Dict[str, Any]] = []
    for row in rows:
        verify_row_provenance(row, provider, verify_files=verify_files)
        migrated_rows.append(migrate_row(row))

    fieldnames = _unified_fieldnames(original_fieldnames)
    new_text = _rows_to_csv_text(migrated_rows, fieldnames)
    old_bytes = csv_path.read_bytes()
    already_current = new_text.encode("utf-8") == old_bytes

    # Re-run the full pairing audit on the migrated rows (comprehensive integrity).
    audit = audit_pairing_rows(
        migrated_rows, expected_count=len(migrated_rows), verify_files=verify_files
    )
    if not audit.get("passed", False):
        raise MigrationError(f"{csv_path}: pairing audit failed after migration")

    summary = {
        "metadata_csv": str(csv_path),
        "rows": len(rows),
        "already_current": already_current,
        "changed": (not already_current) and (not dry_run),
        "dry_run": dry_run,
        "pairing_audit_passed": bool(audit.get("passed", False)),
        "backup_path": None,
    }

    if already_current:
        return summary  # idempotent: nothing to do, no backup, no write
    if dry_run:
        return summary

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_path = csv_path.with_name(f"{csv_path.name}.premigration.{ts}.bak")
    shutil.copy2(csv_path, backup_path)
    summary["backup_path"] = str(backup_path)

    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(new_text, encoding="utf-8")
    os.replace(tmp_path, csv_path)  # atomic on same filesystem
    return summary


def _load_watermark_config(csv_path: Path) -> Dict[str, Any]:
    import json

    for candidate in (
        csv_path.with_name("watermark_config.json"),
        csv_path.parent / "watermark_config.json",
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                break
    return {}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_csv", nargs="+", help="GS metadata.csv file(s) to migrate")
    parser.add_argument("--dry-run", action="store_true", help="validate + report without writing")
    parser.add_argument(
        "--no-verify-files",
        action="store_true",
        help="skip on-disk PNG SHA verification (NOT recommended)",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.metadata_csv:
        try:
            summary = migrate_metadata_file(
                Path(path), dry_run=args.dry_run, verify_files=not args.no_verify_files
            )
            print(summary, flush=True)
        except MigrationError as exc:
            print(f"FAIL CLOSED: {exc}", file=sys.stderr, flush=True)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
