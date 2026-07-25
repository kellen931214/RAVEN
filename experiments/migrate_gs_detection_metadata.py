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
  sampling uniforms are never changed — they are fully **re-derived** and *verified*:
  a fresh ``GsProvider`` is rebuilt from the row's ``gs_secret_index`` /
  ``gs_sampling_seed`` and the sidecar ``watermark_config`` layout (message width,
  channel_copy, hw_copy) at the ``generation_config`` dtype + resolution, and
  ``get_wm_latents()`` is re-run so that ``gs_message/key/nonce/secret_bundle``,
  ``gs_sampling_uniform_sha256``, ``watermarked_latent_sha256``,
  ``watermark_target_sha256`` and the base-latent SHAs all recompute *exactly*.
- The formal mapping ``gs_secret_index == run_id`` and
  ``gs_sampling_seed == base_latent_seed`` is verified per row.
- The sidecar config files are resolved by the metadata filename's shard suffix
  (``metadata.csv`` -> ``watermark_config.json`` / ``generation_config.json``;
  ``metadata.shard-003-of-008.csv`` -> ``watermark_config.shard-003-of-008.json`` /
  ``generation_config.shard-003-of-008.json``). A missing sidecar fails closed — the
  migration never falls back to default parameters. Both configs' canonical SHA-256 is
  recomputed and compared with each row's ``generation_config_sha256`` /
  ``watermark_config_sha256``, and their content (model / revision / scheduler /
  resolution / GS protocol + layout) must agree with the row.
- ``watermarked_latent_sha256``, ``gs_sampling_uniform_sha256``, ``gs_sampling_seed``,
  the secret SHAs, ``base_latent_seed``, the base-latent SHAs,
  ``generation_config_sha256`` and ``watermark_config_sha256`` are never modified.
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
    python experiments/migrate_gs_detection_metadata.py metadata.shard-003-of-008.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
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
    canonical_json_sha256,
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

# SD VAE latent channel count. The generator takes the latent shape from the target
# pipeline (4 channels for SD-2-1-base); a wrong count would surface immediately as a
# GsProvider layout error or a watermarked_latent_sha256 mismatch (fail closed).
LATENT_CHANNELS = 4

# Map the ``generation_config["dtype"]`` string back to a torch dtype so the latent
# re-derivation uses the *exact* dtype used at generation time. The dtype string is
# tied to a verified ``generation_config_sha256``, so it is trustworthy provenance.
_DTYPE_BY_NAME = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,
}

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

# Immutable columns: this migration must never change any of these. Verified
# byte-for-byte after building the migrated row.
IMMUTABLE_FIELDS = (
    "clean_sha256",
    "watermarked_sha256",
    "watermarked_latent_sha256",
    "gs_sampling_uniform_sha256",
    "gs_sampling_seed",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
    "watermark_target_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
    "pairing_sha256",
)

# Provenance columns that must be present and non-empty on every row.
REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "gs_protocol_mode",
    "model_id",
    "model_revision",
    "scheduler_target",
    "resolution",
    "clean_path",
    "watermarked_path",
    "clean_sha256",
    "watermarked_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
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


def _as_int(value: Any, field: str, run_id: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise MigrationError(f"run_id={run_id}: field {field!r}={value!r} is not an integer")


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


# --------------------------------------------------------------------------- #
# Sidecar config resolution (shard-aware) + canonical SHA / content checks
# --------------------------------------------------------------------------- #
def metadata_shard_suffix(csv_path: Path) -> str:
    """Return the shard suffix embedded in a ``metadata{suffix}.csv`` filename.

    ``metadata.csv`` -> ``""``; ``metadata.shard-003-of-008.csv`` ->
    ``".shard-003-of-008"``. The sidecar config files use the identical suffix.
    """
    name = csv_path.name
    if not (name.startswith("metadata") and name.endswith(".csv")):
        raise MigrationError(
            f"unexpected metadata filename (expected metadata*.csv): {name}"
        )
    return name[len("metadata"):-len(".csv")]


def load_sidecar_config(csv_path: Path, suffix: str, stem: str) -> Tuple[Path, Dict[str, Any]]:
    """Load ``{stem}{suffix}.json`` next to the metadata CSV. Fail closed if missing."""
    path = csv_path.with_name(f"{stem}{suffix}.json")
    if not path.is_file():
        raise MigrationError(
            f"{csv_path.name}: required sidecar config {path.name} not found next to "
            f"metadata; refusing to migrate with default parameters (fail closed)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise MigrationError(f"{path}: unreadable/invalid JSON config ({exc})")
    if not isinstance(payload, dict):
        raise MigrationError(f"{path}: config JSON is not an object")
    return path, payload


def resolve_dtype(generation_config: Mapping[str, Any], context: str) -> torch.dtype:
    raw = generation_config.get("dtype")
    if raw is None or str(raw).strip() == "":
        raise MigrationError(
            f"{context}: generation_config missing 'dtype'; cannot re-derive latents"
        )
    dtype = _DTYPE_BY_NAME.get(str(raw).strip())
    if dtype is None:
        raise MigrationError(f"{context}: unrecognized generation_config dtype {raw!r}")
    return dtype


def verify_config_consistency(
    row: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    watermark_config: Mapping[str, Any],
    *,
    gen_sha: str,
    wm_sha: str,
    run_id: str,
) -> None:
    """Verify the sidecar configs match the row's config hashes AND content."""
    # 1) canonical SHA of each sidecar must match what the row recorded.
    if str(row.get("generation_config_sha256")) != gen_sha:
        raise MigrationError(
            f"run_id={run_id}: generation_config_sha256 mismatch vs sidecar config file"
        )
    if str(row.get("watermark_config_sha256")) != wm_sha:
        raise MigrationError(
            f"run_id={run_id}: watermark_config_sha256 mismatch vs sidecar config file"
        )
    # 2) generation_config content vs row.
    for cfg_key, row_key in (
        ("model_id", "model_id"),
        ("model_revision", "model_revision"),
        ("scheduler", "scheduler_target"),
    ):
        cfg_val = str(generation_config.get(cfg_key))
        row_val = str(row.get(row_key))
        if cfg_val != row_val:
            raise MigrationError(
                f"run_id={run_id}: generation_config {cfg_key}={cfg_val!r} != "
                f"metadata {row_key}={row_val!r}"
            )
    cfg_resolution = _as_int(generation_config.get("resolution"), "generation_config.resolution", run_id)
    row_resolution = _as_int(row.get("resolution"), "resolution", run_id)
    if cfg_resolution != row_resolution:
        raise MigrationError(
            f"run_id={run_id}: generation_config resolution={cfg_resolution} != "
            f"metadata resolution={row_resolution}"
        )
    # 3) watermark_config content vs row (protocol + GS layout).
    if str(watermark_config.get("wm_type", "GS")) != "GS":
        raise MigrationError(
            f"run_id={run_id}: watermark_config wm_type is not GS: "
            f"{watermark_config.get('wm_type')!r}"
        )
    if str(watermark_config.get("gs_protocol_mode")) != str(row.get("gs_protocol_mode")):
        raise MigrationError(
            f"run_id={run_id}: watermark_config gs_protocol_mode="
            f"{watermark_config.get('gs_protocol_mode')!r} != metadata "
            f"gs_protocol_mode={row.get('gs_protocol_mode')!r}"
        )
    # GS layout numbers (message_width_in_bytes / channel_copy / hw_copy) drive the
    # re-derived provider; if they disagreed with what produced the images the
    # watermarked_latent / target / uniform SHAs would fail to reproduce below.


def _build_row_provider(
    watermark_config: Mapping[str, Any],
    resolution: int,
    dtype: torch.dtype,
    secret_index: int,
    sampling_seed: int,
) -> GsProvider:
    channel_copy = int(watermark_config.get("channel_copy", 1))
    hw_copy = int(watermark_config.get("hw_copy", 8))
    message_width_in_bytes = int(watermark_config.get("message_width_in_bytes", 32))
    latent_res = int(resolution) // 8
    return GsProvider(
        latent_shape=(1, LATENT_CHANNELS, latent_res, latent_res),
        dtype=dtype,
        device=torch.device("cpu"),
        batch_size=1,
        gs_protocol_mode="official_compatible",
        gs_channel_copy=channel_copy,
        gs_hw_copy=hw_copy,
        message_width_in_bytes=message_width_in_bytes,
        offset=secret_index,
        gs_secret_index=secret_index,
        gs_sampling_seed=sampling_seed,
    )


def verify_row_provenance(
    row: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    watermark_config: Mapping[str, Any],
    *,
    gen_sha: str,
    wm_sha: str,
    dtype: torch.dtype,
    verify_files: bool,
) -> None:
    """Fully re-derive and verify GS provenance for a single row. Fail closed."""
    run_id = str(row.get("run_id", "unknown"))
    for field in REQUIRED_PROVENANCE_FIELDS:
        _require(row, field, run_id)

    if str(row["gs_protocol_mode"]) != "official_compatible":
        raise MigrationError(
            f"run_id={run_id}: gs_protocol_mode={row['gs_protocol_mode']!r}, "
            f"expected official_compatible"
        )

    # Sidecar config hashes + content must agree with the row.
    verify_config_consistency(
        row, generation_config, watermark_config, gen_sha=gen_sha, wm_sha=wm_sha, run_id=run_id
    )

    run_id_int = _as_int(row["run_id"], "run_id", run_id)
    secret_index = _as_int(row["gs_secret_index"], "gs_secret_index", run_id)
    sampling_seed = _as_int(row["gs_sampling_seed"], "gs_sampling_seed", run_id)
    base_latent_seed = _as_int(row["base_latent_seed"], "base_latent_seed", run_id)

    # Formal mapping: secret index == run_id, sampling seed == base latent seed.
    if secret_index != run_id_int:
        raise MigrationError(
            f"run_id={run_id}: gs_secret_index={secret_index} != run_id={run_id_int}"
        )
    if sampling_seed != base_latent_seed:
        raise MigrationError(
            f"run_id={run_id}: gs_sampling_seed={sampling_seed} != "
            f"base_latent_seed={base_latent_seed}"
        )

    # Full re-derivation: rebuild the provider and re-run get_wm_latents().
    resolution = _as_int(generation_config.get("resolution"), "generation_config.resolution", run_id)
    provider = _build_row_provider(watermark_config, resolution, dtype, secret_index, sampling_seed)
    wm = provider.get_wm_latents()
    secret = wm["secret_provenance_list"][0]
    base_sha = tensor_sha256(wm["zT_clean_torch"])
    derived = {
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "gs_sampling_uniform_sha256": wm["sampling_uniform_sha256_list"][0],
        "watermarked_latent_sha256": tensor_sha256(wm["zT_torch"]),
        "watermark_target_sha256": tensor_sha256(wm["barcodes_torch"]),
        "base_latent_sha256": base_sha,
        "clean_base_latent_sha256": base_sha,
        "watermarked_base_latent_sha256": base_sha,
    }
    for field, expected in derived.items():
        if str(row.get(field)) != str(expected):
            raise MigrationError(
                f"run_id={run_id}: {field} mismatch on full re-derivation "
                f"(GS latent/sampling/secret provenance drift)"
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
    for immutable in IMMUTABLE_FIELDS:
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

    # Resolve the shard-suffixed sidecar configs (fail closed if either is missing).
    suffix = metadata_shard_suffix(csv_path)
    gen_path, generation_config = load_sidecar_config(csv_path, suffix, "generation_config")
    wm_path, watermark_config = load_sidecar_config(csv_path, suffix, "watermark_config")
    gen_sha = canonical_json_sha256(generation_config)
    wm_sha = canonical_json_sha256(watermark_config)
    dtype = resolve_dtype(generation_config, str(csv_path))

    migrated_rows: List[Dict[str, Any]] = []
    for row in rows:
        verify_row_provenance(
            row,
            generation_config,
            watermark_config,
            gen_sha=gen_sha,
            wm_sha=wm_sha,
            dtype=dtype,
            verify_files=verify_files,
        )
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
        "shard_suffix": suffix,
        "generation_config": str(gen_path),
        "watermark_config": str(wm_path),
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
