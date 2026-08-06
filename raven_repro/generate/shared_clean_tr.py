#!/usr/bin/env python3
"""Shared canonical-Tree-Ring-clean plumbing for ``shared_tr_clean_v2`` runners.

The V2 contract (Issue #6 / Issue #9) is that every watermarking method consumes
*the same* Tree-Ring source row: the same prompt, the same base latent, the same
clean image on disk and the same generation configuration. Only the
method-specific watermarked image is ever produced.

This module owns the parts of that contract that are identical for every method
— canonical row loading and auditing, base-latent reconstruction and
verification, clean-image verification, generation-config verification, CSV
append/resume plumbing, and the before/after proof that the canonical clean image
was not touched. It deliberately contains **no** watermark algorithm: GaussMarker
and T2SMark live in ``eval_bench_wm/utils/wm/gm_provider.py`` and
``eval_bench_wm/utils/wm/t2s_provider.py``, and Gaussian Shading in
``gs_provider.py``.

Every check here fails closed. Nothing in this module ever writes to, copies,
re-encodes, renames or deletes a canonical clean artifact.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_HERE = Path(__file__).resolve()
WORKSPACE = _HERE.parents[2]                         # repo root
RAVEN_ROOT = _HERE.parents[1]                        # raven_repro/
BENCH_ROOT = WORKSPACE / "eval_bench_wm"             # repo root / eval_bench_wm
for _root in (str(RAVEN_ROOT), str(BENCH_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from raven.pairing_provenance import (  # noqa: E402
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    audit_pairing_rows,
    canonical_json_sha256,
    sha256_path,
    tensor_sha256,
)

#: SD 2.1 latent channel count. The canonical TR cohort is (1, 4, 64, 64).
LATENT_CHANNELS = 4

__all__ = [
    "LATENT_CHANNELS",
    "SHARED_CLEAN_PROTOCOL",
    "SHARED_CLEAN_SOURCE_METHOD",
    "SharedCleanError",
    "CleanImageGuard",
    "append_row",
    "canonical_json_sha256",
    "entrypoint_provenance",
    "existing_completed_rows",
    "finalize_run_manifest",
    "git_provenance",
    "load_tr_rows",
    "preflight_run_manifest",
    "rebuild_shared_clean_latent",
    "save_json",
    "run_manifest_path",
    "select_rows",
    "sha256_path",
    "shard_suffix",
    "str_to_bool",
    "tensor_sha256",
    "verify_generation_config",
    "verify_source_clean_image",
    "verify_source_prompt",
]


class SharedCleanError(RuntimeError):
    """Raised for any shared-clean provenance violation (always fail closed)."""


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def shard_suffix(num_shards: int, shard_index: int) -> str:
    if num_shards == 1:
        return ""
    return f".shard-{shard_index:03d}-of-{num_shards:03d}"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_provenance(root: Optional[Path] = None) -> Dict[str, Any]:
    """Branch / commit / dirty flag for the code that is producing this cohort."""
    root = Path(root) if root is not None else WORKSPACE

    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.stdout.strip()

    status = _run("status", "--porcelain")
    return {
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _run("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def entrypoint_provenance(entrypoint: Path, provider_module: Any) -> Dict[str, Any]:
    """Runner + authoritative provider file identity.

    Recording the provider's own file SHA is what makes "the algorithm came from
    the authoritative provider" auditable after the fact.
    """
    entrypoint = Path(entrypoint).resolve()
    provider_path = Path(provider_module.__file__).resolve()
    return {
        "entrypoint_path": str(entrypoint),
        "entrypoint_sha256": sha256_path(entrypoint),
        "provider_entrypoint_path": str(provider_path),
        "provider_entrypoint_sha256": sha256_path(provider_path),
    }


# --------------------------------------------------------------------------- #
# Canonical Tree-Ring source
# --------------------------------------------------------------------------- #

def load_tr_rows(path: Path) -> List[Dict[str, str]]:
    """Load and fully audit the Tree-Ring source cohort (fail closed)."""
    if not path.is_file():
        raise SharedCleanError(f"Tree-Ring source metadata not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SharedCleanError(f"Tree-Ring source metadata is empty: {path}")
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return rows


def select_rows(
    rows: List[Dict[str, str]],
    *,
    num_shards: int,
    shard_index: int,
    run_ids: Optional[Iterable[int]],
    limit: Optional[int],
) -> List[Dict[str, str]]:
    selected = [row for row in rows if int(row["run_id"]) % num_shards == shard_index]
    if run_ids is not None:
        wanted = {int(value) for value in run_ids}
        selected = [row for row in selected if int(row["run_id"]) in wanted]
        missing = wanted - {int(row["run_id"]) for row in selected}
        if missing:
            raise SharedCleanError(
                f"requested run_ids not present in this shard: {sorted(missing)}"
            )
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise SharedCleanError("no Tree-Ring source rows selected")
    return selected


def rebuild_shared_clean_latent(torch, tr_row: Mapping[str, Any], *, resolution: int, device, dtype):
    """Rebuild the canonical TR base latent and prove it is the recorded one.

    The canonical procedure is a CPU float32 ``torch.randn`` seeded with the
    row's ``base_latent_seed``. The rebuilt tensor must match both
    ``base_latent_sha256`` and ``clean_base_latent_sha256``; anything else means
    this row cannot be reproduced and the run stops.
    """
    run_id = str(tr_row["run_id"])
    base_latent_seed = int(tr_row["base_latent_seed"])
    latent_shape = (1, LATENT_CHANNELS, resolution // 8, resolution // 8)
    generator = torch.Generator(device="cpu").manual_seed(base_latent_seed)
    base_cpu = torch.randn(
        latent_shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    shared_clean_latent = base_cpu.to(device=device, dtype=dtype)
    actual = tensor_sha256(shared_clean_latent)
    for field in ("base_latent_sha256", "clean_base_latent_sha256"):
        expected = str(tr_row[field])
        if actual != expected:
            raise SharedCleanError(
                f"rebuilt TR latent does not match {field} run_id={run_id}: "
                f"expected {expected}, got {actual}"
            )
    return base_cpu, shared_clean_latent, actual


def verify_source_prompt(tr_row: Mapping[str, Any]) -> str:
    """Verify the recorded prompt hash, and return the canonical prompt SHA."""
    run_id = str(tr_row["run_id"])
    prompt = str(tr_row["prompt"])
    if prompt == "":
        raise SharedCleanError(f"TR source row has an empty prompt run_id={run_id}")
    actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    expected = str(tr_row["prompt_sha256"])
    if actual != expected:
        raise SharedCleanError(
            f"TR prompt hash mismatch run_id={run_id}: expected {expected}, got {actual}"
        )
    return actual


def verify_source_clean_image(tr_row: Mapping[str, Any]) -> Path:
    """Verify the canonical clean image exists on disk with the recorded SHA."""
    run_id = str(tr_row["run_id"])
    clean_path = Path(str(tr_row["clean_path"]))
    if not clean_path.is_file():
        raise SharedCleanError(f"TR clean image missing run_id={run_id}: {clean_path}")
    actual = sha256_path(clean_path)
    if actual != str(tr_row["clean_sha256"]):
        raise SharedCleanError(
            f"TR clean image SHA drift run_id={run_id}: "
            f"expected {tr_row['clean_sha256']}, got {actual}"
        )
    return clean_path


def verify_generation_config(
    tr_rows: Iterable[Mapping[str, Any]], generation_config: Mapping[str, Any]
) -> str:
    """The cohort's generation configuration must be *the* TR configuration.

    Not merely "compatible": the shared-clean protocol only means anything if the
    model, revision, scheduler, steps, guidance, resolution and dtype are the
    ones that produced the canonical clean images.
    """
    generation_config_sha256 = canonical_json_sha256(dict(generation_config))
    tr_hashes = {str(row["generation_config_sha256"]) for row in tr_rows}
    if tr_hashes != {generation_config_sha256}:
        raise SharedCleanError(
            "generation config does not match the Tree-Ring cohort: "
            f"method={generation_config_sha256} tr={sorted(tr_hashes)}; the "
            "shared-clean protocol requires an identical model/revision/scheduler/"
            "steps/guidance/resolution/dtype configuration"
        )
    return generation_config_sha256


#: Name of the fail-closed run manifest each cohort directory carries.
RUN_MANIFEST_FILENAME = "run_manifest"


def run_manifest_path(method_dir: Path, suffix: str = "") -> Path:
    return Path(method_dir) / f"{RUN_MANIFEST_FILENAME}{suffix}.json"


def preflight_run_manifest(
    path: Path, early: Mapping[str, Any], *, resume: bool
) -> Optional[Dict[str, Any]]:
    """First gate of a rerun — runs before any pipeline or bundle is built.

    Returns the stored manifest when this run may continue an existing cohort,
    or ``None`` for a brand-new one. It reads; it never writes. That ordering is
    the point: a rerun that will be rejected must be rejected *before* a model is
    loaded or a GM bundle is created, so a failed resume cannot leave a stray
    artifact behind (see the 2026-07-28 GaussMarker entry in DEBUG_CHANGELOG.md).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SharedCleanError(f"run manifest is not valid JSON: {path}: {exc}") from None
    if not resume:
        raise SharedCleanError(
            f"{path} already describes a run of this cohort; pass --resume to "
            "continue it. Nothing was modified."
        )
    for field, value in early.items():
        if str(stored.get(field, "")) != str(value):
            raise SharedCleanError(
                f"run manifest mismatch for {path} field={field}: "
                f"stored={stored.get(field)!r} requested={value!r}. "
                "Nothing was modified."
            )
    return stored


def finalize_run_manifest(
    path: Path, stored: Optional[Mapping[str, Any]], payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Second gate — the full run identity, once the provider and pipeline exist.

    On a compatible resume the stored manifest is returned untouched, so its
    ``created_utc`` keeps recording when the cohort was actually started. On any
    mismatch nothing is written.
    """
    payload = dict(payload)
    run_config_sha256 = canonical_json_sha256(payload)
    if stored is not None:
        if str(stored.get("run_config_sha256", "")) != run_config_sha256:
            raise SharedCleanError(
                f"run configuration changed since {path} was written: "
                f"stored={stored.get('run_config_sha256')} current={run_config_sha256}. "
                "Nothing was modified; use a new output directory for a new "
                "configuration."
            )
        return dict(stored)
    record = dict(payload)
    record["run_config_sha256"] = run_config_sha256
    record["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    save_json(Path(path), record)
    return record


class CleanImageGuard:
    """Prove the canonical clean images were not modified by a run.

    Snapshot the size, mtime and SHA-256 of every clean image the run reads, and
    re-check them afterwards. A single mismatch stops the run: the clean cohort
    is a read-only input shared with Tree-Ring, Gaussian Shading and every other
    method.
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _fingerprint(path: Path) -> Dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_path(path),
        }

    def snapshot(self, path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise SharedCleanError(f"canonical clean image missing: {path}")
        fingerprint = self._fingerprint(path)
        if expected_sha256 is not None and fingerprint["sha256"] != str(expected_sha256):
            raise SharedCleanError(
                f"canonical clean image SHA drift before generation: {path} "
                f"expected {expected_sha256}, got {fingerprint['sha256']}"
            )
        stored = self._snapshots.setdefault(str(path), fingerprint)
        if stored["sha256"] != fingerprint["sha256"]:
            raise SharedCleanError(
                f"canonical clean image changed between reads: {path}"
            )
        return dict(fingerprint)

    def assert_unchanged(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Re-hash the tracked clean images and prove nothing changed."""
        targets = (
            [str(Path(path))] if path is not None else sorted(self._snapshots)
        )
        report: List[Dict[str, Any]] = []
        for key in targets:
            before = self._snapshots.get(key)
            if before is None:
                raise SharedCleanError(f"no clean-image snapshot recorded for {key}")
            current = Path(key)
            if not current.is_file():
                raise SharedCleanError(f"canonical clean image disappeared: {key}")
            after = self._fingerprint(current)
            for field in ("size_bytes", "mtime_ns", "sha256"):
                if after[field] != before[field]:
                    raise SharedCleanError(
                        f"canonical clean image was modified: {key} "
                        f"{field} {before[field]!r} -> {after[field]!r}"
                    )
            report.append({"path": key, "sha256_before": before["sha256"],
                           "sha256_after": after["sha256"], "unchanged": True})
        return report

    def tracked(self) -> List[str]:
        return sorted(self._snapshots)


# --------------------------------------------------------------------------- #
# Metadata CSV plumbing
# --------------------------------------------------------------------------- #

def existing_completed_rows(csv_path: Path, *, resume: bool) -> Dict[int, Dict[str, str]]:
    """Load an existing cohort for resume, re-auditing every stored row."""
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    if not resume:
        raise SharedCleanError(
            f"{csv_path} already has {len(rows)} rows; pass --resume to continue "
            "an existing cohort"
        )
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return {int(row["run_id"]): row for row in rows}


def append_row(csv_path: Path, row: Mapping[str, Any]) -> None:
    """Append one fully-formed row, refusing any schema drift."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fields = list(row.keys())
    if exists:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            stored_fields = list(csv.DictReader(handle).fieldnames or [])
        if stored_fields != fields:
            raise SharedCleanError(
                f"metadata schema mismatch for {csv_path}: "
                f"missing={sorted(set(fields) - set(stored_fields))} "
                f"extra={sorted(set(stored_fields) - set(fields))}"
            )
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())


def assert_resume_fields(
    stored: Mapping[str, Any], checks: Mapping[str, Any], *, run_id: int
) -> None:
    """Refuse to skip an existing row unless every re-derived value still matches."""
    for field, expected in checks.items():
        if str(stored.get(field, "")) != str(expected):
            raise SharedCleanError(
                f"shared-clean resume mismatch run_id={run_id} field={field}: "
                f"stored={stored.get(field)!r} expected={expected!r}"
            )


def assert_recorded_output(stored: Mapping[str, Any], *, run_id: int, label: str) -> None:
    """The previously recorded watermarked image must still be byte-identical."""
    watermarked_path = Path(str(stored["watermarked_path"]))
    if not watermarked_path.is_file():
        raise SharedCleanError(
            f"recorded {label} output missing run_id={run_id}: {watermarked_path}"
        )
    actual = sha256_path(watermarked_path)
    if actual != str(stored["watermarked_sha256"]):
        raise SharedCleanError(
            f"recorded {label} output SHA drift run_id={run_id}: "
            f"expected {stored['watermarked_sha256']}, got {actual}"
        )


def add_common_cli_args(parser: argparse.ArgumentParser, *, default_tr_metadata: Path) -> None:
    """CLI surface shared by every shared-clean runner."""
    parser.add_argument("--tr-metadata", type=Path, default=default_tr_metadata)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default="diffusiondb_shared_tr")
    parser.add_argument("--model-id", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument(
        "--model-revision", type=str, default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
    )
    parser.add_argument("--scheduler-target", type=str, default="DDIM")
    parser.add_argument("--num-inference-steps-target", type=int, default=50)
    parser.add_argument("--guidance-scale-target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--require-free-gpu", type=str_to_bool, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing metadata CSV after re-verifying every stored row",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="restrict generation to these TR run_ids (gates / smoke tests)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-before", type=str_to_bool, default=False)
    parser.add_argument(
        "--smoke-only",
        type=str_to_bool,
        default=False,
        help="label the run incomplete and ineligible for formal reporting",
    )
    parser.add_argument("--min-cpu-mem-gb", type=float, default=64.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=96.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
