"""Canonical serialization, hashing, provenance and JSONL IO for watermark artifacts.

This module is the single authoritative implementation of the primitives that
``.agents/skills/experiment-integrity/SKILL.md`` §6 requires to exist exactly
once:

* canonical (sorted-key, NaN/Inf-rejecting, type-normalizing) JSON,
* SHA256 over that canonical JSON,
* SHA256 of bytes / text / files / numpy arrays / torch tensors,
* git provenance,
* deterministic JSONL reading and writing.

It was extracted verbatim from ``utils/wm/gm_bundle.py`` (GaussMarker, Issue #1)
so that the SFWMark bundles (``utils/wm/sfw_bundle.py``: HSQR now, HSTR under
Issue #4) reuse the very same code instead of growing a second copy. ``gm_bundle``
now delegates here and keeps raising ``GmBundleError``; each caller passes its own
exception class so the fail-closed error type stays method-specific.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import typing
from pathlib import Path

import numpy as np
import torch


class ArtifactError(RuntimeError):
    """Base class for every fail-closed watermark-artifact error."""


def canonicalize(value: typing.Any, error_cls: type = ArtifactError) -> typing.Any:
    """Normalize a value into a deterministic JSON-serializable form.

    Rejects NaN/Inf, normalizes paths to POSIX strings, sorts mapping keys and
    keeps int/bool distinct from float. The very same function is used before
    writing metadata, after reloading it, during resume validation and during
    threshold-compatibility checks so that
    ``hash before serialization == hash after serialization and reload``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise error_cls(f"non-finite float rejected by canonical hashing: {value!r}")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return canonicalize(float(value), error_cls)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        raise error_cls("raw bytes must never enter canonical metadata (possible secret leak)")
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key], error_cls) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, error_cls) for item in value]
    if isinstance(value, torch.Size):
        return [int(item) for item in value]
    raise error_cls(f"unsupported type in canonical metadata: {type(value)!r}")


def canonical_json(
    payload: typing.Mapping[str, typing.Any], error_cls: type = ArtifactError
) -> str:
    """Deterministic JSON text for hashing and on-disk manifests."""
    return json.dumps(
        canonicalize(payload, error_cls),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(
    payload: typing.Mapping[str, typing.Any], error_cls: type = ArtifactError
) -> str:
    return hashlib.sha256(canonical_json(payload, error_cls).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: typing.Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash a numpy array together with its dtype and shape."""
    array = np.ascontiguousarray(array)
    header = f"numpy|{array.dtype.str}|{tuple(int(d) for d in array.shape)}|".encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash a torch tensor together with its dtype and shape (device agnostic)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_complex():
        real = sha256_tensor(tensor.real.contiguous())
        imag = sha256_tensor(tensor.imag.contiguous())
        return hashlib.sha256(f"complex|{real}|{imag}".encode("utf-8")).hexdigest()
    header = f"torch|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_provenance(root: typing.Optional[Path] = None) -> typing.Dict[str, typing.Any]:
    root = Path(root) if root is not None else repo_root()

    def _run(*args: str) -> typing.Optional[str]:
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


def optional_file_sha256(path: typing.Optional[typing.Union[str, Path]]) -> typing.Optional[str]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return sha256_file(path)


def cohort_sha256(
    image_paths: typing.Sequence[typing.Union[str, Path]], error_cls: type = ArtifactError
) -> str:
    """Order-independent-per-name hash of a cohort of image files."""
    entries = []
    for path in image_paths:
        path = Path(path)
        entries.append({"name": path.name, "sha256": sha256_file(path)})
    entries.sort(key=lambda item: (item["name"], item["sha256"]))
    return canonical_sha256({"cohort": entries}, error_cls)


def write_jsonl(
    path: typing.Union[str, Path],
    rows: typing.Iterable[typing.Mapping[str, typing.Any]],
    error_cls: type = ArtifactError,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row, error_cls) + "\n")


def append_jsonl(
    path: typing.Union[str, Path],
    row: typing.Mapping[str, typing.Any],
    error_cls: type = ArtifactError,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(row, error_cls) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: typing.Union[str, Path]) -> typing.List[typing.Dict[str, typing.Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
