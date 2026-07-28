"""Canonical serialization and hashing for eval_bench_wm metadata.

Why this exists next to ``raven_repro/raven/pairing_provenance.py``:
``eval_bench_wm`` is a self-contained benchmark package with its own
requirements and CLI entrypoints, and it is imported without ``raven_repro``
on ``sys.path`` (see ``run_watermark.py`` / ``run_verification.py``). Rather
than making the benchmark depend on the RAVEN experiment package, this module
mirrors the exact same canonicalization rules:

    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

``tests/test_t2s_parity.py::test_canonical_matches_raven_repro`` asserts the two
implementations agree byte-for-byte so they cannot drift apart.

Requirements enforced here (see ``.agents/skills/experiment-integrity``):
dictionary keys are sorted, NaN/Inf are rejected, and serialization is
deterministic so that ``hash(before write) == hash(after reload)``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _reject_non_finite(value: Any, path: str = "$") -> None:
    """Fail closed on NaN/Inf anywhere in the payload."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}: {value!r}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def canonical_json_dumps(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON text: sorted keys, tight separators, no NaN/Inf."""
    _reject_non_finite(payload)
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """SHA256 over the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    """Stream a file through SHA256 without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    """Hash exact tensor shape, dtype, and bytes.

    Shape and dtype are part of the digest so that two tensors with identical
    bytes but different layouts never collide.
    """
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.view(-1).view(value.dtype).numpy().tobytes(order="C"))
    return digest.hexdigest()
