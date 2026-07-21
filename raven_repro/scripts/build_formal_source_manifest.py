#!/usr/bin/env python3
"""Build a content-addressed manifest of every formal evaluation source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CORE_FILES = (
    "experiments/run_raven_formal_eval.py",
    "experiments/run_raven_aligned_color_eval.py",
    "experiments/wait_and_run_raven_eval_all_datasets.py",
    "experiments/build_raven_formal_eval_table.py",
    "raven_repro/raven/attention.py",
    "raven_repro/raven/color_transfer.py",
    "raven_repro/raven/eval_protocol.py",
    "raven_repro/raven/inversion.py",
    "raven_repro/raven/metrics.py",
    "raven_repro/raven/pairing_provenance.py",
    "raven_repro/raven/pipeline_raven.py",
    "raven_repro/raven/quality.py",
    "raven_repro/raven/resource_guard.py",
    "raven_repro/raven/utils.py",
    "raven_repro/raven/warp.py",
    "raven_repro/scripts/build_formal_source_manifest.py",
    "raven_repro/scripts/build_verification_manifest.py",
    "raven_repro/scripts/evaluate_verification.py",
    "raven_repro/scripts/extract_verification_scores.py",
    "raven_repro/scripts/raven_nfpa_tr_eval.py",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def build_payload() -> dict:
    head = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=all").strip())
    tracked = set(git("ls-files").splitlines())
    paths = list(CORE_FILES)
    paths.extend(
        str(path.relative_to(REPO))
        for path in sorted((REPO / "raven_repro" / "tests").glob("test_*.py"))
    )
    paths.extend(
        str(path.relative_to(REPO))
        for path in sorted((REPO / "experiments" / "raven_ablation_configs").glob("*.json"))
    )
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate formal source path")
    files = []
    for relative in sorted(paths):
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "relative_path": relative,
                "tracked_or_untracked": "tracked" if relative in tracked else "untracked",
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
                "git_head": head,
                "git_dirty": dirty,
            }
        )
    return {
        "schema_version": "raven_formal_source_manifest_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "git_dirty": dirty,
        "file_count": len(files),
        "files": files,
    }


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "audit/formal_source_manifest.json")
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build_payload()
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    write_atomic(output, data)
    manifest_sha = hashlib.sha256(data).hexdigest()
    write_atomic(output.with_suffix(".sha256"), f"{manifest_sha}  {output.name}\n".encode("ascii"))
    print(json.dumps({"path": str(output), "sha256": manifest_sha, "file_count": len(payload["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
