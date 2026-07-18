#!/usr/bin/env python
"""Archive audit evidence, then delete contaminated shared-latent outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


KEEP_NAMES = {
    "aggregate_results.json",
    "aggregate_results.md",
    "paper_settings.json",
    "provenance.json",
    "results.json",
    "run_state.json",
    "source_counts.json",
}


def should_keep(path: Path) -> bool:
    return path.name in KEEP_NAMES or path.suffix == ".log"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    outputs = (workspace / "outputs").resolve()
    archive = args.archive.resolve()
    if outputs not in archive.parents:
        raise ValueError(f"archive must be inside {outputs}: {archive}")
    roots = [path.resolve() for path in args.root]
    for root in roots:
        if outputs not in root.parents or root == outputs:
            raise ValueError(f"refusing unsafe contaminated root: {root}")
        if root == archive or archive in root.parents or root in archive.parents:
            raise ValueError(f"archive/root overlap: root={root} archive={archive}")

    inventory = []
    for root in roots:
        files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        inventory.append(
            {
                "root": str(root),
                "exists": root.exists(),
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "audit_files": sum(should_keep(path) for path in files),
            }
        )
    if not args.execute:
        print(json.dumps({"execute": False, "inventory": inventory}, indent=2))
        return 0

    archive.mkdir(parents=True, exist_ok=True)
    for index, root in enumerate(roots):
        if not root.exists():
            continue
        target_root = archive / f"root_{index}_{root.name}"
        for source in (path for path in root.rglob("*") if path.is_file() and should_keep(path)):
            relative = source.relative_to(root)
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.name.startswith("aggregate_results"):
                target = target.with_name(target.name + ".INVALID_SHARED_LATENT")
            shutil.copy2(source, target)
        shutil.rmtree(root)

    reason = {
        "status": "INVALID_SHARED_LATENT_DATASET",
        "invalidated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": (
            "Tree-Ring watermarked samples reused one complete base latent and clean images "
            "were generated from unrelated RNG state. Pairing provenance was absent."
        ),
        "affected_result": "TPR=0.177822 and every detector/quality metric derived from these roots",
        "policy": "Numeric results are retained only under INVALID names; logs/provenance remain for audit. Image data and derived outputs were deleted.",
        "inventory_before_deletion": inventory,
    }
    (archive / "INVALID_REASON.json").write_text(
        json.dumps(reason, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(reason, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
