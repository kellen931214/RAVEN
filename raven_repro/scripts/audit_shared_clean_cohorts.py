#!/usr/bin/env python3
"""Audit TR / GS / GM / T2S ``shared_tr_clean_v2`` cohorts against one TR source.

Every method cohort is first audited on its own terms (``audit_pairing_rows``:
required provenance, duplicate detection, pairing-hash round trip, on-disk file
hashes), then cross-checked against the canonical Tree-Ring rows
(``audit_shared_clean_cohorts``: matching by run_id only, identical prompt,
latent, clean path/SHA and generation configuration, and distinct watermarked
images).

Any drift fails closed with a non-zero exit status. This script contains no
watermark algorithm and no second copy of the audit rules; it is CLI + IO over
``raven.pairing_provenance``.

Example::

    python3 raven_repro/scripts/audit_shared_clean_cohorts.py \\
        --tr-metadata   data/tr/diffusiondb/metadata.csv \\
        --gm-metadata   <gm-cohort>/metadata.csv \\
        --t2s-metadata  <t2s-cohort>/metadata.csv \\
        --output        audit/shared_clean_tr_gm_t2s.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RAVEN_ROOT = REPO_ROOT / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.pairing_provenance import (  # noqa: E402
    SHARED_CLEAN_METHOD_PROTOCOLS,
    audit_pairing_rows,
    audit_shared_clean_cohorts,
    sha256_path,
)

METHOD_ARGS = {"GS": "gs_metadata", "GM": "gm_metadata", "T2S": "t2s_metadata"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"cohort metadata is empty: {path}")
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tr-metadata", type=Path, required=True)
    parser.add_argument("--gs-metadata", type=Path, default=None)
    parser.add_argument("--gm-metadata", type=Path, default=None)
    parser.add_argument("--t2s-metadata", type=Path, default=None)
    parser.add_argument(
        "--verify-files",
        action="store_true",
        default=True,
        help="re-hash every referenced image on disk (default)",
    )
    parser.add_argument(
        "--no-verify-files", dest="verify_files", action="store_false"
    )
    parser.add_argument("--output", type=Path, default=None, help="write the report JSON here")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    tr_path = Path(args.tr_metadata).resolve()
    tr_rows = load_rows(tr_path)
    report: Dict[str, Any] = {
        "audit": "shared_tr_clean_v2_cross_method",
        "tr_metadata_path": str(tr_path),
        "tr_metadata_sha256": sha256_path(tr_path),
        "tr_rows": len(tr_rows),
        "verified_files": bool(args.verify_files),
        "per_method": {},
    }
    report["tr_pairing_audit"] = audit_pairing_rows(
        tr_rows, expected_count=len(tr_rows), verify_files=args.verify_files
    )

    cohorts: Dict[str, List[Dict[str, str]]] = {}
    for method in sorted(SHARED_CLEAN_METHOD_PROTOCOLS):
        value = getattr(args, METHOD_ARGS[method])
        if value is None:
            continue
        path = Path(value).resolve()
        rows = load_rows(path)
        cohorts[method] = rows
        report["per_method"][method] = {
            "metadata_path": str(path),
            "metadata_sha256": sha256_path(path),
            "rows": len(rows),
            "pairing_audit": audit_pairing_rows(
                rows, expected_count=len(rows), verify_files=args.verify_files
            ),
        }

    if not cohorts:
        raise SystemExit(
            "nothing to audit: pass at least one of --gs-metadata / --gm-metadata / "
            "--t2s-metadata"
        )

    report["cross_method"] = audit_shared_clean_cohorts(
        tr_rows,
        cohorts,
        verify_files=args.verify_files,
        require_methods=sorted(cohorts),
    )
    report["passed"] = True

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {output}")
    summary = {key: value for key, value in report["cross_method"].items() if key != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # fail closed with a clear, non-zero exit
        print(f"shared-clean audit FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
