#!/usr/bin/env python3
"""Backfill aggregate-level PSNR/SSIM and attack-success scalars into an
already-attacked, already-verified formal run.

``aggregate_stage`` in ``experiments/run_raven_formal_eval.py`` originally wrote
only the quality-record path and SHA, and no attack-success field, so the
experiment table had nothing to read and printed the absent marker for PSNR,
SSIM and Attack Success. Runs attacked before that fix keep valid per-sample
records; only the reduction to aggregate scalars is missing.

This tool re-derives those scalars **from the existing verified artifacts
only**. It never touches GPU work, never re-scores an image, and never invents a
value:

* the quality records must hash to the SHA the aggregate already recorded;
* the detector payload must hash to the SHA the aggregate already recorded;
* the cohort size and run-ID set must match the immutable attack records;
* any scalar already present in the aggregate must equal the recomputed value.

Both scalars come from the same authoritative reducers the aggregate stage now
calls (``raven.eval_protocol.formal_quality_summary`` /
``gs_attack_success_summary``), so a backfilled run and a freshly aggregated run
cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import (  # noqa: E402
    GS_ATTACK_SUCCESS_FIELD,
    formal_quality_summary,
    gs_attack_success_summary,
    sha256_path,
)

BACKFILL_MARKER = "aggregate_scalar_backfill"


def load_run_ids(run_root: Path) -> set[str]:
    records = run_root / "attack_records_watermarked.jsonl"
    if not records.is_file():
        raise FileNotFoundError(records)
    rows = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id in immutable attack records")
    return set(run_ids)


def backfill(run_root: Path, *, apply: bool) -> dict[str, Any]:
    aggregate_path = run_root / "formal_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    method = str(aggregate["method"])
    expected_count = int(aggregate["N"])

    validated = run_root / "VALIDATED.json"
    if not validated.is_file():
        raise RuntimeError(f"{run_root} is not a validated formal run")

    run_ids = load_run_ids(run_root)
    if len(run_ids) != expected_count:
        raise RuntimeError(
            f"attack record cohort {len(run_ids)} != aggregate N {expected_count}"
        )

    quality_summary = formal_quality_summary(
        aggregate["quality_records"],
        expected_count=expected_count,
        expected_run_ids=run_ids,
        expected_records_sha256=aggregate.get("quality_records_sha256"),
    )

    detector_path = Path(aggregate["detector_result"])
    detector_sha = sha256_path(detector_path)
    if detector_sha != aggregate.get("detector_result_sha256"):
        raise RuntimeError(
            f"detector result SHA mismatch: stored={aggregate.get('detector_result_sha256')} "
            f"actual={detector_sha}"
        )
    detector_payload = json.loads(detector_path.read_text(encoding="utf-8"))

    additions: dict[str, Any] = dict(quality_summary)
    if method == "GS":
        additions.update(gs_attack_success_summary(detector_payload))
        detector_n = int(detector_payload["metric"]["stages"]["attacked"]["N"])
        if detector_n != expected_count:
            raise RuntimeError(
                f"detector attacked cohort {detector_n} != aggregate N {expected_count}"
            )
    elif method != "TR":
        raise RuntimeError(f"no registered aggregate backfill for method {method!r}")

    conflicts = {
        key: (aggregate[key], value)
        for key, value in additions.items()
        if key in aggregate
        and aggregate[key] != value
        and not (
            isinstance(aggregate[key], (int, float))
            and isinstance(value, (int, float))
            and math.isclose(float(aggregate[key]), float(value), rel_tol=0.0, abs_tol=1e-12)
        )
    }
    if conflicts:
        raise RuntimeError(f"{run_root}: recomputed scalars conflict with stored ones: {conflicts}")

    for value in additions.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("refusing to write a non-finite aggregate scalar")

    added = {key: value for key, value in additions.items() if key not in aggregate}
    if apply and added:
        payload = {**aggregate, **additions}
        history = list(payload.get(BACKFILL_MARKER, []))
        history.append(
            {
                "fields": sorted(added),
                "quality_records_sha256": quality_summary["quality_records_sha256"],
                "detector_result_sha256": detector_sha,
                "tool": str(Path(__file__).resolve().relative_to(REPO)),
                "tool_sha256": sha256_path(Path(__file__)),
            }
        )
        payload[BACKFILL_MARKER] = history
        temporary = aggregate_path.with_name(f".{aggregate_path.name}.backfill.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, aggregate_path)
        directory = os.open(aggregate_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    return {
        "run_root": str(run_root),
        "method": method,
        "N": expected_count,
        "applied": bool(apply and added),
        "added_fields": sorted(added),
        "already_present": sorted(set(additions) - set(added)),
        "quality_psnr_mean": quality_summary["quality_psnr_mean"],
        "quality_ssim_mean": quality_summary["quality_ssim_mean"],
        GS_ATTACK_SUCCESS_FIELD: additions.get(GS_ATTACK_SUCCESS_FIELD),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the recomputed scalars. Without it the tool only reports them.",
    )
    args = parser.parse_args()
    results = [backfill(root.resolve(), apply=args.apply) for root in args.run_roots]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
