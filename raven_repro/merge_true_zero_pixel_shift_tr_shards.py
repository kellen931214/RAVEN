#!/usr/bin/env python3
"""Pool two TR detector shards and compute one global 1% FPR operating point."""

from __future__ import annotations

import json
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    import argparse
    from raven.detectors import tr_detector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shard_dir = args.output_dir / "evaluation" / "tr_shards"
    rows = []
    for index in (0, 1):
        path = shard_dir / f"shard_{index}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if len(rows) != 3003:
        raise ValueError(f"Expected 3003 scores across two shards, got {len(rows)}")
    expected = {(cohort, str(run_id)) for cohort in (
        "original_clean", "original_watermarked", "attacked_watermarked"
    ) for run_id in range(1001)}
    actual = {(row["evaluation_cohort"], str(row["run_id"])) for row in rows}
    if actual != expected:
        raise ValueError("Shards do not contain exactly one score per expected cohort/run_id")
    result = tr_detector.aggregate(rows)
    result.update({"stage": "detector", "status": "completed", "available": True,
                   "pooled_shards": [0, 1], "pooled_score_count": len(rows)})
    output = args.output_dir / "results_summary.json"
    summary = json.loads(output.read_text()) if output.is_file() else {}
    summary.setdefault("metrics", {})["tr_detector"] = result
    atomic_json(output, summary)
    atomic_json(args.output_dir / "evaluation" / "tr_detector_aggregate.json", result)
    print(json.dumps(result["detection_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
