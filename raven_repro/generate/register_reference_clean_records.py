#!/usr/bin/env python3
"""Register original clean images as reference-only records for ROC evaluation."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from raven.experiment_io import rebuild_records_jsonl, write_record

    count = 0
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_id = str(row["run_id"])
            clean = Path(row["clean_path"])
            if not clean.is_file():
                raise FileNotFoundError(f"run_id={run_id}: clean image missing: {clean}")
            write_record(args.output_dir, "clean", run_id, {
                "run_id": run_id,
                "role": "clean",
                "dataset": "diffusiondb",
                "method": "GM",
                "input_path": str(clean),
                "output_path": "",
                "prompt": row.get("prompt", ""),
                "prompt_id": row.get("sample_id", ""),
                "prompt_source": "metadata",
                "reference_only_clean": True,
                "source_metadata": row,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            count += 1
    rebuild_records_jsonl(args.output_dir)
    if count != 1001:
        raise RuntimeError(f"expected 1001 clean references, wrote {count}")
    print(f"registered {count} reference-only clean records", flush=True)


if __name__ == "__main__":
    main()
