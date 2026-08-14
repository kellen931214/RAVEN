#!/usr/bin/env python3
"""Score one deterministic half of a completed TR attack cohort on one GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


COHORTS = ("original_clean", "original_watermarked", "attacked_watermarked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {
        (str(row["evaluation_cohort"]), str(row["run_id"]))
        for row in (json.loads(line) for line in path.read_text().splitlines() if line)
    }


def main() -> None:
    args = parse_args()
    # Set before importing torch/detector dependencies to select exactly one
    # physical GPU for this shard.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch
    from raven.detectors import ROW_STATUS_SCORED
    from raven.detectors import tr_detector
    from raven.experiment_io import output_image_path, read_records_jsonl
    from raven.metadata_resolver import MetadataResolver

    records = sorted(
        read_records_jsonl(args.output_dir), key=lambda row: int(row["run_id"])
    )
    if len(records) != 1001:
        raise ValueError(f"Expected 1001 attack records, got {len(records)}")
    split = (len(records) + 1) // 2
    shard_records = records[:split] if args.shard_index == 0 else records[split:]
    expected_count = len(shard_records) * len(COHORTS)

    result_dir = args.output_dir / "evaluation" / "tr_shards"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_path = result_dir / f"shard_{args.shard_index}.jsonl"
    manifest_path = result_dir / f"shard_{args.shard_index}.json"
    resolver = MetadataResolver.from_path(args.metadata)

    # The canonical source CSV omits this explicit zero-valued TR parameter.
    # It is confirmed by the previous canonical cohort and must be present for
    # the detector's strict provider-state contract.
    detector_records = []
    for record in shard_records:
        hydrated = resolver.enrich_record(record, csv_path=str(args.metadata))
        hydrated["w_pattern_const"] = "0.0"
        detector_records.append(hydrated)
    provider_info = tr_detector.load_state(detector_records, device="cuda")

    done = load_done(score_path)
    atomic_json(manifest_path, {
        "shard_index": args.shard_index,
        "cuda_visible_devices": args.cuda_visible_devices,
        "run_id_start": str(shard_records[0]["run_id"]),
        "run_id_end": str(shard_records[-1]["run_id"]),
        "sample_count": len(shard_records),
        "expected_score_count": expected_count,
        "w_pattern_const_source": "canonical TR cohort constant 0.0",
        "provider_config_hash": provider_info["detector_provider_config_hash"],
    })

    with score_path.open("a", encoding="utf-8") as handle:
        completed = len(done)
        for record in detector_records:
            run_id = str(record["run_id"])
            metadata = record["_metadata"]
            entries = (
                ("original_clean", Path(metadata["clean_path"])),
                ("original_watermarked", Path(metadata["watermarked_path"])),
                ("attacked_watermarked", output_image_path(args.output_dir, "watermarked", run_id)),
            )
            for cohort, image_path in entries:
                key = (cohort, run_id)
                if key in done:
                    continue
                score = tr_detector.score_image(
                    provider_info, str(image_path), record=record,
                    evaluation_entry={"evaluation_cohort": cohort},
                )
                row = {
                    "run_id": run_id,
                    "evaluation_cohort": cohort,
                    "image_path": str(image_path),
                    "status": ROW_STATUS_SCORED,
                    **score,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed += 1
                done.add(key)
                del score
                gc.collect()
                torch.cuda.empty_cache()
                if completed % 10 == 0 or completed == 1:
                    print(
                        f"shard={args.shard_index} completed={completed}/{expected_count} "
                        f"last_run_id={run_id} cohort={cohort}",
                        flush=True,
                    )
    print(f"shard={args.shard_index} completed={completed}/{expected_count}", flush=True)


if __name__ == "__main__":
    main()
