#!/usr/bin/env python3
"""Calculate metrics for a completed RGB-zero-padding + REVAN cohort.

The detector receives original clean images as reference-only records for its
1% FPR calibration.  Images are never preloaded; metric stages run serially.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--stages", nargs="+", choices=("fid", "clip", "lpips", "tr_detector"),
        default=("fid", "clip", "lpips", "tr_detector"),
    )
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch
    from raven.experiment_io import read_records_jsonl
    from eval import evaluate_clip, evaluate_detector, evaluate_fid, evaluate_lpips

    if not (args.output_dir / "config.json").is_file():
        raise FileNotFoundError(args.output_dir / "config.json")
    config = json.loads((args.output_dir / "config.json").read_text())
    watermarked_records = read_records_jsonl(args.output_dir)
    if len(watermarked_records) != 1001:
        raise ValueError(f"Expected 1001 completed watermarked records, got {len(watermarked_records)}")

    # Original clean images are detector-only references.  No copies/output
    # images are required because reference_only_clean excludes the attacked
    # clean cohort while retaining original_clean for TPR@1% FPR calibration.
    clean_records = []
    for record in watermarked_records:
        metadata = record.get("source_metadata", {})
        clean_path = Path(metadata.get("clean_path", ""))
        if not clean_path.is_file():
            raise FileNotFoundError(f"run_id={record['run_id']}: clean_path={clean_path}")
        clean_records.append({
            "run_id": record["run_id"],
            "role": "clean",
            "reference_only_clean": True,
            "input_path": str(clean_path),
            "output_path": str(clean_path),
            "prompt": record.get("prompt", ""),
            "method": "TR",
            "source_metadata": metadata,
        })
    detector_records = watermarked_records + clean_records
    evaluation_config = {**config, "method": "TR", "dataset": "diffusiondb"}

    result = {
        "experiment": config.get("experiment"),
        "output_dir": str(args.output_dir.resolve()),
        "watermarked_count": len(watermarked_records),
        "clean_reference_count": len(clean_records),
        "metrics": {},
    }
    stage_runners = {
        "fid": lambda: evaluate_fid(watermarked_records, args.output_dir, "cuda", evaluation_config),
        "clip": lambda: evaluate_clip(watermarked_records, args.output_dir, "cuda", evaluation_config),
        "lpips": lambda: evaluate_lpips(watermarked_records, args.output_dir, "cuda", evaluation_config),
        "tr_detector": lambda: evaluate_detector(detector_records, args.output_dir, "TR", "cuda", evaluation_config),
    }
    destination = args.output_dir / "results_summary.json"
    for name in args.stages:
        run_stage = stage_runners[name]
        print(f"starting={name}", flush=True)
        result["metrics"][name] = run_stage()
        atomic_json(destination, result)
        print(f"finished={name} status={result['metrics'][name].get('status')}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(destination, result)
    print(f"results_summary={destination}", flush=True)


if __name__ == "__main__":
    main()
