#!/usr/bin/env python3
"""ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT.

Evaluate the direct pre-color RAVEN view output from a completed immutable
formal attack run.  This deliberately never relabels the result as the formal
post-color-transfer protocol and never overwrites formal records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import (  # noqa: E402
    CLIP_CONFIG,
    canonical_json_hash,
    current_clip_provenance,
    sha256_path,
    stage_fid_records,
)
from raven.metrics import pair_quality_metrics  # noqa: E402


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_snapshot_rows(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    index = root / "snapshots" / "snapshot_index.jsonl"
    for line in index.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        snapshot = Path(entry["snapshot_path"])
        if not snapshot.is_file() or sha256_path(snapshot) != entry["snapshot_sha256"]:
            raise RuntimeError(f"snapshot drift: {snapshot}")
        with snapshot.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run_id = str(row["run_id"])
                if run_id in rows:
                    raise RuntimeError(f"duplicate snapshot run_id={run_id}")
                rows[run_id] = row
    return rows


def load_records(root: Path, config_hash: str, role: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "attack_cache" / config_hash).glob(f"*/{role}/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(record["run_id"])
        if run_id in result:
            raise RuntimeError(f"duplicate {role} record run_id={run_id}")
        result[run_id] = record
    return result


def require_image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.convert("RGB").size != (512, 512):
            raise ValueError(f"expected 512x512 image: {path}")
    return sha256_path(path)


def build_variant_records(formal_root: Path, expected_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    run_config = json.loads((formal_root / "run_config.json").read_text(encoding="utf-8"))
    source = load_snapshot_rows(formal_root)
    wm = load_records(formal_root, run_config["attack_config_hash"], "watermarked")
    clean = load_records(formal_root, run_config["attack_config_hash"], "clean")
    run_ids = set(source)
    if len(run_ids) != expected_count or set(wm) != run_ids or set(clean) != run_ids:
        raise RuntimeError("formal source snapshot/attack record coverage mismatch")
    variant_payload = {
        "variant": "shift_no_color_transfer",
        "output_color_transfer": False,
        "output_color_transfer_mode": "none",
        "output_source": "view_guided_output.png",
        "base_attack_config_hash": run_config["attack_config_hash"],
        "source_code_manifest_sha256": run_config["source_code_manifest_sha256"],
    }
    variant_hash = canonical_json_hash(variant_payload)
    variant_wm: list[dict[str, Any]] = []
    variant_clean: list[dict[str, Any]] = []
    for run_id in sorted(run_ids, key=int):
        for base, destination in ((wm[run_id], variant_wm), (clean[run_id], variant_clean)):
            pre_color = Path(base["debug_info_path"]).parent / "view_guided_output.png"
            pre_color_sha = require_image(pre_color)
            variant = {
                **base,
                "attack_config_hash": variant_hash,
                "formal_config_hash": variant_hash,
                "attacked_path": str(pre_color.resolve()),
                "attacked_sha256": pre_color_sha,
                "output_sha256": pre_color_sha,
                "evaluation_variant": variant_payload["variant"],
                "output_color_transfer": False,
                "output_color_transfer_mode": "none",
                "output_source": variant_payload["output_source"],
                "source_attack_config_hash": base["attack_config_hash"],
                "source_final_output_sha256": base["attacked_sha256"],
            }
            destination.append(variant)
    return variant_wm, variant_clean, variant_hash


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def configure_single_gpu(physical_gpu: int) -> None:
    """Constrain clean-fid and child detector processes to one validated GPU."""
    if physical_gpu < 0:
        raise ValueError("--gpu must be a non-negative physical GPU index")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    # This must run before importing torch through clean-fid or detector helpers.
    configure_single_gpu(args.gpu)
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    variant_wm, variant_clean, variant_hash = build_variant_records(
        formal_root, args.expected_count
    )
    output_root.mkdir(parents=True)
    provenance = {
        "status": "ablation_only_not_formal",
        "variant": "shift_no_color_transfer",
        "formal_source_root": str(formal_root),
        "formal_run_config_sha256": sha256_path(formal_root / "run_config.json"),
        "ablation_entrypoint": str(Path(__file__).resolve()),
        "ablation_entrypoint_sha256": sha256_path(Path(__file__)),
        "variant_config_hash": variant_hash,
        "sample_count": args.expected_count,
        "physical_gpu": args.gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "created_utc": utc_now(),
        "clip": CLIP_CONFIG,
        "quality_reference": "watermarked input",
        "quality_overlap": "effective source flow inverse warp",
        "fid_reference": "original watermarked images",
        "detector_protocol": "formal Tree-Ring complex-L1, strict score < threshold",
    }
    write_json(output_root / "provenance.json", provenance)
    wm_records = output_root / "attack_records_no_color_watermarked.jsonl"
    clean_records = output_root / "attack_records_no_color_clean.jsonl"
    write_jsonl(wm_records, variant_wm)
    write_jsonl(clean_records, variant_clean)
    manifest = output_root / "verification" / "manifest.csv"
    snapshot_index = formal_root / "snapshots" / "snapshot_index.jsonl"
    run([
        sys.executable, str(REPO / "raven_repro/scripts/build_verification_manifest.py"),
        "--dataset", "diffusiondb", "--method", "TR", "--metadata", str(snapshot_index),
        "--attack-records", str(wm_records), "--snapshot-manifest", str(snapshot_index),
        "--output", str(manifest),
    ])
    run([
        sys.executable, str(REPO / "raven_repro/scripts/raven_nfpa_tr_eval.py"),
        "score-formal", "--manifest", str(manifest), "--attacked-clean-records", str(clean_records),
        "--output-dir", str(output_root / "verification" / "tr_nfpa"), "--device", args.device,
    ])
    quality_rows: list[dict[str, Any]] = []
    for record in variant_wm:
        with Image.open(record["watermarked_path"]) as reference, Image.open(record["attacked_path"]) as attacked:
            metric = pair_quality_metrics(
                reference.convert("RGB"), attacked.convert("RGB"),
                record["effective_source_flow_dx_image_px"],
                record["effective_source_flow_dy_image_px"],
            )
        quality_rows.append({"run_id": record["run_id"], **metric})
    quality_root = output_root / "metrics" / "quality"
    quality_root.mkdir(parents=True)
    write_jsonl(quality_root / "quality_records.jsonl", quality_rows)
    fid_root, fid_manifest = stage_fid_records(
        variant_wm, formal_output=output_root, quality_config_hash=variant_hash,
        expected_count=args.expected_count,
    )
    from raven.quality import clean_fid, openclip_text_image_scores
    fid_result = clean_fid(fid_root / "reference_watermarked", fid_root / "attacked", device=args.device)
    fid_result.update({"image_count": args.expected_count, "manifest_hash": fid_manifest["manifest_hash"], "metric_name": "ablation_fid_watermarked_vs_shift_no_color"})
    write_json(fid_root / "fid_result.json", fid_result)
    clip = openclip_text_image_scores(
        [record["attacked_path"] for record in variant_wm],
        [record["prompt"] for record in variant_wm], device=args.device,
        model_name=CLIP_CONFIG["clip_model_name"], pretrained=CLIP_CONFIG["clip_pretrained"],
    )
    write_json(output_root / "metrics" / "clip_result.json", {**clip, **current_clip_provenance()})
    detector = json.loads((output_root / "verification" / "tr_nfpa" / "aggregate_results.json").read_text())
    aggregate = {
        **provenance,
        "status": "ablation_complete_not_formal",
        "detector": detector,
        "fid": fid_result,
        "clip": {**clip, **current_clip_provenance()},
        "quality_count": len(quality_rows),
        "quality_psnr_mean": sum(float(row["overlap_psnr"]) for row in quality_rows) / len(quality_rows),
        "quality_ssim_mean": sum(float(row["overlap_ssim"]) for row in quality_rows) / len(quality_rows),
    }
    if not all(math.isfinite(float(value)) for value in (aggregate["quality_psnr_mean"], aggregate["quality_ssim_mean"], fid_result["value"], clip["mean"])):
        raise RuntimeError("non-finite no-color aggregate metric")
    write_json(output_root / "aggregate_results.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
