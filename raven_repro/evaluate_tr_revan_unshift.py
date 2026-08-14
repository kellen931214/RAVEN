#!/usr/bin/env python3
"""Evaluate formal TR REVAN attacks and their inverse-shifted counterparts.

The detector uses the repository TR scorer on GPU one image at a time and
holds the formal raw complex-L1 decision rule fixed: ``raw_l1 < 75.68``.
LPIPS and CLIP are paired against the original TR-watermarked image; FID uses
clean-fid's formal legacy_tensorflow setting over the same watermarked cohort.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_bench_wm"))

LOG = logging.getLogger("tr_revan_unshift_eval")
THRESHOLD = 75.68


def rows_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = {str(row["run_id"]): row for row in csv.DictReader(handle)}
    if len(result) != 1001:
        raise ValueError(f"expected 1001 TR metadata rows, got {len(result)}")
    for row in result.values():
        row["w_pattern_const"] = "0.0"
    return result


def memory_note() -> str:
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        free = os.sysconf("SC_AVPHYS_PAGES") * page
        total = os.sysconf("SC_PHYS_PAGES") * page
        return f"ram_available_gib={free / 2**30:.1f}/{total / 2**30:.1f}"
    except (AttributeError, OSError, ValueError):
        return "ram_available_gib=unavailable"


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("no scores")
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
        "detected_count": sum(value < THRESHOLD for value in values),
        "tpr_at_fixed_1pct_fpr_threshold": sum(value < THRESHOLD for value in values) / len(values),
        "threshold_raw_complex_l1": THRESHOLD,
        "comparison_operator": "<",
    }


def source_l1_stats(path: Path) -> dict[str, dict[str, Any]]:
    watermarked: list[float] = []
    attacked: list[float] = []
    for row in rows_jsonl(path):
        watermarked.append(float(row["watermarked_l1_full_precision"]))
        attacked.append(float(row["attacked_watermarked_l1_full_precision"]))
    if len(watermarked) != 1001 or len(attacked) != 1001:
        raise ValueError("formal score file is not N=1001")
    return {"watermarked": stats(watermarked), "attacked": stats(attacked)}


def score_inverse(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]], out: Path, device: str) -> dict[str, Any]:
    import torch
    from raven.detectors import tr_detector

    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, float] = {}
    if out.is_file():
        for row in rows_jsonl(out):
            rid = str(row["run_id"])
            if rid in existing:
                raise ValueError(f"duplicate existing detector score for run_id={rid}")
            existing[rid] = float(row["raw_complex_l1"])
        known_ids = {str(record["run_id"]) for record in records}
        if not set(existing).issubset(known_ids):
            raise ValueError("existing detector scores do not match the requested cohort")
        LOG.info("resuming TR inverse detector from %d/%d persisted scores", len(existing), len(records))
    provider = tr_detector.load_state([metadata[record["run_id"]] for record in records], device)
    values: list[float] = list(existing.values())
    try:
        with out.open("a", encoding="utf-8") as handle:
            for index, record in enumerate(records, start=1):
                rid = str(record["run_id"])
                if rid in existing:
                    continue
                result = tr_detector.score_image(
                    provider, record["inverse_shifted_attacked_path"], record=metadata[rid]
                )
                raw = float(result["raw_score"])
                values.append(raw)
                handle.write(json.dumps({
                    "run_id": rid,
                    "raw_complex_l1": raw,
                    "canonical_score": float(result["canonical_score"]),
                    "detected_at_fixed_threshold": raw < THRESHOLD,
                    "image_path": record["inverse_shifted_attacked_path"],
                }, sort_keys=True) + "\n")
                if index % 25 == 0:
                    handle.flush()
                    LOG.info("TR inverse score %d/%d %s", index, len(records), memory_note())
    finally:
        del provider
        gc.collect()
        torch.cuda.empty_cache()
    return stats(values)


def pair_lpips(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]], variant: str, out: Path, device: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import lpips
    from PIL import Image

    model = lpips.LPIPS(net="alex").to(device).eval()
    values: list[float] = []
    try:
        with out.open("w", encoding="utf-8") as handle, torch.no_grad():
            for index, record in enumerate(records, start=1):
                rid = str(record["run_id"])
                reference = Path(metadata[rid]["watermarked_path"])
                target = Path(record["source_attacked_path"] if variant == "attacked" else record["inverse_shifted_attacked_path"])
                def tensor(path: Path):
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
                    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
                ref_tensor, target_tensor = tensor(reference), tensor(target)
                value = float(model(ref_tensor, target_tensor).item())
                values.append(value)
                handle.write(json.dumps({"run_id": rid, "lpips_alex": value}, sort_keys=True) + "\n")
                del ref_tensor, target_tensor
                if index % 25 == 0:
                    handle.flush(); torch.cuda.empty_cache()
                    LOG.info("LPIPS %s %d/%d %s", variant, index, len(records), memory_note())
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()
    return {"count": len(values), "mean": sum(values) / len(values), "median": float(median(values)), "min": min(values), "max": max(values), "model": "alex", "reference": "original_TR_watermarked"}


def pair_clip(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]], variant: str, out: Path, device: str) -> dict[str, Any]:
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
    model.eval()
    values: list[float] = []
    try:
        with out.open("w", encoding="utf-8") as handle, torch.no_grad():
            for index, record in enumerate(records, start=1):
                rid = str(record["run_id"])
                reference = Path(metadata[rid]["watermarked_path"])
                target = Path(record["source_attacked_path"] if variant == "attacked" else record["inverse_shifted_attacked_path"])
                with Image.open(reference) as image:
                    ref_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                with Image.open(target) as image:
                    target_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                ref_features = model.encode_image(ref_tensor)
                target_features = model.encode_image(target_tensor)
                ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
                value = float((ref_features * target_features).sum().cpu().item())
                values.append(value)
                handle.write(json.dumps({"run_id": rid, "clip_image_image_cosine": value}, sort_keys=True) + "\n")
                del ref_tensor, target_tensor, ref_features, target_features
                if index % 25 == 0:
                    handle.flush(); torch.cuda.empty_cache()
                    LOG.info("CLIP %s %d/%d %s", variant, index, len(records), memory_note())
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()
    return {"count": len(values), "mean": sum(values) / len(values), "median": float(median(values)), "min": min(values), "max": max(values), "model": "ViT-bigG-14", "pretrained": "laion2b_s39b_b160k", "metric": "image-image cosine similarity", "reference": "original_TR_watermarked"}


def fid_inverse(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]], stage: Path, device: str) -> dict[str, Any]:
    from raven.evaluation.metrics import clean_fid
    reference_dir, inverse_dir = stage / "reference_watermarked", stage / "inverse_shifted"
    if stage.exists():
        shutil.rmtree(stage)
    reference_dir.mkdir(parents=True); inverse_dir.mkdir()
    for index, record in enumerate(records, start=1):
        rid = str(record["run_id"])
        name = f"{int(rid):06d}.png"
        os.symlink(Path(metadata[rid]["watermarked_path"]), reference_dir / name)
        os.symlink(Path(record["inverse_shifted_attacked_path"]), inverse_dir / name)
        if index % 250 == 0:
            LOG.info("FID staging %d/%d", index, len(records))
    try:
        return clean_fid(reference_dir, inverse_dir, device=device, mode="legacy_tensorflow", secondary_modes=())
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse-records", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--formal-l1-scores", type=Path, required=True)
    parser.add_argument("--formal-attacked-fid", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+", choices=["detector", "lpips", "clip", "fid"], default=["detector", "lpips", "clip", "fid"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    records = list(rows_jsonl(args.inverse_records))
    if len(records) != 1001 or len({record["run_id"] for record in records}) != 1001:
        raise ValueError("inverse records must be an N=1001 unique cohort")
    metadata = load_metadata(args.metadata)
    for record in records:
        if not Path(record["inverse_shifted_attacked_path"]).is_file() or not Path(record["source_attacked_path"]).is_file():
            raise FileNotFoundError(f"missing paired output for run_id={record['run_id']}")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    result.update({
        "N": len(records), "fixed_threshold_raw_complex_l1": THRESHOLD,
        "threshold_protocol": "formal NFPA rounded2 threshold; strict raw_l1 < threshold; no recalibration",
        "source_formal_l1_scores": str(args.formal_l1_scores),
        "watermarked": source_l1_stats(args.formal_l1_scores)["watermarked"],
        "attacked": source_l1_stats(args.formal_l1_scores)["attacked"],
        "attacked_fid_legacy_tensorflow": args.formal_attacked_fid,
    })
    if "detector" in args.stages:
        result["inverse_shifted"] = score_inverse(records, metadata, output / "inverse_tr_l1_scores.jsonl", args.device)
    if "lpips" in args.stages:
        result["attacked_lpips"] = pair_lpips(records, metadata, "attacked", output / "attacked_lpips.jsonl", args.device)
        result["inverse_shifted_lpips"] = pair_lpips(records, metadata, "inverse_shifted", output / "inverse_lpips.jsonl", args.device)
    if "clip" in args.stages:
        result["attacked_clip"] = pair_clip(records, metadata, "attacked", output / "attacked_clip.jsonl", args.device)
        result["inverse_shifted_clip"] = pair_clip(records, metadata, "inverse_shifted", output / "inverse_clip.jsonl", args.device)
    if "fid" in args.stages:
        result["inverse_shifted_fid"] = fid_inverse(records, metadata, output / "fid_stage", args.device)
    if "inverse_shifted" in result:
        before = result["watermarked"]["tpr_at_fixed_1pct_fpr_threshold"]
        attacked = result["attacked"]["tpr_at_fixed_1pct_fpr_threshold"]
        inverse = result["inverse_shifted"]["tpr_at_fixed_1pct_fpr_threshold"]
        result["delta_tpr_inverse_minus_attacked"] = inverse - attacked
        result["recovery_ratio"] = (inverse - attacked) / (before - attacked) if before != attacked else None
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("completed results=%s", result_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
