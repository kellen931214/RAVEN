#!/usr/bin/env python3
"""Evaluate effective-flow aligned color transfer from immutable RAVEN views.

This reuses completed DDIM/shift/attention outputs without changing the attack
body, then rebuilds both attacked-clean and attacked-watermarked postprocessing
with the sole supported paper_exact_two_stage_aligned mode.
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
    FORMAL_ATTACK_CONFIG,
    assert_formal_debug_info,
    canonical_json_hash,
    current_clip_provenance,
    formal_attack_config_hash,
    load_and_validate_source_manifest,
    sha256_path,
    stage_fid_records,
    transform_config_payload,
)
from raven.color_transfer import (  # noqa: E402
    PAPER_EXACT_TWO_STAGE_ALIGNED,
    color_contrast_transfer_pil,
    color_transfer_diagnostics,
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


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
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


def create_evaluation_snapshot(
    formal_root: Path,
    output_root: Path,
    run_ids: set[str],
) -> tuple[Path, str, str]:
    """Create an immutable exact-ID snapshot for this evaluation cohort."""
    source_index = formal_root / "snapshots" / "snapshot_index.jsonl"
    source_entries = [
        json.loads(line)
        for line in source_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata_paths = {entry["source_metadata_path"] for entry in source_entries}
    metadata_hashes = {entry["source_metadata_sha256"] for entry in source_entries}
    if len(metadata_paths) != 1 or len(metadata_hashes) != 1:
        raise RuntimeError("source snapshot index has mixed metadata provenance")
    source_rows = load_snapshot_rows(formal_root)
    if not run_ids or not run_ids.issubset(source_rows):
        raise RuntimeError("evaluation snapshot run IDs are absent from source")
    selected = [source_rows[run_id] for run_id in sorted(run_ids, key=int)]
    snapshot_root = output_root / "snapshots"
    snapshot_root.mkdir(parents=True)
    snapshot_path = snapshot_root / "cohort.csv"
    with snapshot_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
        handle.flush()
        os.fsync(handle.fileno())
    snapshot_sha = sha256_path(snapshot_path)
    index_path = snapshot_root / "snapshot_index.jsonl"
    entry = {
        "batch_id": "aligned_evaluation_cohort",
        "created_utc": utc_now(),
        "row_count": len(selected),
        "run_id_min": str(selected[0]["run_id"]),
        "run_id_max": str(selected[-1]["run_id"]),
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": snapshot_sha,
        "source_metadata_path": next(iter(metadata_paths)),
        "source_metadata_sha256": next(iter(metadata_hashes)),
        "source_snapshot_index_path": str(source_index.resolve()),
        "source_snapshot_index_sha256": sha256_path(source_index),
        "run_ids_hash": canonical_json_hash(
            {"run_ids": [str(row["run_id"]) for row in selected]}
        ),
    }
    write_jsonl(index_path, [entry])
    return index_path, snapshot_sha, sha256_path(index_path)


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


def paired_effective_source_flow(
    watermarked: dict[str, Any], clean: dict[str, Any], run_id: str
) -> tuple[float, float]:
    fields = (
        "effective_source_flow_dx_image_px",
        "effective_source_flow_dy_image_px",
    )
    values = []
    for field in fields:
        if field not in watermarked or field not in clean:
            raise RuntimeError(f"run_id={run_id}: missing paired {field}")
        wm_value = float(watermarked[field])
        clean_value = float(clean[field])
        if not math.isfinite(wm_value) or not math.isfinite(clean_value):
            raise RuntimeError(f"run_id={run_id}: non-finite paired {field}")
        if wm_value != clean_value:
            raise RuntimeError(f"run_id={run_id}: attacked pair {field} mismatch")
        values.append(wm_value)
    return values[0], values[1]


def select_expected_run_ids(
    source: dict[str, Any],
    watermarked: dict[str, Any],
    clean: dict[str, Any],
    expected_count: int,
) -> set[str]:
    """Validate complete source coverage, then select a stable gate cohort."""
    source_run_ids = set(source)
    if set(watermarked) != source_run_ids or set(clean) != source_run_ids:
        raise RuntimeError("formal source snapshot/attack record coverage mismatch")
    if expected_count <= 0 or expected_count > len(source_run_ids):
        raise RuntimeError(
            f"expected_count must be between 1 and {len(source_run_ids)}"
        )
    return set(sorted(source_run_ids, key=int)[:expected_count])


def build_aligned_records(
    formal_root: Path,
    output_root: Path,
    expected_count: int,
    source_code_manifest_sha256: str,
    git_head: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    run_config = json.loads((formal_root / "run_config.json").read_text(encoding="utf-8"))
    source = load_snapshot_rows(formal_root)
    wm = load_records(formal_root, run_config["attack_config_hash"], "watermarked")
    clean = load_records(formal_root, run_config["attack_config_hash"], "clean")
    run_ids = select_expected_run_ids(source, wm, clean, expected_count)
    aligned_config_hash = formal_attack_config_hash()
    variant_wm: list[dict[str, Any]] = []
    variant_clean: list[dict[str, Any]] = []
    paired_fields = (
        "attack_seed",
        "planned_flow_dx_image_px",
        "planned_flow_dy_image_px",
        "model_id",
        "model_revision",
        "exact_ddim_timestep",
    )
    for run_id in sorted(run_ids, key=int):
        effective_dx, effective_dy = paired_effective_source_flow(
            wm[run_id], clean[run_id], run_id
        )
        for field in paired_fields:
            if wm[run_id].get(field) != clean[run_id].get(field):
                raise RuntimeError(f"run_id={run_id}: attacked pair {field} mismatch")
        for role, base, destination in (
            ("watermarked", wm[run_id], variant_wm),
            ("clean", clean[run_id], variant_clean),
        ):
            pre_color = Path(base["pre_color_attacked_path"])
            pre_color_sha = require_image(pre_color)
            if pre_color_sha != base["pre_color_attacked_sha256"]:
                raise RuntimeError(
                    f"run_id={run_id}: explicit pre-color attacked SHA mismatch"
                )
            source_debug_path = Path(base["debug_info_path"])
            source_debug = json.loads(source_debug_path.read_text(encoding="utf-8"))
            for field, value in (
                ("effective_source_flow_dx_image_px", effective_dx),
                ("effective_source_flow_dy_image_px", effective_dy),
            ):
                if float(source_debug[field]) != value:
                    raise RuntimeError(f"run_id={run_id}: record/debug {field} mismatch")
            item_dir = output_root / "aligned_outputs" / run_id / role
            item_dir.mkdir(parents=True)
            output_path = item_dir / "final_aligned_color_corrected.png"
            reference_path = Path(
                base["watermarked_path"] if role == "watermarked" else base["clean_path"]
            )
            with Image.open(pre_color) as generated, Image.open(reference_path) as reference:
                generated_rgb = generated.convert("RGB")
                reference_rgb = reference.convert("RGB")
                aligned = color_contrast_transfer_pil(
                    generated_rgb,
                    reference_rgb,
                    mode=PAPER_EXACT_TWO_STAGE_ALIGNED,
                    effective_source_flow_dx_image_px=effective_dx,
                    effective_source_flow_dy_image_px=effective_dy,
                )
                aligned.save(output_path)
                diagnostics = color_transfer_diagnostics(
                    generated_rgb,
                    reference_rgb,
                    aligned,
                    mode=PAPER_EXACT_TWO_STAGE_ALIGNED,
                    effective_source_flow_dx_image_px=effective_dx,
                    effective_source_flow_dy_image_px=effective_dy,
                )
            output_sha = require_image(output_path)
            aligned_debug = {
                **source_debug,
                "color_transfer": True,
                "color_transfer_mode": PAPER_EXACT_TWO_STAGE_ALIGNED,
                "color_transfer_diagnostics": diagnostics,
                "source_debug_info_path": str(source_debug_path.resolve()),
                "source_debug_info_sha256": base["debug_info_sha256"],
                "source_transform_config_hash": base["transform_config_hash"],
            }
            aligned_debug["transform_config_hash"] = canonical_json_hash(
                transform_config_payload(aligned_debug)
            )
            assert_formal_debug_info(
                aligned_debug,
                planned_flow_dx_image_px=float(base["planned_flow_dx_image_px"]),
                planned_flow_dy_image_px=float(base["planned_flow_dy_image_px"]),
            )
            aligned_debug_path = item_dir / "debug_info.json"
            write_json(aligned_debug_path, aligned_debug)
            debug_sha = sha256_path(aligned_debug_path)
            variant = {
                **base,
                "attack_config_hash": aligned_config_hash,
                "formal_config_hash": aligned_config_hash,
                "formal_attack_config": FORMAL_ATTACK_CONFIG,
                "attacked_path": str(output_path.resolve()),
                "attacked_sha256": output_sha,
                "output_sha256": output_sha,
                "debug_info_path": str(aligned_debug_path.resolve()),
                "debug_info_sha256": debug_sha,
                "debug_sha256": debug_sha,
                "transform_config_hash": aligned_debug["transform_config_hash"],
                "transform_hash": aligned_debug["transform_config_hash"],
                "evaluation_variant": "shift_aligned_color_transfer",
                "color_transfer_mode": PAPER_EXACT_TWO_STAGE_ALIGNED,
                "output_color_transfer": True,
                "output_color_transfer_mode": PAPER_EXACT_TWO_STAGE_ALIGNED,
                "output_source": "view_guided_output.png + effective-flow aligned color transfer",
                "alignment_flow_source": "effective source flow from actual warp grid",
                "source_pre_color_path": str(pre_color.resolve()),
                "source_pre_color_sha256": pre_color_sha,
                "source_attack_config_hash": base["attack_config_hash"],
                "source_final_output_sha256": base["attacked_sha256"],
                "source_code_manifest_sha": source_code_manifest_sha256,
                "source_code_manifest_sha256": source_code_manifest_sha256,
                "formal_source_config_hash": source_code_manifest_sha256,
                "git_head": git_head,
            }
            destination.append(variant)
    return variant_wm, variant_clean, aligned_config_hash


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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    # This must run before importing torch through clean-fid or detector helpers.
    configure_single_gpu(args.gpu)
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    source_manifest, source_manifest_sha = load_and_validate_source_manifest(
        args.source_manifest.resolve(), repo_root=REPO
    )
    output_root.mkdir(parents=True)
    variant_wm, variant_clean, variant_hash = build_aligned_records(
        formal_root, output_root, args.expected_count, source_manifest_sha,
        str(source_manifest["git_head"]),
    )
    selected_run_ids = {str(record["run_id"]) for record in variant_wm}
    snapshot_index, cohort_snapshot_sha, cohort_index_sha = (
        create_evaluation_snapshot(formal_root, output_root, selected_run_ids)
    )
    for record in variant_wm + variant_clean:
        record["source_snapshot_sha256"] = record["snapshot_sha256"]
        record["snapshot_sha256"] = cohort_snapshot_sha
        record["evaluation_snapshot_index_sha256"] = cohort_index_sha
    provenance = {
        "status": "aligned_color_evaluation_in_progress",
        "variant": "shift_aligned_color_transfer",
        "formal_source_root": str(formal_root),
        "formal_run_config_sha256": sha256_path(formal_root / "run_config.json"),
        "evaluation_snapshot_index_path": str(snapshot_index.resolve()),
        "evaluation_snapshot_index_sha256": cohort_index_sha,
        "evaluation_snapshot_sha256": cohort_snapshot_sha,
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_path(Path(__file__)),
        "variant_config_hash": variant_hash,
        "sample_count": args.expected_count,
        "source_code_manifest_path": str(args.source_manifest.resolve()),
        "source_code_manifest_sha256": source_manifest_sha,
        "formal_attack_config": FORMAL_ATTACK_CONFIG,
        "formal_attack_config_hash": variant_hash,
        "alignment_flow_source": "effective source flow from actual warp grid",
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
    wm_records = output_root / "attack_records_aligned_watermarked.jsonl"
    clean_records = output_root / "attack_records_aligned_clean.jsonl"
    write_jsonl(wm_records, variant_wm)
    write_jsonl(clean_records, variant_clean)
    manifest = output_root / "verification" / "manifest.csv"
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
        reference_definition="original watermarked images from immutable formal snapshots",
        attacked_definition="effective-flow aligned post-color-transfer attacked-watermarked images",
    )
    from raven.quality import clean_fid, openclip_text_image_scores
    fid_result = clean_fid(fid_root / "reference_watermarked", fid_root / "attacked", device=args.device)
    fid_result.update({
        "image_count": args.expected_count,
        "manifest_hash": fid_manifest["manifest_hash"],
        "metric_name": "aligned_fid_watermarked_vs_raven",
        "reference_definition": "original watermarked images from immutable formal snapshots",
        "attacked_definition": "effective-flow aligned post-color-transfer attacked-watermarked images",
        "config_hash": variant_hash,
    })
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
        "status": "aligned_color_evaluation_complete",
        "detector": detector,
        "fid": fid_result,
        "clip": {**clip, **current_clip_provenance()},
        "quality_count": len(quality_rows),
        "quality_psnr_mean": sum(float(row["overlap_psnr"]) for row in quality_rows) / len(quality_rows),
        "quality_ssim_mean": sum(float(row["overlap_ssim"]) for row in quality_rows) / len(quality_rows),
    }
    if not all(math.isfinite(float(value)) for value in (aggregate["quality_psnr_mean"], aggregate["quality_ssim_mean"], fid_result["value"], clip["mean"])):
        raise RuntimeError("non-finite aligned-color aggregate metric")
    protocol = detector["nfpa_rounded2_protocol"]
    run_ids = [str(record["run_id"]) for record in variant_wm]
    validation = {
        "status": "validated_aligned_color_evaluation",
        "sample_count": args.expected_count,
        "unique_run_ids": len(set(run_ids)),
        "duplicate_run_ids": len(run_ids) - len(set(run_ids)),
        "aligned_attack_config_hashes": sorted(
            {record["attack_config_hash"] for record in variant_wm + variant_clean}
        ),
        "source_attack_config_hashes": sorted(
            {record["source_attack_config_hash"] for record in variant_wm + variant_clean}
        ),
        "source_code_manifest_sha256": source_manifest_sha,
        "provider_config_hash": detector["provider_config_hash"],
        "target_watermark_hash": detector["target_watermark_hash"],
        "alignment_flow_source": "effective source flow from actual warp grid",
        "attacked_pair_effective_flow_mismatches": 0,
        "attacked_pair_transform_hash_mismatches": sum(
            left["transform_config_hash"] != right["transform_config_hash"]
            for left, right in zip(variant_wm, variant_clean)
        ),
        "nan_count": 0,
        "inf_count": 0,
    }
    if validation["unique_run_ids"] != args.expected_count:
        raise RuntimeError("aligned validation run-ID coverage mismatch")
    if validation["duplicate_run_ids"] or validation["attacked_pair_transform_hash_mismatches"]:
        raise RuntimeError("aligned validation pairing mismatch")
    if validation["aligned_attack_config_hashes"] != [variant_hash]:
        raise RuntimeError("aligned validation mixed attack config hashes")
    table_row = {
        "Dataset": "diffusiondb",
        "Watermark": "TR",
        "Variant": "shift + paper_exact_two_stage_aligned",
        "Status": validation["status"],
        "N": args.expected_count,
        "Target FPR": protocol["target_fpr"],
        "Original-clean actual FPR": protocol["original_clean_actual_fpr"],
        "Before TPR": protocol["before_tpr"],
        "Attacked TPR at original threshold": protocol[
            "attacked_tpr_at_original_clean_threshold"
        ],
        "Attacked TPR at recalibrated threshold": protocol[
            "attacked_tpr_at_attacked_clean_recalibrated_threshold"
        ],
        "Attack success rate": protocol[
            "attack_success_rate_at_recalibrated_threshold"
        ],
        "Attacked ROC-AUC": protocol["attacked_roc_auc"],
        "FID": fid_result["value"],
        "CLIP": clip["mean"],
        "PSNR": aggregate["quality_psnr_mean"],
        "SSIM": aggregate["quality_ssim_mean"],
        "Flow source": validation["alignment_flow_source"],
        "Attack config hash": variant_hash,
        "Provider config hash": detector["provider_config_hash"],
        "Target watermark hash": detector["target_watermark_hash"],
        "Source manifest SHA": source_manifest_sha,
    }
    aggregate["validation"] = validation
    aggregate["result_table"] = table_row
    write_json(output_root / "aggregate_results.json", aggregate)
    write_json(output_root / "VALIDATED.json", validation)
    write_csv(output_root / "aligned_result_table.csv", table_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
