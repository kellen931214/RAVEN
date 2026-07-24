#!/usr/bin/env python3
"""Evaluate explicit pre-color RAVEN outputs without rerunning the attack.

This evaluator isolates the effect of color transfer by scoring the exact
``view_guided_output`` (pre-color) image already saved by a completed formal
attack. It never re-runs inversion, denoising or warp.

Two methods are supported and dispatch to their own detector:

* ``TR`` reuses the formal Tree-Ring / NFPA protocol, which requires the
  attacked-clean records for attacked-clean FPR recalibration.
* ``GS`` reuses the Gaussian Shading verification scripts and does NOT read or
  require attacked-clean records. GS detector results are kept in GS-named
  fields and never written into TR/NFPA field names.

The CLI ``--dataset`` and ``--method`` must match the source formal root's
``run_config.json`` exactly; a mismatch fails closed.
"""

from __future__ import annotations

import argparse
import csv as _csv
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import (  # noqa: E402
    CLIP_CONFIG,
    NO_COLOR_FID_ATTACKED_DEFINITION,
    bind_pre_color_attack_record,
    canonical_json_hash,
    current_clip_provenance,
    load_and_validate_source_manifest,
    require_clean_git_worktree,
    stage_no_color_fid_records,
)
from raven.metrics import pair_quality_metrics  # noqa: E402
from raven.pairing_provenance import audit_pairing_rows  # noqa: E402

from experiments.run_raven_aligned_color_eval import (  # noqa: E402
    create_evaluation_snapshot,
    load_snapshot_rows,
    paired_effective_source_flow,
    select_expected_run_ids,
    write_csv,
    write_json,
    write_jsonl,
)
from experiments.run_raven_formal_eval import attack_records  # noqa: E402


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def configure_single_gpu(physical_gpu: int) -> None:
    if physical_gpu < 0:
        raise ValueError("--gpu must be non-negative")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def select_watermarked_run_ids(
    source: dict[str, Any], watermarked: dict[str, Any], expected_count: int
) -> set[str]:
    """GS coverage check: only attacked-watermarked records are required."""
    source_run_ids = set(source)
    if set(watermarked) != source_run_ids:
        raise RuntimeError(
            "formal source snapshot/attacked-watermarked record coverage mismatch"
        )
    if expected_count <= 0 or expected_count > len(source_run_ids):
        raise RuntimeError(f"expected_count must be between 1 and {len(source_run_ids)}")
    return set(sorted(source_run_ids, key=int)[:expected_count])


def uniform_manifest_provider_hash(manifest: Path) -> str:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        hashes = {row.get("provider_config_hash", "") for row in _csv.DictReader(handle)}
    hashes.discard("")
    if len(hashes) != 1:
        raise RuntimeError(f"non-uniform manifest provider config hash: {sorted(hashes)}")
    return next(iter(hashes))


def bound_no_color_record(base: dict[str, Any], variant_hash: str) -> dict[str, Any]:
    """Bind the explicit pre-color path/SHA and stamp no-color provenance."""
    record = bind_pre_color_attack_record(base)
    record.update(
        {
            "output_sha256": record["attacked_sha256"],
            "evaluation_variant": "no_color_transfer",
            "evaluation_variant_hash": variant_hash,
            "metric_image_source": "pre_color_attacked_path",
            "output_color_transfer": False,
            "output_color_transfer_mode": "none",
            "source_final_post_color_path": base["attacked_path"],
            "source_final_post_color_sha256": base["attacked_sha256"],
            "source_attack_config_hash": base["attack_config_hash"],
        }
    )
    return record


def compute_quality_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality_rows: list[dict[str, Any]] = []
    for record in records:
        with Image.open(record["watermarked_path"]) as reference, Image.open(
            record["attacked_path"]
        ) as attacked:
            metric = pair_quality_metrics(
                reference.convert("RGB"),
                attacked.convert("RGB"),
                record["effective_source_flow_dx_image_px"],
                record["effective_source_flow_dy_image_px"],
            )
        quality_rows.append({"run_id": record["run_id"], **metric})
    return quality_rows


def compute_fid_and_clip(
    variant_wm: list[dict[str, Any]],
    *,
    output_root: Path,
    variant_hash: str,
    expected_count: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fid_root, fid_manifest = stage_no_color_fid_records(
        variant_wm,
        formal_output=output_root,
        quality_config_hash=variant_hash,
        expected_count=expected_count,
    )
    from raven.quality import clean_fid, openclip_text_image_scores

    fid = clean_fid(
        fid_root / "reference_watermarked", fid_root / "attacked", device=device
    )
    fid.update(
        {
            "image_count": expected_count,
            "manifest_hash": fid_manifest["manifest_hash"],
            "reference_definition": fid_manifest["reference_definition"],
            "attacked_definition": fid_manifest["attacked_definition"],
            "config_hash": variant_hash,
        }
    )
    write_json(fid_root / "fid_result.json", fid)
    clip = openclip_text_image_scores(
        [record["attacked_path"] for record in variant_wm],
        [record["prompt"] for record in variant_wm],
        device=device,
        model_name=CLIP_CONFIG["clip_model_name"],
        pretrained=CLIP_CONFIG["clip_pretrained"],
    )
    clip = {**clip, **current_clip_provenance()}
    write_json(output_root / "metrics" / "clip_result.json", clip)
    return fid, clip


def evaluate_tr_no_color(
    args: argparse.Namespace,
    *,
    head: str,
    source_manifest_sha: str,
    formal_root: Path,
    output_root: Path,
    run_config: dict[str, Any],
    source_rows: dict[str, dict[str, str]],
) -> int:
    """Formal Tree-Ring / NFPA no-color evaluation (requires attacked-clean)."""
    wm_by_id = {
        str(row["run_id"]): row
        for row in attack_records(formal_root, run_config, "watermarked")
    }
    clean_by_id = {
        str(row["run_id"]): row
        for row in attack_records(formal_root, run_config, "clean")
    }
    run_ids = select_expected_run_ids(
        source_rows, wm_by_id, clean_by_id, args.expected_count
    )
    variant_hash = canonical_json_hash(
        {
            "metric_image_source": "explicit pre-color attacked image",
            "source_attack_config_hash": run_config["attack_config_hash"],
            "source_code_manifest_sha256": source_manifest_sha,
            "protocol": "raven_formal_no_color_v1",
        }
    )

    variant_wm: list[dict[str, Any]] = []
    variant_clean: list[dict[str, Any]] = []
    for run_id in sorted(run_ids, key=int):
        paired_effective_source_flow(wm_by_id[run_id], clean_by_id[run_id], run_id)
        if wm_by_id[run_id]["pairing_sha256"] != clean_by_id[run_id]["pairing_sha256"]:
            raise RuntimeError(f"run_id={run_id}: attacked pair pairing_sha256 mismatch")
        for base, destination in (
            (wm_by_id[run_id], variant_wm),
            (clean_by_id[run_id], variant_clean),
        ):
            destination.append(bound_no_color_record(base, variant_hash))

    output_root.mkdir(parents=True)
    snapshot_index, snapshot_sha, index_sha = create_evaluation_snapshot(
        formal_root, output_root, run_ids
    )
    for record in variant_wm + variant_clean:
        record["source_snapshot_sha256"] = record["snapshot_sha256"]
        record["snapshot_sha256"] = snapshot_sha
        record["evaluation_snapshot_index_sha256"] = index_sha

    provenance = {
        "status": "no_color_evaluation_in_progress",
        "variant": "no_color_transfer",
        "method": "TR",
        "dataset": args.dataset,
        "formal_source_root": str(formal_root),
        "source_attack_config_hash": run_config["attack_config_hash"],
        "evaluation_variant_hash": variant_hash,
        "sample_count": args.expected_count,
        "source_code_manifest_path": str(args.source_manifest.resolve()),
        "source_code_manifest_sha256": source_manifest_sha,
        "git_head": head,
        "physical_gpu": args.gpu,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quality_reference": "watermarked input",
        "quality_overlap": "effective source flow inverse warp",
        "fid_reference": "original watermarked images",
        "fid_attacked_definition": NO_COLOR_FID_ATTACKED_DEFINITION,
        "clip": CLIP_CONFIG,
    }
    write_json(output_root / "provenance.json", provenance)
    wm_records = output_root / "attack_records_no_color_watermarked.jsonl"
    clean_records = output_root / "attack_records_no_color_clean.jsonl"
    write_jsonl(wm_records, variant_wm)
    write_jsonl(clean_records, variant_clean)

    manifest = output_root / "verification" / "manifest.csv"
    manifest_command = [
        sys.executable,
        str(REPO / "raven_repro/scripts/build_verification_manifest.py"),
        "--dataset", args.dataset,
        "--method", "TR",
        "--metadata", str(snapshot_index),
        "--attack-records", str(wm_records),
        "--snapshot-manifest", str(snapshot_index),
    ]
    if run_config.get("attack_config_source_path"):
        manifest_command.extend(["--attack-config", run_config["attack_config_source_path"]])
    manifest_command.extend(["--output", str(manifest)])
    run(manifest_command)
    run([
        sys.executable,
        str(REPO / "raven_repro/scripts/raven_nfpa_tr_eval.py"),
        "score-formal",
        "--manifest", str(manifest),
        "--attacked-clean-records", str(clean_records),
        "--output-dir", str(output_root / "verification" / "tr_nfpa"),
        "--device", args.device,
    ])

    quality_rows = compute_quality_rows(variant_wm)
    quality_root = output_root / "metrics" / "quality"
    quality_root.mkdir(parents=True)
    write_jsonl(quality_root / "quality_records.jsonl", quality_rows)

    fid, clip = compute_fid_and_clip(
        variant_wm,
        output_root=output_root,
        variant_hash=variant_hash,
        expected_count=args.expected_count,
        device=args.device,
    )

    detector = json.loads(
        (output_root / "verification" / "tr_nfpa" / "aggregate_results.json").read_text()
    )
    protocol = detector["nfpa_rounded2_protocol"]
    values = [
        *(float(row["overlap_psnr"]) for row in quality_rows),
        *(float(row["overlap_ssim"]) for row in quality_rows),
        float(fid["value"]),
        float(clip["mean"]),
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite no-color metric")
    quality_psnr_mean = sum(float(row["overlap_psnr"]) for row in quality_rows) / len(
        quality_rows
    )
    quality_ssim_mean = sum(float(row["overlap_ssim"]) for row in quality_rows) / len(
        quality_rows
    )
    aggregate = {
        **provenance,
        "status": "no_color_evaluation_complete",
        "detector": detector,
        "fid": fid,
        "clip": clip,
        "quality_count": len(quality_rows),
        "quality_psnr_mean": quality_psnr_mean,
        "quality_ssim_mean": quality_ssim_mean,
    }
    validation = {
        "status": "validated_no_color_evaluation",
        "method": "TR",
        "evaluation_variant": "no_color_transfer",
        "output_color_transfer": False,
        "sample_count": args.expected_count,
        "unique_run_ids": len(run_ids),
        "source_attack_config_hash": run_config["attack_config_hash"],
        "source_formal_root": str(formal_root),
        "source_pre_color_sha256_set_hash": canonical_json_hash(
            {"pre_color_sha256": sorted(r["pre_color_attacked_sha256"] for r in variant_wm)}
        ),
        "evaluation_variant_hash": variant_hash,
        "source_code_manifest_sha256": source_manifest_sha,
        "git_head": head,
        "provider_config_hash": detector["provider_config_hash"],
        "target_watermark_hash": detector["target_watermark_hash"],
        "metric_image_source": "pre_color_attacked_path",
        "pairing_mismatches": 0,
        "sha_mismatches": 0,
        "nan_count": 0,
        "inf_count": 0,
    }
    table = {
        "Dataset": args.dataset,
        "Watermark": "TR",
        "Variant": run_config["attack_config"]["variant_name"] + " + no color",
        "N": args.expected_count,
        "Original-clean actual FPR": protocol["original_clean_actual_fpr"],
        "Attacked-clean recalibrated actual FPR": protocol["attacked_clean_actual_fpr"],
        "Before TPR": protocol["before_tpr"],
        "Attacked TPR at original threshold": protocol[
            "attacked_tpr_at_original_clean_threshold"
        ],
        "Attacked TPR at recalibrated threshold": protocol[
            "attacked_tpr_at_attacked_clean_recalibrated_threshold"
        ],
        "Attack success at original threshold": 1.0
        - protocol["attacked_tpr_at_original_clean_threshold"],
        "Attack success at recalibrated threshold": protocol[
            "attack_success_rate_at_recalibrated_threshold"
        ],
        "ROC-AUC": protocol["attacked_roc_auc"],
        "FID": fid["value"],
        "CLIP": clip["mean"],
        "PSNR": quality_psnr_mean,
        "SSIM": quality_ssim_mean,
        "Source manifest SHA": source_manifest_sha,
    }
    aggregate["validation"] = validation
    aggregate["result_table"] = table
    write_json(output_root / "aggregate_results.json", aggregate)
    write_json(output_root / "VALIDATED.json", validation)
    write_csv(output_root / "no_color_result_table.csv", table)
    return 0


def evaluate_gs_no_color(
    args: argparse.Namespace,
    *,
    head: str,
    source_manifest_sha: str,
    formal_root: Path,
    output_root: Path,
    run_config: dict[str, Any],
    source_rows: dict[str, dict[str, str]],
) -> int:
    """Gaussian Shading no-color evaluation. Never reads attacked-clean records."""
    wm_by_id = {
        str(row["run_id"]): row
        for row in attack_records(formal_root, run_config, "watermarked")
    }
    run_ids = select_watermarked_run_ids(source_rows, wm_by_id, args.expected_count)
    variant_hash = canonical_json_hash(
        {
            "metric_image_source": "explicit pre-color attacked image",
            "source_attack_config_hash": run_config["attack_config_hash"],
            "source_code_manifest_sha256": source_manifest_sha,
            "protocol": "raven_formal_no_color_gs_v1",
            "method": "GS",
        }
    )

    variant_wm: list[dict[str, Any]] = []
    for run_id in sorted(run_ids, key=int):
        variant_wm.append(bound_no_color_record(wm_by_id[run_id], variant_hash))

    output_root.mkdir(parents=True)
    snapshot_index, snapshot_sha, index_sha = create_evaluation_snapshot(
        formal_root, output_root, run_ids
    )
    for record in variant_wm:
        record["source_snapshot_sha256"] = record["snapshot_sha256"]
        record["snapshot_sha256"] = snapshot_sha
        record["evaluation_snapshot_index_sha256"] = index_sha

    provenance = {
        "status": "no_color_evaluation_in_progress",
        "variant": "no_color_transfer",
        "method": "GS",
        "dataset": args.dataset,
        "formal_source_root": str(formal_root),
        "source_attack_config_hash": run_config["attack_config_hash"],
        "evaluation_variant_hash": variant_hash,
        "sample_count": args.expected_count,
        "source_code_manifest_path": str(args.source_manifest.resolve()),
        "source_code_manifest_sha256": source_manifest_sha,
        "git_head": head,
        "physical_gpu": args.gpu,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quality_reference": "watermarked input",
        "quality_overlap": "effective source flow inverse warp",
        "fid_reference": "original watermarked images",
        "fid_attacked_definition": NO_COLOR_FID_ATTACKED_DEFINITION,
        "clip": CLIP_CONFIG,
        "attacked_clean_used": False,
        "detector": "gaussian_shading_verification",
    }
    write_json(output_root / "provenance.json", provenance)
    wm_records = output_root / "attack_records_no_color_watermarked.jsonl"
    write_jsonl(wm_records, variant_wm)

    verification = output_root / "verification"
    manifest = verification / "manifest.csv"
    manifest_command = [
        sys.executable,
        str(REPO / "raven_repro/scripts/build_verification_manifest.py"),
        "--dataset", args.dataset,
        "--method", "GS",
        "--metadata", str(snapshot_index),
        "--attack-records", str(wm_records),
        "--snapshot-manifest", str(snapshot_index),
    ]
    if run_config.get("attack_config_source_path"):
        manifest_command.extend(["--attack-config", run_config["attack_config_source_path"]])
    manifest_command.extend(["--output", str(manifest)])
    run(manifest_command)

    scores = verification / "scores.csv"
    run([
        sys.executable,
        str(REPO / "raven_repro/scripts/extract_verification_scores.py"),
        "--method", "GS",
        "--metadata", str(manifest),
        "--output", str(scores),
        "--model-revision", run_config["attack_config"]["model_revision"],
        "--device", args.device,
    ])
    verification_result = verification / "verification_result.json"
    run([
        sys.executable,
        str(REPO / "raven_repro/scripts/evaluate_verification.py"),
        "--method", "GS",
        "--records", str(scores),
        "--output-json", str(verification_result),
        "--output-rows", str(verification / "verification_rows.json"),
    ])

    quality_rows = compute_quality_rows(variant_wm)
    quality_root = output_root / "metrics" / "quality"
    quality_root.mkdir(parents=True)
    write_jsonl(quality_root / "quality_records.jsonl", quality_rows)

    fid, clip = compute_fid_and_clip(
        variant_wm,
        output_root=output_root,
        variant_hash=variant_hash,
        expected_count=args.expected_count,
        device=args.device,
    )

    detector = json.loads(verification_result.read_text())
    if detector.get("method") != "GS":
        raise RuntimeError("GS verification result is not method=GS")
    gs_metric = detector["metric"]
    provider_config_hash = uniform_manifest_provider_hash(manifest)
    quality_psnr_mean = sum(float(row["overlap_psnr"]) for row in quality_rows) / len(
        quality_rows
    )
    quality_ssim_mean = sum(float(row["overlap_ssim"]) for row in quality_rows) / len(
        quality_rows
    )
    values = [
        *(float(row["overlap_psnr"]) for row in quality_rows),
        *(float(row["overlap_ssim"]) for row in quality_rows),
        float(fid["value"]),
        float(clip["mean"]),
        float(gs_metric["macro_bit_accuracy_before"]),
        float(gs_metric["macro_bit_accuracy_attacked"]),
        float(gs_metric["before_roc_auc"]),
        float(gs_metric["attacked_roc_auc"]),
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite GS no-color metric")

    aggregate = {
        **provenance,
        "status": "no_color_evaluation_complete",
        "gs_verification_result": detector,
        "fid": fid,
        "clip": clip,
        "quality_count": len(quality_rows),
        "quality_psnr_mean": quality_psnr_mean,
        "quality_ssim_mean": quality_ssim_mean,
    }
    validation = {
        "status": "validated_no_color_evaluation",
        "method": "GS",
        "evaluation_variant": "no_color_transfer",
        "output_color_transfer": False,
        "sample_count": args.expected_count,
        "unique_run_ids": len(run_ids),
        "source_attack_config_hash": run_config["attack_config_hash"],
        "source_formal_root": str(formal_root),
        "source_pre_color_sha256_set_hash": canonical_json_hash(
            {"pre_color_sha256": sorted(r["pre_color_attacked_sha256"] for r in variant_wm)}
        ),
        "evaluation_variant_hash": variant_hash,
        "source_code_manifest_sha256": source_manifest_sha,
        "git_head": head,
        "provider_config_hash": provider_config_hash,
        "metric_image_source": "pre_color_attacked_path",
        "attacked_clean_used": False,
        "gs_macro_bit_accuracy_before": gs_metric["macro_bit_accuracy_before"],
        "gs_macro_bit_accuracy_attacked": gs_metric["macro_bit_accuracy_attacked"],
        "pairing_mismatches": 0,
        "sha_mismatches": 0,
        "nan_count": 0,
        "inf_count": 0,
    }
    table = {
        "Dataset": args.dataset,
        "Watermark": "GS",
        "Variant": run_config["attack_config"]["variant_name"] + " + no color",
        "N": args.expected_count,
        "GS macro bit accuracy before": gs_metric["macro_bit_accuracy_before"],
        "GS macro bit accuracy attacked": gs_metric["macro_bit_accuracy_attacked"],
        "GS before TPR at clean-calibrated threshold": gs_metric[
            "before_tpr_at_clean_calibrated_threshold"
        ],
        "GS attacked TPR at clean-calibrated threshold": gs_metric[
            "attacked_tpr_at_clean_calibrated_threshold"
        ],
        "GS before ROC-AUC": gs_metric["before_roc_auc"],
        "GS attacked ROC-AUC": gs_metric["attacked_roc_auc"],
        "GS official one-bit attacked rate": gs_metric["official_onebit_rates"]["attacked"],
        "GS official traceability attacked rate": gs_metric["official_traceability_rates"][
            "attacked"
        ],
        "FID": fid["value"],
        "CLIP": clip["mean"],
        "PSNR": quality_psnr_mean,
        "SSIM": quality_ssim_mean,
        "Source manifest SHA": source_manifest_sha,
    }
    aggregate["validation"] = validation
    aggregate["result_table"] = table
    write_json(output_root / "aggregate_results.json", aggregate)
    write_json(output_root / "VALIDATED.json", validation)
    write_csv(output_root / "no_color_result_table.csv", table)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=["TR", "GS"])
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    args.method = args.method.upper()

    if not args.device.startswith("cuda"):
        raise RuntimeError("formal no-color evaluation forbids CPU fallback")
    configure_single_gpu(args.gpu)
    head = require_clean_git_worktree(REPO)
    _, source_manifest_sha = load_and_validate_source_manifest(
        args.source_manifest.resolve(), repo_root=REPO
    )
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        validated = output_root / "VALIDATED.json"
        if validated.is_file():
            payload = json.loads(validated.read_text(encoding="utf-8"))
            if (
                payload.get("status") == "validated_no_color_evaluation"
                and payload.get("method") == args.method
                and payload.get("source_code_manifest_sha256") == source_manifest_sha
                and payload.get("git_head") == head
                and payload.get("sample_count") == args.expected_count
            ):
                print(json.dumps({"status": "reused", "output_root": str(output_root)}))
                return 0
        raise FileExistsError(output_root)

    run_config = json.loads((formal_root / "run_config.json").read_text(encoding="utf-8"))
    if run_config["source_code_manifest_sha256"] != source_manifest_sha:
        raise RuntimeError("source attack/current runtime source manifest mismatch")
    if run_config["git_head"] != head:
        raise RuntimeError("source attack/current commit mismatch")
    if str(run_config.get("dataset")) != args.dataset:
        raise RuntimeError(
            f"--dataset={args.dataset!r} does not match formal run_config "
            f"dataset={run_config.get('dataset')!r}"
        )
    if str(run_config.get("method")).upper() != args.method:
        raise RuntimeError(
            f"--method={args.method!r} does not match formal run_config "
            f"method={run_config.get('method')!r}"
        )
    source_rows = load_snapshot_rows(formal_root)
    audit_pairing_rows(
        source_rows.values(), expected_count=args.expected_count, verify_files=True
    )

    common = dict(
        head=head,
        source_manifest_sha=source_manifest_sha,
        formal_root=formal_root,
        output_root=output_root,
        run_config=run_config,
        source_rows=source_rows,
    )
    if args.method == "TR":
        return evaluate_tr_no_color(args, **common)
    if args.method == "GS":
        return evaluate_gs_no_color(args, **common)
    raise AssertionError(args.method)


if __name__ == "__main__":
    raise SystemExit(main())
