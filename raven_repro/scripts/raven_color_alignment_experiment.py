#!/usr/bin/env python
"""ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT. Color alignment experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.color_transfer import (
    PAPER_EXACT_TWO_STAGE,
    PAPER_EXACT_TWO_STAGE_ALIGNED,
    PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
    align_original_chroma_to_generated,
    color_contrast_transfer,
    color_transfer_diagnostics,
)
from raven.metrics import pair_quality_metrics, roc_auc, summarize_numeric
from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
from raven.gpu_utils import setup_run_logging
from scripts.raven_nfpa_tr_eval import (
    MODEL_ID,
    MODEL_REVISION,
    assert_attack_pair_config_match,
    complex_l1_score,
    nfpa_rate,
    nfpa_threshold,
)

VARIANTS = (
    PAPER_EXACT_TWO_STAGE,
    PAPER_EXACT_TWO_STAGE_ALIGNED,
    PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND,
)
ALPHA = 0.5
FORMAL_CONFIG = {
    "dataset": "DiffusionDB",
    "steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "prompt": "",
    "negative_prompt": "",
    "inversion_mode": "ddim",
    "view_guided_attention": True,
    "shift_rule": "each axis independently sampled from [24,32] or [-32,-24] image pixels",
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "latent_sampling": "nearest",
    "padding": "reflection",
    "baseline_color_transfer": PAPER_EXACT_TWO_STAGE,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        converted.load()
    return converted


def require_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.convert("RGB").size != (512, 512):
            raise ValueError(f"unexpected image size: {path}: {image.size}")


def validate_debug(path: Path, run_id: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid debug_info run_id={run_id}: {path}: {exc}") from exc
    required = {
        "flow_dx_image_px", "flow_dy_image_px", "visual_shift_dx_image_px",
        "visual_shift_dy_image_px", "warp_mode", "interpolation_mode",
        "padding_mode", "transform_config_hash", "exact_timestep",
        "strength", "guidance_scale", "inversion_mode",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"debug_info missing run_id={run_id}: {missing}")
    if float(payload["visual_shift_dx_image_px"]) != -float(payload["flow_dx_image_px"]):
        raise ValueError(f"x flow/visual convention mismatch run_id={run_id}")
    if float(payload["visual_shift_dy_image_px"]) != -float(payload["flow_dy_image_px"]):
        raise ValueError(f"y flow/visual convention mismatch run_id={run_id}")
    return payload


def audit_sources(p1_dir: Path, nfpa_dir: Path, expected_count: int) -> dict:
    manifest_rows = load_manifest(p1_dir / "diagnostic_manifest.csv")
    plan = json.loads((p1_dir / "shift_plan.json").read_text(encoding="utf-8"))
    wm_rows = load_jsonl(p1_dir / "attack_records.jsonl")
    clean_rows = load_jsonl(nfpa_dir / "attacked_clean_records.jsonl")
    counts = {
        "manifest": len(manifest_rows),
        "shift_plan": len(plan.get("samples", [])),
        "attacked_watermarked": len(wm_rows),
        "attacked_clean": len(clean_rows),
    }
    if any(value != expected_count for value in counts.values()):
        raise ValueError(f"source count audit failed: expected={expected_count}, counts={counts}")
    ids = {
        "manifest": {str(row["run_id"]) for row in manifest_rows},
        "plan": {str(row["run_id"]) for row in plan["samples"]},
        "watermarked": {str(row["run_id"]) for row in wm_rows},
        "clean": {str(row["run_id"]) for row in clean_rows},
    }
    if any(len(value) != expected_count for value in ids.values()) or len({frozenset(value) for value in ids.values()}) != 1:
        raise ValueError("run_id sets are duplicate or inconsistent")
    wm_map = {str(row["run_id"]): row for row in wm_rows}
    clean_map = {str(row["run_id"]): row for row in clean_rows}
    manifest_map = {str(row["run_id"]): row for row in manifest_rows}
    for run_id in sorted(ids["manifest"], key=int):
        wm = wm_map[run_id]
        clean = clean_map[run_id]
        assert_attack_pair_config_match(clean, wm, run_id)
        wm_debug = Path(wm["debug_info_path"])
        clean_debug = Path(clean["debug_info_path"])
        wm_info = validate_debug(wm_debug, run_id)
        clean_info = validate_debug(clean_debug, run_id)
        hash_checks = (
            (Path(wm["clean_path"]), wm["clean_sha256"], "clean"),
            (Path(wm["watermarked_path"]), wm["watermarked_sha256"], "watermarked"),
            (Path(wm["attacked_path"]), wm["attacked_sha256"], "attacked-watermarked"),
            (Path(clean["attacked_clean_path"]), clean["attacked_clean_sha256"], "attacked-clean"),
        )
        for path, expected_hash, stage in hash_checks:
            require_image(path)
            if sha256_path(path) != expected_hash:
                raise ValueError(f"{stage} SHA drift run_id={run_id}: {path}")
        for path in (
            wm_debug.parent / "view_guided_output.png",
            clean_debug.parent / "view_guided_output.png",
        ):
            require_image(path)
        if wm_info["transform_config_hash"] != clean_info["transform_config_hash"]:
            raise ValueError(f"debug transform hash mismatch run_id={run_id}")
    return {
        "counts": counts,
        "config_and_hash_audit": "passed",
        "manifest_rows": manifest_rows,
        "manifest_map": manifest_map,
        "watermarked_map": wm_map,
        "clean_map": clean_map,
        "plan": plan,
    }


def output_for_variant(
    variant: str,
    pre_color: Image.Image,
    reference: Image.Image,
    existing_baseline: Path,
    output_path: Path,
    flow_dx: float,
    flow_dy: float,
    verify_baseline: bool,
) -> tuple[Image.Image, dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if variant == PAPER_EXACT_TWO_STAGE:
        if verify_baseline:
            recomputed = color_contrast_transfer(pre_color, reference, mode=variant)
            existing = np.asarray(open_rgb(existing_baseline), dtype=np.uint8)
            if not np.array_equal(recomputed, existing):
                raise ValueError(f"baseline recomputation differs from formal output: {existing_baseline}")
        shutil.copy2(existing_baseline, output_path)
        output = open_rgb(output_path)
        diagnostics = color_transfer_diagnostics(pre_color, reference, output, mode=variant)
    else:
        array = color_contrast_transfer(
            pre_color, reference, mode=variant,
            flow_dx_image_px=flow_dx, flow_dy_image_px=flow_dy, alpha=ALPHA,
        )
        output = Image.fromarray(array, mode="RGB")
        output.save(output_path)
        diagnostics = color_transfer_diagnostics(
            pre_color, reference, output, mode=variant,
            flow_dx_image_px=flow_dx, flow_dy_image_px=flow_dy, alpha=ALPHA,
        )
    return output, diagnostics


def score_finite(torch, provider, detector_pipe, path: Path) -> float:
    score = float(complex_l1_score(torch, provider, detector_pipe, path, 50)["score"])
    if not math.isfinite(score):
        raise ValueError(f"non-finite complex L1: {path}: {score}")
    return score


def direction_smoke(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    visual_dir = output_dir / "alignment_direction_validation"
    visual_dir.mkdir()
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    original = np.stack((xx * 2.0, yy * -2.0), axis=2).astype(np.float32)
    generated = np.full_like(original, 127.0)
    records = []
    for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8), (8, 8), (8, -8), (-8, 8), (-8, -8)):
        aligned, valid = align_original_chroma_to_generated(original, generated, dx, dy)
        expected = generated.copy()
        for y, x in zip(*np.nonzero(valid)):
            expected[y, x] = original[y + dy, x + dx]
        if not np.array_equal(aligned, expected):
            raise AssertionError(f"direction smoke mismatch dx={dx} dy={dy}")
        display = np.zeros((height, width, 3), dtype=np.uint8)
        display[..., 0] = np.clip(aligned[..., 0], 0, 255).astype(np.uint8)
        display[..., 1] = np.clip(aligned[..., 1] + 128, 0, 255).astype(np.uint8)
        display[..., 2] = valid.astype(np.uint8) * 255
        Image.fromarray(display).save(visual_dir / f"aligned_dx_{dx:+d}_dy_{dy:+d}.png")
        Image.fromarray(valid.astype(np.uint8) * 255).save(
            visual_dir / f"valid_mask_dx_{dx:+d}_dy_{dy:+d}.png"
        )
        records.append({
            "flow_dx_image_px": dx,
            "flow_dy_image_px": dy,
            "visual_shift_dx_image_px": -dx,
            "visual_shift_dy_image_px": -dy,
            "valid_ratio": float(valid.mean()),
            "passed": True,
        })

    checker = ((xx // 8 + yy // 8) % 2).astype(np.uint8)
    original_rgb = np.stack(
        (checker * 210 + 20, (1 - checker) * 180 + 30, (xx * 4).clip(0, 255)),
        axis=2,
    ).astype(np.uint8)
    generated_rgb = np.full_like(original_rgb, 127)
    smoke_dx, smoke_dy = 8, -8
    target_x0, target_x1 = max(0, -smoke_dx), min(width, width - smoke_dx)
    target_y0, target_y1 = max(0, -smoke_dy), min(height, height - smoke_dy)
    source_rgb = original_rgb[
        target_y0 + smoke_dy:target_y1 + smoke_dy,
        target_x0 + smoke_dx:target_x1 + smoke_dx,
    ].astype(np.float32)
    generated_rgb[target_y0:target_y1, target_x0:target_x1] = np.clip(
        source_rgb * np.array([0.72, 0.88, 0.64], dtype=np.float32) + np.array([28, 8, 36]),
        0,
        255,
    ).astype(np.uint8)
    variant_hashes = {}
    variant_diagnostics = {}
    for variant in VARIANTS:
        kwargs = {} if variant == PAPER_EXACT_TWO_STAGE else {
            "flow_dx_image_px": smoke_dx,
            "flow_dy_image_px": smoke_dy,
            "alpha": ALPHA,
        }
        output = color_contrast_transfer(generated_rgb, original_rgb, mode=variant, **kwargs)
        path = output_dir / f"{variant}.png"
        Image.fromarray(output).save(path)
        diagnostics = color_transfer_diagnostics(
            generated_rgb, original_rgb, output, mode=variant, **kwargs
        )
        if any(not math.isfinite(value) for value in diagnostics.values() if isinstance(value, float)):
            raise ValueError(f"non-finite direction smoke diagnostics: {variant}")
        variant_hashes[variant] = sha256_path(path)
        variant_diagnostics[variant] = diagnostics
    if variant_hashes[PAPER_EXACT_TWO_STAGE_ALIGNED] == variant_hashes[PAPER_EXACT_TWO_STAGE]:
        raise AssertionError("fully aligned smoke output unexpectedly equals baseline")
    if variant_hashes[PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND] == variant_hashes[PAPER_EXACT_TWO_STAGE]:
        raise AssertionError("aligned blend smoke output unexpectedly equals baseline")
    if variant_hashes[PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND] == variant_hashes[PAPER_EXACT_TWO_STAGE_ALIGNED]:
        raise AssertionError("aligned blend smoke output unexpectedly equals fully aligned")
    smoke_result = {
        "directions": len(records),
        "variants": list(VARIANTS),
        "variant_hashes": variant_hashes,
        "variant_diagnostics": variant_diagnostics,
        "finite": True,
        "variants_not_identical": True,
        "passed": True,
    }
    return {"direction_checks": records, **smoke_result}


def aggregate(rows: list[dict]) -> dict:
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS}
    first_rows = by_variant[PAPER_EXACT_TWO_STAGE]
    original_clean = [row["original_clean_l1"] for row in first_rows]
    watermarked = [row["original_watermarked_l1"] for row in first_rows]
    before_threshold = nfpa_threshold(original_clean)
    before = {
        "threshold": before_threshold,
        "actual_fpr": nfpa_rate(original_clean, before_threshold),
        "tpr": nfpa_rate(watermarked, before_threshold),
        "roc_auc": roc_auc([-v for v in watermarked], [-v for v in original_clean]),
    }
    metrics = (
        "overlap_psnr_vs_watermarked", "overlap_ssim_vs_watermarked",
        "overlap_psnr_vs_clean", "overlap_ssim_vs_clean",
        "raw_full_psnr_vs_watermarked", "raw_full_ssim_vs_watermarked",
        "output_saturated_pixel_ratio", "output_rgb_out_of_gamut_ratio_before_clip",
        "final_output_L_mean", "final_output_L_std",
        "final_output_L_mean_abs_error_vs_original",
        "final_output_L_std_abs_error_vs_original",
        "mean_abs_a_difference_vs_generated", "mean_abs_b_difference_vs_generated",
        "mean_chroma_delta_e76_vs_generated",
        "attacked_watermarked_l1", "attacked_clean_l1",
    )
    summary: dict[str, Any] = {"before_attack": before, "variants": {}, "paired_vs_baseline": {}}
    baseline_map = {row["run_id"]: row for row in first_rows}
    for variant, items in by_variant.items():
        attacked_clean = [row["attacked_clean_l1"] for row in items]
        attacked_wm = [row["attacked_watermarked_l1"] for row in items]
        threshold = nfpa_threshold(attacked_clean)
        variant_summary = {
            "n": len(items),
            "after_attack": {
                "threshold": threshold,
                "actual_fpr": nfpa_rate(attacked_clean, threshold),
                "tpr": nfpa_rate(attacked_wm, threshold),
                "attack_success_rate": 1.0 - nfpa_rate(attacked_wm, threshold),
                "roc_auc": roc_auc([-v for v in attacked_wm], [-v for v in attacked_clean]),
                "threshold_negative_source": f"{variant} attacked-clean outputs",
            },
        }
        for metric in metrics:
            values = [row.get(metric) for row in items if row.get(metric) is not None]
            variant_summary[metric] = summarize_numeric(values) if values else None
        summary["variants"][variant] = variant_summary
        if variant == PAPER_EXACT_TWO_STAGE:
            continue
        comparisons = {}
        directions = {
            "overlap_psnr_vs_watermarked": "higher",
            "overlap_ssim_vs_watermarked": "higher",
            "output_saturated_pixel_ratio": "lower",
            "output_rgb_out_of_gamut_ratio_before_clip": "lower",
            "attacked_watermarked_l1": "higher",
        }
        item_map = {row["run_id"]: row for row in items}
        for metric, better in directions.items():
            diffs = [item_map[k][metric] - baseline_map[k][metric] for k in baseline_map]
            improved = sum(value > 0 for value in diffs) if better == "higher" else sum(value < 0 for value in diffs)
            comparisons[metric] = {
                "difference_variant_minus_baseline": summarize_numeric(diffs),
                "improved_samples": improved,
                "N": len(diffs),
                "improved_ratio": improved / len(diffs),
                "better_direction": better,
            }
        summary["paired_vs_baseline"][variant] = comparisons
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# Experiment 1: Shift-aligned chroma transfer",
        "",
        "Formal RAVEN diffusion settings were not changed or rerun. Aligned modes are offline unpublished implementation-detail experiments.",
        "",
        "| Variant | N | PSNR overlap vs WM | SSIM overlap vs WM | Saturated | RGB OOG | Attacked-WM L1 | Attacked-clean L1 | After threshold | FPR | TPR | Success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        stats = summary["variants"][variant]
        after = stats["after_attack"]
        lines.append(
            f"| {variant} | {stats['n']} | "
            f"{stats['overlap_psnr_vs_watermarked']['mean']:.4f} | "
            f"{stats['overlap_ssim_vs_watermarked']['mean']:.6f} | "
            f"{stats['output_saturated_pixel_ratio']['mean']:.6f} | "
            f"{stats['output_rgb_out_of_gamut_ratio_before_clip']['mean']:.6f} | "
            f"{stats['attacked_watermarked_l1']['mean']:.6f} | "
            f"{stats['attacked_clean_l1']['mean']:.6f} | "
            f"{after['threshold']:.6f} | {after['actual_fpr']:.6f} | "
            f"{after['tpr']:.6f} | {after['attack_success_rate']:.6f} |"
        )
    before = summary["before_attack"]
    lines.extend([
        "",
        f"Before threshold: {before['threshold']:.8f}; actual FPR: {before['actual_fpr']:.6f}; TPR: {before['tpr']:.6f}; ROC-AUC: {before['roc_auc']:.6f}.",
        "",
        "Full-image PSNR/SSIM are diagnostics only; inverse-warp valid-overlap values are the formal quality comparison.",
    ])
    return "\n".join(lines) + "\n"


def run_experiment(args) -> int:
    limit_cpu_threads(1)
    guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
    guard.check("before Experiment 1 source audit")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    setup_run_logging(out)
    for name in ("outputs", "alignment_direction_validation"):
        (out / name).mkdir()
    audit = audit_sources(args.source_p1_dir.resolve(), args.source_nfpa_dir.resolve(), args.expected_count)
    selected_ids = sorted(audit["manifest_map"], key=int)[: min(args.count, args.expected_count)]
    if len(selected_ids) < args.validation_count:
        raise ValueError(f"need at least {args.validation_count} complete samples, got {len(selected_ids)}")
    smoke_tmp = out / "_direction_smoke_tmp"
    smoke = direction_smoke(smoke_tmp)
    for path in (smoke_tmp / "alignment_direction_validation").iterdir():
        shutil.move(str(path), out / "alignment_direction_validation" / path.name)
    shutil.rmtree(smoke_tmp)

    first_manifest = audit["manifest_map"][selected_ids[0]]
    detector_args = SimpleNamespace(eval_repo=args.eval_repo, device=args.device)
    torch, provider, detector_pipe = load_detector(detector_args, first_manifest)
    rows: list[dict] = []
    hashes_by_variant: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
    started = utc_now()
    for sample_index, run_id in enumerate(selected_ids, start=1):
            guard.check(f"before Experiment 1 sample {sample_index}/{len(selected_ids)}")
            wm_record = audit["watermarked_map"][run_id]
            clean_record = audit["clean_map"][run_id]
            assert_attack_pair_config_match(clean_record, wm_record, run_id)
            flow_dx = float(wm_record["flow_dx_image_px"])
            flow_dy = float(wm_record["flow_dy_image_px"])
            wm_debug = Path(wm_record["debug_info_path"])
            clean_debug = Path(clean_record["debug_info_path"])
            wm_pre_path = wm_debug.parent / "view_guided_output.png"
            clean_pre_path = clean_debug.parent / "view_guided_output.png"
            original_clean_path = Path(wm_record["clean_path"])
            original_wm_path = Path(wm_record["watermarked_path"])
            original_clean = open_rgb(original_clean_path)
            original_wm = open_rgb(original_wm_path)
            wm_pre = open_rgb(wm_pre_path)
            clean_pre = open_rgb(clean_pre_path)
            original_clean_l1 = score_finite(torch, provider, detector_pipe, original_clean_path)
            original_wm_l1 = score_finite(torch, provider, detector_pipe, original_wm_path)
            for variant in VARIANTS:
                wm_out_path = out / "outputs" / variant / "attacked_watermarked" / f"{int(run_id):06d}.png"
                clean_out_path = out / "outputs" / variant / "attacked_clean" / f"{int(run_id):06d}.png"
                wm_out, diagnostics = output_for_variant(
                    variant, wm_pre, original_wm, Path(wm_record["attacked_path"]),
                    wm_out_path, flow_dx, flow_dy, sample_index <= args.validation_count,
                )
                clean_out, _ = output_for_variant(
                    variant, clean_pre, original_clean, Path(clean_record["attacked_clean_path"]),
                    clean_out_path, flow_dx, flow_dy, sample_index <= args.validation_count,
                )
                quality_wm = pair_quality_metrics(original_wm, wm_out, flow_dx, flow_dy)
                quality_clean = pair_quality_metrics(original_clean, wm_out, flow_dx, flow_dy)
                attacked_wm_l1 = score_finite(torch, provider, detector_pipe, wm_out_path)
                attacked_clean_l1 = score_finite(torch, provider, detector_pipe, clean_out_path)
                record = {
                    "dataset": wm_record["dataset"],
                    "run_id": run_id,
                    "variant": variant,
                    "alpha": 1.0 if variant == PAPER_EXACT_TWO_STAGE_ALIGNED else (ALPHA if variant == PAPER_EXACT_TWO_STAGE_ALIGNED_BLEND else None),
                    "source_transform_config_hash": wm_record["transform_config_hash"],
                    "flow_dx_image_px": flow_dx,
                    "flow_dy_image_px": flow_dy,
                    "visual_shift_dx_image_px": -flow_dx,
                    "visual_shift_dy_image_px": -flow_dy,
                    "source_pre_color_watermarked_path": str(wm_pre_path.resolve()),
                    "source_pre_color_clean_path": str(clean_pre_path.resolve()),
                    "output_attacked_watermarked_path": str(wm_out_path.resolve()),
                    "output_attacked_clean_path": str(clean_out_path.resolve()),
                    "output_attacked_watermarked_sha256": sha256_path(wm_out_path),
                    "output_attacked_clean_sha256": sha256_path(clean_out_path),
                    "original_clean_l1": original_clean_l1,
                    "original_watermarked_l1": original_wm_l1,
                    "attacked_clean_l1": attacked_clean_l1,
                    "attacked_watermarked_l1": attacked_wm_l1,
                    "overlap_psnr_vs_watermarked": quality_wm["overlap_psnr"],
                    "overlap_ssim_vs_watermarked": quality_wm["overlap_ssim"],
                    "raw_full_psnr_vs_watermarked": quality_wm["raw_full_psnr"],
                    "raw_full_ssim_vs_watermarked": quality_wm["raw_full_ssim"],
                    "overlap_psnr_vs_clean": quality_clean["overlap_psnr"],
                    "overlap_ssim_vs_clean": quality_clean["overlap_ssim"],
                    "raw_full_psnr_vs_clean": quality_clean["raw_full_psnr"],
                    "raw_full_ssim_vs_clean": quality_clean["raw_full_ssim"],
                    "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    **diagnostics,
                }
                if any(not math.isfinite(value) for value in record.values() if isinstance(value, float)):
                    raise ValueError(f"non-finite result run_id={run_id} variant={variant}")
                rows.append(record)
                hashes_by_variant[variant].append(record["output_attacked_watermarked_sha256"])
                print(
                    f"[Experiment1 {sample_index}/{len(selected_ids)}] run_id={run_id} "
                    f"variant={variant} psnr={record['overlap_psnr_vs_watermarked']:.3f} "
                    f"ssim={record['overlap_ssim_vs_watermarked']:.4f} l1={attacked_wm_l1:.6f}",
                    flush=True,
                )
            if sample_index == args.validation_count:
                for variant in VARIANTS[1:]:
                    if hashes_by_variant[variant] == hashes_by_variant[PAPER_EXACT_TWO_STAGE]:
                        raise ValueError(f"10-sample validation outputs identical to baseline: {variant}")
            del original_clean, original_wm, wm_pre, clean_pre, wm_out, clean_out
            gc.collect()
            torch.cuda.empty_cache()

    summary = aggregate(rows)
    repo = Path(__file__).resolve().parents[2]
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, check=True, capture_output=True).stdout.strip()
    provenance = {
        "created_utc": started,
        "completed_utc": utc_now(),
        "git_commit_sha": git_head,
        "source_pre_color_output_directory": str(args.source_p1_dir.resolve()),
        "source_attacked_clean_directory": str(args.source_nfpa_dir.resolve()),
        "dataset": "DiffusionDB",
        "sample_count": len(selected_ids),
        "baseline_config": FORMAL_CONFIG,
        "shift_convention": {
            "flow": "generated[y,x] corresponds to original[y+flow_dy,x+flow_dx]",
            "visual_shift": "(-flow_dx,-flow_dy)",
            "inverse_warp_overlap": "valid correspondence only",
        },
        "alignment_formula": "generated chroma[y,x] uses original chroma[y+flow_dy,x+flow_dx]",
        "alpha": ALPHA,
        "overlap_rule": "non-overlap retains generated a_opt/b_opt; no wrap or reflected correspondence",
        "color_transfer_modes": list(VARIANTS),
        "diffusion_reexecuted": False,
        "ddim_unet_reexecuted": False,
        "waiter_script": str(args.waiter_script.resolve()),
        "experiment_start_time": started,
        "source_audit": {"counts": audit["counts"], "config_and_hash_audit": audit["config_and_hash_audit"]},
        "threshold_calibration_protocol": "per-variant attacked-clean negatives; strict complex L1 score < threshold at empirical 1% FPR",
        "classification": {
            "formal_paper_setting": FORMAL_CONFIG,
            "unpublished_implementation_detail_experiment": list(VARIANTS[1:]),
            "diagnostic_only": ["full-image PSNR", "full-image SSIM", "direction visualizations"],
            "formal_comparison": ["inverse-warp valid-overlap PSNR", "inverse-warp valid-overlap SSIM", "NFPA-style complex L1"],
        },
    }
    write_json(out / "results.json", {
        "completed_utc": utc_now(),
        "created_utc": started,
        "sample_count": len(selected_ids),
        "smoke": smoke,
        "summary": summary,
        "provenance": provenance,
    })
    del provider, detector_pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-p1-dir", type=Path)
    parser.add_argument("--source-nfpa-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1001)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--eval-repo", type=Path, default=Path("eval_bench_wm"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--waiter-script", type=Path, default=Path(__file__).with_name("wait_for_images_then_run_color_alignment.sh"))
    parser.add_argument("--min-cpu-mem-gb", type=float, default=64.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=96.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
    parser.add_argument("--direction-smoke-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.direction_smoke_only:
        print(json.dumps(direction_smoke(args.output_dir.resolve()), indent=2))
        return 0
    if args.source_p1_dir is None or args.source_nfpa_dir is None:
        raise ValueError("--source-p1-dir and --source-nfpa-dir are required")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
