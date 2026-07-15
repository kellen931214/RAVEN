#!/usr/bin/env python
"""RAVEN-paper / NFPA-gap-fill warp and overlap-quality diagnostics.

This script keeps completed P1 outputs immutable. It can recompute quality
metrics from existing DiffusionDB P1 outputs and run a small paired validation
for the new RAVEN-paper / NFPA-gap-fill transform.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import pair_quality_metrics, summarize_numeric
from raven.pipeline_raven import RavenPipeline
from raven.utils import load_image
from scripts import raven_p1_full as p1
from scripts.raven_nfpa_tr_eval import complex_l1_score

MODEL_ID = p1.MODEL_ID
MODEL_REVISION = p1.MODEL_REVISION
IMPLEMENTATION_CLASSIFICATION = (
    "RAVEN paper-faithful settings with NFPA-based gap filling "
    "for underspecified warp implementation details."
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_text(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd or Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def image_path_near_debug(debug_info_path: Path, filename: str) -> Path:
    path = debug_info_path.parent / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def quality_record(reference_path: Path, pre_color_path: Path, post_color_path: Path, flow_dx: float, flow_dy: float, suffix: str) -> dict:
    reference = Image.open(reference_path).convert("RGB")
    pre = Image.open(pre_color_path).convert("RGB")
    post = Image.open(post_color_path).convert("RGB")
    pre_metrics = pair_quality_metrics(reference, pre, flow_dx, flow_dy)
    post_metrics = pair_quality_metrics(reference, post, flow_dx, flow_dy)
    return {
        **prefix_metrics(f"pre_color_vs_{suffix}", pre_metrics),
        **prefix_metrics(f"post_color_vs_{suffix}", post_metrics),
        f"primary_quality_reference_{suffix}": suffix,
        f"primary_overlap_protocol_{suffix}": "inverse_warp_valid_correspondence",
    }


def aggregate_quality(rows: list[dict]) -> dict:
    metric_keys = [
        key for key in rows[0]
        if key.endswith(("psnr", "ssim", "area_ratio"))
        or key.endswith(("valid_overlap_width", "valid_overlap_height"))
    ]
    return {key: summarize_numeric(row[key] for row in rows if key in row) for key in metric_keys}


def command_recompute_quality(args) -> int:
    p1_dir = args.p1_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir()
    records = load_jsonl(p1_dir / "attack_records.jsonl")
    if args.count is not None:
        records = records[:args.count]
    rows: list[dict] = []
    with (output_dir / "per_sample_quality.jsonl").open("x", encoding="utf-8") as handle:
        for index, row in enumerate(records, start=1):
            debug_info_path = Path(row["debug_info_path"])
            pre_color_path = image_path_near_debug(debug_info_path, "view_guided_output.png")
            post_color_path = Path(row["attacked_path"])
            for path_key in ("watermarked_path", "clean_path"):
                if sha256_path(Path(row[path_key])) != row[path_key.replace("path", "sha256")]:
                    raise ValueError(f"SHA drift for {path_key} run_id={row['run_id']}")
            if sha256_path(post_color_path) != row["attacked_sha256"]:
                raise ValueError(f"attacked SHA drift run_id={row['run_id']}")
            flow_dx = float(row["flow_dx_image_px"])
            flow_dy = float(row["flow_dy_image_px"])
            record = {
                "dataset": row["dataset"],
                "run_id": str(row["run_id"]),
                "mode": row["mode"],
                "source_p1_dir": str(p1_dir),
                "watermarked_path": row["watermarked_path"],
                "watermarked_sha256": row["watermarked_sha256"],
                "clean_path": row["clean_path"],
                "clean_sha256": row["clean_sha256"],
                "pre_color_path": str(pre_color_path.resolve()),
                "pre_color_sha256": sha256_path(pre_color_path),
                "post_color_path": str(post_color_path.resolve()),
                "post_color_sha256": row["attacked_sha256"],
                "flow_dx_image_px": flow_dx,
                "flow_dy_image_px": flow_dy,
                "visual_dx_image_px": -flow_dx,
                "visual_dy_image_px": -flow_dy,
                **quality_record(Path(row["watermarked_path"]), pre_color_path, post_color_path, flow_dx, flow_dy, "watermarked"),
                **quality_record(Path(row["clean_path"]), pre_color_path, post_color_path, flow_dx, flow_dy, "clean"),
            }
            append_jsonl(handle, record)
            rows.append(record)
            if index % 100 == 0 or index == len(records):
                print(f"[quality {index}/{len(records)}] run_id={row['run_id']} post_overlap_psnr_wm={record['post_color_vs_watermarked_overlap_psnr']:.3f}", flush=True)
    write_csv(output_dir / "per_sample_quality.csv", rows)
    summary = {
        "protocol": "raven_inverse_warp_overlap_quality_v1",
        "source_p1_dir": str(p1_dir),
        "source_attack_records_sha256": sha256_path(p1_dir / "attack_records.jsonl"),
        "n": len(rows),
        "primary_paper_comparable_fields": [
            "post_color_vs_watermarked_overlap_psnr",
            "post_color_vs_watermarked_overlap_ssim",
        ],
        "stats": aggregate_quality(rows),
    }
    write_json(output_dir / "quality_summary.json", summary)
    lines = [
        "# Existing P1 Overlap Quality Recompute",
        "",
        f"Source P1 dir: `{p1_dir}`",
        f"N: {len(rows)}",
        "",
        "Primary paper-comparable quality fields use watermarked input vs final post-color-transfer attacked output, cropped by inverse-warp valid overlap.",
        "",
        "| Metric | Mean | Median | Std | Min | Max | NaN | Inf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("post_color_vs_watermarked_overlap_psnr", "post_color_vs_watermarked_overlap_ssim", "pre_color_vs_watermarked_overlap_psnr", "pre_color_vs_watermarked_overlap_ssim", "post_color_vs_watermarked_raw_full_psnr", "post_color_vs_watermarked_raw_full_ssim"):
        stats = summary["stats"][key]
        lines.append(
            f"| {key} | {stats['mean']:.6f} | {stats['median']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} | {stats['nan_count']} | {stats['inf_count']} |"
        )
    (output_dir / "quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_common_provenance(output_dir, args, extra={"command": "recompute-quality"})
    print(json.dumps({"output_dir": str(output_dir), "n": len(rows)}, indent=2))
    return 0


def write_common_provenance(output_dir: Path, args, extra: dict | None = None) -> None:
    repo = Path(__file__).resolve().parents[2]
    diff = run_text(["git", "diff"], cwd=repo)
    (output_dir / "git_diff.patch").write_text(diff + ("\n" if diff else ""), encoding="utf-8")
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "python": sys.executable,
        "git_head": run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "git_status_short": run_text(["git", "status", "--short"], cwd=repo).splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        **(extra or {}),
    }
    write_json(output_dir / "provenance.json", payload)


def load_detector(args, first_manifest_row: dict):
    import torch

    if not (args.eval_repo / "utils" / "pipe" / "pipe_utils.py").is_file():
        raise FileNotFoundError(args.eval_repo)
    sys.path.insert(0, str(args.eval_repo.resolve()))
    from raven.resource_guard import limit_cpu_threads
    from scripts.extract_verification_scores import provider_class, provider_kwargs
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    device = torch.device(args.device)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=MODEL_ID,
        resolution=512,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        revision=MODEL_REVISION,
    )
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **provider_kwargs("TR", first_manifest_row),
    )
    return torch, provider, pipe


def score_l1(torch, provider, pipe, path: Path) -> dict:
    torch.cuda.reset_peak_memory_stats()
    result = complex_l1_score(torch, provider, pipe, path, 50)
    return result


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def command_validate(args) -> int:
    import torch

    p1_dir = args.p1_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for name in ("outputs", "logs", "configs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    manifest = {str(row["run_id"]): row for row in load_csv_rows(p1_dir / "diagnostic_manifest.csv")}
    p1_records = load_jsonl(p1_dir / "attack_records.jsonl")[: args.count]
    old_l1 = {}
    if args.l1_records and args.l1_records.is_file():
        old_l1 = {str(row["run_id"]): row for row in load_jsonl(args.l1_records)}
    modes = [
        {"mode": "A_old_P1_latent_grid_nearest_reflection", "reuse_old": True, "sampling_mode": "nearest"},
        {"mode": "B_RAVEN_paper_NFPA_gap_fill_nearest", "reuse_old": False, "sampling_mode": "nearest"},
        {"mode": "C_RAVEN_paper_NFPA_gap_fill_bilinear", "reuse_old": False, "sampling_mode": "bilinear"},
    ]
    write_json(output_dir / "configs" / "fixed_conditions.json", {
        "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
        "raven_paper_settings": {
            "ddim_inversion_steps": 50,
            "strength": 0.15,
            "guidance_scale": 2.5,
            "inversion_prompt": "",
            "reconstruction_prompt": "",
            "shift_unit": "image_pixels",
            "shift_range": "[24,32] or [-32,-24] independently per axis",
            "color_transfer": "CIELAB color and contrast transfer",
        },
        "nfpa_gap_fill": {
            "grid": "512x512 image coordinate grid",
            "normalization": "x_norm = 2*(x+dx)/W - 1; y_norm = 2*(y+dy)/H - 1",
            "coordinate_resize": "bilinear, align_corners=False",
            "padding_mode": "reflection",
            "main_latent_sampling_mode": "nearest",
        },
        "modes": modes,
    })
    pipe = RavenPipeline(model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype)
    torch_score = provider = detector_pipe = None
    if args.score:
        first_manifest = manifest[str(p1_records[0]["run_id"])]
        torch_score, provider, detector_pipe = load_detector(args, first_manifest)
    rows: list[dict] = []
    with (output_dir / "per_sample_results.jsonl").open("x", encoding="utf-8") as handle:
        for index, old in enumerate(p1_records, start=1):
            run_id = str(old["run_id"])
            manifest_row = manifest[run_id]
            watermarked = load_image(Path(old["watermarked_path"]), size=512)
            clean = load_image(Path(old["clean_path"]), size=512)
            for mode in modes:
                started = time.monotonic()
                if mode["reuse_old"]:
                    final_path = Path(old["attacked_path"])
                    debug_info_path = Path(old["debug_info_path"])
                    pre_color_path = image_path_near_debug(debug_info_path, "view_guided_output.png")
                    attacked_sha = old["attacked_sha256"]
                    debug_info = json.loads(debug_info_path.read_text())
                    peak_gpu = int(old.get("peak_gpu_memory_bytes") or 0)
                else:
                    item_dir = output_dir / "outputs" / mode["mode"] / f"{int(run_id):06d}"
                    if item_dir.exists():
                        raise FileExistsError(item_dir)
                    torch.cuda.reset_peak_memory_stats()
                    pipe.run(
                        input_image=watermarked,
                        output_dir=item_dir,
                        steps=50,
                        strength=0.15,
                        guidance_scale=2.5,
                        shift_space="image_pixels",
                        warp_mode="raven_paper_nfpa_gap_fill",
                        padding_mode="reflection",
                        latent_sampling_mode=mode["sampling_mode"],
                        shift_x=float(old["flow_dx_image_px"]),
                        shift_y=float(old["flow_dy_image_px"]),
                        view_guided_attention=True,
                        color_transfer=True,
                        seed=int(old["seed"]),
                        prompt="",
                        negative_prompt="",
                        debug=args.debug,
                        inversion_mode="ddim",
                    )
                    final_path = item_dir / "final_color_corrected.png"
                    pre_color_path = item_dir / "view_guided_output.png"
                    debug_info_path = item_dir / "debug_info.json"
                    debug_info = json.loads(debug_info_path.read_text())
                    attacked_sha = sha256_path(final_path)
                    peak_gpu = int(torch.cuda.max_memory_allocated())
                flow_dx = float(old["flow_dx_image_px"])
                flow_dy = float(old["flow_dy_image_px"])
                attacked = load_image(final_path, size=None)
                quality = {
                    **quality_record(Path(old["watermarked_path"]), pre_color_path, final_path, flow_dx, flow_dy, "watermarked"),
                    **quality_record(Path(old["clean_path"]), pre_color_path, final_path, flow_dx, flow_dy, "clean"),
                }
                if args.score:
                    if mode["reuse_old"] and run_id in old_l1:
                        before_l1 = float(old_l1[run_id]["watermarked_l1"])
                        after_l1 = float(old_l1[run_id]["attacked_watermarked_l1"])
                    else:
                        before_l1 = float(score_l1(torch_score, provider, detector_pipe, Path(old["watermarked_path"]))["score"])
                        after_l1 = float(score_l1(torch_score, provider, detector_pipe, final_path)["score"])
                else:
                    before_l1 = after_l1 = None
                record = {
                    "dataset": old["dataset"],
                    "run_id": run_id,
                    "mode": mode["mode"],
                    "implementation_classification": IMPLEMENTATION_CLASSIFICATION if not mode["reuse_old"] else "legacy P1 latent-grid reference",
                    "watermarked_path": old["watermarked_path"],
                    "watermarked_sha256": old["watermarked_sha256"],
                    "clean_path": old["clean_path"],
                    "attacked_path": str(final_path.resolve()),
                    "attacked_sha256": attacked_sha,
                    "pre_color_path": str(pre_color_path.resolve()),
                    "pre_color_sha256": sha256_path(pre_color_path),
                    "debug_info_path": str(debug_info_path.resolve()),
                    "flow_dx_image_px": flow_dx,
                    "flow_dy_image_px": flow_dy,
                    "visual_dx_image_px": -flow_dx,
                    "visual_dy_image_px": -flow_dy,
                    "warp_mode": debug_info.get("warp_mode"),
                    "transform_setting_name": debug_info.get("transform_setting_name"),
                    "latent_sampling_mode": debug_info.get("interpolation_mode"),
                    "padding_mode": debug_info.get("padding_mode"),
                    "align_corners": debug_info.get("align_corners"),
                    "transform_config_hash": debug_info.get("transform_config_hash"),
                    "tree_ring_l1_before": before_l1,
                    "tree_ring_l1_after": after_l1,
                    "tree_ring_l1_delta_after_minus_before": (after_l1 - before_l1) if before_l1 is not None else None,
                    "runtime_seconds": float(time.monotonic() - started),
                    "peak_gpu_memory_bytes": peak_gpu,
                    "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    **quality,
                }
                append_jsonl(handle, record)
                rows.append(record)
                print(f"[validate {len(rows)}/{args.count * len(modes)}] mode={mode['mode']} run_id={run_id} post_overlap_psnr_wm={record['post_color_vs_watermarked_overlap_psnr']:.3f}", flush=True)
    write_csv(output_dir / "per_sample_results.csv", rows)
    aggregate = aggregate_validation(rows)
    write_json(output_dir / "aggregate_results.json", aggregate)
    (output_dir / "aggregate_results.md").write_text(render_validation_md(aggregate), encoding="utf-8")
    write_audit_report(output_dir, aggregate)
    write_common_provenance(output_dir, args, extra={"command": "validate", "n_samples": args.count})
    del pipe
    if detector_pipe is not None:
        del provider, detector_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, indent=2))
    return 0


def aggregate_validation(rows: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    summary = {}
    for mode, items in by_mode.items():
        summary[mode] = {
            "n": len(items),
            "tree_ring_l1_before": summarize_numeric(row["tree_ring_l1_before"] for row in items if row["tree_ring_l1_before"] is not None),
            "tree_ring_l1_after": summarize_numeric(row["tree_ring_l1_after"] for row in items if row["tree_ring_l1_after"] is not None),
            "tree_ring_l1_delta_after_minus_before": summarize_numeric(row["tree_ring_l1_delta_after_minus_before"] for row in items if row["tree_ring_l1_delta_after_minus_before"] is not None),
            "post_color_vs_watermarked_overlap_psnr": summarize_numeric(row["post_color_vs_watermarked_overlap_psnr"] for row in items),
            "post_color_vs_watermarked_overlap_ssim": summarize_numeric(row["post_color_vs_watermarked_overlap_ssim"] for row in items),
            "pre_color_vs_watermarked_overlap_psnr": summarize_numeric(row["pre_color_vs_watermarked_overlap_psnr"] for row in items),
            "pre_color_vs_watermarked_overlap_ssim": summarize_numeric(row["pre_color_vs_watermarked_overlap_ssim"] for row in items),
            "post_color_vs_watermarked_raw_full_psnr": summarize_numeric(row["post_color_vs_watermarked_raw_full_psnr"] for row in items),
            "post_color_vs_watermarked_raw_full_ssim": summarize_numeric(row["post_color_vs_watermarked_raw_full_ssim"] for row in items),
            "runtime_seconds": summarize_numeric(row["runtime_seconds"] for row in items),
        }
    return {
        "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
        "mode_summary": summary,
    }


def render_validation_md(aggregate: dict) -> str:
    lines = [
        "# RAVEN-paper / NFPA-gap-fill Validation",
        "",
        f"Implementation classification: {IMPLEMENTATION_CLASSIFICATION}",
        "",
        "| Mode | N | Mean L1 before | Mean L1 after | Mean delta | Post-color overlap PSNR vs WM | Post-color overlap SSIM vs WM | Runtime mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, stats in aggregate["mode_summary"].items():
        before = stats["tree_ring_l1_before"]["mean"]
        after = stats["tree_ring_l1_after"]["mean"]
        delta = stats["tree_ring_l1_delta_after_minus_before"]["mean"]
        lines.append(
            f"| {mode} | {stats['n']} | {before if before is not None else float('nan'):.6f} | "
            f"{after if after is not None else float('nan'):.6f} | {delta if delta is not None else float('nan'):.6f} | "
            f"{stats['post_color_vs_watermarked_overlap_psnr']['mean']:.3f} | "
            f"{stats['post_color_vs_watermarked_overlap_ssim']['mean']:.4f} | "
            f"{stats['runtime_seconds']['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_audit_report(output_dir: Path, aggregate: dict) -> None:
    lines = [
        "# RAVEN-paper / NFPA-gap-fill Audit Report",
        "",
        "## 1. RAVEN Paper-defined Items",
        "Stable Diffusion 2.1 image-to-image reconstruction, 50-step DDIM inversion, strength 0.15, CFG 2.5, empty inversion/reconstruction prompts, bounded global diagonal translation with independent dx/dy sampled from [24,32] or [-32,-24] image pixels, fixed shift plan, and CIELAB color/contrast transfer are treated as RAVEN-defined settings.",
        "",
        "## 2. RAVEN Paper-underspecified Items",
        "Low-level grid construction, coordinate normalization, coordinate-grid resize, inverse grid-sampling convention, padding, and latent value sampling are treated as underspecified by RAVEN and filled from NFPA utils.py.",
        "",
        "## 3. NFPA Gap Fill",
        "NFPA utils.py `coords_grid`, `warp_single_latent`, `create_motion_field`, and `create_motion_field_and_warp_latents_xy` use x/y coordinate channels, `coords0 + reference_flow`, `/W` and `/H` normalization, bilinear coordinate-grid resize, and `grid_sample(..., mode=\"nearest\", padding_mode=\"reflection\")` with PyTorch's effective `align_corners=False` default. The local implementation makes `align_corners=False` explicit.",
        "",
        "## 4. Explicitly Excluded NFPA Settings",
        "NFPA adaptive X/Y search, `max_warp_latents`, `NFP_XY=40`, NFPA checkpoint defaults, NFPA 10-step inference, and NFPA detector calibration are not used for this RAVEN reproduction setting.",
        "",
        "## 5. Coordinate Grid Formula",
        "Grid shape is `[N, 2, 512, 512]`, channel 0 is X and channel 1 is Y. Sampling coordinates are `base_coords + flow`; normalization is `x_norm = 2 * (x + dx) / W - 1` and `y_norm = 2 * (y + dy) / H - 1`, then resized bilinearly to latent resolution.",
        "",
        "## 6. Flow and Visual Direction",
        "Because `grid_sample` uses inverse sampling, `attacked[y, x]` samples from `reference[y + flow_dy, x + flow_dx]`. Therefore visual content motion is approximately `(-flow_dx, -flow_dy)`.",
        "",
        "## 7. Overlap Crop Formula",
        "Quality metrics use only pixels satisfying `0 <= x < W`, `0 <= x + dx < W`, `0 <= y < H`, and `0 <= y + dy < H`; padding/reflection regions are excluded.",
        "",
        "## 8. Validation Summary",
        render_validation_md(aggregate),
        "## Implementation classification",
        f"Implementation classification: {IMPLEMENTATION_CLASSIFICATION}",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recompute = sub.add_parser("recompute-quality")
    recompute.add_argument("--p1-dir", type=Path, required=True)
    recompute.add_argument("--output-dir", type=Path, required=True)
    recompute.add_argument("--count", type=int)
    recompute.set_defaults(func=command_recompute_quality)

    validate = sub.add_parser("validate")
    validate.add_argument("--p1-dir", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--count", type=int, default=10)
    validate.add_argument("--device", default="cuda")
    validate.add_argument("--dtype", default="float16")
    validate.add_argument("--eval-repo", type=Path, default=Path("eval_bench_wm"))
    validate.add_argument("--l1-records", type=Path, default=Path("outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/nfpa_l1_scores.jsonl"))
    validate.add_argument("--score", action="store_true")
    validate.add_argument("--debug", action="store_true")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
