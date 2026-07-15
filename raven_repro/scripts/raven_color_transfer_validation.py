#!/usr/bin/env python
"""Color-transfer-only validation using existing view-guided RAVEN outputs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.color_transfer import color_contrast_transfer, color_transfer_diagnostics
from raven.metrics import pair_quality_metrics, summarize_numeric
from scripts.raven_nfpa_tr_eval import MODEL_ID, MODEL_REVISION, complex_l1_score


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def output_luminance_diagnostics(output: Image.Image, original: Image.Image) -> dict:
    from skimage import color

    output_arr = np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0
    original_arr = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    output_l = color.rgb2lab(output_arr)[..., 0]
    original_l = color.rgb2lab(original_arr)[..., 0]
    output_u8 = np.asarray(output.convert("RGB"), dtype=np.uint8)
    saturated = np.any((output_u8 == 0) | (output_u8 == 255), axis=2)
    return {
        "final_output_L_mean": float(output_l.mean()),
        "final_output_L_std": float(output_l.std()),
        "L_w_mean": float(original_l.mean()),
        "L_w_std": float(original_l.std()),
        "final_output_L_mean_abs_error_vs_original": abs(float(output_l.mean()) - float(original_l.mean())),
        "final_output_L_std_abs_error_vs_original": abs(float(output_l.std()) - float(original_l.std())),
        "output_saturated_pixel_ratio": float(saturated.mean()),
    }


def source_rows(source_validation_dir: Path, count: int) -> list[dict]:
    rows = [
        row for row in load_jsonl(source_validation_dir / "per_sample_results.jsonl")
        if row["mode"] == "B_RAVEN_paper_NFPA_gap_fill_nearest"
    ]
    if len(rows) < count:
        raise ValueError(f"expected at least {count} nearest validation rows, got {len(rows)}")
    return rows[:count]


def aggregate(rows: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["color_transfer_mode"], []).append(row)
    summary = {}
    for mode, items in by_mode.items():
        summary[mode] = {
            "n": len(items),
            "tree_ring_complex_l1": summarize_numeric(row["tree_ring_complex_l1"] for row in items),
            "overlap_psnr_vs_watermarked": summarize_numeric(row["overlap_psnr_vs_watermarked"] for row in items),
            "overlap_ssim_vs_watermarked": summarize_numeric(row["overlap_ssim_vs_watermarked"] for row in items),
            "output_saturated_pixel_ratio": summarize_numeric(row["output_saturated_pixel_ratio"] for row in items),
            "luminance_mean_abs_error": summarize_numeric(row["final_output_L_mean_abs_error_vs_original"] for row in items),
            "luminance_std_abs_error": summarize_numeric(row["final_output_L_std_abs_error_vs_original"] for row in items),
        }
    return {"mode_summary": summary}


def render_md(summary: dict) -> str:
    lines = [
        "# Color Transfer Validation",
        "",
        "Existing `view_guided_output.png` files were reused; no DDIM inversion or denoising was rerun.",
        "",
        "| Mode | N | Mean complex L1 | Overlap PSNR | Overlap SSIM | Saturated ratio | L mean error | L std error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, stats in summary["mode_summary"].items():
        lines.append(
            f"| {mode} | {stats['n']} | {stats['tree_ring_complex_l1']['mean']:.6f} | "
            f"{stats['overlap_psnr_vs_watermarked']['mean']:.3f} | "
            f"{stats['overlap_ssim_vs_watermarked']['mean']:.4f} | "
            f"{stats['output_saturated_pixel_ratio']['mean']:.6f} | "
            f"{stats['luminance_mean_abs_error']['mean']:.6f} | "
            f"{stats['luminance_std_abs_error']['mean']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--eval-repo", type=Path, default=Path("eval_bench_wm"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    for name in ("outputs", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rows = source_rows(args.source_validation_dir, args.count)
    first = rows[0]
    first_manifest = {
        "w_seed": "999999",
        "w_channel": "3",
        "w_pattern": "ring",
        "w_mask_shape": "circle",
        "w_radius": "10",
        "w_measurement": "l1_complex",
        "w_injection": "complex",
    }
    torch, provider, detector_pipe = load_detector(args, first_manifest)
    modes = ("no_color_transfer", "direct_stats", "paper_exact_two_stage")
    results: list[dict] = []
    with (out / "per_sample_results.jsonl").open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            run_id = str(row["run_id"])
            pre_color_path = Path(row["pre_color_path"])
            watermarked_path = Path(row["watermarked_path"])
            watermarked = Image.open(watermarked_path).convert("RGB")
            pre_color = Image.open(pre_color_path).convert("RGB")
            flow_dx = float(row["flow_dx_image_px"])
            flow_dy = float(row["flow_dy_image_px"])
            for mode in modes:
                mode_dir = out / "outputs" / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                output_path = mode_dir / f"{int(run_id):06d}.png"
                if mode == "no_color_transfer":
                    shutil.copy2(pre_color_path, output_path)
                    output_image = pre_color
                    diagnostics = {
                        "color_transfer_mode": mode,
                        "L_opt_mean": None,
                        "L_opt_std": None,
                        "L_c_mean": None,
                        "L_c_std": None,
                        "L_final_before_clip_min": None,
                        "L_final_before_clip_max": None,
                        "L_final_after_clip_min": None,
                        "L_final_after_clip_max": None,
                        **output_luminance_diagnostics(output_image, watermarked),
                    }
                else:
                    output_array = color_contrast_transfer(pre_color, watermarked, mode=mode)
                    output_image = Image.fromarray(output_array, mode="RGB")
                    output_image.save(output_path)
                    diagnostics = color_transfer_diagnostics(pre_color, watermarked, output_image, mode=mode)
                quality = pair_quality_metrics(watermarked, output_image, flow_dx, flow_dy)
                score = complex_l1_score(torch, provider, detector_pipe, output_path, 50)["score"]
                record = {
                    "dataset": row["dataset"],
                    "run_id": run_id,
                    "source_validation_dir": str(args.source_validation_dir.resolve()),
                    "source_pre_color_path": str(pre_color_path.resolve()),
                    "source_pre_color_sha256": sha256_path(pre_color_path),
                    "watermarked_path": str(watermarked_path.resolve()),
                    "watermarked_sha256": row["watermarked_sha256"],
                    "output_path": str(output_path.resolve()),
                    "output_sha256": sha256_path(output_path),
                    "color_transfer_mode": mode,
                    "flow_dx_image_px": flow_dx,
                    "flow_dy_image_px": flow_dy,
                    "tree_ring_complex_l1": float(score),
                    "overlap_psnr_vs_watermarked": quality["overlap_psnr"],
                    "overlap_ssim_vs_watermarked": quality["overlap_ssim"],
                    "raw_full_psnr_vs_watermarked": quality["raw_full_psnr"],
                    "raw_full_ssim_vs_watermarked": quality["raw_full_ssim"],
                    **diagnostics,
                }
                append_jsonl(handle, record)
                results.append(record)
                print(f"[color {len(results)}/{len(rows)*len(modes)}] mode={mode} run_id={run_id} l1={score:.6f} psnr={quality['overlap_psnr']:.3f}", flush=True)
    write_csv(out / "per_sample_results.csv", results)
    summary = aggregate(results)
    write_json(out / "aggregate_results.json", summary)
    (out / "aggregate_results.md").write_text(render_md(summary), encoding="utf-8")
    write_json(out / "provenance.json", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "source_validation_dir": str(args.source_validation_dir.resolve()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "note": "Color-transfer-only validation; reused existing view_guided_output.png files.",
    })
    del provider, detector_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"output_dir": str(out), "rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
