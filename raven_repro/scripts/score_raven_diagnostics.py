#!/usr/bin/env python
"""Score timestamped RAVEN diagnostic outputs with the validated Tree-Ring detector."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_verification_scores import (
    canonical_score,
    evaluate_image,
    provider_class,
    provider_kwargs,
    raw_score,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-records", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--calibrated-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default="c6a5e9bab8d874d081de76fa270ae0aefa5410ff")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.eval_repo.resolve()))

    import torch
    from raven.resource_guard import limit_cpu_threads
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    attack_rows = [json.loads(line) for line in args.attack_records.read_text().splitlines() if line.strip()]
    with args.baseline_records.open(newline="", encoding="utf-8") as handle:
        baseline_by_id = {row["run_id"]: row for row in csv.DictReader(handle)}
    calibrated = json.loads(args.calibrated_metrics.read_text())
    threshold = float(calibrated["metric"]["threshold"])
    legacy_threshold = float(calibrated["metric"]["legacy_threshold"])
    selected_ids = {row["run_id"] for row in attack_rows}
    if not selected_ids <= baseline_by_id.keys():
        raise ValueError("attack records contain run_ids absent from baseline records")

    device = torch.device(args.device)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        revision=args.model_revision,
    )
    first_baseline = baseline_by_id[next(iter(selected_ids))]
    kwargs = provider_kwargs("TR", first_baseline)
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **kwargs,
    )

    output_path = args.output_dir / "detector_records.jsonl"
    scored = []
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    started = time.monotonic()
    with output_path.open("x", encoding="utf-8") as output:
        for index, attack in enumerate(attack_rows, start=1):
            baseline = baseline_by_id[attack["run_id"]]
            result = evaluate_image(torch, provider, pipe, Path(attack["attacked_path"]), args.steps)
            attacked_raw = raw_score("TR", result)
            attacked_canonical = canonical_score("TR", attacked_raw, result)
            diagnostic = (result.get("p_value_diagnostics") or [{}])[0]
            before_raw = float(baseline["watermarked_raw_score"])
            before_canonical = float(baseline["watermarked_canonical_score"])
            record = {
                **attack,
                "watermarked_raw_detector_score": before_raw,
                "watermarked_canonical_detector_score": before_canonical,
                "attacked_raw_detector_score": attacked_raw,
                "attacked_canonical_detector_score": attacked_canonical,
                "canonical_score_delta": attacked_canonical - before_canonical,
                "calibrated_threshold": threshold,
                "before_prediction": before_canonical >= threshold,
                "after_prediction": attacked_canonical >= threshold,
                "legacy_raw_threshold": legacy_threshold,
                "legacy_before_prediction": before_raw <= legacy_threshold,
                "legacy_after_prediction": attacked_raw <= legacy_threshold,
                "tr_log_p": diagnostic.get("log_p"),
                "tr_statistic": diagnostic.get("statistic"),
                "tr_df": diagnostic.get("df"),
                "tr_sigma": diagnostic.get("sigma"),
                "tr_lambda": diagnostic.get("lambda"),
                "tr_p_underflow": bool(diagnostic.get("p_underflow", False)),
            }
            if not math.isfinite(attacked_canonical):
                raise ValueError(f"non-finite attacked score for {attack['config']}/{attack['run_id']}")
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            scored.append(record)
            print(
                f"[{index}/{len(attack_rows)}] {attack['config']} run_id={attack['run_id']} "
                f"before={before_canonical:.6f} after={attacked_canonical:.6f}",
                flush=True,
            )

    summaries = []
    for config in dict.fromkeys(item["config"] for item in scored):
        rows = [item for item in scored if item["config"] == config]
        summaries.append({
            "config": config,
            "mode": rows[0]["mode"],
            "attention": rows[0]["attention"],
            "shift": f"{rows[0]['shift_sampling']} ({abs(int(rows[0]['image_dx']))} px)",
            "N": len(rows),
            "mean_score_before": mean(item["watermarked_canonical_detector_score"] for item in rows),
            "mean_score_after": mean(item["attacked_canonical_detector_score"] for item in rows),
            "mean_delta": mean(item["canonical_score_delta"] for item in rows),
            "detect_before": mean(item["before_prediction"] for item in rows),
            "detect_after": mean(item["after_prediction"] for item in rows),
            "legacy_detect_before": mean(item["legacy_before_prediction"] for item in rows),
            "legacy_detect_after": mean(item["legacy_after_prediction"] for item in rows),
            "psnr": mean(item["psnr"] for item in rows),
            "ssim": mean(item["ssim"] for item in rows),
            "underflow_count": sum(item["tr_p_underflow"] for item in rows),
        })

    summary = {
        "method": "TR",
        "score_direction": "canonical=-log10(p), higher means more watermark",
        "calibrated_threshold_source": str(args.calibrated_metrics.resolve()),
        "calibrated_threshold": threshold,
        "target_fpr": float(calibrated["metric"]["target_FPR"]),
        "actual_fpr": float(calibrated["metric"]["actual_FPR"]),
        "legacy_raw_threshold": legacy_threshold,
        "N": len(scored),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
        "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "nan_count": sum(not math.isfinite(item["attacked_canonical_detector_score"]) for item in scored),
        "inf_count": sum(math.isinf(item["attacked_canonical_detector_score"]) for item in scored),
        "underflow_count": sum(item["tr_p_underflow"] for item in scored),
        "configs": summaries,
        "records_path": str(output_path.resolve()),
        "clip_status": "skipped: OpenCLIP model weights were not verified in local cache",
    }
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    header = "| Mode | Attention | Shift | N | Mean score before | Mean score after | Mean delta | Detect before | Detect after | PSNR | SSIM |"
    separator = "| --- | --- | --- | -: | -: | -: | -: | -: | -: | -: | -: |"
    lines = [header, separator]
    for item in summaries:
        lines.append(
            f"| {item['mode']} | {item['attention']} | {item['shift']} | {item['N']} | "
            f"{item['mean_score_before']:.6f} | {item['mean_score_after']:.6f} | "
            f"{item['mean_delta']:.6f} | {item['detect_before']:.3f} | "
            f"{item['detect_after']:.3f} | {item['psnr']:.3f} | {item['ssim']:.4f} |"
        )
    (args.output_dir / "diagnostic_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    del provider, pipe
    gc.collect()
    torch.cuda.empty_cache() if args.device == "cuda" else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
