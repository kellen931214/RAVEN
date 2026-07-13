#!/usr/bin/env python3
"""Generate watermarked images with eval_bench_wm providers only.

This produces watermarked source images and validates the watermark before any
RAVEN attack. It intentionally does not run RAVEN or removal baselines.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
RAVEN_ROOT = WORKSPACE / "raven_repro"
BENCH_ROOT = WORKSPACE / "eval_bench_wm"
for root in (str(RAVEN_ROOT), str(BENCH_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from raven.gpu_utils import configure_gpu, finalize_gpu_logging, utc_timestamp, write_experiment_records

PAPER_WM_METHODS_IN_BENCH = ["GS", "TR", "RID", "HSTR", "HSQR"]
PAPER_WM_NAMES = {
    "GS": "Gaussian Shading",
    "TR": "Tree-Ring",
    "RID": "RingID",
    "HSTR": "HSTR",
    "HSQR": "HSQR",
}
METRIC_MAP = {
    "GS": "bit_accuracy",
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_name", type=str, default="mscoco")
    parser.add_argument("--prompts_csv", type=str, default=str(WORKSPACE / "data" / "prompts" / "mscoco_5000.csv"))
    parser.add_argument("--output_dir", type=str, default=str(WORKSPACE / "data" / "watermarked"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--require_free_gpu", type=str_to_bool, default=True)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=64.0)
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=96.0)
    parser.add_argument("--max_process_ram_gb", type=float, default=16.0)
    return parser


def parse_args() -> argparse.Namespace:
    from utils.pipe import pipe_utils
    from utils.wm.gs_provider import parser as gs_parser
    from utils.wm.hstr_provider import parser as hstr_parser
    from utils.wm.hsqr_provider import parser as hsqr_parser
    from utils.wm.ringid_provider import parser as ringid_parser
    from utils.wm.tr_provider import parser as tr_parser

    parent_parsers = [base_parser(), gs_parser, tr_parser, ringid_parser, hstr_parser, hsqr_parser]
    parser = argparse.ArgumentParser(
        description="Generate paper-overlap watermarked images with eval_bench_wm providers",
        parents=parent_parsers,
        conflict_handler="resolve",
    )
    parser.add_argument("--wm_types", nargs="+", default=PAPER_WM_METHODS_IN_BENCH, choices=PAPER_WM_METHODS_IN_BENCH)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--modelid_target", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--validate_before", type=str_to_bool, default=True)
    return parser.parse_args()


def bool_for_json(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_prompts(path: Path, start_index: int, num_pairs: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prompt CSV has no header: {path}")
        prompt_field = "prompt" if "prompt" in reader.fieldnames else reader.fieldnames[0]
        for row_index, row in enumerate(reader):
            if row_index < start_index:
                continue
            prompt = (row.get(prompt_field) or "").strip()
            if not prompt:
                continue
            rows.append({
                "run_id": str(row_index),
                "prompt": prompt,
                "prompt_id": str(row.get("id", row_index)),
                "source": str(row.get("source", "")),
            })
            if len(rows) >= num_pairs:
                break
    if len(rows) < num_pairs:
        raise ValueError(f"Only found {len(rows)} prompts in {path}; requested {num_pairs}")
    return rows


def existing_completed_rows(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                run_id = int(row["run_id"])
            except Exception:
                continue
            if str(row.get("watermarked_image_path", "")) and Path(row["watermarked_image_path"]).exists():
                done.add(run_id)
    return done


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def detection_value(results: Dict[str, Any], wm_type: str) -> Any:
    return results[METRIC_MAP[wm_type]]


def compact_results(results: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bit_accuracy",
        "p_value",
        "value",
        "l1_dist",
        "detection_success",
        "log_message",
        "prc_threshold",
        "message_bits_str_initial",
        "message_bits_str_recovered",
    ]
    return {key: results.get(key) for key in keys}


def generate_watermarked(pipe_provider_target: Any, wm_provider: Any, prompt: str, wm_zT: Any, args: argparse.Namespace) -> Image.Image:
    if hasattr(wm_provider, "generate"):
        generated = wm_provider.generate(
            pipe_provider_target=pipe_provider_target,
            prompts=prompt,
            latents=wm_zT,
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
    else:
        generated = pipe_provider_target.generate(
            prompts=prompt,
            latents=wm_zT,
            num_inference_steps=args.num_inference_steps_target,
            guidance_scale=args.guidance_scale_target,
        )
    return generated["images_PIL"][0]


def summarize_metadata(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"completed": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"completed": 0}

    def parse_bool(value: str) -> Optional[bool]:
        if str(value).lower() in {"true", "1"}:
            return True
        if str(value).lower() in {"false", "0"}:
            return False
        return None

    successes = [parse_bool(row.get("before_detection_successful", "")) for row in rows]
    successes_valid = [x for x in successes if x is not None]
    return {
        "completed": len(rows),
        "before_detection_rate": (sum(successes_valid) / len(successes_valid)) if successes_valid else None,
        "metadata_csv": str(csv_path),
    }


def run_method(args: argparse.Namespace, dataset_dir: Path, wm_type: str, prompt_rows: List[Dict[str, str]], guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from utils.imprint_utils import validate
    from utils.pipe import pipe_utils
    from utils.utils import check_if_detection_successful, get_detection_threshold, seed_everything, set_random_seed
    from utils.wm.wm_utils import WmProviders

    wm_provider_cls = WmProviders[wm_type].value
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    method_dir = dataset_dir / wm_type
    method_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = method_dir / "metadata.csv"
    summary_json = method_dir / "summary.json"
    completed = existing_completed_rows(metadata_csv)
    detection_threshold = get_detection_threshold(wm_type, args.modelid_target)

    print(f"[{wm_type}] loading target pipeline: {args.modelid_target}", flush=True)
    set_random_seed(args.seed)
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
    )
    provider_kwargs = vars(args).copy()
    for reserved_key in ("latent_shape", "dtype", "device"):
        provider_kwargs.pop(reserved_key, None)
    wm_provider = wm_provider_cls(
        latent_shape=pipe_provider_target.get_latent_shape(),
        dtype=pipe_provider_target.get_dtype(),
        device=device,
        **provider_kwargs,
    )
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]
    message_bits_str_initial = (
        wm_initial_results["message_bits_str_list"][0]
        if "message_bits_str_list" in wm_initial_results
        else None
    )

    rows_written = 0
    try:
        for local_idx, prompt_row in enumerate(prompt_rows):
            run_id = int(prompt_row["run_id"])
            item_dir = method_dir / f"{run_id:06d}"
            image_path = item_dir / "watermarked.png"
            if run_id in completed and image_path.exists():
                if local_idx == 0 or (local_idx + 1) % 25 == 0:
                    print(f"[{wm_type}] skip existing run_id={run_id}", flush=True)
                continue

            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/before")
            if wm_type in ["PRC", "SPH"]:
                seed_everything(args.seed)

            print(f"[{wm_type}] generating {local_idx + 1}/{len(prompt_rows)} run_id={run_id}", flush=True)
            if image_path.exists():
                watermarked_image = Image.open(image_path).convert("RGB")
            else:
                watermarked_image = generate_watermarked(pipe_provider_target, wm_provider, prompt_row["prompt"], wm_zT, args)
                item_dir.mkdir(parents=True, exist_ok=True)
                watermarked_image.save(image_path)

            if args.validate_before:
                before = validate(
                    out_dir=str(method_dir),
                    image_to_verify_PIL=watermarked_image,
                    original_PIL=watermarked_image,
                    wm_provider=wm_provider,
                    pipe_provider_target=pipe_provider_target,
                    num_inference_steps_target=args.num_inference_steps_target,
                    step=-1,
                    message_bits_str_initial=message_bits_str_initial,
                    do_psnr=False,
                    do_ssim=False,
                    do_msssim=False,
                    do_lpips=False,
                    device=str(device),
                )
                before_successful = check_if_detection_successful(
                    wm_type=wm_type,
                    threshold=detection_threshold,
                    value=detection_value(before, wm_type),
                )
                before_metric_value = detection_value(before, wm_type)
                before_compact = compact_results(before)
            else:
                before_successful = None
                before_metric_value = None
                before_compact = {}

            row = {
                "dataset_name": args.dataset_name,
                "run_id": run_id,
                "prompt_id": prompt_row["prompt_id"],
                "prompt": prompt_row["prompt"],
                "source": prompt_row["source"],
                "wm_type": wm_type,
                "wm_name": PAPER_WM_NAMES[wm_type],
                "target_model": args.modelid_target,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                "detection_threshold": detection_threshold,
                "detection_metric": METRIC_MAP[wm_type],
                "before_detection_successful": bool_for_json(before_successful),
                "before_detection_metric_value": bool_for_json(before_metric_value),
                "watermarked_image_path": str(image_path),
            }
            row.update({f"before_{key}": bool_for_json(value) for key, value in before_compact.items()})
            append_row(metadata_csv, row)
            rows_written += 1

            print(
                f"[{wm_type}] run_id={run_id} before={before_successful} "
                f"metric={before_metric_value}",
                flush=True,
            )
            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/done")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        try:
            pipe_provider_target.stash_pipe()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    summary = summarize_metadata(metadata_csv)
    summary.update({
        "wm_type": wm_type,
        "wm_name": PAPER_WM_NAMES[wm_type],
        "rows_written_this_run": rows_written,
        "target_images_requested": len(prompt_rows),
        "paper_setting_note": "Watermarked image generation only; RAVEN/removal not run in this step.",
    })
    save_json(summary_json, summary)
    print(f"[{wm_type}] summary: {summary}", flush=True)
    return summary


def main() -> int:
    first = base_parser().parse_known_args()[0]
    dataset_dir = Path(first.output_dir) / first.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_timestamp()
    gpu_record = configure_gpu(first.gpu, first.device, dataset_dir, require_free_gpu=first.require_free_gpu)

    status = "failed"
    summaries: Dict[str, Any] = {}
    args_dict: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        args = parse_args()
        args_dict = vars(args).copy()
        os.environ.setdefault("TQDM_DISABLE", "1")

        from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
        import torch

        limit_cpu_threads(1)
        guard = CpuMemoryGuard(
            min_available_gib=args.min_cpu_mem_gb,
            warn_available_gib=args.warn_cpu_mem_gb,
            max_process_rss_gib=args.max_process_ram_gb,
        )
        guard.check("startup")

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        prompt_rows = load_prompts(Path(args.prompts_csv), args.start_index, args.num_pairs)

        settings = {
            "paper_dataset_size_used_for_detection": args.num_pairs,
            "model_setup": {
                "model_id_requested_by_paper": "stabilityai/stable-diffusion-2-1-base",
                "model_id_used": args.modelid_target,
                "resolution": args.resolution,
                "cfg_scale": args.guidance_scale_target,
                "steps": args.num_inference_steps_target,
                "scheduler": args.scheduler_target,
            },
            "watermark_methods": args.wm_types,
            "paper_watermark_methods_not_available_in_eval_bench_wm": [
                "DwtDct", "DwtDctSvd", "RivaGAN", "StegaStamp", "StableSignature",
                "Zodiac", "ROBIN", "TrustMark", "VINE",
            ],
            "stage": "generate_watermarked_images_only",
        }
        save_json(dataset_dir / "paper_settings.json", settings)

        for wm_type in args.wm_types:
            summaries[wm_type] = run_method(args, dataset_dir, wm_type, prompt_rows, guard, device)
            guard.check(f"{args.dataset_name}/{wm_type}/method_complete")
        status = "completed"
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        finished_at = utc_timestamp()
        gpu_record = finalize_gpu_logging(dataset_dir, gpu_record)
        extra = {"method_summaries": summaries}
        if error:
            extra["error"] = error
        write_experiment_records(dataset_dir, args_dict or vars(first), gpu_record, started_at, finished_at, status, extra)


if __name__ == "__main__":
    raise SystemExit(main())
