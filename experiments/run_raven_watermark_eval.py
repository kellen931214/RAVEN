#!/usr/bin/env python3
"""Run eval_bench_wm watermark generation, RAVEN attack, and detector eval.

This script intentionally runs only watermark methods that are both mentioned in
RAVEN Section 5.1 and implemented in eval_bench_wm: GS, TR, RID, HSTR, HSQR.
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
from typing import Any, Dict, Iterable, List, Optional

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
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
    "GS": "bit_accuracy",
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
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--prompts_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(WORKSPACE / "outputs" / "raven_watermark_eval"))
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
        description="RAVEN-only watermark benchmark using eval_bench_wm providers",
        parents=parent_parsers,
        conflict_handler="resolve",
    )
    parser.add_argument("--wm_types", nargs="+", default=PAPER_WM_METHODS_IN_BENCH, choices=PAPER_WM_METHODS_IN_BENCH)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_images", type=str_to_bool, default=True)

    parser.add_argument("--modelid_target", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--raven_model_id", type=str, default=None)
    parser.add_argument("--raven_dtype", type=str, default="fp16")
    parser.add_argument("--raven_steps", type=int, default=50)
    parser.add_argument("--raven_strength", type=float, default=0.15)
    parser.add_argument("--raven_guidance_scale", type=float, default=2.5)
    parser.add_argument("--shift_min", type=int, default=24)
    parser.add_argument("--shift_max", type=int, default=32)
    parser.add_argument("--shift_sign", type=str, default="random", choices=["positive", "negative", "random"])
    parser.add_argument("--shift_space", type=str, default="image_pixels", choices=["image_pixels", "latent_pixels"])
    parser.add_argument("--padding_mode", type=str, default="reflection", choices=["reflection", "zeros", "replicate", "circular"])
    parser.add_argument("--view_guided_attention", type=str_to_bool, default=True)
    parser.add_argument("--color_transfer", type=str_to_bool, default=True)
    parser.add_argument("--debug", type=str_to_bool, default=False)
    return parser.parse_args()


def load_prompts(path: Path, start_index: int, num_pairs: int) -> List[str]:
    prompts: List[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prompt CSV has no header: {path}")
        prompt_field = "prompt" if "prompt" in reader.fieldnames else reader.fieldnames[0]
        for row_id, row in enumerate(reader):
            if row_id < start_index:
                continue
            prompt = (row.get(prompt_field) or "").strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= num_pairs:
                break
    if len(prompts) < num_pairs:
        raise ValueError(f"Only found {len(prompts)} prompts in {path}; requested {num_pairs}")
    return prompts


def compact_results(results: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bit_accuracy",
        "p_value",
        "value",
        "l1_dist",
        "detection_success",
        "log_message",
        "prc_threshold",
        "psnr",
        "ssim",
        "msssim",
        "lpips",
        "message_bits_str_initial",
        "message_bits_str_recovered",
    ]
    return {key: results.get(key) for key in keys}


def detection_value(results: Dict[str, Any], wm_type: str) -> Any:
    return results[METRIC_MAP[wm_type]]


def bool_for_json(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def existing_run_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                done.add(int(row["run_id"]))
            except Exception:
                continue
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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_csv(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"completed": 0}
    rows = []
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

    before = [parse_bool(row.get("before_detection_successful", "")) for row in rows]
    after = [parse_bool(row.get("after_detection_successful", "")) for row in rows]
    before_valid = [x for x in before if x is not None]
    after_valid = [x for x in after if x is not None]
    attacked_only = [b and not a for b, a in zip(before, after) if b is not None and a is not None]
    return {
        "completed": len(rows),
        "before_detection_rate": (sum(before_valid) / len(before_valid)) if before_valid else None,
        "after_detection_rate": (sum(after_valid) / len(after_valid)) if after_valid else None,
        "raven_suppression_rate_on_detected": (sum(attacked_only) / max(1, sum(1 for x in before if x is True))) if before_valid else None,
    }


def move_pipe_to_cpu(pipe_obj: Any) -> None:
    if pipe_obj is None:
        return
    try:
        pipe_obj.pipe = pipe_obj.pipe.to("cpu")
    except Exception:
        pass


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


def run_method(args: argparse.Namespace, dataset_dir: Path, wm_type: str, prompts: List[str], guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from raven.pipeline_raven import RavenPipeline
    from utils.imprint_utils import validate
    from utils.pipe import pipe_utils
    from utils.utils import check_if_detection_successful, get_detection_threshold, seed_everything, set_random_seed
    from utils.wm.wm_utils import WmProviders

    wm_provider_cls = WmProviders[wm_type].value
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    method_dir = dataset_dir / wm_type
    images_root = method_dir / "images"
    results_csv = method_dir / "results.csv"
    summary_json = method_dir / "summary.json"
    method_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    completed = existing_run_ids(results_csv)
    detection_threshold = get_detection_threshold(wm_type, args.modelid_target)
    set_random_seed(args.seed)

    print(f"[{wm_type}] loading target watermark pipeline: {args.modelid_target}", flush=True)
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
    )

    wm_provider = wm_provider_cls(
        latent_shape=pipe_provider_target.get_latent_shape(),
        dtype=pipe_provider_target.get_dtype(),
        device=device,
        **vars(args),
    )
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]
    message_bits_str_initial = (
        wm_initial_results["message_bits_str_list"][0]
        if "message_bits_str_list" in wm_initial_results
        else None
    )

    raven_model_id = args.raven_model_id or args.modelid_target
    print(f"[{wm_type}] loading RAVEN pipeline: {raven_model_id}", flush=True)
    raven_pipe = RavenPipeline(model_id=raven_model_id, device=str(device), dtype=args.raven_dtype)
    raven_pipe.pipe = raven_pipe.pipe.to("cpu")
    torch.cuda.empty_cache()

    rows_written = 0
    try:
        for local_idx, prompt in enumerate(prompts):
            run_id = args.start_index + local_idx
            if run_id in completed:
                if (run_id + 1) % 25 == 0 or run_id == args.start_index:
                    print(f"[{wm_type}] skip existing run_id={run_id}", flush=True)
                continue

            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/before")
            if wm_type in ["PRC"]:
                seed_everything(args.seed)

            print(f"[{wm_type}] run {local_idx + 1}/{len(prompts)} global_run_id={run_id}", flush=True)
            watermarked_image = generate_watermarked(pipe_provider_target, wm_provider, prompt, wm_zT, args)

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

            pipe_provider_target.stash_pipe()
            torch.cuda.empty_cache()
            gc.collect()

            item_dir = images_root / f"{run_id:06d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            if args.save_images:
                watermarked_image.save(item_dir / "watermarked.png")

            raven_pipe.pipe = raven_pipe.pipe.to(device)
            attacked_image = raven_pipe.run(
                input_image=watermarked_image,
                output_dir=item_dir,
                steps=args.raven_steps,
                strength=args.raven_strength,
                guidance_scale=args.raven_guidance_scale,
                shift_min=args.shift_min,
                shift_max=args.shift_max,
                shift_sign=args.shift_sign,
                shift_space=args.shift_space,
                padding_mode=args.padding_mode,
                view_guided_attention=args.view_guided_attention,
                color_transfer=args.color_transfer,
                seed=args.seed + run_id,
                prompt="",
                negative_prompt="",
                debug=args.debug,
            )
            raven_pipe.pipe = raven_pipe.pipe.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/after_raven")
            after = validate(
                out_dir=str(method_dir),
                image_to_verify_PIL=attacked_image,
                original_PIL=watermarked_image,
                wm_provider=wm_provider,
                pipe_provider_target=pipe_provider_target,
                num_inference_steps_target=args.num_inference_steps_target,
                step=run_id,
                message_bits_str_initial=message_bits_str_initial,
                do_psnr=True,
                do_ssim=True,
                do_msssim=True,
                do_lpips=False,
                device=str(device),
            )
            after_successful = check_if_detection_successful(
                wm_type=wm_type,
                threshold=detection_threshold,
                value=detection_value(after, wm_type),
            )
            pipe_provider_target.stash_pipe()
            torch.cuda.empty_cache()
            gc.collect()

            before_compact = compact_results(before)
            after_compact = compact_results(after)
            row = {
                "dataset_name": args.dataset_name,
                "run_id": run_id,
                "prompt": prompt,
                "wm_type": wm_type,
                "wm_name": PAPER_WM_NAMES[wm_type],
                "target_model": args.modelid_target,
                "scheduler_target": args.scheduler_target,
                "num_inference_steps_target": args.num_inference_steps_target,
                "guidance_scale_target": args.guidance_scale_target,
                "resolution": args.resolution,
                "attack_method": "RAVEN",
                "raven_model_id": raven_model_id,
                "raven_steps": args.raven_steps,
                "raven_strength": args.raven_strength,
                "raven_guidance_scale": args.raven_guidance_scale,
                "shift_min": args.shift_min,
                "shift_max": args.shift_max,
                "shift_sign": args.shift_sign,
                "shift_space": args.shift_space,
                "padding_mode": args.padding_mode,
                "view_guided_attention": args.view_guided_attention,
                "color_transfer": args.color_transfer,
                "detection_threshold": detection_threshold,
                "detection_metric": METRIC_MAP[wm_type],
                "before_detection_successful": bool_for_json(before_successful),
                "after_detection_successful": bool_for_json(after_successful),
                "before_detection_metric_value": bool_for_json(detection_value(before, wm_type)),
                "after_detection_metric_value": bool_for_json(detection_value(after, wm_type)),
                "watermarked_image_path": str(item_dir / "watermarked.png") if args.save_images else "",
                "raven_output_path": str(item_dir / ("final_color_corrected.png" if args.color_transfer else "final.png")),
            }
            row.update({f"before_{key}": bool_for_json(value) for key, value in before_compact.items()})
            row.update({f"after_{key}": bool_for_json(value) for key, value in after_compact.items()})
            append_row(results_csv, row)
            rows_written += 1

            print(
                f"[{wm_type}] run_id={run_id} before={before_successful} after={after_successful} "
                f"before_metric={float(detection_value(before, wm_type)):.6g} "
                f"after_metric={float(detection_value(after, wm_type)):.6g}",
                flush=True,
            )

            guard.check(f"{args.dataset_name}/{wm_type}/run_id={run_id}/done")
    finally:
        try:
            raven_pipe.pipe = raven_pipe.pipe.to("cpu")
        except Exception:
            pass
        try:
            pipe_provider_target.stash_pipe()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    summary = summarize_csv(results_csv)
    summary.update({
        "wm_type": wm_type,
        "wm_name": PAPER_WM_NAMES[wm_type],
        "rows_written_this_run": rows_written,
        "results_csv": str(results_csv),
        "paper_setting_note": "Detector threshold comes from eval_bench_wm; RAVEN settings match Section 5.1 for steps/CFG/strength/shift.",
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

        prompts = load_prompts(Path(args.prompts_csv), args.start_index, args.num_pairs)
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
            "raven": {
                "cfg_scale": args.raven_guidance_scale,
                "steps": args.raven_steps,
                "strength": args.raven_strength,
                "prompt": "",
                "shift_min": args.shift_min,
                "shift_max": args.shift_max,
                "shift_sign": args.shift_sign,
                "shift_space": args.shift_space,
                "padding_mode": args.padding_mode,
                "view_guided_attention": args.view_guided_attention,
                "color_transfer": args.color_transfer,
            },
            "paper_watermark_methods_available_in_eval_bench_wm": args.wm_types,
            "paper_watermark_methods_not_available_in_eval_bench_wm": [
                "DwtDct", "DwtDctSvd", "RivaGAN", "StegaStamp", "StableSignature",
                "Zodiac", "ROBIN", "TrustMark", "VINE",
            ],
        }
        save_json(dataset_dir / "paper_settings.json", settings)

        for wm_type in args.wm_types:
            summaries[wm_type] = run_method(args, dataset_dir, wm_type, prompts, guard, device)
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
