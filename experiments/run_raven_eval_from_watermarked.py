#!/usr/bin/env python3
"""Evaluate RAVEN on already-generated watermarked images.

Input comes from data/watermarked/<dataset>/<WM_TYPE>/metadata.csv. The script
runs RAVEN only, then evaluates watermark detection plus quality metrics. It does
not generate new watermarked images and does not run other removal attacks.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import gc
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
RESULT_FIELDS = [
    "dataset_name", "run_id", "prompt_id", "prompt", "source", "wm_type", "wm_name",
    "target_model", "threshold_mode", "threshold_source", "detection_threshold", "detection_metric",
    "watermarked_image_path", "raven_output_path", "debug_info_path",
    "raven_model_id", "raven_steps", "raven_strength", "raven_guidance_scale",
    "shift_min", "shift_max", "shift_sign", "shift_space", "padding_mode",
    "view_guided_attention", "color_transfer", "dx", "dy",
    "before_detection_successful", "after_detection_successful",
    "before_detection_metric_value", "after_detection_metric_value",
    "before_bit_accuracy", "after_bit_accuracy", "before_p_value", "after_p_value",
    "before_l1_dist", "after_l1_dist", "before_value", "after_value",
    "overlap_psnr", "overlap_ssim", "clip_score", "clip_model", "error",
]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    from utils.pipe import pipe_utils
    from utils.wm.gs_provider import parser as gs_parser
    from utils.wm.hstr_provider import parser as hstr_parser
    from utils.wm.hsqr_provider import parser as hsqr_parser
    from utils.wm.ringid_provider import parser as ringid_parser
    from utils.wm.tr_provider import parser as tr_parser

    parents = [gs_parser, tr_parser, ringid_parser, hstr_parser, hsqr_parser]
    parser = argparse.ArgumentParser(
        description="Run RAVEN eval from existing watermarked images",
        parents=parents,
        conflict_handler="resolve",
    )
    parser.add_argument("--dataset_name", type=str, default="mscoco")
    parser.add_argument("--watermarked_dir", type=str, default=str(WORKSPACE / "data" / "watermarked" / "mscoco"))
    parser.add_argument("--output_dir", type=str, default=str(WORKSPACE / "outputs" / "raven_eval"))
    parser.add_argument("--wm_types", nargs="+", default=PAPER_WM_METHODS_IN_BENCH, choices=PAPER_WM_METHODS_IN_BENCH)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_id", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--threshold_mode", choices=["eval_bench_wm", "paper_1pct"], default="eval_bench_wm")

    parser.add_argument("--raven_model_id", type=str, default=None)
    parser.add_argument("--raven_dtype", type=str, default="fp16")
    parser.add_argument("--raven_steps", type=int, default=50)
    parser.add_argument("--raven_strength", type=float, default=0.15)
    parser.add_argument("--raven_guidance_scale", type=float, default=2.5)
    parser.add_argument("--shift_min", type=int, default=24)
    parser.add_argument("--shift_max", type=int, default=32)
    parser.add_argument("--shift_sign", choices=["positive", "negative", "random"], default="random")
    parser.add_argument("--shift_space", choices=["image_pixels", "latent_pixels"], default="image_pixels")
    parser.add_argument("--padding_mode", choices=["reflection", "zeros", "replicate", "circular"], default="reflection")
    parser.add_argument("--view_guided_attention", type=str_to_bool, default=True)
    parser.add_argument("--color_transfer", type=str_to_bool, default=True)

    parser.add_argument("--compute_clip", type=str_to_bool, default=True)
    parser.add_argument("--clip_model", type=str, default="ViT-g-14")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s12b_b42k")
    parser.add_argument("--compute_fid", type=str_to_bool, default=True)
    parser.add_argument("--fid_batch_size", type=int, default=16)
    parser.add_argument("--fid_num_workers", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--require_free_gpu", type=str_to_bool, default=True)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=64.0)
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=96.0)
    parser.add_argument("--max_process_ram_gb", type=float, default=16.0)
    return parser.parse_args()


def bool_for_json(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_bool(value: Any) -> Optional[bool]:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def parse_float(value: Any) -> Optional[float]:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def read_existing_completed(csv_path: Path) -> set[int]:
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
            output_path = row.get("raven_output_path")
            if output_path and Path(output_path).exists() and not row.get("error"):
                done.add(run_id)
    return done


def acquire_dataset_lock(dataset_dir: Path) -> Path:
    lock_path = dataset_dir / ".eval.lock"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip() if lock_path.exists() else "unknown"
        raise RuntimeError(f"Eval lock already exists for {dataset_dir}: {lock_path} owner={owner}")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.write(f"dataset_dir={dataset_dir}\n")
        handle.write(f"started_at={utc_timestamp()}\n")
    def cleanup_lock() -> None:
        try:
            if lock_path.exists():
                content = lock_path.read_text(encoding="utf-8", errors="replace")
                if f"pid={os.getpid()}" in content:
                    lock_path.unlink()
        except Exception:
            pass
    atexit.register(cleanup_lock)
    return lock_path


def load_metadata_rows(watermarked_dir: Path, wm_type: str, start_index: int, num_pairs: int) -> List[Dict[str, str]]:
    metadata_csv = watermarked_dir / wm_type / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV for {wm_type}: {metadata_csv}")
    rows: List[Dict[str, str]] = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                run_id = int(row["run_id"])
            except Exception:
                continue
            if run_id < start_index:
                continue
            image_path = Path(row.get("watermarked_image_path", ""))
            if not image_path.exists():
                continue
            rows.append(row)
            if len(rows) >= num_pairs:
                break
    if not rows:
        raise ValueError(f"No usable watermarked rows found for {wm_type} in {metadata_csv}")
    return rows


def detection_value(results: Dict[str, Any], wm_type: str) -> Any:
    return results[METRIC_MAP[wm_type]]


def compact_after(results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "after_bit_accuracy": results.get("bit_accuracy"),
        "after_p_value": results.get("p_value"),
        "after_l1_dist": results.get("l1_dist"),
        "after_value": results.get("value"),
        "after_detection_metric_value": detection_value(results, results["wm_type"]),
    }


def load_rgb(path: Path):
    import numpy as np

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def crop_overlap(a, b, dx: int, dy: int):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    x0_a = max(0, dx)
    x1_a = w + min(0, dx)
    y0_a = max(0, dy)
    y1_a = h + min(0, dy)
    x0_b = max(0, -dx)
    x1_b = w - max(0, dx)
    y0_b = max(0, -dy)
    y1_b = h - max(0, dy)
    return a[y0_a:y1_a, x0_a:x1_a], b[y0_b:y1_b, x0_b:x1_b]


def psnr(a, b) -> float:
    import numpy as np

    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a, b) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def overlap_quality(watermarked_path: Path, raven_output_path: Path, debug_info_path: Path) -> Tuple[int, int, float, float]:
    dx = dy = 0
    if debug_info_path.exists():
        info = json.loads(debug_info_path.read_text(encoding="utf-8"))
        dx = int(info.get("dx", 0))
        dy = int(info.get("dy", 0))
    original = load_rgb(watermarked_path)
    attacked = load_rgb(raven_output_path)
    original_crop, attacked_crop = crop_overlap(original, attacked, dx, dy)
    return dx, dy, psnr(original_crop, attacked_crop), ssim(original_crop, attacked_crop)


class ClipScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip
        import torch

        self.torch = torch
        self.open_clip = open_clip
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def score(self, image_path: Path, prompt: str) -> float:
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        text = self.tokenizer([prompt]).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return float((image_features @ text_features.T).item())


def prepare_fid_dirs(method_dir: Path, rows: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    watermarked_fid = method_dir / "fid_inputs" / "watermarked"
    raven_fid = method_dir / "fid_inputs" / "raven"
    watermarked_fid.mkdir(parents=True, exist_ok=True)
    raven_fid.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("error"):
            continue
        run_id = int(row["run_id"])
        pairs = [
            (Path(row["watermarked_image_path"]), watermarked_fid / f"{run_id:06d}.png"),
            (Path(row["raven_output_path"]), raven_fid / f"{run_id:06d}.png"),
        ]
        for src, dst in pairs:
            if not src.exists():
                continue
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)
    return watermarked_fid, raven_fid


def compute_fid(method_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace, device: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        from cleanfid import fid

        watermarked_fid, raven_fid = prepare_fid_dirs(method_dir, rows)
        return float(fid.compute_fid(
            fdir1=str(watermarked_fid),
            fdir2=str(raven_fid),
            mode="clean",
            num_workers=args.fid_num_workers,
            batch_size=args.fid_batch_size,
            device=device,
            verbose=False,
            use_dataparallel=False,
        )), None
    except Exception as exc:
        return None, repr(exc)


def summarize_rows(rows: List[Dict[str, Any]], fid_value: Optional[float], fid_error: Optional[str], args: argparse.Namespace, wm_type: str) -> Dict[str, Any]:
    def mean(values: Iterable[Any]) -> Optional[float]:
        parsed = [parse_float(v) for v in values]
        parsed = [v for v in parsed if v is not None and not math.isnan(v)]
        return float(sum(parsed) / len(parsed)) if parsed else None

    before_success = [parse_bool(row.get("before_detection_successful")) for row in rows if not row.get("error")]
    after_success = [parse_bool(row.get("after_detection_successful")) for row in rows if not row.get("error")]
    before_success = [value for value in before_success if value is not None]
    after_success = [value for value in after_success if value is not None]
    detected_then_removed = [
        parse_bool(row.get("before_detection_successful")) is True and parse_bool(row.get("after_detection_successful")) is False
        for row in rows if not row.get("error")
    ]
    return {
        "wm_type": wm_type,
        "wm_name": PAPER_WM_NAMES[wm_type],
        "completed": len([row for row in rows if not row.get("error")]),
        "errors": len([row for row in rows if row.get("error")]),
        "threshold_mode": args.threshold_mode,
        "threshold_source": "eval_bench_wm_current_thresholds" if args.threshold_mode == "eval_bench_wm" else "paper_1pct_interface_not_recalibrated",
        "before_detection_rate": (sum(before_success) / len(before_success)) if before_success else None,
        "after_detection_rate": (sum(after_success) / len(after_success)) if after_success else None,
        "raven_suppression_rate_on_detected": (sum(detected_then_removed) / max(1, sum(1 for row in rows if parse_bool(row.get("before_detection_successful")) is True))),
        "mean_before_bit_accuracy": mean(row.get("before_bit_accuracy") for row in rows),
        "mean_after_bit_accuracy": mean(row.get("after_bit_accuracy") for row in rows),
        "mean_overlap_psnr": mean(row.get("overlap_psnr") for row in rows),
        "mean_overlap_ssim": mean(row.get("overlap_ssim") for row in rows),
        "mean_clip_score": mean(row.get("clip_score") for row in rows),
        "fid_watermarked_vs_raven": fid_value,
        "fid_error": fid_error,
        "clip_model": f"{args.clip_model}:{args.clip_pretrained}" if args.compute_clip else "disabled",
    }


def make_provider(args: argparse.Namespace, wm_type: str, pipe_provider_target: Any, device: Any):
    from utils.wm.wm_utils import WmProviders

    wm_provider_cls = WmProviders[wm_type].value
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)
    provider_kwargs = vars(args).copy()
    for reserved_key in ("latent_shape", "dtype", "device"):
        provider_kwargs.pop(reserved_key, None)
    provider = wm_provider_cls(
        latent_shape=pipe_provider_target.get_latent_shape(),
        dtype=pipe_provider_target.get_dtype(),
        device=device,
        **provider_kwargs,
    )
    initial = provider.get_wm_latents()
    message_bits = initial["message_bits_str_list"][0] if "message_bits_str_list" in initial else None
    return provider, message_bits


def run_method(args: argparse.Namespace, wm_type: str, dataset_dir: Path, guard: Any, device: Any) -> Dict[str, Any]:
    import torch
    from raven.pipeline_raven import RavenPipeline
    from utils.imprint_utils import validate
    from utils.pipe import pipe_utils
    from utils.utils import check_if_detection_successful, get_detection_threshold, set_random_seed

    method_dir = dataset_dir / wm_type
    method_dir.mkdir(parents=True, exist_ok=True)
    results_csv = method_dir / "results.csv"
    summary_json = method_dir / "summary.json"
    completed = read_existing_completed(results_csv)
    metadata_rows = load_metadata_rows(Path(args.watermarked_dir), wm_type, args.start_index, args.num_pairs)
    detection_threshold = get_detection_threshold(wm_type, args.model_id)
    threshold_source = "eval_bench_wm_current_thresholds" if args.threshold_mode == "eval_bench_wm" else "paper_1pct_interface_not_recalibrated"

    set_random_seed(args.seed)
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
    )
    wm_provider, message_bits = make_provider(args, wm_type, pipe_provider_target, device)
    pipe_provider_target.stash_pipe()
    torch.cuda.empty_cache()
    gc.collect()

    raven_model_id = args.raven_model_id or args.model_id
    # Keep the fp16 RAVEN pipeline on the accelerator. Moving fp16 diffusers
    # pipelines to CPU is unsupported and pushes process RSS near the guard limit.
    raven_pipe = RavenPipeline(model_id=raven_model_id, device=str(device), dtype=args.raven_dtype)
    torch.cuda.empty_cache()
    gc.collect()

    clip_scorer = None
    rows_for_summary: List[Dict[str, Any]] = []
    if results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            rows_for_summary.extend(list(csv.DictReader(handle)))

    try:
        for local_idx, meta in enumerate(metadata_rows):
            run_id = int(meta["run_id"])
            output_item_dir = method_dir / f"{run_id:06d}"
            raven_output_path = output_item_dir / ("final_color_corrected.png" if args.color_transfer else "final.png")
            if run_id in completed and raven_output_path.exists():
                print(f"[{wm_type}] skip existing run_id={run_id}", flush=True)
                continue

            guard.check(f"eval/{args.dataset_name}/{wm_type}/run_id={run_id}/before")
            watermarked_path = Path(meta["watermarked_image_path"])
            watermarked_image = Image.open(watermarked_path).convert("RGB")
            output_item_dir.mkdir(parents=True, exist_ok=True)

            attacked_image = raven_pipe.run(
                input_image=watermarked_image,
                output_dir=output_item_dir,
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
                debug=False,
            )
            torch.cuda.empty_cache()
            gc.collect()

            after = validate(
                out_dir=str(method_dir),
                image_to_verify_PIL=attacked_image,
                original_PIL=watermarked_image,
                wm_provider=wm_provider,
                pipe_provider_target=pipe_provider_target,
                num_inference_steps_target=args.num_inference_steps_target,
                step=run_id,
                message_bits_str_initial=message_bits,
                do_psnr=False,
                do_ssim=False,
                do_msssim=False,
                do_lpips=False,
                device=str(device),
            )
            after["wm_type"] = wm_type
            after_success = check_if_detection_successful(wm_type, detection_threshold, detection_value(after, wm_type))
            pipe_provider_target.stash_pipe()
            torch.cuda.empty_cache()
            gc.collect()

            debug_info_path = output_item_dir / "debug_info.json"
            dx, dy, overlap_psnr, overlap_ssim = overlap_quality(watermarked_path, raven_output_path, debug_info_path)
            clip_score = ""
            if args.compute_clip:
                if clip_scorer is None:
                    clip_scorer = ClipScorer(args.clip_model, args.clip_pretrained, str(device))
                clip_score = clip_scorer.score(raven_output_path, meta.get("prompt", ""))

            row = {
                "dataset_name": args.dataset_name,
                "run_id": run_id,
                "prompt_id": meta.get("prompt_id", ""),
                "prompt": meta.get("prompt", ""),
                "source": meta.get("source", ""),
                "wm_type": wm_type,
                "wm_name": PAPER_WM_NAMES[wm_type],
                "target_model": args.model_id,
                "threshold_mode": args.threshold_mode,
                "threshold_source": threshold_source,
                "detection_threshold": detection_threshold,
                "detection_metric": METRIC_MAP[wm_type],
                "watermarked_image_path": str(watermarked_path),
                "raven_output_path": str(raven_output_path),
                "debug_info_path": str(debug_info_path),
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
                "dx": dx,
                "dy": dy,
                "before_detection_successful": meta.get("before_detection_successful", ""),
                "after_detection_successful": bool_for_json(after_success),
                "before_detection_metric_value": meta.get("before_detection_metric_value", ""),
                "after_detection_metric_value": bool_for_json(detection_value(after, wm_type)),
                "before_bit_accuracy": meta.get("before_bit_accuracy", ""),
                "after_bit_accuracy": bool_for_json(after.get("bit_accuracy")),
                "before_p_value": meta.get("before_p_value", ""),
                "after_p_value": bool_for_json(after.get("p_value")),
                "before_l1_dist": meta.get("before_l1_dist", ""),
                "after_l1_dist": bool_for_json(after.get("l1_dist")),
                "before_value": meta.get("before_value", ""),
                "after_value": bool_for_json(after.get("value")),
                "overlap_psnr": overlap_psnr,
                "overlap_ssim": overlap_ssim,
                "clip_score": clip_score,
                "clip_model": f"{args.clip_model}:{args.clip_pretrained}" if args.compute_clip else "disabled",
                "error": "",
            }
            append_row(results_csv, row)
            rows_for_summary.append(row)
            print(f"[{wm_type}] run_id={run_id} after={after_success} overlap_psnr={overlap_psnr:.4f} overlap_ssim={overlap_ssim:.4f}", flush=True)
            try:
                watermarked_image.close()
            except Exception:
                pass
            try:
                attacked_image.close()
            except Exception:
                pass
            del watermarked_image, attacked_image, after
            torch.cuda.empty_cache()
            gc.collect()
            guard.check(f"eval/{args.dataset_name}/{wm_type}/run_id={run_id}/done")
    finally:
        try:
            pipe_provider_target.stash_pipe()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    fid_value = fid_error = None
    if args.compute_fid:
        fid_value, fid_error = compute_fid(method_dir, rows_for_summary, args, str(device))
    summary = summarize_rows(rows_for_summary, fid_value, fid_error, args, wm_type)
    summary["results_csv"] = str(results_csv)
    save_json(summary_json, summary)
    return summary


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.output_dir) / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_timestamp()
    gpu_record = configure_gpu(args.gpu, args.device, dataset_dir, require_free_gpu=args.require_free_gpu)
    lock_path = acquire_dataset_lock(dataset_dir)
    print(f"Acquired eval lock: {lock_path}", flush=True)

    status = "failed"
    summaries: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        os.environ.setdefault("TQDM_DISABLE", "1")
        from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
        import torch

        limit_cpu_threads(1)
        guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
        guard.check("eval startup")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)

        for wm_type in args.wm_types:
            summaries[wm_type] = run_method(args, wm_type, dataset_dir, guard, device)
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
        write_experiment_records(dataset_dir, vars(args), gpu_record, started_at, finished_at, status, extra)


if __name__ == "__main__":
    raise SystemExit(main())
