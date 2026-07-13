#!/usr/bin/env python3
"""Wait for data/GPU availability, then run RAVEN eval sequentially per dataset."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE = Path("/workspace")
RAVEN_ROOT = WORKSPACE / "raven_repro"
sys.path.insert(0, str(RAVEN_ROOT))

from raven.gpu_utils import query_gpu_status

LOG_DIR = RAVEN_ROOT / "logs"
METHODS = ["GS", "TR", "RID", "HSTR", "HSQR"]
DATASET_TARGETS = {
    "mscoco": 1000,
    "diffusiondb": 1001,
    "sd_prompts": 8192,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(path: Path, message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def is_compatible(gpu) -> bool:
    return "blackwell" not in gpu.name.lower()


def idle_gpu_ids(wait_log: Path) -> List[str]:
    status = query_gpu_status()
    if status.get("error"):
        log(wait_log, f"WARNING nvidia-smi failed: {status['error']}")
        return []
    gpus = [gpu for gpu in status.get("gpus") or [] if is_compatible(gpu)]
    idle = [gpu for gpu in gpus if gpu.is_idle]
    idle = sorted(idle, key=lambda gpu: (gpu.memory_used_mib, gpu.utilization_gpu_percent, int(gpu.index)))
    return [str(gpu.index) for gpu in idle]


def count_ready_rows(dataset: str, method: str) -> int:
    metadata_csv = WORKSPACE / "data" / "watermarked" / dataset / method / "metadata.csv"
    if not metadata_csv.exists():
        return 0
    count = 0
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row.get("watermarked_image_path", ""))
            if image_path.exists():
                count += 1
    return count


def dataset_ready(dataset: str, target: int, wait_log: Path) -> bool:
    counts: Dict[str, int] = {method: count_ready_rows(dataset, method) for method in METHODS}
    ready = all(count >= target for count in counts.values())
    counts_text = " ".join(f"{method}={counts[method]}/{target}" for method in METHODS)
    log(wait_log, f"data_status dataset={dataset} ready={ready} {counts_text}")
    return ready


def build_eval_cmd(dataset: str, target: int, gpu_id: str) -> List[str]:
    return [
        "ionice", "-c2", "-n7", "nice", "-n", "10",
        sys.executable, str(WORKSPACE / "experiments" / "run_raven_eval_from_watermarked.py"),
        "--dataset_name", dataset,
        "--watermarked_dir", str(WORKSPACE / "data" / "watermarked" / dataset),
        "--output_dir", str(WORKSPACE / "outputs" / "raven_eval"),
        "--wm_types", *METHODS,
        "--num_pairs", str(target),
        "--model_id", "RedbeardNZ/stable-diffusion-2-1-base",
        "--scheduler_target", "DDIM",
        "--num_inference_steps_target", "50",
        "--resolution", "512",
        "--threshold_mode", "eval_bench_wm",
        "--raven_steps", "50",
        "--raven_strength", "0.15",
        "--raven_guidance_scale", "2.5",
        "--shift_min", "24",
        "--shift_max", "32",
        "--shift_sign", "random",
        "--shift_space", "image_pixels",
        "--padding_mode", "reflection",
        "--view_guided_attention", "true",
        "--color_transfer", "true",
        "--compute_clip", "true",
        "--compute_fid", "true",
        "--device", "cuda",
        "--gpu", gpu_id,
        "--require_free_gpu", "false",
        "--min_cpu_mem_gb", "64",
        "--warn_cpu_mem_gb", "96",
        "--max_process_ram_gb", "16",
    ]


def run_table(dataset: str, env: Dict[str, str], eval_log: Path, wait_log: Path) -> int:
    eval_dir = WORKSPACE / "outputs" / "raven_eval" / dataset
    md_path = eval_dir / "eval_summary_table.md"
    csv_path = eval_dir / "eval_summary_table.csv"
    cmd = [
        sys.executable,
        str(WORKSPACE / "experiments" / "build_raven_eval_table.py"),
        "--eval_dir", str(eval_dir),
    ]
    with eval_log.open("a", encoding="utf-8") as handle:
        rc = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(WORKSPACE), env=env)
        if rc == 0 and md_path.exists():
            handle.write(f"\n[{now()}] summary_table dataset={dataset}\n")
            handle.write(md_path.read_text(encoding="utf-8"))
            handle.write(f"[{now()}] summary_table_files md={md_path} csv={csv_path}\n")
            handle.flush()
    log(wait_log, f"table_finished dataset={dataset} rc={rc} md={md_path} csv={csv_path}")
    return rc


def run_dataset(dataset: str, target: int, args: argparse.Namespace, env: Dict[str, str], wait_log: Path) -> int:
    eval_log = LOG_DIR / f"raven_eval_{dataset}.log"
    pid_file = LOG_DIR / f"raven_eval_{dataset}.pid"

    while not dataset_ready(dataset, target, wait_log):
        log(wait_log, f"waiting_for_watermarked_data dataset={dataset} poll_seconds={args.poll_seconds}")
        time.sleep(args.poll_seconds)

    while True:
        ids = idle_gpu_ids(wait_log)
        if ids:
            gpu_id = ids[0]
            log(wait_log, f"idle_gpu_found dataset={dataset} gpu={gpu_id}; launching eval log={eval_log}")
            cmd = build_eval_cmd(dataset, target, gpu_id)
            with eval_log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{now()}] eval_start dataset={dataset} target={target} gpu={gpu_id}\n")
                handle.flush()
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(WORKSPACE), env=env)
                pid_file.write_text(str(proc.pid) + "\n", encoding="utf-8")
                rc = proc.wait()
                handle.write(f"[{now()}] eval_finished dataset={dataset} pid={proc.pid} rc={rc}\n")
                handle.flush()
            log(wait_log, f"eval_finished dataset={dataset} pid={proc.pid} rc={rc}")
            if rc != 0:
                return rc
            table_rc = run_table(dataset, env, eval_log, wait_log)
            if table_rc != 0:
                return table_rc
            return 0
        log(wait_log, f"waiting_for_idle_gpu dataset={dataset} poll_seconds={args.poll_seconds}")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential RAVEN eval queue for all paper datasets")
    parser.add_argument("--datasets", nargs="+", default=["mscoco", "diffusiondb", "sd_prompts"], choices=sorted(DATASET_TARGETS))
    parser.add_argument("--poll_seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    wait_log = LOG_DIR / "raven_eval_all_datasets_waiter.log"
    (LOG_DIR / "raven_eval_all_datasets_waiter.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update({
        "TQDM_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("RAVEN_CUDA_VISIBLE_DEVICES_APPLIED", None)

    log(wait_log, f"waiter_start datasets={','.join(args.datasets)} methods={','.join(METHODS)}")
    for dataset in args.datasets:
        target = DATASET_TARGETS[dataset]
        rc = run_dataset(dataset, target, args, env, wait_log)
        if rc != 0:
            log(wait_log, f"queue_stop dataset={dataset} rc={rc}")
            return rc
    log(wait_log, "queue_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
