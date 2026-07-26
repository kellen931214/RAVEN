#!/usr/bin/env python3
"""Parallel launcher for MS-COCO watermarked image generation.

Runs one watermark method per idle compatible GPU and resumes existing outputs.
The launcher itself does not generate images; it coordinates child calls to
experiments/generate_watermarked_images.py with explicit --gpu ids selected from
current nvidia-smi status.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE = Path(__file__).resolve().parents[1]
RAVEN_ROOT = WORKSPACE / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.gpu_utils import query_gpu_status

METHODS = ["GS", "TR", "RID", "HSTR", "HSQR"]
LOG_DIR = WORKSPACE / "raven_repro" / "logs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "watermark_generation_mscoco_parallel.log").open("a", encoding="utf-8") as fh:
        fh.write(f"[{now()}] {message}\n")
        fh.flush()
    print(f"[{now()}] {message}", flush=True)


def is_compatible(gpu) -> bool:
    return "blackwell" not in gpu.name.lower()


def idle_compatible_gpus() -> List:
    status = query_gpu_status()
    if status.get("error"):
        raise RuntimeError(f"nvidia-smi failed: {status['error']}")
    gpus = [gpu for gpu in status.get("gpus") or [] if is_compatible(gpu)]
    idle = [gpu for gpu in gpus if gpu.is_idle]
    return sorted(idle, key=lambda gpu: (gpu.memory_used_mib, gpu.utilization_gpu_percent, int(gpu.index)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel MS-COCO watermark generation launcher")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    # Canonical layout: empty output_dir lets the generator route each method to
    # its own data root (data/tr/, data/gs/).
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--prompts_csv", type=str,
                        default=str(WORKSPACE / "data" / "clean" / "mscoco" / "inputs" / "mscoco_5000.csv"))
    parser.add_argument("--max_parallel", type=int, default=0, help="0 means use all currently idle compatible GPUs")
    parser.add_argument("--min_cpu_mem_gb", type=float, default=64.0)
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=96.0)
    parser.add_argument("--max_process_ram_gb", type=float, default=16.0)
    return parser.parse_args()


@dataclass
class RunningJob:
    method: str
    gpu_id: str
    process: subprocess.Popen
    log_handle: object


def child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update({
        "TQDM_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Child scripts set CUDA_VISIBLE_DEVICES only because --gpu is explicit.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("RAVEN_CUDA_VISIBLE_DEVICES_APPLIED", None)
    return env


def launch_method(args: argparse.Namespace, method: str, gpu_id: str) -> RunningJob:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"watermark_generation_mscoco_{method}.log"
    fh = log_path.open("a", encoding="utf-8")
    cmd = [
        "ionice", "-c2", "-n7",
        "nice", "-n", "10",
        sys.executable,
        str(WORKSPACE / "experiments" / "generate_watermarked_images.py"),
        "--dataset_name", "mscoco",
        "--prompts_csv", args.prompts_csv,
        *(["--output_dir", args.output_dir] if args.output_dir else []),
        "--wm_types", method,
        "--num_pairs", str(args.num_pairs),
        "--modelid_target", args.model_id,
        "--resolution", "512",
        "--scheduler_target", "DDIM",
        "--num_inference_steps_target", "50",
        "--guidance_scale_target", "7.5",
        "--seed", "42",
        "--device", "cuda",
        "--gpu", str(gpu_id),
        "--require_free_gpu", "false",
        "--min_cpu_mem_gb", str(args.min_cpu_mem_gb),
        "--warn_cpu_mem_gb", str(args.warn_cpu_mem_gb),
        "--max_process_ram_gb", str(args.max_process_ram_gb),
        "--validate_before", "true",
    ]
    log(f"launch method={method} gpu={gpu_id} log={log_path}")
    process = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=child_env(), cwd=str(WORKSPACE))
    return RunningJob(method=method, gpu_id=str(gpu_id), process=process, log_handle=fh)


def terminate_jobs(jobs: List[RunningJob], signum, frame) -> None:
    log(f"received signal={signum}; terminating children")
    for job in jobs:
        if job.process.poll() is None:
            job.process.terminate()
    time.sleep(5)
    for job in jobs:
        if job.process.poll() is None:
            job.process.kill()
    raise SystemExit(128 + int(signum))


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "watermark_generation_mscoco_parallel.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    gpus = idle_compatible_gpus()
    if not gpus:
        log("ERROR no idle compatible GPU available; refusing to start full parallel generation")
        return 2
    if args.max_parallel > 0:
        gpus = gpus[: args.max_parallel]
    log("selected_idle_gpus=" + ",".join(f"{gpu.index}:{gpu.name}:{gpu.memory_used_mib}MiB" for gpu in gpus))

    pending = list(args.methods)
    running: List[RunningJob] = []
    available_gpu_ids = [gpu.index for gpu in gpus]
    signal.signal(signal.SIGTERM, lambda s, f: terminate_jobs(running, s, f))
    signal.signal(signal.SIGINT, lambda s, f: terminate_jobs(running, s, f))

    exit_code = 0
    try:
        while pending or running:
            while pending and available_gpu_ids:
                method = pending.pop(0)
                gpu_id = available_gpu_ids.pop(0)
                running.append(launch_method(args, method, gpu_id))

            time.sleep(10)
            still_running: List[RunningJob] = []
            for job in running:
                rc = job.process.poll()
                if rc is None:
                    still_running.append(job)
                    continue
                job.log_handle.close()
                available_gpu_ids.append(job.gpu_id)
                if rc == 0:
                    log(f"complete method={job.method} gpu={job.gpu_id} rc=0")
                else:
                    log(f"FAILED method={job.method} gpu={job.gpu_id} rc={rc}")
                    exit_code = rc if exit_code == 0 else exit_code
            running = still_running
        log(f"parallel launcher finished exit_code={exit_code}")
        return exit_code
    finally:
        for job in running:
            try:
                job.log_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
