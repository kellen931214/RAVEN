#!/usr/bin/env python3
"""Wait for idle GPUs, then generate remaining dataset watermarked images.

Datasets covered by default:
- DiffusionDB: 1,001 prompts
- SD-Prompts: 8,192 prompts

Each job is one (dataset, watermark method) child process using
experiments/generate_watermarked_images.py with an explicit GPU id. The launcher
only uses GPUs that are idle according to nvidia-smi and compatible with this
Torch build.
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
from typing import Dict, List, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
RAVEN_ROOT = WORKSPACE / "raven_repro"
if str(RAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEN_ROOT))

from raven.gpu_utils import query_gpu_status

METHODS = ["GS", "TR", "RID", "HSTR", "HSQR"]
DATASETS = {
    "diffusiondb": {
        "prompts_csv": WORKSPACE / "data" / "clean" / "diffusiondb" / "inputs" / "diffusiondb_1001_prompts.csv",
        "num_pairs": 1001,
    },
    "sd_prompts": {
        "prompts_csv": WORKSPACE / "data" / "clean" / "sd_prompts" / "inputs" / "sd_prompts_8192.csv",
        "num_pairs": 8192,
    },
}
LOG_DIR = WORKSPACE / "raven_repro" / "logs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (LOG_DIR / "watermark_generation_remaining_queue.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
    print(line, flush=True)


def is_compatible(gpu) -> bool:
    return "blackwell" not in gpu.name.lower()


def idle_compatible_gpus() -> List:
    status = query_gpu_status()
    if status.get("error"):
        log(f"WARNING nvidia-smi failed: {status['error']}")
        return []
    gpus = [gpu for gpu in status.get("gpus") or [] if is_compatible(gpu)]
    idle = [gpu for gpu in gpus if gpu.is_idle]
    return sorted(idle, key=lambda gpu: (gpu.memory_used_mib, gpu.utilization_gpu_percent, int(gpu.index)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue DiffusionDB and SD-Prompts watermarked image generation")
    parser.add_argument("--datasets", nargs="+", default=["diffusiondb", "sd_prompts"], choices=sorted(DATASETS.keys()))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    # Canonical layout: empty output_dir lets the generator route each method to
    # its own data root (data/tr/, data/gs/).
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model_id", type=str, default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--min_cpu_mem_gb", type=float, default=64.0)
    parser.add_argument("--warn_cpu_mem_gb", type=float, default=96.0)
    parser.add_argument("--max_process_ram_gb", type=float, default=16.0)
    return parser.parse_args()


@dataclass
class PendingJob:
    dataset: str
    method: str
    prompts_csv: Path
    num_pairs: int


@dataclass
class RunningJob:
    pending: PendingJob
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
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("RAVEN_CUDA_VISIBLE_DEVICES_APPLIED", None)
    return env


def build_jobs(args: argparse.Namespace) -> List[PendingJob]:
    jobs: List[PendingJob] = []
    for dataset in args.datasets:
        config = DATASETS[dataset]
        for method in args.methods:
            jobs.append(PendingJob(dataset, method, Path(config["prompts_csv"]), int(config["num_pairs"])))
    return jobs


def launch_job(args: argparse.Namespace, job: PendingJob, gpu_id: str) -> RunningJob:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"watermark_generation_{job.dataset}_{job.method}.log"
    fh = log_path.open("a", encoding="utf-8")
    cmd = [
        "ionice", "-c2", "-n7",
        "nice", "-n", "10",
        sys.executable,
        str(WORKSPACE / "experiments" / "generate_watermarked_images.py"),
        "--dataset_name", job.dataset,
        "--prompts_csv", str(job.prompts_csv),
        *(["--output_dir", args.output_dir] if args.output_dir else []),
        "--wm_types", job.method,
        "--num_pairs", str(job.num_pairs),
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
    log(f"launch dataset={job.dataset} method={job.method} gpu={gpu_id} log={log_path}")
    process = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=child_env(), cwd=str(WORKSPACE))
    return RunningJob(job, str(gpu_id), process, fh)


def terminate_jobs(jobs: List[RunningJob], signum: int) -> None:
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
    (LOG_DIR / "watermark_generation_remaining_queue.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    pending = build_jobs(args)
    running: List[RunningJob] = []
    exit_code = 0

    signal.signal(signal.SIGTERM, lambda s, f: terminate_jobs(running, s))
    signal.signal(signal.SIGINT, lambda s, f: terminate_jobs(running, s))
    log("queue_start datasets=" + ",".join(args.datasets) + " methods=" + ",".join(args.methods))

    while pending or running:
        still_running: List[RunningJob] = []
        used_gpu_ids = set()
        for job in running:
            rc = job.process.poll()
            if rc is None:
                still_running.append(job)
                used_gpu_ids.add(job.gpu_id)
                continue
            job.log_handle.close()
            if rc == 0:
                log(f"complete dataset={job.pending.dataset} method={job.pending.method} gpu={job.gpu_id} rc=0")
            else:
                log(f"FAILED dataset={job.pending.dataset} method={job.pending.method} gpu={job.gpu_id} rc={rc}")
                exit_code = rc if exit_code == 0 else exit_code
        running = still_running

        if pending and len(running) < args.max_parallel:
            idle = [gpu for gpu in idle_compatible_gpus() if gpu.index not in used_gpu_ids]
            slots = max(0, args.max_parallel - len(running))
            if idle and slots:
                selected = idle[:slots]
                log("idle_gpus=" + ",".join(f"{gpu.index}:{gpu.name}:{gpu.memory_used_mib}MiB" for gpu in selected))
                for gpu in selected:
                    if not pending:
                        break
                    running.append(launch_job(args, pending.pop(0), gpu.index))
                    used_gpu_ids.add(gpu.index)
            else:
                log(f"waiting_for_idle_gpu pending={len(pending)} running={len(running)}")

        if pending or running:
            time.sleep(args.poll_seconds)

    log(f"queue_finished exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
