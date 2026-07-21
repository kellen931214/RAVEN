#!/usr/bin/env python3
"""Dispatch strict 1001-sample RAVEN variants onto idle validated GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import formal_attack_config_hash, load_formal_attack_config

FORMAL_RUNNER = REPO / "experiments/run_raven_formal_eval.py"
NO_COLOR_RUNNER = REPO / "experiments/run_raven_no_color_eval.py"
SOURCE_MANIFEST = REPO / "audit/formal_source_manifest.json"

JOBS = {
    "ddim_no_shift_no_color": {
        "config": "ddim_no_shift.json",
        "formal_stages": ["snapshot", "attack-watermarked", "attack-clean"],
        "no_color": True,
    },
    "ddpm_nearest_shift_aligned_color": {
        "config": "ddpm_reflection.json",
        "formal_stages": [
            "snapshot", "attack-watermarked", "attack-clean", "verify",
            "quality", "fid", "clip", "aggregate", "validate",
        ],
        "no_color": False,
    },
    "ddim_bilinear_shift_aligned_color": {
        "config": "nfpa_bilinear_reflection.json",
        "formal_stages": [
            "snapshot", "attack-watermarked", "attack-clean", "verify",
            "quality", "fid", "clip", "aggregate", "validate",
        ],
        "no_color": False,
    },
    "ddim_nearest_shift_aligned_and_no_color": {
        "config": "nfpa_nearest_reflection_ddim_aligned.json",
        "formal_stages": [
            "snapshot", "attack-watermarked", "attack-clean", "verify",
            "quality", "fid", "clip", "aggregate", "validate",
        ],
        "no_color": True,
    },
}


class GPUUnavailable(RuntimeError):
    pass


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / (1024 * 1024)


def nvidia_query() -> tuple[list[dict[str, int | str]], set[str]]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if gpu_result.returncode:
        raise GPUUnavailable(
            f"nvidia-smi GPU query failed ({gpu_result.returncode}): "
            f"{gpu_result.stderr.strip()}"
        )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if process_result.returncode:
        raise GPUUnavailable(
            f"nvidia-smi process query failed ({process_result.returncode}): "
            f"{process_result.stderr.strip()}"
        )
    gpus = []
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, free, total, utilization = [part.strip() for part in line.split(",")]
        gpus.append({
            "index": int(index),
            "uuid": uuid,
            "free_mib": int(free),
            "total_mib": int(total),
            "utilization": int(utilization),
        })
    active_uuids = {
        line.split(",", 1)[0].strip()
        for line in process_result.stdout.splitlines()
        if line.strip()
    }
    return gpus, active_uuids


def idle_candidates(
    gpus: list[dict[str, int | str]],
    active_uuids: set[str],
    *,
    min_free_mib: int,
    max_utilization: int,
    locally_reserved: set[int],
) -> list[dict[str, int | str]]:
    result = [
        gpu
        for gpu in gpus
        if gpu["uuid"] not in active_uuids
        and int(gpu["index"]) not in locally_reserved
        and int(gpu["free_mib"]) >= min_free_mib
        and int(gpu["utilization"]) <= max_utilization
    ]
    return sorted(result, key=lambda gpu: int(gpu["free_mib"]), reverse=True)


def probe_cuda(physical_gpu: int) -> None:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    code = (
        "import torch; "
        "assert torch.cuda.is_available(); "
        "assert torch.cuda.device_count() == 1; "
        "x=torch.ones(16, device='cuda'); "
        "y=(x*x).sum(); "
        "torch.cuda.synchronize(); "
        "assert float(y)==16.0"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    if result.returncode:
        raise GPUUnavailable(
            f"CUDA probe failed for physical GPU {physical_gpu}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def formal_base(
    args: argparse.Namespace, job: dict[str, Any], root: Path, gpu: int, count: int
) -> list[str]:
    return [
        sys.executable,
        str(FORMAL_RUNNER),
        "--dataset", "diffusiondb",
        "--method", "TR",
        "--source-metadata", str(args.source_metadata),
        "--immutable-source-snapshot-index", str(args.snapshot_index),
        "--source-manifest", str(SOURCE_MANIFEST),
        "--attack-config", str(REPO / "experiments/raven_ablation_configs" / job["config"]),
        "--output-root", str(root),
        "--expected-count", str(count),
        "--batch-size", str(args.batch_size),
        "--device", "cuda",
        "--dtype", "float16",
        "--gpu", str(gpu),
    ]


def run_logged(command: list[str], log, *, cwd: Path = REPO) -> None:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def validated(
    root: Path, *, status: str, count: int, expected_attack_hash: str
) -> bool:
    path = root / "VALIDATED.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest_sha = SOURCE_MANIFEST.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).split()[0]
    if (
        payload.get("status") != status
        or payload.get("sample_count", payload.get("N")) != count
        or payload.get("source_code_manifest_sha256") != manifest_sha
    ):
        return False
    if status == "validated_formal_result":
        config_path = root / "run_config.json"
        if not config_path.is_file():
            return False
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return (
            config.get("git_head") == head
            and config.get("source_code_manifest_sha256") == manifest_sha
            and config.get("attack_config_hash") == expected_attack_hash
        )
    return (
        payload.get("git_head") == head
        and payload.get("source_attack_config_hash") == expected_attack_hash
    )


def run_one_cohort(
    args: argparse.Namespace,
    *,
    job_name: str,
    job: dict[str, Any],
    gpu: int,
    count: int,
    cohort_name: str,
    log,
) -> tuple[Path, Path | None]:
    cohort_root = args.state_root / job_name / cohort_name
    root = cohort_root / "formal_attack"
    no_color_root = cohort_root / "no_color_eval"
    attack_config = load_formal_attack_config(
        REPO / "experiments/raven_ablation_configs" / job["config"]
    )
    expected_attack_hash = formal_attack_config_hash(attack_config)
    color_complete = validated(
        root,
        status="validated_formal_result",
        count=count,
        expected_attack_hash=expected_attack_hash,
    )
    no_color_complete = validated(
        no_color_root,
        status="validated_no_color_evaluation",
        count=count,
        expected_attack_hash=expected_attack_hash,
    )
    if not color_complete:
        base = formal_base(args, job, root, gpu, count)
        for stage in job["formal_stages"]:
            command = [*base]
            if (root / "run_config.json").exists():
                command.append("--resume")
            command.extend(["--stage", stage])
            run_logged(command, log)
    if job["no_color"] and not no_color_complete:
        command = [
            sys.executable,
            str(NO_COLOR_RUNNER),
            "--formal-root", str(root),
            "--output-root", str(no_color_root),
            "--expected-count", str(count),
            "--source-manifest", str(SOURCE_MANIFEST),
            "--device", "cuda",
            "--gpu", str(gpu),
        ]
        run_logged(command, log)
    expected_status = (
        "validated_no_color_evaluation" if job["no_color"] else "validated_formal_result"
    )
    expected_root = no_color_root if job["no_color"] else root
    if not validated(
        expected_root,
        status=expected_status,
        count=count,
        expected_attack_hash=expected_attack_hash,
    ):
        raise RuntimeError(
            f"{job_name} {cohort_name} did not produce {expected_status}"
        )
    return root, no_color_root if job["no_color"] else None


def worker(args: argparse.Namespace) -> int:
    job = JOBS[args.worker_job]
    log_path = args.state_root / "logs" / f"{args.worker_job}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        run_one_cohort(
            args,
            job_name=args.worker_job,
            job=job,
            gpu=args.worker_gpu,
            count=10,
            cohort_name="smoke10",
            log=log,
        )
        root, no_color_root = run_one_cohort(
            args,
            job_name=args.worker_job,
            job=job,
            gpu=args.worker_gpu,
            count=args.expected_count,
            cohort_name=f"full{args.expected_count}",
            log=log,
        )
        log.write(f"completed_utc={utc_stamp()}\n")
    atomic_json(
        args.state_root / "status" / f"{args.worker_job}.json",
        {
            "status": "completed",
            "job": args.worker_job,
            "gpu": args.worker_gpu,
            "formal_root": str(root),
            "no_color_root": str(no_color_root) if no_color_root else None,
            "completed_utc": utc_stamp(),
        },
    )
    return 0


def parent(args: argparse.Namespace) -> int:
    args.state_root.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "waiting_for_idle_gpus",
        "created_utc": utc_stamp(),
        "repo": str(REPO),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_metadata": str(args.source_metadata),
        "snapshot_index": str(args.snapshot_index),
        "expected_count": args.expected_count,
        "jobs": list(JOBS),
    }
    atomic_json(args.state_root / "dispatcher_state.json", state)
    pending = list(JOBS)
    active: dict[str, tuple[subprocess.Popen, int, Any]] = {}
    failed = False
    while pending or active:
        for name, (process, gpu, handle) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[name]
            if status != 0:
                failed = True
                atomic_json(
                    args.state_root / "status" / f"{name}.json",
                    {"status": "failed", "exit_code": status, "gpu": gpu},
                )
        if failed:
            if active:
                time.sleep(args.poll_seconds)
                continue
            state["status"] = "failed"
            atomic_json(args.state_root / "dispatcher_state.json", state)
            return 1
        if pending and available_ram_gib() >= args.min_available_ram_gib:
            try:
                gpus, active_uuids = nvidia_query()
            except GPUUnavailable as exc:
                state["status"] = "stopped_gpu_unavailable"
                state["error"] = str(exc)
                atomic_json(args.state_root / "dispatcher_state.json", state)
                return 2
            candidates = idle_candidates(
                gpus,
                active_uuids,
                min_free_mib=args.min_free_mib,
                max_utilization=args.max_utilization,
                locally_reserved={gpu for _, gpu, _ in active.values()},
            )
            while candidates and pending:
                candidate = candidates.pop(0)
                gpu = int(candidate["index"])
                try:
                    probe_cuda(gpu)
                except GPUUnavailable as exc:
                    state["status"] = "stopped_gpu_unavailable"
                    state["error"] = str(exc)
                    atomic_json(args.state_root / "dispatcher_state.json", state)
                    return 2
                name = pending.pop(0)
                worker_log = args.state_root / "logs" / f"{name}.worker.log"
                handle = worker_log.open("a", encoding="utf-8")
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--state-root", str(args.state_root),
                    "--source-metadata", str(args.source_metadata),
                    "--snapshot-index", str(args.snapshot_index),
                    "--expected-count", str(args.expected_count),
                    "--batch-size", str(args.batch_size),
                    "--poll-seconds", str(args.poll_seconds),
                    "--min-free-mib", str(args.min_free_mib),
                    "--max-utilization", str(args.max_utilization),
                    "--min-available-ram-gib", str(args.min_available_ram_gib),
                    "--worker-job", name,
                    "--worker-gpu", str(gpu),
                ]
                process = subprocess.Popen(
                    command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT
                )
                active[name] = (process, gpu, handle)
                atomic_json(
                    args.state_root / "status" / f"{name}.json",
                    {
                        "status": "running",
                        "pid": process.pid,
                        "gpu": gpu,
                        "free_mib_at_launch": candidate["free_mib"],
                        "started_utc": utc_stamp(),
                    },
                )
        state["status"] = "running" if active else "waiting_for_idle_gpus"
        state["pending"] = pending
        state["active"] = {
            name: {"pid": process.pid, "gpu": gpu}
            for name, (process, gpu, _) in active.items()
        }
        atomic_json(args.state_root / "dispatcher_state.json", state)
        if pending or active:
            time.sleep(args.poll_seconds)
    state["status"] = "completed"
    state["completed_utc"] = utc_stamp()
    atomic_json(args.state_root / "dispatcher_state.json", state)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-root", type=Path, required=True)
    result.add_argument("--source-metadata", type=Path, required=True)
    result.add_argument("--snapshot-index", type=Path, required=True)
    result.add_argument("--expected-count", type=int, default=1001)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--poll-seconds", type=int, default=60)
    result.add_argument("--min-free-mib", type=int, default=18000)
    result.add_argument("--max-utilization", type=int, default=5)
    result.add_argument("--min-available-ram-gib", type=float, default=64.0)
    result.add_argument("--worker-job", choices=JOBS, default=None)
    result.add_argument("--worker-gpu", type=int, default=None)
    return result


def main() -> int:
    args = parser().parse_args()
    args.state_root = args.state_root.resolve()
    args.source_metadata = args.source_metadata.resolve()
    args.snapshot_index = args.snapshot_index.resolve()
    if args.worker_job is not None:
        if args.worker_gpu is None:
            raise ValueError("--worker-gpu is required for a worker")
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
