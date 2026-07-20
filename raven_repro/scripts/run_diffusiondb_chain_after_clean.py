#!/usr/bin/env python
"""ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT.

Run the historical paired DiffusionDB Tree-Ring experiment.

The historical implementation waited for independently generated clean images.
That workflow is intentionally removed: clean and watermarked images are now
generated together from one unique per-sample base latent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.gpu_utils import setup_run_logging
from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def available_gpus(min_free_mib: int) -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        index, uuid, name, free_mib, utilization = [
            part.strip() for part in line.split(",", 4)
        ]
        # The installed torch build does not support sm_120 Blackwell kernels.
        if "Blackwell" in name:
            continue
        item = {
            "index": index,
            "uuid": uuid,
            "name": name,
            "free_mib": int(free_mib),
            "utilization": int(utilization),
        }
        if item["free_mib"] >= min_free_mib and item["utilization"] <= 5:
            candidates.append(item)
    return sorted(candidates, key=lambda item: (-item["free_mib"], int(item["index"])))


def wait_for_gpus(
    requested: str,
    count: int,
    poll_seconds: int,
    min_free_mib: int,
) -> list[dict]:
    while True:
        candidates = available_gpus(min_free_mib)
        if requested != "auto":
            allowed = {value.strip() for value in requested.split(",") if value.strip()}
            candidates = [item for item in candidates if item["index"] in allowed]
        if len(candidates) >= count:
            selected = candidates[:count]
            print(f"[{utc_now()}] selected GPUs {selected}", flush=True)
            return selected
        print(
            f"[{utc_now()}] waiting for compatible idle GPU "
            f"requested={requested} count={count} min_free_mib={min_free_mib}",
            flush=True,
        )
        time.sleep(poll_seconds)


def wait_for_gpu(requested: str, poll_seconds: int, min_free_mib: int) -> dict:
    return wait_for_gpus(requested, 1, poll_seconds, min_free_mib)[0]


def mark_failed(root: Path, state: dict, stage: str, returncode: int) -> None:
    state["status"] = "failed"
    state["failed_stage"] = stage
    state["failed_returncode"] = returncode
    state["failed_utc"] = utc_now()
    write_json(root / "run_state.json", state)


def run_logged_subprocess(command: list[str], *, env: dict[str, str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def paired_shard_metadata_complete(
    root: Path, expected_count: int, num_shards: int
) -> bool:
    metadata_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"
    for shard_index in range(num_shards):
        path = metadata_dir / f"metadata.shard-{shard_index:03d}-of-{num_shards:03d}.csv"
        if not path.is_file():
            return False
        with path.open(newline="", encoding="utf-8-sig") as handle:
            run_ids = [int(row["run_id"]) for row in csv.DictReader(handle)]
        expected_ids = list(range(shard_index, expected_count, num_shards))
        if run_ids != expected_ids:
            return False
    return True


def run_paired_generation_shards(
    args: argparse.Namespace,
    *,
    root: Path,
    prompt_copy: Path,
    state: dict,
    guard: CpuMemoryGuard,
    env: dict[str, str],
    python: str,
) -> None:
    if "paired_generation" in state["completed_stages"]:
        print(f"[{utc_now()}] skip completed stage=paired_generation", flush=True)
        return
    num_shards = int(args.generation_workers)
    if num_shards <= 0 or num_shards > args.expected_count:
        raise ValueError("--generation-workers must be in [1, expected-count]")

    # Always rerun preparation on resume. It audits committed rows and moves
    # crash-written images without metadata to an invalid quarantine.
    prepare_command = [
        python,
        "-u",
        "raven_repro/scripts/paired_generation_shards.py",
        "prepare",
        "--root",
        str(root),
        "--prompts-csv",
        str(prompt_copy),
        "--expected-count",
        str(args.expected_count),
        "--num-shards",
        str(num_shards),
    ]
    guard.check("before paired generation shard preparation")
    print(f"[{utc_now()}] stage=paired_generation_prepare", flush=True)
    prepared_returncode = run_logged_subprocess(prepare_command, env=env)
    if prepared_returncode != 0:
        mark_failed(root, state, "paired_generation_prepare", prepared_returncode)
        raise RuntimeError("paired generation shard preparation failed")

    merge_command = [
        python,
        "-u",
        "raven_repro/scripts/paired_generation_shards.py",
        "merge",
        "--root",
        str(root),
        "--prompts-csv",
        str(prompt_copy),
        "--expected-count",
        str(args.expected_count),
        "--num-shards",
        str(num_shards),
    ]
    if paired_shard_metadata_complete(root, args.expected_count, num_shards):
        print(
            f"[{utc_now()}] all generation shard metadata complete after prepare; "
            "recovering directly through formal merge audit",
            flush=True,
        )
        merged_returncode = run_logged_subprocess(merge_command, env=env)
        if merged_returncode != 0:
            mark_failed(root, state, "paired_generation_merge", merged_returncode)
            raise RuntimeError("paired generation shard merge failed")
        for worker_state in state.get("generation_workers", {}).values():
            if worker_state.get("status") == "running":
                worker_state["status"] = "completed_recovered_from_metadata"
                worker_state["finished_utc"] = utc_now()
        state["completed_stages"].append("paired_generation")
        state["last_completed_utc"] = utc_now()
        write_json(root / "run_state.json", state)
        guard.check("after paired generation shard merge")
        return

    selected_gpus = wait_for_gpus(
        str(args.visible_gpu),
        num_shards,
        args.gpu_poll_seconds,
        args.min_free_gpu_mib,
    )
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers: list[dict] = []
    for shard_index, gpu in enumerate(selected_gpus):
        shard_log = log_dir / f"paired_generation_shard_{shard_index:03d}.log"
        log_handle = shard_log.open("a", encoding="utf-8", buffering=1)
        command = [
            python,
            "-u",
            "experiments/generate_watermarked_images.py",
            "--dataset_name",
            "diffusiondb",
            "--prompts_csv",
            str(prompt_copy),
            "--output_dir",
            str(root / "data" / "watermarked"),
            "--clean_output_dir",
            str(root / "data" / "generated"),
            "--device",
            args.device,
            "--gpu",
            gpu["index"],
            "--require_free_gpu",
            "false",
            "--min_cpu_mem_gb",
            str(args.min_cpu_mem_gib),
            "--warn_cpu_mem_gb",
            "40",
            "--max_process_ram_gb",
            str(args.max_process_rss_gib),
            "--wm_types",
            "TR",
            "--num_pairs",
            str(args.expected_count),
            "--num_shards",
            str(num_shards),
            "--shard_index",
            str(shard_index),
            "--start_index",
            "0",
            "--seed",
            "42",
            "--modelid_target",
            "RedbeardNZ/stable-diffusion-2-1-base",
            "--model_revision",
            "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
            "--scheduler_target",
            "DDIM",
            "--num_inference_steps_target",
            "50",
            "--guidance_scale_target",
            "7.5",
            "--resolution",
            "512",
            "--validate_before",
            "true",
        ]
        print(
            f"[{utc_now()}] launch generation shard={shard_index}/{num_shards} "
            f"gpu={gpu['index']} log={shard_log}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        workers.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "log": str(shard_log),
                "log_handle": log_handle,
                "process": process,
            }
        )
    state["generation_workers"] = {
        str(worker["shard_index"]): {
            "gpu": worker["gpu"],
            "pid": worker["process"].pid,
            "log": worker["log"],
            "status": "running",
        }
        for worker in workers
    }
    write_json(root / "run_state.json", state)

    failed: list[dict] = []
    try:
        while True:
            running = False
            for worker in workers:
                returncode = worker["process"].poll()
                status = state["generation_workers"][str(worker["shard_index"])]
                if returncode is None:
                    running = True
                elif status["status"] == "running":
                    status["status"] = "completed" if returncode == 0 else "failed"
                    status["returncode"] = returncode
                    status["finished_utc"] = utc_now()
                    if returncode != 0:
                        failed.append(worker)
                    write_json(root / "run_state.json", state)
            if failed or not running:
                break
            time.sleep(5)
        if failed:
            for worker in workers:
                if worker["process"].poll() is None:
                    worker["process"].terminate()
            for worker in workers:
                try:
                    worker["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker["process"].kill()
            for worker in failed:
                path = Path(worker["log"])
                tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
                print(
                    f"[{utc_now()}] FAILED shard={worker['shard_index']} log={path}\n"
                    + "\n".join(tail),
                    file=sys.stderr,
                    flush=True,
                )
            returncode = int(failed[0]["process"].returncode or 1)
            mark_failed(root, state, "paired_generation", returncode)
            raise RuntimeError("one or more paired generation shards failed")
    finally:
        for worker in workers:
            worker["log_handle"].close()

    print(f"[{utc_now()}] stage=paired_generation_merge", flush=True)
    merged_returncode = run_logged_subprocess(merge_command, env=env)
    if merged_returncode != 0:
        mark_failed(root, state, "paired_generation_merge", merged_returncode)
        raise RuntimeError("paired generation shard merge failed")
    state["completed_stages"].append("paired_generation")
    state["last_completed_utc"] = utc_now()
    write_json(root / "run_state.json", state)
    guard.check("after paired generation shard merge")


def run_stage(
    name: str,
    command: list[str],
    *,
    root: Path,
    state: dict,
    guard: CpuMemoryGuard,
    env: dict[str, str],
) -> None:
    if name in state["completed_stages"]:
        print(f"[{utc_now()}] skip completed stage={name}", flush=True)
        return
    guard.check(f"before stage {name}")
    stage_log = root / "logs" / f"{name}.log"
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[{utc_now()}] stage={name} log={stage_log} command={' '.join(command)}",
        flush=True,
    )
    with stage_log.open("a", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode != 0:
        state["status"] = "failed"
        state["failed_stage"] = name
        state["failed_returncode"] = returncode
        state["failed_utc"] = utc_now()
        write_json(root / "run_state.json", state)
        raise RuntimeError(f"stage failed ({returncode}): {name}")
    state["completed_stages"].append(name)
    state["last_completed_utc"] = utc_now()
    write_json(root / "run_state.json", state)
    guard.check(f"after stage {name}")


def run_parallel_gpu_stages(
    stages: list[tuple[str, list[str]]],
    *,
    args: argparse.Namespace,
    root: Path,
    state: dict,
    guard: CpuMemoryGuard,
    env: dict[str, str],
) -> None:
    pending = [
        (name, command)
        for name, command in stages
        if name not in state["completed_stages"]
    ]
    if not pending:
        for name, _ in stages:
            print(f"[{utc_now()}] skip completed stage={name}", flush=True)
        return

    guard.check("before parallel clean/watermarked attacks")
    selected_gpus = wait_for_gpus(
        str(args.visible_gpu),
        len(pending),
        args.gpu_poll_seconds,
        args.min_free_gpu_mib,
    )
    workers: list[dict] = []

    def pump_output(name: str, stream, log_handle) -> None:
        for line in stream:
            log_handle.write(line)
            print(f"[{name}] {line}", end="", flush=True)

    for (name, command), gpu in zip(pending, selected_gpus):
        stage_env = env.copy()
        stage_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        stage_env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        stage_log = root / "logs" / f"{name}.log"
        stage_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = stage_log.open("a", encoding="utf-8", buffering=1)
        print(
            f"[{utc_now()}] launch parallel stage={name} gpu={gpu['index']} "
            f"uuid={gpu['uuid']} log={stage_log} command={' '.join(command)}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=stage_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_thread = threading.Thread(
            target=pump_output,
            args=(name, process.stdout, log_handle),
            name=f"{name}-log-pump",
            daemon=True,
        )
        output_thread.start()
        workers.append(
            {
                "name": name,
                "gpu": gpu,
                "log": str(stage_log),
                "log_handle": log_handle,
                "process": process,
                "output_thread": output_thread,
            }
        )
        state.setdefault("stage_gpus", {})[name] = gpu
        state.setdefault("attack_workers", {})[name] = {
            "gpu": gpu,
            "pid": process.pid,
            "log": str(stage_log),
            "status": "running",
        }
    write_json(root / "run_state.json", state)

    failed: list[dict] = []
    try:
        while True:
            running = False
            for worker in workers:
                returncode = worker["process"].poll()
                worker_state = state["attack_workers"][worker["name"]]
                if returncode is None:
                    running = True
                elif worker_state["status"] == "running":
                    worker_state["status"] = "completed" if returncode == 0 else "failed"
                    worker_state["returncode"] = returncode
                    worker_state["finished_utc"] = utc_now()
                    if returncode != 0:
                        failed.append(worker)
                    write_json(root / "run_state.json", state)
            if failed or not running:
                break
            time.sleep(5)
        if failed:
            for worker in workers:
                if worker["process"].poll() is None:
                    worker["process"].terminate()
            for worker in workers:
                try:
                    worker["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker["process"].kill()
            returncode = int(failed[0]["process"].returncode or 1)
            mark_failed(root, state, "parallel_attacks", returncode)
            raise RuntimeError(
                "parallel attack failed: "
                + ", ".join(worker["name"] for worker in failed)
            )
        for worker in workers:
            if worker["name"] not in state["completed_stages"]:
                state["completed_stages"].append(worker["name"])
        state["last_completed_utc"] = utc_now()
        write_json(root / "run_state.json", state)
    finally:
        for worker in workers:
            worker["output_thread"].join(timeout=10)
            worker["log_handle"].close()
    guard.check("after parallel clean/watermarked attacks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1001)
    parser.add_argument("--visible-gpu", default="auto")
    parser.add_argument("--generation-workers", type=int, default=2)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--min-free-gpu-mib", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--min-cpu-mem-gib", type=float, default=24.0)
    parser.add_argument("--max-process-rss-gib", type=float, default=48.0)
    parser.add_argument(
        "--invalid-archive",
        type=Path,
        default=Path("outputs/invalid/shared_latent_20260716"),
    )
    args = parser.parse_args()

    limit_cpu_threads(1)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    setup_run_logging(root, filename="run.log")
    guard = CpuMemoryGuard(
        min_available_gib=args.min_cpu_mem_gib,
        max_process_rss_gib=args.max_process_rss_gib,
        warn_available_gib=40.0,
    )
    guard.check("paired formal orchestrator startup")
    prompt_copy = root / "inputs" / "diffusiondb_1001_prompts.csv"
    prompt_copy.parent.mkdir(parents=True, exist_ok=True)
    if prompt_copy.exists():
        if prompt_copy.read_bytes() != args.prompts_csv.resolve().read_bytes():
            raise ValueError("existing prompt copy differs from requested prompts")
    else:
        shutil.copy2(args.prompts_csv.resolve(), prompt_copy)

    state_path = root / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "protocol": "diffusiondb_tree_ring_paired_formal_v1",
            "status": "running",
            "created_utc": utc_now(),
            "expected_count": args.expected_count,
            "completed_stages": [],
        }
        write_json(state_path, state)
    if int(state["expected_count"]) != args.expected_count:
        raise ValueError("run_state expected_count mismatch")
    state["status"] = "running"
    state["resumed_utc"] = utc_now()
    state.pop("failed_stage", None)
    state.pop("failed_returncode", None)
    state.pop("failed_utc", None)
    state["run_log"] = str((root / "run.log").resolve())
    write_json(state_path, state)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    python = sys.executable
    paired_metadata = root / "data" / "watermarked" / "diffusiondb" / "TR" / "metadata.csv"
    manifest = root / "inputs" / "diffusiondb_tr_manifest_1001.csv"
    p1_dir = root / "p1_1001"
    nfpa_dir = root / "nfpa_1001"
    quality_dir = root / "quality_decomposition_1001"

    run_paired_generation_shards(
        args,
        root=root,
        prompt_copy=prompt_copy,
        state=state,
        guard=guard,
        env=env,
        python=python,
    )

    setup_stages = [
        (
            "paired_manifest",
            False,
            [
                python, "raven_repro/scripts/build_diffusiondb_tr_manifest.py",
                "--paired-metadata", str(paired_metadata),
                "--expected-count", str(args.expected_count),
                "--output", str(manifest),
            ],
        ),
        (
            "p1_plan",
            False,
            [
                python, "-u", "raven_repro/scripts/raven_p1_full.py", "plan-dataset",
                "--dataset", "diffusiondb",
                "--manifest", str(manifest),
                "--output-dir", str(p1_dir),
                "--expected-count", str(args.expected_count),
                "--plan-seed", "2026071401",
            ],
        ),
        (
            "nfpa_prepare_protocol",
            False,
            [
                python, "-u", "raven_repro/scripts/raven_nfpa_tr_eval.py", "prepare-protocol",
                "--dataset", "diffusiondb",
                "--p1-dir", str(p1_dir),
                "--output-dir", str(nfpa_dir),
                "--expected-count", str(args.expected_count),
            ],
        ),
    ]
    for name, _, command in setup_stages:
        run_stage(name, command, root=root, state=state, guard=guard, env=env)

    run_parallel_gpu_stages(
        [
            (
                "attacked_watermarked",
                [
                    python, "-u", "raven_repro/scripts/raven_p1_full.py", "attack",
                    "--output-dir", str(p1_dir),
                    "--device", args.device,
                    "--dtype", args.dtype,
                    "--resume",
                ],
            ),
            (
                "attacked_clean",
                [
                    python, "-u", "raven_repro/scripts/raven_nfpa_tr_eval.py", "attack-clean",
                    "--output-dir", str(nfpa_dir),
                    "--device", args.device,
                    "--dtype", args.dtype,
                    "--resume",
                ],
            ),
        ],
        args=args,
        root=root,
        state=state,
        guard=guard,
        env=env,
    )

    downstream_stages = [
        (
            "nfpa_sync_watermarked",
            False,
            [
                python, "-u", "raven_repro/scripts/raven_nfpa_tr_eval.py", "sync-watermarked",
                "--p1-dir", str(p1_dir),
                "--output-dir", str(nfpa_dir),
                "--expected-count", str(args.expected_count),
            ],
        ),
        (
            "quality_and_detector_metrics",
            True,
            [
                python, "-u", "raven_repro/scripts/quality_decomposition_experiment.py",
                "--source-root", str(root),
                "--output-dir", str(quality_dir),
                "--count", str(args.expected_count),
                "--device", args.device,
                "--dtype", args.dtype,
                "--quality-device", args.device,
                "--tree-ring-repo", "external/tree-ring-watermark",
            ],
        ),
    ]
    if args.expected_count == 1001:
        downstream_stages.append(
            (
                "invalidate_legacy_shared_latent_outputs",
                False,
                [
                    python,
                    "-u",
                    "raven_repro/scripts/invalidate_shared_latent_outputs.py",
                    "--root",
                    "outputs/raven_color_alignment_ablation/diffusiondb/20260716T082019Z",
                    "--root",
                    "outputs/raven_color_alignment_ablation/diffusiondb/20260716T125600Z",
                    "--root",
                    "outputs/quality_decomposition/diffusiondb",
                    "--archive",
                    str(args.invalid_archive),
                    "--execute",
                ],
            )
        )
    for name, needs_gpu, command in downstream_stages:
        stage_env = env.copy()
        stage_command = list(command)
        if needs_gpu and name not in state["completed_stages"]:
            selected = wait_for_gpu(
                str(args.visible_gpu), args.gpu_poll_seconds, args.min_free_gpu_mib
            )
            if "__GPU__" in stage_command:
                stage_command = [
                    selected["index"] if value == "__GPU__" else value
                    for value in stage_command
                ]
            else:
                stage_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                stage_env["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
            state.setdefault("stage_gpus", {})[name] = selected
            write_json(state_path, state)
        run_stage(name, stage_command, root=root, state=state, guard=guard, env=stage_env)

    aggregate = quality_dir / "aggregate_results.json"
    if not aggregate.is_file():
        raise FileNotFoundError(aggregate)
    result = json.loads(aggregate.read_text(encoding="utf-8"))
    if not (result.get("pairing_audit") or {}).get("passed"):
        raise ValueError("final aggregate missing pairing audit")
    state["status"] = "completed"
    state["completed_utc"] = utc_now()
    state["aggregate_results"] = str(aggregate)
    write_json(state_path, state)
    (root / "FORMAL_RUN_COMPLETE").write_text(utc_now() + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
