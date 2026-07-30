#!/usr/bin/env python3
"""Detached RID/HSTR/HSQR shared-clean smoke and formal waiter for Issue #17."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.eval_protocol import (  # noqa: E402
    formal_attack_config_hash,
    formal_output_root,
    formal_run_key,
    load_formal_attack_config,
    method_data_root,
    sha256_path,
)

METHODS = ("RID", "HSTR", "HSQR")
STAGES = ("snapshot", "attack-watermarked", "verify", "quality", "fid", "clip", "aggregate", "validate")
GENERATOR = {
    "RID": REPO / "experiments/generate_rid_from_tr_shared_clean.py",
    "HSTR": REPO / "experiments/generate_hstr_from_tr_shared_clean.py",
    "HSQR": REPO / "experiments/generate_hsqr_from_tr_shared_clean.py",
}
SOURCE_MANIFEST = REPO / "audit/formal_source_manifest.json"
FORMAL_RUNNER = REPO / "experiments/run_raven_formal_eval.py"
TABLE_UPDATER = REPO / "experiments/update_experiment_table.py"
AUDIT_SCRIPT = REPO / "raven_repro/scripts/audit_shared_clean_cohorts.py"
DEFAULT_TR_METADATA = Path("/workspace/RAVEN/data/tr/diffusiondb/metadata.csv")
STATE_ROOT = REPO / "logs/issue17_fourier_waiter"


class WaiterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    generation: Path
    formal: Path
    metadata: Path
    bundle: Path


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_event(state_root: Path, line: str) -> None:
    path = state_root / "events.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc()} {line}\n")
        handle.flush()
        os.fsync(handle.fileno())


class AtomicLock:
    def __init__(self, path: Path, payload: dict[str, Any] | None = None):
        self.path = path
        self.payload = payload or {}
        self.fd: int | None = None

    def __enter__(self) -> "AtomicLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        record = {"pid": os.getpid(), "created_utc": utc(), **self.payload}
        os.write(self.fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(self.fd)
        return self

    def close_without_unlink(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def lock_path(state_root: Path, name: str) -> Path:
    return state_root / "locks" / name


def run(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values.get("MemAvailable", 0) / (1024 * 1024)


def parse_gpu_query() -> tuple[list[dict[str, Any]], set[str]]:
    base = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if base.returncode:
        raise WaiterError(f"nvidia-smi failed ({base.returncode}): {base.stderr.strip() or base.stdout.strip()}")
    gpu_query = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True)
    if gpu_query.returncode:
        raise WaiterError(f"nvidia-smi GPU query failed ({gpu_query.returncode}): {gpu_query.stderr.strip()}")
    proc_query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if proc_query.returncode:
        raise WaiterError(f"nvidia-smi process query failed ({proc_query.returncode}): {proc_query.stderr.strip()}")
    gpus = []
    for line in gpu_query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, name, used, free, total, util = [part.strip() for part in line.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "name": name, "used_mib": int(used), "free_mib": int(free), "total_mib": int(total), "utilization": int(util)})
    active = {line.split(",", 1)[0].strip() for line in proc_query.stdout.splitlines() if line.strip()}
    return gpus, active


def torch_probe(gpu: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = (
        "import json, torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; "
        "cap=torch.cuda.get_device_capability(0); name=torch.cuda.get_device_name(0); "
        "assert cap != (12, 0), 'Blackwell sm_120 is not supported for this run'; "
        "x=torch.ones(64, device='cuda'); y=(x*x).sum(); torch.cuda.synchronize(); assert float(y)==64.0; "
        "print(json.dumps({'name': name, 'capability': cap, 'probe_sum': float(y)}))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if result.returncode:
        raise WaiterError(f"CUDA probe failed for GPU {gpu}: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def idle_supported_gpus(state_root: Path, *, max_utilization: int, max_used_mib: int) -> list[dict[str, Any]]:
    gpus, active = parse_gpu_query()
    result = []
    for gpu in gpus:
        if gpu["uuid"] in active:
            continue
        if gpu["utilization"] > max_utilization or gpu["used_mib"] >= max_used_mib:
            continue
        if "blackwell" in gpu["name"].lower() or gpu["index"] == 6:
            continue
        if lock_path(state_root, f"gpu-{gpu['index']}.lock").exists():
            continue
        result.append(gpu)
    return sorted(result, key=lambda item: item["free_mib"], reverse=True)


def paths_for(method: str, *, smoke: bool, source_manifest_sha: str, attack_hash: str) -> Paths:
    method_l = method.lower()
    if smoke:
        root = Path(f"/tmp/raven-{method_l}-issue17-smoke")
        gen = root / "generation" / method
        formal = root / "formal"
        bundle = root / "bundle"
    else:
        gen = method_data_root(method) / "diffusiondb_shared_tr" / method
        run_key = formal_run_key(source_manifest_sha, attack_hash)
        formal = formal_output_root(method, "diffusiondb_shared_tr", "formal", run_key)
        bundle = gen / "bundle"
    return Paths(generation=gen, formal=formal, metadata=gen / "metadata.csv", bundle=bundle)


def csv_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.is_file() else 0


def stage_complete(root: Path, stage: str, expected_count: int) -> bool:
    config = root / "run_config.json"
    if stage == "snapshot":
        index = root / "snapshots/snapshot_index.jsonl"
        if not index.is_file():
            return False
        return sum(int(json.loads(line)["row_count"]) for line in index.read_text().splitlines() if line) == expected_count
    if not config.is_file():
        return False
    cfg = json.loads(config.read_text(encoding="utf-8"))
    qhash = cfg["quality_config_hash"]
    ahash = cfg["attack_config_hash"]
    if stage == "attack-watermarked":
        return all((root / "attack_cache" / ahash / str(i) / "watermarked" / "record.json").is_file() for i in range(expected_count))
    if stage == "verify":
        return csv_count(root / "verification/scores.csv") == expected_count
    if stage == "quality":
        return line_count(root / "metrics/quality" / qhash / "quality_records.jsonl") == expected_count
    if stage == "fid":
        return (root / "metrics/fid" / qhash / "fid_result.json").is_file()
    if stage == "clip":
        return (root / "metrics/clip" / qhash / "clip_result.json").is_file()
    if stage == "aggregate":
        return (root / "formal_aggregate.json").is_file()
    if stage == "validate":
        valid = root / "VALIDATED.json"
        return valid.is_file() and json.loads(valid.read_text(encoding="utf-8")).get("N") == expected_count
    raise ValueError(stage)


def generation_command(method: str, paths: Paths, args: argparse.Namespace, gpu: int, *, smoke: bool) -> list[str]:
    command = [
        sys.executable, str(GENERATOR[method]), "--tr-metadata", str(args.tr_metadata), "--output-dir", str(paths.generation),
        f"--{method.lower()}-bundle-dir", str(paths.bundle), f"--{method.lower()}-create-bundle", "true", "--device", "cuda", "--gpu", str(gpu),
        "--require-free-gpu", "true", "--min-cpu-mem-gb", str(args.min_cpu_mem_gb), "--warn-cpu-mem-gb", str(args.warn_cpu_mem_gb),
        "--max-process-ram-gb", str(args.max_process_ram_gb),
    ]
    if smoke:
        command.extend(["--run-ids", "0", "1", "--smoke-only", "true"])
    if paths.metadata.exists():
        command.append("--resume")
    return command


def formal_command(method: str, paths: Paths, args: argparse.Namespace, gpu: int, stage: str, count: int) -> list[str]:
    command = [
        sys.executable, str(FORMAL_RUNNER), "--dataset", "diffusiondb_shared_tr", "--method", method, "--source-metadata", str(paths.metadata),
        "--source-manifest", str(args.source_manifest), "--output-root", str(paths.formal), "--expected-count", str(count), "--batch-size", str(args.batch_size),
        "--device", "cuda", "--dtype", "float16", "--gpu", str(gpu), "--verify-min-cpu-mem-gb", str(args.min_cpu_mem_gb),
        "--verify-warn-cpu-mem-gb", str(args.warn_cpu_mem_gb), "--verify-max-process-ram-gb", str(args.max_process_ram_gb), "--stage", stage,
    ]
    if (paths.formal / "run_config.json").exists():
        command.append("--resume")
    return command


def audit_command(args: argparse.Namespace, output: Path, paths: dict[str, Paths], *, smoke: bool) -> list[str]:
    command = [sys.executable, str(AUDIT_SCRIPT), "--tr-metadata", str(args.tr_metadata), "--output", str(output)]
    for method in METHODS:
        command.extend([f"--{method.lower()}-metadata", str(paths[method].metadata)])
    if smoke:
        command.extend(["--expected-run-ids", "0", "1"])
    else:
        command.append("--expect-full-tr-cohort")
    return command


def run_pipeline(method: str, args: argparse.Namespace, gpu: int, *, smoke: bool) -> Paths:
    source_sha = sha256_path(args.source_manifest)
    attack_hash = formal_attack_config_hash(load_formal_attack_config(None))
    paths = paths_for(method, smoke=smoke, source_manifest_sha=source_sha, attack_hash=attack_hash)
    count = 2 if smoke else args.expected_count
    log = args.state_root / "logs" / f"{method.lower()}-{'smoke' if smoke else 'formal'}.log"
    with AtomicLock(lock_path(args.state_root, f"method-{method.lower()}.lock"), {"method": method, "smoke": smoke}):
        append_event(args.state_root, f"METHOD START method={method} mode={'smoke' if smoke else 'formal'} gpu={gpu}")
        append_event(args.state_root, f"STAGE START method={method} stage=generation")
        run(generation_command(method, paths, args, gpu, smoke=smoke), log)
        if csv_count(paths.metadata) != count:
            raise WaiterError(f"{method} metadata row count mismatch at {paths.metadata}: expected {count}, found {csv_count(paths.metadata)}")
        append_event(args.state_root, f"STAGE OK method={method} stage=generation")
        for stage in STAGES:
            append_event(args.state_root, f"STAGE START method={method} stage={stage}")
            if not stage_complete(paths.formal, stage, count):
                run(formal_command(method, paths, args, gpu, stage, count), log)
            if not stage_complete(paths.formal, stage, count):
                raise WaiterError(f"{method} stage did not validate as complete: {stage} root={paths.formal}")
            marker = paths.formal / "waiter_state" / f"{stage}.complete"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"completed_utc={utc()}\n", encoding="utf-8")
            append_event(args.state_root, f"STAGE OK method={method} stage={stage}")
        append_event(args.state_root, f"STAGE START method={method} stage=update-table")
        run([sys.executable, str(TABLE_UPDATER), "--run-root", str(paths.formal)], log)
        append_event(args.state_root, f"TABLE UPDATED method={method} root={paths.formal}")
        append_event(args.state_root, f"METHOD OK method={method} mode={'smoke' if smoke else 'formal'}")
    return paths


def worker(args: argparse.Namespace) -> int:
    gpu_lock = Path(args.gpu_lock_file)
    try:
        paths = run_pipeline(args.worker_method, args, args.worker_gpu, smoke=args.worker_mode == "smoke")
        atomic_json(args.state_root / "status" / f"{args.worker_method.lower()}-{args.worker_mode}.json", {"status": "ok", "method": args.worker_method, "mode": args.worker_mode, "gpu": args.worker_gpu, "paths": {k: str(v) for k, v in paths.__dict__.items()}, "finished_utc": utc()})
        return 0
    except Exception as exc:
        append_event(args.state_root, f"STAGE FAILED rc=1 method={args.worker_method} mode={args.worker_mode} error={exc}")
        atomic_json(args.state_root / "status" / f"{args.worker_method.lower()}-{args.worker_mode}.json", {"status": "failed", "method": args.worker_method, "mode": args.worker_mode, "gpu": args.worker_gpu, "error": str(exc), "finished_utc": utc()})
        return 1
    finally:
        try:
            gpu_lock.unlink()
        except FileNotFoundError:
            pass
        append_event(args.state_root, f"GPU RELEASED gpu={args.worker_gpu} method={args.worker_method}")


def write_waiter_state(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    atomic_json(args.state_root / "waiter_state.json", {"updated_utc": utc(), **payload})


def wait_for_gpu(args: argparse.Namespace, reserved: set[int]) -> tuple[int, Path] | None:
    if available_ram_gib() < args.min_cpu_mem_gb:
        return None
    candidates = idle_supported_gpus(args.state_root, max_utilization=args.max_utilization, max_used_mib=args.max_used_mib)
    for candidate in candidates:
        gpu = int(candidate["index"])
        if gpu in reserved:
            continue
        append_event(args.state_root, f"GPU AVAILABLE gpu={gpu} free_mib={candidate['free_mib']} util={candidate['utilization']}")
        try:
            probe = torch_probe(gpu)
            latest = [item for item in idle_supported_gpus(args.state_root, max_utilization=args.max_utilization, max_used_mib=args.max_used_mib) if int(item["index"]) == gpu]
            if not latest:
                continue
            gpu_lock = lock_path(args.state_root, f"gpu-{gpu}.lock")
            claim = AtomicLock(gpu_lock, {"gpu": gpu, "probe": probe})
            claim.__enter__()
            claim.close_without_unlink()
            append_event(args.state_root, f"GPU CLAIMED gpu={gpu} name={probe['name']} capability={probe['capability']}")
            return gpu, gpu_lock
        except (FileExistsError, WaiterError):
            continue
    return None


def spawn_worker(args: argparse.Namespace, method: str, mode: str, gpu: int, gpu_lock: Path) -> subprocess.Popen:
    log = args.state_root / "logs" / f"{method.lower()}-{mode}.worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    command = [
        sys.executable, str(Path(__file__).resolve()), "--state-root", str(args.state_root), "--tr-metadata", str(args.tr_metadata),
        "--source-manifest", str(args.source_manifest), "--expected-count", str(args.expected_count), "--batch-size", str(args.batch_size),
        "--poll-seconds", str(args.poll_seconds), "--worker-method", method, "--worker-mode", mode, "--worker-gpu", str(gpu),
        "--gpu-lock-file", str(gpu_lock), "--min-cpu-mem-gb", str(args.min_cpu_mem_gb), "--warn-cpu-mem-gb", str(args.warn_cpu_mem_gb),
        "--max-process-ram-gb", str(args.max_process_ram_gb),
    ]
    process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(args.state_root / "status" / f"{method.lower()}-{mode}.json", {"status": "running", "pid": process.pid, "gpu": gpu, "started_utc": utc()})
    return process


def supervise_group(args: argparse.Namespace, methods: list[str], mode: str) -> dict[str, Paths]:
    pending = list(methods)
    active: dict[str, tuple[subprocess.Popen, int]] = {}
    completed: dict[str, Paths] = {}
    while pending or active:
        if (args.state_root / "STOP_WAITER").exists():
            write_waiter_state(args, {"status": "stopped_waiter_only", "active": {k: {"pid": p.pid, "gpu": g} for k, (p, g) in active.items()}, "pending": pending})
            return completed
        for method, (process, gpu) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            del active[method]
            if rc:
                append_event(args.state_root, f"SMOKE FAILED rc={rc} method={method}" if mode == "smoke" else f"STAGE FAILED rc={rc} method={method}")
                raise WaiterError(f"{method} {mode} worker failed rc={rc}")
            status = json.loads((args.state_root / "status" / f"{method.lower()}-{mode}.json").read_text(encoding="utf-8"))
            completed[method] = Paths(**{key: Path(value) for key, value in status["paths"].items()})
            if mode == "smoke":
                append_event(args.state_root, f"SMOKE OK method={method}")
        while pending:
            got = wait_for_gpu(args, {gpu for _, gpu in active.values()})
            if got is None:
                break
            gpu, gpu_lock = got
            method = pending.pop(0)
            append_event(args.state_root, f"SMOKE START method={method} gpu={gpu}" if mode == "smoke" else f"FULL RUN START method={method} gpu={gpu}")
            active[method] = (spawn_worker(args, method, mode, gpu, gpu_lock), gpu)
        write_waiter_state(args, {"status": f"{mode}_running" if active else "waiting_for_idle_gpu", "pending": pending, "active": {k: {"pid": p.pid, "gpu": g} for k, (p, g) in active.items()}, "completed": sorted(completed)})
        if pending or active:
            time.sleep(args.poll_seconds)
    return completed


def parent(args: argparse.Namespace) -> int:
    args.state_root.mkdir(parents=True, exist_ok=True)
    if lock_path(args.state_root, "waiter.lock").exists():
        raise WaiterError(f"existing waiter lock: {lock_path(args.state_root, 'waiter.lock')}")
    with AtomicLock(lock_path(args.state_root, "waiter.lock"), {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD")}):
        append_event(args.state_root, f"WAITER START pid={os.getpid()} head={git('rev-parse', 'HEAD')}")
        smoke_paths = supervise_group(args, list(METHODS), "smoke")
        if set(smoke_paths) != set(METHODS):
            raise WaiterError("waiter stopped before all smokes completed")
        run(audit_command(args, args.state_root / "audit" / "smoke_cross_method.json", smoke_paths, smoke=True), args.state_root / "logs/cross-audit-smoke.log")
        formal_paths = supervise_group(args, list(METHODS), "formal")
        if set(formal_paths) != set(METHODS):
            raise WaiterError("waiter stopped before all formal runs completed")
        run(audit_command(args, args.state_root / "audit" / "formal_cross_method.json", formal_paths, smoke=False), args.state_root / "logs/cross-audit-formal.log")
        summary = {"status": "all_methods_ok", "completed_utc": utc(), "frozen_commit_sha": git("rev-parse", "HEAD"), "smoke": {m: {k: str(v) for k, v in p.__dict__.items()} for m, p in smoke_paths.items()}, "formal": {m: {k: str(v) for k, v in p.__dict__.items()} for m, p in formal_paths.items()}, "tables": {m: str(REPO / "outputs" / m.lower() / "diffusiondb_shared_tr" / "_table" / "experiment_results.md") for m in METHODS}}
        atomic_json(args.state_root / "summary.json", summary)
        append_event(args.state_root, "ALL METHODS OK")
    return 0


def status(args: argparse.Namespace) -> int:
    for path in (args.state_root / "waiter_state.json", args.state_root / "summary.json"):
        if path.is_file():
            print(path.read_text(encoding="utf-8"), end="")
            return 0
    print(f"no waiter state at {args.state_root}")
    return 1


def stop_waiter(args: argparse.Namespace) -> int:
    (args.state_root / "STOP_WAITER").write_text(f"requested_utc={utc()}\n", encoding="utf-8")
    print(f"wrote {args.state_root / 'STOP_WAITER'}")
    return 0


def stop_method(args: argparse.Namespace) -> int:
    if not args.method:
        raise SystemExit("--method is required with --stop-method")
    found = False
    for mode in ("smoke", "formal"):
        path = args.state_root / "status" / f"{args.method.lower()}-{mode}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = payload.get("pid")
        if payload.get("status") == "running" and pid:
            os.killpg(int(pid), signal.SIGTERM)
            print(f"sent SIGTERM to {args.method} {mode} process group {pid}")
            found = True
    return 0 if found else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-root", type=Path, default=STATE_ROOT)
    p.add_argument("--tr-metadata", type=Path, default=DEFAULT_TR_METADATA)
    p.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    p.add_argument("--expected-count", type=int, default=1001)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-utilization", type=int, default=5)
    p.add_argument("--max-used-mib", type=int, default=1024)
    p.add_argument("--min-cpu-mem-gb", type=float, default=64.0)
    p.add_argument("--warn-cpu-mem-gb", type=float, default=80.0)
    p.add_argument("--max-process-ram-gb", type=float, default=16.0)
    p.add_argument("--worker-method", choices=METHODS, default=None)
    p.add_argument("--worker-mode", choices=["smoke", "formal"], default=None)
    p.add_argument("--worker-gpu", type=int, default=None)
    p.add_argument("--gpu-lock-file", type=Path, default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop-waiter", action="store_true")
    p.add_argument("--stop-method", action="store_true")
    p.add_argument("--method", choices=METHODS, default=None)
    return p


def main() -> int:
    args = parser().parse_args()
    args.state_root = args.state_root.resolve()
    args.tr_metadata = args.tr_metadata.resolve()
    args.source_manifest = args.source_manifest.resolve()
    if args.status:
        return status(args)
    if args.stop_waiter:
        return stop_waiter(args)
    if args.stop_method:
        return stop_method(args)
    if args.worker_method:
        if args.worker_gpu is None or args.gpu_lock_file is None or args.worker_mode is None:
            raise SystemExit("worker requires --worker-gpu, --worker-mode and --gpu-lock-file")
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
