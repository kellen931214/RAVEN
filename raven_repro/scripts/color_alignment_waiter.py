#!/usr/bin/env python
"""Wait for the formal image/attack chain, audit it, then launch Experiment 1 once."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.raven_color_alignment_experiment import audit_sources

FATAL_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"\bKilled\b"),
    re.compile(r"fatal error", re.IGNORECASE),
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_generation_pid(args) -> int | None:
    if args.generation_pid_file:
        path = args.generation_pid_file
        if not path.is_file():
            return None
        return int(path.read_text(encoding="utf-8").strip())
    return args.generation_pid


def read_completion_marker(path: Path) -> bool:
    return path.is_file()


def scan_log(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        return [], []
    fatal: list[str] = []
    tail: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip()
            tail.append(stripped)
            if len(tail) > 40:
                tail.pop(0)
            if any(pattern.search(stripped) for pattern in FATAL_PATTERNS):
                fatal.append(stripped)
    return fatal, tail


def find_partial_files(*roots: Path) -> list[str]:
    partial: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix.lower() in {".tmp", ".part", ".partial"} or path.name.startswith(".nfs")):
                partial.append(str(path))
                if len(partial) >= 20:
                    return partial
    return partial


def build_experiment_command(args) -> list[str]:
    return [
        str(args.python_executable),
        "-u",
        str(args.repo_root / "raven_repro/scripts/raven_color_alignment_experiment.py"),
        "--source-p1-dir", str(args.source_p1_dir),
        "--source-nfpa-dir", str(args.source_nfpa_dir),
        "--output-dir", str(args.experiment_output_dir),
        "--expected-count", str(args.expected_count),
        "--count", str(args.experiment_count),
        "--validation-count", str(args.validation_count),
        "--eval-repo", str(args.eval_repo),
        "--device", args.device,
        "--waiter-script", str(args.repo_root / "raven_repro/scripts/wait_for_images_then_run_color_alignment.sh"),
        "--min-cpu-mem-gb", str(args.min_cpu_mem_gb),
        "--warn-cpu-mem-gb", str(args.warn_cpu_mem_gb),
        "--max-process-ram-gb", str(args.max_process_ram_gb),
    ]


def launch_once(args) -> int:
    if (args.experiment_output_dir / "results.json").is_file():
        log("Experiment 1 already completed; refusing duplicate launch")
        return 0
    if args.experiment_output_dir.exists():
        raise FileExistsError(f"Experiment output already exists without final results: {args.experiment_output_dir}")

    command = build_experiment_command(args)
    args.experiment_output_dir.mkdir(parents=True, exist_ok=False)
    log_path = args.experiment_output_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
    time.sleep(1.0)
    if process.poll() is not None:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Experiment 1 exited immediately code={process.returncode}:\n{tail}")
    log(f"Experiment 1 launched pid={process.pid} log={log_path}")
    return process.pid


def one_iteration(args) -> str:
    fatal, tail = scan_log(args.generation_log)
    if fatal:
        raise RuntimeError(f"generation log contains fatal error: {fatal[-1]}")
    complete = read_completion_marker(args.completion_marker)
    pid = resolve_generation_pid(args)
    alive = pid_alive(pid)
    if not complete:
        if pid is not None and not alive:
            raise RuntimeError(f"generation process pid={pid} exited before completion marker {args.completion_marker}")
        log(f"waiting for image/attack chain pid={pid} alive={alive} marker={args.completion_marker}")
        return "waiting"
    if args.success_log_token and not any(args.success_log_token in line for line in tail):
        if alive:
            log(f"completion marker present but waiting for log token: {args.success_log_token}")
            return "waiting"
        raise RuntimeError(f"completion marker present but generation log lacks token: {args.success_log_token}")
    if alive:
        log(f"completion marker present; waiting for generation process pid={pid} to exit normally")
        return "waiting"

    partial = find_partial_files(args.source_p1_dir, args.source_nfpa_dir)
    if partial:
        raise RuntimeError(f"partial/temporary files found: {partial}")
    log("completion signal verified; auditing required images, records, debug JSON, hashes, and configs")
    audit = audit_sources(args.source_p1_dir, args.source_nfpa_dir, args.expected_count)
    log(f"source audit passed counts={audit['counts']}")
    launch_once(args)
    return "launched"


def run_waiter(args) -> int:
    args.repo_root = args.repo_root.resolve()
    for name in (
        "generation_log", "image_output_dir", "completion_marker",
        "source_p1_dir", "source_nfpa_dir", "experiment_output_dir",
        "eval_repo",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (args.repo_root / value).resolve())
    if not args.python_executable.is_absolute():
        raise ValueError("--python-executable must be an absolute path")
    if not args.repo_root.is_dir() or not args.generation_log.parent.is_dir():
        raise FileNotFoundError("repository root or generation log directory does not exist")
    while True:
        status = one_iteration(args)
        if status == "launched":
            return 0
        if args.check_once:
            return 2
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generation-pid", type=int)
    group.add_argument("--generation-pid-file", type=Path)
    parser.add_argument("--generation-log", type=Path, required=True)
    parser.add_argument("--image-output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--success-log-token", default="chain complete")
    parser.add_argument("--source-p1-dir", type=Path, required=True)
    parser.add_argument("--source-nfpa-dir", type=Path, required=True)
    parser.add_argument("--experiment-output-dir", type=Path, required=True)
    parser.add_argument("--eval-repo", type=Path, default=Path("eval_bench_wm"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-cpu-mem-gb", type=float, default=64.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=96.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
    parser.add_argument("--check-once", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_waiter(build_parser().parse_args()))
