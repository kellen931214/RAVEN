#!/usr/bin/env python3
"""Incrementally snapshot and run only the formal RAVEN evaluation entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = REPO / "experiments" / "run_raven_formal_eval.py"
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import formal_attack_config_hash  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--datasets", nargs="+", required=True)
    result.add_argument("--methods", nargs="+", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--poll-seconds", type=int, default=60)
    result.add_argument("--formal-runner", type=Path, default=DEFAULT_RUNNER)
    result.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO / "audit" / "formal_source_manifest.json",
    )
    result.add_argument("--source-template", default="data/watermarked/{dataset}/{method}/metadata.csv")
    result.add_argument("--output-root", type=Path, default=Path("outputs/raven_formal_eval"))
    result.add_argument("--expected-count", action="append", default=[], metavar="DATASET=COUNT")
    result.add_argument("--device", default="cuda")
    result.add_argument("--gpu", default=None)
    result.add_argument("--once", action="store_true", help="Poll once, useful for supervised schedulers")
    return result


def expected_counts(values: list[str]) -> dict[str, int]:
    result = {"diffusiondb": 1001, "mscoco": 1000}
    for value in values:
        dataset, count = value.split("=", 1)
        result[dataset] = int(count)
    return result


def count_snapshots(root: Path) -> int:
    index = root / "snapshots" / "snapshot_index.jsonl"
    if not index.is_file():
        return 0
    return sum(
        int(json.loads(line)["row_count"])
        for line in index.read_text(encoding="utf-8").splitlines()
        if line
    )


class CohortLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(f"cohort is already locked: {self.path}") from exc
        os.write(self.fd, f"{os.getpid()}\n".encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink()


def run_stage(args, dataset: str, method: str, source: Path, output: Path, count: int, stage: str) -> None:
    command = [
        sys.executable, str(args.formal_runner.resolve()), "--dataset", dataset,
        "--method", method, "--source-metadata", str(source.resolve()),
        "--output-root", str(output.resolve()), "--expected-count", str(count),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--source-manifest", str(args.source_manifest.resolve()),
        "--stage", stage, "--resume",
    ]
    if args.gpu is not None:
        command.extend(["--gpu", str(args.gpu)])
    subprocess.run(command, cwd=REPO, check=True)


def process_cohort(args, dataset: str, method: str, count: int) -> None:
    source = (REPO / args.source_template.format(dataset=dataset, method=method)).resolve()
    if not source.is_file():
        print(f"waiting: missing source metadata {source}", flush=True)
        return
    output = (args.output_root / dataset / method).resolve()
    lock_key = hashlib.sha256(
        f"{dataset}|{method}|{source}|{formal_attack_config_hash()}".encode()
    ).hexdigest()[:16]
    with CohortLock(args.output_root / "locks" / f"{dataset}_{method}_{lock_key}.lock"):
        run_stage(args, dataset, method, source, output, count, "snapshot")
        run_stage(args, dataset, method, source, output, count, "attack-watermarked")
        if method == "TR":
            run_stage(args, dataset, method, source, output, count, "attack-clean")
        completed = count_snapshots(output)
        if completed < count:
            print(f"preliminary only: {dataset}/{method} immutable={completed}/{count}", flush=True)
            return
        if (output / "VALIDATED.json").is_file():
            print(f"validated: {dataset}/{method} N={completed}", flush=True)
            return
        for stage in ("verify", "quality", "fid", "clip", "aggregate", "validate"):
            marker = output / "waiter_state" / f"{stage}.complete"
            if marker.is_file():
                continue
            run_stage(args, dataset, method, source, output, count, stage)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"completed_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


def main() -> int:
    args = parser().parse_args()
    if args.batch_size <= 0 or args.poll_seconds <= 0:
        raise ValueError("batch-size and poll-seconds must be positive")
    if args.formal_runner.resolve() != DEFAULT_RUNNER.resolve():
        raise ValueError("formal waiter may only invoke experiments/run_raven_formal_eval.py")
    counts = expected_counts(args.expected_count)
    while True:
        for dataset in args.datasets:
            if dataset not in counts:
                raise ValueError(f"missing --expected-count for {dataset}")
            for method in args.methods:
                process_cohort(args, dataset, method, counts[dataset])
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
