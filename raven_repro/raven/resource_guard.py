"""CPU memory safety checks for long-running experiments."""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from pathlib import Path

KIB_PER_GIB = 1024 * 1024


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0])
    except FileNotFoundError:
        pass
    return values


def _current_rss_kib() -> int:
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage)


def limit_cpu_threads(num_threads: int = 1) -> None:
    """Keep CPU-side libraries from fanning out across the shared server."""
    value = str(int(num_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(int(num_threads))
    except RuntimeError:
        # PyTorch only allows setting interop threads before parallel work starts.
        pass


@dataclass
class CpuMemorySnapshot:
    total_gib: float
    available_gib: float
    process_rss_gib: float


class CpuMemoryPressure(MemoryError):
    """Raised when hard CPU RAM limits are crossed."""


class CpuMemoryGuard:
    def __init__(
        self,
        min_available_gib: float = 16.0,
        max_process_rss_gib: float = 40.0,
        warn_available_gib: float | None = None,
    ):
        self.min_available_gib = float(min_available_gib)
        self.max_process_rss_gib = float(max_process_rss_gib)
        self.warn_available_gib = float(warn_available_gib) if warn_available_gib is not None else None
        if self.warn_available_gib is not None and self.warn_available_gib < self.min_available_gib:
            raise ValueError("warn_available_gib must be greater than or equal to min_available_gib")

    def snapshot(self) -> CpuMemorySnapshot:
        info = _read_meminfo()
        total_kib = info.get("MemTotal", 0)
        available_kib = info.get("MemAvailable", 0)
        return CpuMemorySnapshot(
            total_gib=total_kib / KIB_PER_GIB,
            available_gib=available_kib / KIB_PER_GIB,
            process_rss_gib=_current_rss_kib() / KIB_PER_GIB,
        )

    def check(self, label: str = "memory check") -> CpuMemorySnapshot:
        snap = self.snapshot()
        warn_text = ""
        if self.warn_available_gib is not None:
            warn_text = f" warn_at={self.warn_available_gib:.1f}GiB"
        print(
            f"CPU memory {label}: available={snap.available_gib:.1f}GiB/"
            f"{snap.total_gib:.1f}GiB process_rss={snap.process_rss_gib:.2f}GiB "
            f"hard_min={self.min_available_gib:.1f}GiB max_rss={self.max_process_rss_gib:.1f}GiB"
            f"{warn_text}",
            flush=True,
        )
        if snap.total_gib > 0 and snap.available_gib < self.min_available_gib:
            raise CpuMemoryPressure(
                f"Stopping before server RAM pressure: MemAvailable {snap.available_gib:.1f}GiB "
                f"is below hard stop {self.min_available_gib:.1f}GiB"
            )
        if snap.process_rss_gib > self.max_process_rss_gib:
            raise CpuMemoryPressure(
                f"Stopping before process memory leak hurts server: process RSS {snap.process_rss_gib:.1f}GiB "
                f"exceeds limit {self.max_process_rss_gib:.1f}GiB"
            )
        if (
            self.warn_available_gib is not None
            and snap.total_gib > 0
            and snap.available_gib < self.warn_available_gib
        ):
            print(
                f"WARNING: CPU RAM soft threshold crossed: MemAvailable {snap.available_gib:.1f}GiB "
                f"is below warning threshold {self.warn_available_gib:.1f}GiB; continuing until "
                f"hard stop {self.min_available_gib:.1f}GiB",
                flush=True,
            )
        return snap
