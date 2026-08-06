"""GPU selection and experiment output helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IDLE_MEMORY_MIB = 256
IDLE_UTILIZATION_PERCENT = 10


@dataclass
class GpuProcess:
    gpu_uuid: str
    pid: str
    process_name: str
    used_memory_mib: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_uuid": self.gpu_uuid,
            "pid": self.pid,
            "process_name": self.process_name,
            "used_memory_mib": self.used_memory_mib,
        }


@dataclass
class GpuInfo:
    index: str
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu_percent: int
    active_processes: List[GpuProcess] = field(default_factory=list)

    @property
    def is_idle(self) -> bool:
        return (
            self.memory_used_mib <= IDLE_MEMORY_MIB
            and self.utilization_gpu_percent <= IDLE_UTILIZATION_PERCENT
            and not self.active_processes
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "utilization_gpu_percent": self.utilization_gpu_percent,
            "is_idle": self.is_idle,
            "active_processes": [process.to_dict() for process in self.active_processes],
        }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class _TeeStream:
    """Line-buffered mirror used to retain both parent and child logs."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(bool(getattr(stream, "isatty", lambda: False)()) for stream in self.streams)


def setup_run_logging(
    output_dir: str | os.PathLike[str],
    filename: str = "run.log",
    *,
    mirror_console: bool = True,
) -> Path:
    """Append stdout/stderr to a run log and optionally mirror inherited output."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / filename
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    if mirror_console:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, stream)
        sys.stderr = _TeeStream(original_stderr, stream)
    else:
        sys.stdout = stream
        sys.stderr = stream
    return log_path


def _run_command(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _parse_mib(value: str) -> Optional[int]:
    value = value.strip()
    if value in {"", "[Not Supported]", "[N/A]", "N/A"}:
        return None
    if value.lower().endswith("mib"):
        value = value[:-3].strip()
    try:
        return int(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int:
    stripped = value.strip()
    if stripped in {"", "[Not Supported]", "[N/A]", "N/A"}:
        return 0
    return int(stripped)


def _parse_csv_lines(text: str) -> Iterable[List[str]]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        yield [part.strip() for part in line.split(",")]


def query_gpu_status() -> Dict[str, Any]:
    """Inspect GPU status using nvidia-smi and return raw plus parsed data."""
    timestamp = utc_timestamp()
    raw = _run_command(["nvidia-smi"])
    status: Dict[str, Any] = {
        "timestamp": timestamp,
        "raw_nvidia_smi": raw.stdout,
        "raw_nvidia_smi_stderr": raw.stderr,
        "nvidia_smi_returncode": raw.returncode,
        "gpus": [],
        "error": None,
    }
    if raw.returncode != 0:
        status["error"] = raw.stderr.strip() or "nvidia-smi failed"
        return status

    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query.returncode != 0:
        status["error"] = gpu_query.stderr.strip() or "nvidia-smi GPU query failed"
        return status

    processes_by_uuid: Dict[str, List[GpuProcess]] = {}
    process_query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if process_query.returncode == 0:
        for parts in _parse_csv_lines(process_query.stdout):
            if len(parts) < 4:
                continue
            process = GpuProcess(
                gpu_uuid=parts[0],
                pid=parts[1],
                process_name=parts[2],
                used_memory_mib=_parse_mib(parts[3]),
            )
            processes_by_uuid.setdefault(process.gpu_uuid, []).append(process)
    else:
        status["process_query_error"] = process_query.stderr.strip() or "nvidia-smi process query failed"

    gpus: List[GpuInfo] = []
    for parts in _parse_csv_lines(gpu_query.stdout):
        if len(parts) < 6:
            continue
        gpu = GpuInfo(
            index=parts[0],
            uuid=parts[1],
            name=parts[2],
            memory_used_mib=_parse_int(parts[3]),
            memory_total_mib=_parse_int(parts[4]),
            utilization_gpu_percent=_parse_int(parts[5]),
            active_processes=processes_by_uuid.get(parts[1], []),
        )
        gpus.append(gpu)

    status["gpus"] = gpus
    return status


def _is_torch_compatible_gpu(gpu: GpuInfo) -> bool:
    """Avoid GPUs known to be unsupported by this environment's torch build."""
    # The installed torch 2.5.1+cu124 build does not support Blackwell sm_120 here.
    # Do not import torch before CUDA_VISIBLE_DEVICES is set; that makes masking unreliable.
    return "blackwell" not in gpu.name.lower()


def get_free_gpu(status: Optional[Dict[str, Any]] = None) -> Optional[GpuInfo]:
    """Return an idle compatible GPU when possible, otherwise the least-used visible GPU."""
    status = status or query_gpu_status()
    gpus = list(status.get("gpus") or [])
    if not gpus:
        return None

    compatible_gpus = [gpu for gpu in gpus if _is_torch_compatible_gpu(gpu)]
    if not compatible_gpus:
        compatible_gpus = gpus

    idle_gpus = [gpu for gpu in compatible_gpus if gpu.is_idle]
    no_process_gpus = [gpu for gpu in compatible_gpus if not gpu.active_processes]
    candidates = idle_gpus or no_process_gpus or compatible_gpus
    return sorted(
        candidates,
        key=lambda gpu: (
            gpu.memory_used_mib,
            gpu.utilization_gpu_percent,
            len(gpu.active_processes),
            int(gpu.index),
        ),
    )[0]


def _format_gpu(gpu: GpuInfo) -> str:
    processes = gpu.active_processes
    process_text = "none"
    if processes:
        process_text = "; ".join(
            f"pid={proc.pid} mem={proc.used_memory_mib}MiB name={proc.process_name}" for proc in processes
        )
    return (
        f"GPU {gpu.index}: {gpu.name} | memory {gpu.memory_used_mib}/{gpu.memory_total_mib} MiB | "
        f"utilization {gpu.utilization_gpu_percent}% | active processes: {process_text}"
    )




def configure_gpu(
    gpu_arg: Optional[str],
    device: str,
    output_dir: str | os.PathLike[str],
    require_free_gpu: bool = False,
) -> Dict[str, Any]:
    """Apply --gpu selection, set CUDA_VISIBLE_DEVICES when explicit, and log status."""
    status_before = query_gpu_status()
    selected: Optional[GpuInfo] = None
    selected_gpu_id: Optional[str] = None
    all_busy = False

    wants_cuda = str(device).startswith("cuda")
    normalized_gpu_arg = None if gpu_arg is None else str(gpu_arg).strip()

    if wants_cuda and normalized_gpu_arg:
        if normalized_gpu_arg == "auto":
            selected = get_free_gpu(status_before)
            if selected is None:
                raise RuntimeError(f"Unable to select GPU automatically: {status_before.get('error') or 'no GPUs visible'}")
            selected_gpu_id = selected.index
            all_busy = not any(gpu.is_idle for gpu in status_before.get("gpus") or [])
            if all_busy and require_free_gpu:
                raise RuntimeError(
                    "All visible GPUs appear busy. Refusing to start a full experiment; wait for an idle GPU or choose one manually."
                )
            if all_busy:
                print(
                    "WARNING: All visible GPUs appear busy; using the least busy GPU for this run. "
                    "For full experiments, wait for an idle GPU or pass --require_free_gpu true.",
                    flush=True,
                )
        else:
            selected_gpu_id = normalized_gpu_arg
            for gpu in status_before.get("gpus") or []:
                if gpu.index == selected_gpu_id:
                    selected = gpu
                    break
        visible_gpu = selected.uuid if selected is not None and selected.uuid else selected_gpu_id
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        if os.environ.get("RAVEN_CUDA_VISIBLE_DEVICES_APPLIED") != visible_gpu:
            os.environ["RAVEN_CUDA_VISIBLE_DEVICES_APPLIED"] = visible_gpu
            print(
                f"Re-executing with CUDA_VISIBLE_DEVICES={visible_gpu} for selected GPU {selected_gpu_id}",
                flush=True,
            )
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ.copy())
        print(f"Using GPU {selected_gpu_id} via CUDA_VISIBLE_DEVICES={visible_gpu}", flush=True)
    elif wants_cuda and not normalized_gpu_arg:
        print(
            "WARNING: --device cuda was requested without --gpu; CUDA_VISIBLE_DEVICES was left unchanged.",
            flush=True,
        )

    return {
        "requested_gpu": normalized_gpu_arg,
        "selected_gpu_id": selected_gpu_id,
        "selected_gpu": selected.to_dict() if selected is not None else None,
        "all_gpus_busy": all_busy,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status_before": _serializable_status(status_before),
    }


def finalize_gpu_logging(output_dir: str | os.PathLike[str], gpu_record: Dict[str, Any]) -> Dict[str, Any]:
    status_after = query_gpu_status()
    gpu_record["status_after"] = _serializable_status(status_after)
    return gpu_record


def write_results_json(
    output_dir: str | os.PathLike[str],
    args: Dict[str, Any],
    gpu_record: Dict[str, Any],
    started_at: str,
    finished_at: str,
    status: str,
    extra_summary: Optional[Dict[str, Any]] = None,
    filename: str = "results.json",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "args": _json_safe(args),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "gpu": _json_safe(gpu_record),
    }
    if extra_summary:
        results.update(_json_safe(extra_summary))
    (out / filename).write_text(json.dumps(results, indent=2, sort_keys=True))


def _serializable_status(status: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(status)
    result["gpus"] = [gpu.to_dict() for gpu in status.get("gpus") or []]
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


write_experiment_records = write_results_json


# --------------------------------------------------------------------------- #
# CPU memory guard (generation orchestrators only)
# --------------------------------------------------------------------------- #
_KIB_PER_GIB = 1024 * 1024


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
            total_gib=total_kib / _KIB_PER_GIB,
            available_gib=available_kib / _KIB_PER_GIB,
            process_rss_gib=_current_rss_kib() / _KIB_PER_GIB,
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
