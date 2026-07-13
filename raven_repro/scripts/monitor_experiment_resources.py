#!/usr/bin/env python3
import os
import signal
import time
import os
from datetime import datetime, timezone
from pathlib import Path

PID_FILE = Path(os.environ.get('PID_FILE', '/workspace/raven_repro/logs/full_experiment.pid'))
LOG_FILE = Path(os.environ.get('LOG_FILE', '/workspace/raven_repro/logs/resource_monitor.log'))
MIN_AVAILABLE_GIB = float(os.environ.get('MIN_AVAILABLE_GIB', '32'))
WARN_AVAILABLE_GIB = os.environ.get('WARN_AVAILABLE_GIB')
WARN_AVAILABLE_GIB = float(WARN_AVAILABLE_GIB) if WARN_AVAILABLE_GIB is not None else None
MAX_TREE_RSS_GIB = float(os.environ.get('MAX_TREE_RSS_GIB', '40'))
MAX_TREE_CPU_PERCENT = float(os.environ.get('MAX_TREE_CPU_PERCENT', '0'))
WARN_TREE_CPU_PERCENT = os.environ.get('WARN_TREE_CPU_PERCENT')
WARN_TREE_CPU_PERCENT = float(WARN_TREE_CPU_PERCENT) if WARN_TREE_CPU_PERCENT is not None else None
INTERVAL_SECONDS = int(os.environ.get('INTERVAL_SECONDS', '30'))
KIB_PER_GIB = 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open('a', encoding='utf-8') as fh:
        fh.write(f'[{now()}] {message}\n')
        fh.flush()


def read_root_pid():
    try:
        return int(PID_FILE.read_text(encoding='utf-8').strip())
    except Exception as exc:
        log(f'ERROR cannot read pid file {PID_FILE}: {exc}')
        return None


def pid_exists(pid):
    return pid is not None and Path(f'/proc/{pid}').exists()


def process_name(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
    except Exception:
        return '<unknown>'


def all_ppids():
    mapping = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / 'stat').read_text(encoding='utf-8', errors='replace')
            after = stat.rsplit(')', 1)[1].strip().split()
            ppid = int(after[1])
            mapping.setdefault(ppid, []).append(int(proc.name))
        except Exception:
            continue
    return mapping


def process_tree(root):
    mapping = all_ppids()
    seen = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in seen or not pid_exists(pid):
            continue
        seen.add(pid)
        stack.extend(mapping.get(pid, []))
    return sorted(seen)


def proc_cpu_jiffies(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8', errors='replace')
        fields = stat.rsplit(')', 1)[1].strip().split()
        return int(fields[11]) + int(fields[12])
    except Exception:
        return 0


def rss_kib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def meminfo():
    data = {}
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            key, rest = line.split(':', 1)
            data[key] = int(rest.strip().split()[0])
    except Exception as exc:
        log(f'ERROR cannot read /proc/meminfo: {exc}')
    return data


def terminate_tree(pids, reason):
    log(f'GUARD_TRIGGERED reason={reason}; sending SIGTERM to experiment pids={pids}')
    for pid in sorted(pids, reverse=True):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f'SIGTERM sent pid={pid} cmd={process_name(pid)}')
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            log(f'ERROR no permission to terminate pid={pid}: {exc}')
    time.sleep(10)
    for pid in sorted(pids, reverse=True):
        if pid_exists(pid):
            log(f'WARNING pid still alive after SIGTERM pid={pid} cmd={process_name(pid)}')


def main():
    root = read_root_pid()
    log(f'monitor_start root_pid={root} min_available_gib={MIN_AVAILABLE_GIB} warn_available_gib={WARN_AVAILABLE_GIB} max_tree_rss_gib={MAX_TREE_RSS_GIB} max_tree_cpu_percent={MAX_TREE_CPU_PERCENT} warn_tree_cpu_percent={WARN_TREE_CPU_PERCENT} interval={INTERVAL_SECONDS}s')
    if not pid_exists(root):
        log('root process is not running; exiting')
        return 0
    clock_ticks = os.sysconf(os.sysconf_names.get('SC_CLK_TCK', 'SC_CLK_TCK'))
    previous_cpu_jiffies = None
    previous_time = None
    while True:
        if not pid_exists(root):
            log('root process finished; exiting')
            return 0
        pids = process_tree(root)
        info = meminfo()
        available_kib = info.get('MemAvailable', 0)
        total_kib = info.get('MemTotal', 0)
        tree_rss_kib = sum(rss_kib(pid) for pid in pids)
        tree_cpu_jiffies = sum(proc_cpu_jiffies(pid) for pid in pids)
        current_time = time.monotonic()
        tree_cpu_percent = None
        if previous_cpu_jiffies is not None and previous_time is not None:
            elapsed = max(current_time - previous_time, 1e-6)
            tree_cpu_percent = ((tree_cpu_jiffies - previous_cpu_jiffies) / clock_ticks) / elapsed * 100.0
        previous_cpu_jiffies = tree_cpu_jiffies
        previous_time = current_time
        available_gib = available_kib / KIB_PER_GIB
        total_gib = total_kib / KIB_PER_GIB
        tree_rss_gib = tree_rss_kib / KIB_PER_GIB
        cpu_text = 'n/a' if tree_cpu_percent is None else f'{tree_cpu_percent:.1f}'
        log(f'status total_gib={total_gib:.1f} available_gib={available_gib:.1f} tree_rss_gib={tree_rss_gib:.2f} tree_cpu_percent={cpu_text} pids={pids}')
        if WARN_AVAILABLE_GIB is not None and available_gib < WARN_AVAILABLE_GIB:
            log(f'WARNING MemAvailable {available_gib:.1f}GiB below warning threshold {WARN_AVAILABLE_GIB:.1f}GiB; continuing until hard stop {MIN_AVAILABLE_GIB:.1f}GiB')
        if available_gib < MIN_AVAILABLE_GIB:
            terminate_tree(pids, f'MemAvailable {available_gib:.1f}GiB below {MIN_AVAILABLE_GIB:.1f}GiB')
            return 2
        if tree_rss_gib > MAX_TREE_RSS_GIB:
            terminate_tree(pids, f'experiment RSS {tree_rss_gib:.1f}GiB above {MAX_TREE_RSS_GIB:.1f}GiB')
            return 3
        if tree_cpu_percent is not None and WARN_TREE_CPU_PERCENT is not None and tree_cpu_percent > WARN_TREE_CPU_PERCENT:
            log(f'WARNING experiment CPU {tree_cpu_percent:.1f}% above warning threshold {WARN_TREE_CPU_PERCENT:.1f}%')
        if tree_cpu_percent is not None and MAX_TREE_CPU_PERCENT > 0 and tree_cpu_percent > MAX_TREE_CPU_PERCENT:
            terminate_tree(pids, f'experiment CPU {tree_cpu_percent:.1f}% above {MAX_TREE_CPU_PERCENT:.1f}%')
            return 4
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    raise SystemExit(main())
