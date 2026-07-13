#!/usr/bin/env python3
"""Wait for an idle compatible GPU, then launch full MS-COCO RAVEN eval."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path('/workspace')
RAVEN_ROOT = WORKSPACE / 'raven_repro'
sys.path.insert(0, str(RAVEN_ROOT))
from raven.gpu_utils import query_gpu_status

LOG_DIR = RAVEN_ROOT / 'logs'
LOG_FILE = LOG_DIR / 'raven_eval_mscoco_waiter.log'
PID_FILE = LOG_DIR / 'raven_eval_mscoco_waiter.pid'


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f'[{now()}] {message}'
    with LOG_FILE.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
        fh.flush()
    print(line, flush=True)


def is_compatible(gpu):
    return 'blackwell' not in gpu.name.lower()


def idle_gpu_ids():
    status = query_gpu_status()
    if status.get('error'):
        log(f"WARNING nvidia-smi failed: {status['error']}")
        return []
    gpus = [gpu for gpu in status.get('gpus') or [] if is_compatible(gpu)]
    idle = [gpu for gpu in gpus if gpu.is_idle]
    idle = sorted(idle, key=lambda gpu: (gpu.memory_used_mib, gpu.utilization_gpu_percent, int(gpu.index)))
    return [gpu.index for gpu in idle]


def main():
    PID_FILE.write_text(str(os.getpid()) + '\n', encoding='utf-8')
    poll_seconds = int(os.environ.get('RAVEN_EVAL_WAIT_POLL_SECONDS', '60'))
    log('waiter_start target=mscoco wm_types=GS,TR,RID,HSTR,HSQR')
    while True:
        ids = idle_gpu_ids()
        if ids:
            gpu_id = ids[0]
            log(f'idle_gpu_found gpu={gpu_id}; launching eval')
            env = os.environ.copy()
            env.update({
                'TQDM_DISABLE': '1',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false',
            })
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.pop('RAVEN_CUDA_VISIBLE_DEVICES_APPLIED', None)
            cmd = [
                'ionice', '-c2', '-n7', 'nice', '-n', '10',
                sys.executable, str(WORKSPACE / 'experiments' / 'run_raven_eval_from_watermarked.py'),
                '--dataset_name', 'mscoco',
                '--watermarked_dir', str(WORKSPACE / 'data' / 'watermarked' / 'mscoco'),
                '--output_dir', str(WORKSPACE / 'outputs' / 'raven_eval'),
                '--wm_types', 'GS', 'TR', 'RID', 'HSTR', 'HSQR',
                '--num_pairs', '1000',
                '--model_id', 'RedbeardNZ/stable-diffusion-2-1-base',
                '--scheduler_target', 'DDIM',
                '--num_inference_steps_target', '50',
                '--resolution', '512',
                '--threshold_mode', 'eval_bench_wm',
                '--raven_steps', '50',
                '--raven_strength', '0.15',
                '--raven_guidance_scale', '2.5',
                '--shift_min', '24',
                '--shift_max', '32',
                '--shift_sign', 'random',
                '--shift_space', 'image_pixels',
                '--padding_mode', 'reflection',
                '--view_guided_attention', 'true',
                '--color_transfer', 'true',
                '--compute_clip', 'true',
                '--compute_fid', 'true',
                '--device', 'cuda',
                '--gpu', gpu_id,
                '--require_free_gpu', 'false',
                '--min_cpu_mem_gb', '64',
                '--warn_cpu_mem_gb', '96',
                '--max_process_ram_gb', '16',
            ]
            out = LOG_DIR / 'raven_eval_mscoco.log'
            with out.open('a', encoding='utf-8') as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(WORKSPACE), env=env)
                (LOG_DIR / 'raven_eval_mscoco.pid').write_text(str(proc.pid) + '\n', encoding='utf-8')
                log(f'eval_started pid={proc.pid} gpu={gpu_id} log={out}')
                rc = proc.wait()
                log(f'eval_finished pid={proc.pid} rc={rc}')
                if rc == 0:
                    table_cmd = [
                        sys.executable,
                        str(WORKSPACE / 'experiments' / 'build_raven_eval_table.py'),
                        '--eval_dir', str(WORKSPACE / 'outputs' / 'raven_eval' / 'mscoco'),
                    ]
                    with out.open('a', encoding='utf-8') as table_fh:
                        table_rc = subprocess.call(table_cmd, stdout=table_fh, stderr=subprocess.STDOUT, cwd=str(WORKSPACE), env=env)
                    log(f'table_finished rc={table_rc} md=/workspace/outputs/raven_eval/mscoco/eval_summary_table.md csv=/workspace/outputs/raven_eval/mscoco/eval_summary_table.csv')
                    return table_rc
                return rc
        log(f'waiting_for_idle_gpu poll_seconds={poll_seconds}')
        time.sleep(poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
