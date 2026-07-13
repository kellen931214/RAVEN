#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
LOG_DIR="/workspace/raven_repro/logs"
mkdir -p "${LOG_DIR}"

if [[ "${RAVEN_LOW_PRIORITY:-0}" != "1" ]]; then
  export RAVEN_LOW_PRIORITY=1
  exec ionice -c2 -n7 nice -n 10 bash "$0" "$@"
fi

export TQDM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

printf '[%s] RAVEN eval smoke: RAM before\n' "$(date -Iseconds)"
free -h
printf '[%s] RAVEN eval smoke: GPU before\n' "$(date -Iseconds)"
nvidia-smi

python "${ROOT_DIR}/experiments/run_raven_eval_from_watermarked.py" \
  --dataset_name mscoco \
  --watermarked_dir "${ROOT_DIR}/data/watermarked/mscoco" \
  --output_dir "${ROOT_DIR}/outputs/raven_eval_smoke" \
  --wm_types GS TR RID HSTR HSQR \
  --num_pairs 1 \
  --model_id RedbeardNZ/stable-diffusion-2-1-base \
  --scheduler_target DDIM \
  --num_inference_steps_target 50 \
  --resolution 512 \
  --threshold_mode eval_bench_wm \
  --raven_steps 50 \
  --raven_strength 0.15 \
  --raven_guidance_scale 2.5 \
  --shift_min 24 \
  --shift_max 32 \
  --shift_sign random \
  --shift_space image_pixels \
  --padding_mode reflection \
  --view_guided_attention true \
  --color_transfer true \
  --compute_clip true \
  --compute_fid true \
  --device cuda \
  --gpu auto \
  --require_free_gpu false \
  --min_cpu_mem_gb 64 \
  --warn_cpu_mem_gb 96 \
  --max_process_ram_gb 16
