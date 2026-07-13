#!/usr/bin/env bash
set -euo pipefail

if [ "${RAVEN_LOW_PRIORITY:-0}" != "1" ]; then
  export RAVEN_LOW_PRIORITY=1
  if command -v ionice >/dev/null 2>&1; then
    exec ionice -c2 -n7 nice -n 10 bash "$0" "$@"
  else
    exec nice -n 10 bash "$0" "$@"
  fi
fi

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

python experiments/generate_images.py \
  --prompts_csv data/prompts/smoke.csv \
  --output_dir data/generated/smoke \
  --model_id RedbeardNZ/stable-diffusion-2-1-base \
  --height 512 \
  --width 512 \
  --steps 10 \
  --guidance_scale 7.5 \
  --scheduler ddim \
  --device cuda \
  --gpu auto \
  --dtype fp16 \
  --batch_size 1 \
  --limit 2 \
  --min_cpu_mem_gb 32 \
  --max_process_ram_gb 16 \
  --warn_cpu_mem_gb 64

python experiments/run_raven_experiments.py \
  --input_dir data/generated/smoke \
  --output_dir outputs/raven/smoke \
  --model_id RedbeardNZ/stable-diffusion-2-1-base \
  --steps 10 \
  --strength 0.15 \
  --guidance_scale 2.5 \
  --shift_min 24 \
  --shift_max 32 \
  --shift_sign random \
  --shift_space image_pixels \
  --padding_mode reflection \
  --view_guided_attention true \
  --color_transfer true \
  --seed 42 \
  --device cuda \
  --gpu auto \
  --min_cpu_mem_gb 32 \
  --max_process_ram_gb 16 \
  --warn_cpu_mem_gb 64
