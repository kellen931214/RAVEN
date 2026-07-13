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

MODEL_ID="RedbeardNZ/stable-diffusion-2-1-base"
HEIGHT=512
WIDTH=512
STEPS=50
GUIDANCE_SCALE=7.5
SCHEDULER="ddim"
DEVICE="cuda"
GPU="auto"
DTYPE="fp16"
BATCH_SIZE=1
MIN_CPU_MEM_GB=64
WARN_CPU_MEM_GB=96
MAX_PROCESS_RAM_GB=16

RAVEN_STRENGTH=0.15
RAVEN_GUIDANCE_SCALE=2.5
RAVEN_SHIFT_MIN=24
RAVEN_SHIFT_MAX=32
RAVEN_SHIFT_SIGN="random"
RAVEN_SHIFT_SPACE="image_pixels"
RAVEN_PADDING_MODE="reflection"
RAVEN_VIEW_GUIDED_ATTENTION="true"
RAVEN_COLOR_TRANSFER="true"
RAVEN_SEED=42

generate_dataset() {
  local prompts_csv="$1"
  local output_dir="$2"
  python experiments/generate_images.py \
    --prompts_csv "$prompts_csv" \
    --output_dir "$output_dir" \
    --model_id "$MODEL_ID" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --steps "$STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --scheduler "$SCHEDULER" \
    --device "$DEVICE" \
    --gpu "$GPU" \
    --require_free_gpu true \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --min_cpu_mem_gb "$MIN_CPU_MEM_GB" \
    --max_process_ram_gb "$MAX_PROCESS_RAM_GB" \
    --warn_cpu_mem_gb "$WARN_CPU_MEM_GB"
}

run_raven_dataset() {
  local input_dir="$1"
  local output_dir="$2"
  python experiments/run_raven_experiments.py \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --model_id "$MODEL_ID" \
    --steps "$STEPS" \
    --strength "$RAVEN_STRENGTH" \
    --guidance_scale "$RAVEN_GUIDANCE_SCALE" \
    --shift_min "$RAVEN_SHIFT_MIN" \
    --shift_max "$RAVEN_SHIFT_MAX" \
    --shift_sign "$RAVEN_SHIFT_SIGN" \
    --shift_space "$RAVEN_SHIFT_SPACE" \
    --padding_mode "$RAVEN_PADDING_MODE" \
    --view_guided_attention "$RAVEN_VIEW_GUIDED_ATTENTION" \
    --color_transfer "$RAVEN_COLOR_TRANSFER" \
    --seed "$RAVEN_SEED" \
    --device "$DEVICE" \
    --gpu "$GPU" \
    --require_free_gpu true \
    --min_cpu_mem_gb "$MIN_CPU_MEM_GB" \
    --max_process_ram_gb "$MAX_PROCESS_RAM_GB" \
    --warn_cpu_mem_gb "$WARN_CPU_MEM_GB"
}

# Paper datasets: 5,000 MS-COCO captions, 1,001 DiffusionDB prompts, 8,192 SD-Prompts.
generate_dataset data/prompts/mscoco_5000.csv data/generated/mscoco
run_raven_dataset data/generated/mscoco outputs/raven/mscoco

generate_dataset data/prompts/diffusiondb_1001.csv data/generated/diffusiondb
run_raven_dataset data/generated/diffusiondb outputs/raven/diffusiondb

generate_dataset data/prompts/sd_prompts_8192.csv data/generated/sd_prompts
run_raven_dataset data/generated/sd_prompts outputs/raven/sd_prompts
