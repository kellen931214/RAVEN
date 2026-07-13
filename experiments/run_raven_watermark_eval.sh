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

MODEL_ID="${MODEL_ID:-RedbeardNZ/stable-diffusion-2-1-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/raven_watermark_eval}"
NUM_PAIRS="${NUM_PAIRS:-1000}"
WM_TYPES=(GS TR RID HSTR HSQR)

COMMON_ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --modelid_target "${MODEL_ID}"
  --raven_model_id "${MODEL_ID}"
  --resolution 512
  --scheduler_target DDIM
  --num_inference_steps_target 50
  --guidance_scale_target 7.5
  --raven_steps 50
  --raven_strength 0.15
  --raven_guidance_scale 2.5
  --shift_min 24
  --shift_max 32
  --shift_sign random
  --shift_space image_pixels
  --padding_mode reflection
  --view_guided_attention true
  --color_transfer true
  --seed 42
  --device cuda
  --gpu auto
  --require_free_gpu true
  --num_pairs "${NUM_PAIRS}"
  --wm_types "${WM_TYPES[@]}"
  --min_cpu_mem_gb 64
  --warn_cpu_mem_gb 96
  --max_process_ram_gb 16
  --save_images true
)

run_dataset() {
  local name="$1"
  local prompts="$2"
  echo "[$(date -Iseconds)] Starting RAVEN watermark eval: ${name} prompts=${prompts} num_pairs=${NUM_PAIRS}" | tee -a "${LOG_DIR}/raven_watermark_eval.log"
  python "${ROOT_DIR}/experiments/run_raven_watermark_eval.py" \
    --dataset_name "${name}" \
    --prompts_csv "${prompts}" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/raven_watermark_eval_${name}.log"
  echo "[$(date -Iseconds)] Finished RAVEN watermark eval: ${name}" | tee -a "${LOG_DIR}/raven_watermark_eval.log"
}

run_dataset "mscoco" "/workspace/data/prompts/mscoco_5000.csv"
run_dataset "diffusiondb" "/workspace/data/prompts/diffusiondb_1001.csv"
run_dataset "sd_prompts" "/workspace/data/prompts/sd_prompts_8192.csv"
