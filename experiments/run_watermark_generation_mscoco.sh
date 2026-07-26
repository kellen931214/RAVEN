#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
LOG_DIR="/workspace/raven_repro/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/watermark_generation_mscoco.log"

if [[ "${RAVEN_LOW_PRIORITY:-0}" != "1" ]]; then
  export RAVEN_LOW_PRIORITY=1
  exec ionice -c2 -n7 nice -n 10 bash "$0" "$@"
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

export TQDM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-RedbeardNZ/stable-diffusion-2-1-base}"
# Canonical layout (migration 2026-07-26): leaving OUTPUT_DIR empty lets the
# generator route each method to its own root (data/tr/, data/gs/); clean images
# always go to data/clean/<dataset>/.
OUTPUT_DIR="${OUTPUT_DIR:-}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-${ROOT_DIR}/data/clean}"
NUM_PAIRS="${NUM_PAIRS:-1000}"

printf '[%s] Starting MS-COCO watermarked image generation\n' "$(date -Iseconds)"
printf '[%s] Log file: %s\n' "$(date -Iseconds)" "${LOG_FILE}"
printf '[%s] Checking RAM before launch\n' "$(date -Iseconds)"
free -h
printf '[%s] Checking GPU before launch\n' "$(date -Iseconds)"
nvidia-smi

python "${ROOT_DIR}/experiments/generate_watermarked_images.py" \
  --dataset_name mscoco \
  --prompts_csv "${ROOT_DIR}/data/clean/mscoco/inputs/mscoco_5000.csv" \
  ${OUTPUT_DIR:+--output_dir "${OUTPUT_DIR}"} \
  --clean_output_dir "${CLEAN_OUTPUT_DIR}" \
  --wm_types GS TR RID HSTR HSQR \
  --num_pairs "${NUM_PAIRS}" \
  --modelid_target "${MODEL_ID}" \
  --resolution 512 \
  --scheduler_target DDIM \
  --num_inference_steps_target 50 \
  --guidance_scale_target 7.5 \
  --seed 42 \
  --device cuda \
  --gpu auto \
  --require_free_gpu true \
  --min_cpu_mem_gb 64 \
  --warn_cpu_mem_gb 96 \
  --max_process_ram_gb 16 \
  --validate_before true

printf '[%s] Finished MS-COCO watermarked image generation\n' "$(date -Iseconds)"
printf '[%s] Checking RAM after completion\n' "$(date -Iseconds)"
free -h
printf '[%s] Checking GPU after completion\n' "$(date -Iseconds)"
nvidia-smi
