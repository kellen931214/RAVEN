#!/usr/bin/env bash
# Run one GM or T2S formal RAVEN evaluation end to end on one GPU.
#
# Every stage is a separate invocation of the single auditable entrypoint, in
# protocol order, and the experiment-table updater is attached as a completion
# hook after the validate stage succeeds -- not as monitoring. --resume is
# passed throughout, so re-running this script continues the same
# content-addressed run root instead of creating a second one.
#
# usage: run_gm_t2s_formal_eval.sh <METHOD> <VARIANT-SLUG> <ATTACK-CONFIG> <GPU>
set -euo pipefail

METHOD="$1"
VARIANT="$2"
ATTACK_CONFIG="$3"
GPU="$4"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASET="diffusiondb_shared_tr"
EXPECTED_COUNT=1001
SOURCE_MANIFEST="audit/formal_source_manifest_gm_t2s_20260729.json"

case "$METHOD" in
  GM)  SOURCE_METADATA="data/gm/${DATASET}/GM/metadata.csv" ;;
  T2S) SOURCE_METADATA="data/t2s/${DATASET}/T2S/metadata.csv" ;;
  *)   echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

# Hard stop before any model load if the GPU is not usable. AGENTS.md forbids
# CPU fallback, and the runner itself refuses a non-cuda device, but failing
# here keeps a dead GPU from being discovered halfway through a stage.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" python - <<'PREFLIGHT'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("GPU preflight failed: no visible CUDA device")
try:
    torch.zeros(8, device="cuda").sum().item()
except Exception as exc:  # noqa: BLE001 - any CUDA failure is a hard stop
    sys.exit(f"GPU preflight failed: {exc}")
print(f"GPU preflight OK: {torch.cuda.get_device_name(0)}")
PREFLIGHT

run_stage() {
  echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${METHOD}/${VARIANT} stage=$1"
  python experiments/run_raven_formal_eval.py \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --source-metadata "$SOURCE_METADATA" \
    --source-manifest "$SOURCE_MANIFEST" \
    --attack-config "$ATTACK_CONFIG" \
    --variant "$VARIANT" \
    --expected-count "$EXPECTED_COUNT" \
    --device cuda \
    --gpu "$GPU" \
    --resume \
    --stage "$1"
}

for STAGE in snapshot attack-watermarked verify quality fid clip aggregate validate; do
  run_stage "$STAGE"
done

# Resolve the run root the same way the runner does, so the completion hook can
# never be pointed at a different directory than the one just validated.
OUT="$(python - "$METHOD" "$DATASET" "$VARIANT" "$ATTACK_CONFIG" "$SOURCE_MANIFEST" <<'RESOLVE'
import sys
sys.path.insert(0, "raven_repro")
from raven.eval_protocol import (
    formal_attack_config_hash,
    formal_output_root,
    formal_run_key,
    load_formal_attack_config,
    sha256_path,
)

method, dataset, variant, attack_config, source_manifest = sys.argv[1:6]
run_key = formal_run_key(
    sha256_path(source_manifest),
    formal_attack_config_hash(load_formal_attack_config(attack_config)),
)
print(formal_output_root(method, dataset, variant, run_key))
RESOLVE
)"

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${METHOD}/${VARIANT} recording experiment table row"
python experiments/update_experiment_table.py --run-root "$OUT"
echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${METHOD}/${VARIANT} COMPLETE run_root=$OUT"
