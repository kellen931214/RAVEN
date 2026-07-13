# Watermark Verification Workflow

This workflow evaluates existing DiffusionDB and MS-COCO images. It does not
generate images or rerun RAVEN. Run one dataset and one method at a time.

## 1. Select the existing Python environment

The current `/usr/bin/python` can build and evaluate manifests but does not
contain the detector dependencies. Set `PYTHON_BIN` to the existing environment
used for the original runs, then verify it without installing anything:

```bash
export PYTHON_BIN=/absolute/path/to/existing/python
"$PYTHON_BIN" -c 'import diffusers, scipy, torch, tqdm; print(torch.__version__, tqdm.__version__)'
"$PYTHON_BIN" raven_repro/scripts/extract_verification_scores.py --help
```

Confirm that the installed tqdm supports `TQDM_DISABLE=1`. The extractor also
passes `disable_tqdm=True` directly to the Diffusers pipeline.

## 2. Build strict pairing manifests

DiffusionDB Tree-Ring:

```bash
/usr/bin/python raven_repro/scripts/build_verification_manifest.py \
  --dataset diffusiondb --method TR \
  --metadata data/watermarked/diffusiondb/TR/metadata.csv \
  --raven-results outputs/raven_eval/diffusiondb/TR/results.csv \
  --clean-dir data/generated/diffusiondb \
  --watermark-config data/watermarked/diffusiondb/experiment_config.json \
  --clean-config data/generated/diffusiondb/experiment_config.json \
  --raven-config outputs/raven_eval/diffusiondb/experiment_config.json \
  --workspace-root /workspace/kellen \
  --output outputs/verification/diffusiondb/TR/pairs.csv
```

Use the same command with `mscoco` paths after DiffusionDB completes. The
builder refuses to overwrite an existing manifest and verifies run ID, prompt,
prompt ID, image existence, and SHA-256 hashes.

## 3. Extract scores

Start with `--limit 10`, then use 100, then omit `--limit` for the formal run.
Use a new output path for every stage.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="raven_repro/logs/TR_diffusiondb_10_${TS}.log"
PID="raven_repro/logs/TR_diffusiondb_10_${TS}.pid"

nohup env \
  TQDM_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PYTHON_BIN" -u raven_repro/scripts/extract_verification_scores.py \
  --method TR \
  --metadata outputs/verification/diffusiondb/TR/pairs.csv \
  --output "outputs/verification/diffusiondb/TR/raw_scores_10_${TS}.csv" \
  --eval-repo eval_bench_wm \
  --model-id RedbeardNZ/stable-diffusion-2-1-base \
  --scheduler DDIM --steps 50 --resolution 512 --device cuda \
  --limit 10 --target-fpr 0.01 \
  --min-cpu-mem-gb 92 --warn-cpu-mem-gb 110 --max-process-ram-gb 16 \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

Do not launch MS-COCO while DiffusionDB has a live PID. The extractor uses
batch size one, no DataLoader workers, one detector pipeline, and fsyncs every
score row without retaining latents across samples.

## 4. Calibrate and report

```bash
/usr/bin/python raven_repro/scripts/evaluate_verification.py \
  --method TR \
  --records outputs/verification/diffusiondb/TR/raw_scores_full.csv \
  --target-fpr 0.01 \
  --output-json outputs/verification/diffusiondb/TR/metrics.json
```

The report contains both `legacy_fixed_threshold_detect_rate` and
`calibrated_TPR_at_1pct_FPR`. Only the calibrated field is paper-comparable.
For GS, also pass `--output-rows` to save decoded bits, ground-truth bits,
error indices, key, nonce, offset, and per-sample bit accuracy.

## Order

Run DiffusionDB TR smoke, diagnostic, and full evaluation first. Then run
MS-COCO TR. Continue sequentially with RID, HSTR, HSQR, and GS. Quality metrics
and any RAVEN ablation remain separate later stages.
