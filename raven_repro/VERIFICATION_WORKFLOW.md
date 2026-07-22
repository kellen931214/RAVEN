# Formal Watermark Verification Workflow

The only formal orchestrator is `experiments/run_raven_formal_eval.py`. Detector
scripts are implementation helpers and must not be invoked as alternative formal runners.

## Inputs And Snapshot

The `snapshot` stage reads only complete CSV lines, checks unique run IDs, required
prompt/prompt ID fields, clean and watermarked image decodability, image SHA-256, and
provider configuration. It writes fsynced immutable batch CSVs and an append-only
`snapshot_index.jsonl`. Later stages never reread the live source CSV.

## Formal Stage Order

```text
snapshot -> attack-watermarked -> attack-clean (TR only) -> verify
         -> quality -> fid -> clip -> aggregate -> validate
```

Every stage uses the same `--dataset`, `--method`, `--source-metadata`, `--output-root`,
`--expected-count`, `--batch-size`, `--device`, and optional `--gpu`. Add `--resume`
after the initial snapshot. Attack resume validates manifest, input, config, model
revision, seed, planned flow, attacked SHA, debug SHA, and transform hash; drift is a
hard error rather than a skip or rerun.

## Detector Protocol

Raw detector scores are stored with their direction and converted to a canonical
higher-means-watermark direction where applicable. The clean negatives from the same
immutable cohort calibrate the requested 1% FPR; reports preserve target FPR, actual
empirical FPR, false-positive count, threshold, before/attacked TPR, and ROC-AUC.
Legacy fixed thresholds use `legacy_fixed_threshold_*` fields only.

Tree-Ring uses complex L1 and attacks clean negatives with the identical transform.
It reports `full_precision_protocol` and `nfpa_rounded2_protocol` separately. NFPA
rounded2 is primary, threshold detection is strict `<`, and attacked TPR is named
separately at the original-clean and attacked-clean-recalibrated thresholds. One
canonical provider config and target watermark are required per cohort.

## Quality, FID, And CLIP

Primary PSNR/SSIM compares watermarked input with the final post-color-transfer output
over an inverse-warp overlap derived from effective source flow measured from the exact
sampling grid. FID uses a new config-hash directory, exact completed run-ID sets, the
watermarked inputs as reference, and attacked outputs as comparison. CLIP is centralized
as prompt-image cosine using `ViT-bigG-14/laion2b_s39b_b160k`, original prompts, and
L2-normalized embeddings; mixed provenance fails aggregation.

## Memory Safety

Check `free -h` and GPU availability before every gate. The runner processes one attack
image at a time, limits CPU threads, avoids DataLoader workers and dataset caches, and
hashes files incrementally. Use fresh timestamped roots for 2-, 10-, and 30-sample gates.

## Color Transfer Modes

`paper_exact_two_stage` is the paper-faithful, unaligned mode. Following RAVEN
Section 4.2.4, stage 1 builds `x_c = F_RGB(L_a, a_o, b_o)` from attacked
luminance and original chroma at the same pixel coordinates. After converting
`x_c` back to LAB, stage 2 uses
`L' = sigma(L_o)/sigma(L_c) * (L_c - mu(L_c)) + mu(L_o)` and reconstructs with
`a_o,b_o`. This mode does not accept planned or effective flow.

`paper_exact_two_stage_aligned` is the existing effective-flow aligned
ablation. It uses `effective_source_flow_dx_image_px` and
`effective_source_flow_dy_image_px` derived from the actual warp grid for both
attacked-clean and attacked-watermarked outputs. Planned flow and visual shift
remain provenance fields only. Quality overlap uses effective source flow for
both modes, independently of color-transfer alignment.

Use `experiments/run_raven_color_transfer_eval.py` to reprocess immutable
pre-color formal attack views without rerunning DDIM, shift, attention, or
sampling. Attack records, debug info, transform hashes, FID definitions, and
aggregate tables preserve the explicit selected mode.
