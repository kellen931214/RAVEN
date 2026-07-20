# Debug Changelog

This file records implementation bugs, validated non-bugs, ablations, and the evidence used to verify each change. Large logs and generated outputs are not copied here; paths point to the source artifacts.

## Current Status

| Date | Area | Status | Evidence |
| --- | --- | --- | --- |
| 2026-07-18 | Formal evaluation protocol audit | Implemented immutable snapshots, explicit formal attack/debug assertions, effective-grid quality flow, strict resume/FID/provider/CLIP provenance, full/rounded TR reporting, formal waiter/table, and quarantined pre-audit derived outputs. Complete CPU suite: 148 passed; new 2/10/30 GPU gates remain blocked by unavailable NVML, so full eval is not safe. | `audit/formal_eval_protocol.md`; `audit/current_eval_processes.md`; `outputs/legacy_invalid/20260718T072817Z/DO_NOT_USE.md` |
| 2026-07-17 | Tree-Ring paired generation and formal provenance | Shared-latent source and every derived `TPR=0.177822` result rejected. Per-sample paired latent generation, two-GPU pair sharding, orphan quarantine, fail-closed provenance gates, paired attack config hashes, and two aligned-color-only variants implemented; formal rerun in progress. | `raven_repro/raven/pairing_provenance.py`; `raven_repro/scripts/paired_generation_shards.py`; `raven_repro/scripts/run_diffusiondb_chain_after_clean.py`; `outputs/raven_paired_formal_smoke/diffusiondb/20260717T033000Z/data/watermarked/diffusiondb/TR/shard_merge_audit.json` |
| 2026-07-15 | DiffusionDB latest RAVEN-paper/NFPA-gap-fill Tree-Ring L1 rerun preparation | Fixed attacked-clean config drift, verified 2-sample smoke and 10-sample validation; full 1001 run prepared for nohup. | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `raven_repro/scripts/raven_p1_full.py`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.json` |
| 2026-07-15 | RAVEN-paper / NFPA-gap-fill warp and inverse-overlap quality | Implemented and verified on focused tests plus 10-sample DiffusionDB validation. Existing 1001 DiffusionDB P1 outputs were quality-recomputed without rerunning attack. | `raven_repro/raven/warp.py`; `raven_repro/raven/metrics.py`; `raven_repro/scripts/raven_paper_nfpa_gap_fill_eval.py`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md` |
| 2026-07-15 | RAVEN exact two-stage color transfer | Implemented and verified. Existing 10 validation pre-color outputs were reused; no DDIM inversion or denoising was rerun. | `raven_repro/raven/color_transfer.py`; `raven_repro/scripts/raven_color_transfer_validation.py`; `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.md` |
| 2026-07-14 | NFPA-style Tree-Ring complex L1 evaluation | Completed for DiffusionDB only. MS-COCO was not run after scope was corrected. | `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/aggregate_results.json` |

## Confirmed Issues And Fixes

### 2026-07-17 - Corrected Paired DDIM-Shift No-Color Evaluation

| Field | Details |
| --- | --- |
| Problem | The corrected paired 1001-sample run intentionally evaluated only aligned-color and blended-aligned-color variants. The only existing `ddim_shift_no_color` aggregate came from the rejected shared-latent/unpaired source and could not be reused. |
| Source validity | The new no-color evaluation is restricted to `outputs/raven_paired_formal/diffusiondb/20260717T014700Z`, whose pairing audit has 1001 unique base-latent seeds/hashes, zero duplicate base latents, paired clean/watermarked base hashes, matching target/mask/config hashes, and complete image SHA provenance. |
| Fix | `quality_decomposition_experiment.py` now supports explicit `--variants ddim_shift_no_color`. It directly reuses each verified clean/WM `view_guided_output.png` before color transfer, records `color_transfer_mode=none`, and keeps the existing alignment+blend default unchanged. Detector, threshold, CLIP-vs-prompt, FID, inverse-overlap PSNR/SSIM, pairing, and attack-config gates remain shared with the authoritative evaluator. Fresh FID staging now records a canonical manifest SHA and rejects duplicate paths, missing files, stale targets, broken links, and staging file-set drift. |
| Attack pairing | Attacked-clean and attacked-watermarked inputs must match run ID, attack seed, dx/dy, exact DDIM timestep, strength, guidance, inversion mode/prompts, warp, sampling, padding, normalization, model revision, pairing hash, and canonical attack-config hash before scoring. |
| Evaluation definitions | Tree-Ring score is official complex L1 with lower-is-watermarked and strict `< threshold`; the threshold is calibrated from no-color attacked-clean scores at target FPR 1%. CLIP is attacked-watermarked no-color output versus the source prompt. FID compares original watermarked images with no-color attacked-watermarked images. PSNR/SSIM compare the same pair over inverse-warp valid overlap. |
| Memory controls | No DDIM/UNet attack rerun is needed. Images are processed one at a time with one CPU thread, no DataLoader or dataset cache, RAM guards, incremental records, and fresh temporary FID staging. |
| Legacy handling | Every old shared-latent no-color result remains `INVALID_SHARED_LATENT`; no old detector or quality value is reused. |
| Validation | Focused tests passed (`19 passed`) and the complete suite passed (`101 passed`, 12 expected warnings). The first smoke was quarantined under `invalid/incomplete_no_color_smoke_gpu_mapping_20260717T174140Z` after logical GPU index 0 mapped to a full 48 GiB device; no metric output was produced. The UUID-bound 2-sample smoke completed with two unique base latents, pairing/config/hash audits passing, direct pre-color paths, matching detector target, finite metrics, zero NaN/Inf, and fresh FID staging manifest SHA `99405e947f1db8c2c636e0d42a5c6bdceb29441d0f49d6dfb0e8ac6a8d97182a`. Evidence: `quality_decomposition_no_color_smoke2_20260717T174412Z/run.log` and `aggregate_results.json`. The corrected 1001-sample run is active at `quality_decomposition_no_color_1001_20260717T174846Z/` with PID recorded in `pid` and progress in `run.log`. |
| Git provenance | Branch `agent/cleanup-quality-decomposition`, base HEAD `3330e67c0a9538268691ac32fe00ecca79abef50`; working tree dirty; not committed or pushed; results remain non-release until publication. |

### 2026-07-17 - Tree-Ring Generator Reused One Full Latent And Clean Images Were Not Paired

| Field | Details |
| --- | --- |
| Problem | `experiments/generate_watermarked_images.py` previously called `get_wm_latents()` once outside the sample loop and reused the returned complete `wm_zT` for all 1001 prompts. Clean images were generated by a separate sequential RNG workflow, so a clean/watermarked pair did not share a base latent. The old manifest then wrote `generation_seed=42+run_id` without any latent hash proving that claim. |
| Impact | The source cohort was non-independent and unpaired. Every detector or quality result derived from `outputs/raven_color_alignment_ablation/diffusiondb/20260716T082019Z`, including aligned-color `TPR=0.177822`, is invalid and must not be reported. |
| Core logic changed | Each sample now creates a unique base latent from its own seed, injects Tree-Ring into a clone, and generates clean/watermarked images with identical prompt/model/scheduler/steps/guidance and the same exact base latent. Generation is sharded by run ID across two GPUs; each worker owns complete pairs and writes an independent shard metadata/log/results file. A merge gate requires exact run-ID coverage and audits every row before producing formal `metadata.csv`. Per-row provenance records base seed/hash, clean and watermarked base hashes, watermarked latent hash, target/mask hashes, image SHA-256 values, generation/watermark config hashes, and a canonical pairing hash. P1, attacked-clean, detector, and quality entry points reject missing provenance, duplicate latent hashes, file drift, target drift, or clean/watermarked attack-config mismatch. Crash-written images without a committed metadata row are moved to `invalid/orphaned_unrecorded/` and regenerated; they are never resumed as valid data. |
| Formal variants | The corrected formal rerun evaluates only `alignment_color` (`paper_exact_two_stage_aligned`, alpha 1.0) and `blend_alignment_color` (`paper_exact_two_stage_aligned_blend`, alpha 0.5). No no-shift or no-color metric branch is run. CLIP is image-text cosine between `wm_attack` and the source prompt. |
| Memory controls | Each of the two workers remains sample-at-a-time with no DataLoader or dataset cache, one CPU thread per process, incremental disk writes, RAM guards, and a waiter that requires two idle compatible GPUs before launch. The two workers duplicate only the diffusion pipeline, not the dataset in CPU memory. |
| Verification | The complete test suite passed (`93 passed`, 12 expected warnings). A two-sample/two-GPU smoke assigned run 0 to GPU 0 and run 1 to GPU 2, produced one audited row per shard, merged with `count=2`, two unique latent seeds/hashes and image hashes, no duplicate latent, and matching target/mask/config/revision. The merged manifest independently passed the same pairing audit. Evidence: `outputs/raven_paired_formal_smoke/diffusiondb/20260717T033000Z/run.log`, `logs/paired_generation_shard_000.log`, `logs/paired_generation_shard_001.log`, and `data/watermarked/diffusiondb/TR/shard_merge_audit.json`. The interrupted unrecorded run 1 was retained under `outputs/raven_paired_formal/diffusiondb/20260717T014700Z/invalid/orphaned_unrecorded/20260717T032839Z/`. |
| Legacy handling | After the corrected formal run succeeds, logs/provenance and invalid aggregate files are archived under `outputs/invalid/shared_latent_20260716/`; contaminated images and derived data are deleted by `invalidate_shared_latent_outputs.py`. |
| Status | Code fix, two-GPU pair sharding, root `run.log`, orphan recovery, and fail-closed merge gate implemented. Two-GPU generation/manifest smoke passed; corrected 1001-sample rerun is pending/resumable. |

### 2026-07-13 - Partial Inversion Was Treated Ambiguously As DDIM

| Field | Details |
| --- | --- |
| Problem | The original partial inversion path used random forward noising via `scheduler.add_noise(clean_latents, noise, timestep)` for the Equation (4) interpretation. That is not true DDIM inversion and should not be the formal reproduction setting. |
| Impact | RAVEN attack diagnostics mixed a random forward-noising ablation with the formal DDIM-inversion reproduction path, making Tree-Ring suppression results hard to interpret. |
| Core logic changed | `ddim` became the primary reproduction mode. `forward_noise` is retained as a labeled ablation. DDIM inversion records denoise scheduler, inverse scheduler, prediction type, target timestep, inverse timestep sequence, and `eta=0.0`. |
| Verification | Round-trip audit reconstructed a real image through VAE encode -> DDIM inversion to timestep 121 -> DDIM denoise -> VAE decode with PSNR 30.3395, SSIM 0.9634, latent MAE 0.02739. |
| Evidence | `raven_repro/raven/inversion.py`; `raven_repro/scripts/audit_ddim_roundtrip.py`; `outputs/raven_diagnostics/20260713T171750Z/roundtrip/roundtrip.json` |
| Status | Fixed and verified for the diagnostic setting. |

### 2026-07-13 - Shift Sign And Overlap Crop Convention Were Wrong/Ambiguous

| Field | Details |
| --- | --- |
| Problem | The local shift convention and overlap crop were ambiguous and previously matched inverse-sampling wording rather than the requested visual convention: positive x should move content right and positive y should move content down for the integer RAVEN translation mode. |
| Impact | PSNR/SSIM overlap comparisons and direction-based diagnostics could compare the wrong regions or report misleading direction metadata. |
| Core logic changed | `crop_overlap` was updated to the right/down visual convention. Integer translation uses explicit slicing with zero padding and no wrap-around. Tests now validate impulse movement for `(+x,0)`, `(-x,0)`, `(0,+y)`, `(0,-y)` and no circular wrap. |
| Verification | Unit tests in `raven_repro/tests/test_warp.py` and `raven_repro/tests/test_overlap_metrics.py`; diagonal interpretation ablation recorded valid overlap and per-direction outcomes. |
| Evidence | `raven_repro/raven/metrics.py`; `raven_repro/raven/warp.py`; `outputs/raven_diagonal_interpretation/20260714T071247Z/aggregate_results.md` |
| Status | Fixed for integer/latent-grid RAVEN modes; NFPA exact mode separately records flow direction and visual direction because NFPA uses inverse sampling. |

### 2026-07-13 - Attention Processor State Could Leak Across Runs

| Field | Details |
| --- | --- |
| Problem | Attention processors were not restored as an exact original mapping after a run; cross-attention processors could be replaced by generic defaults and debug state could leak across repeated ablations. |
| Impact | Attention-on/off and sampling/padding comparisons risked cross-run contamination, making output differences unreliable. |
| Core logic changed | The pipeline stores the original UNet attention processor mapping, installs view-guided processors only for self-attention, preserves existing cross-attention processors, and restores the exact mapping before/after runs. Debug metadata records processor counts and call counts. |
| Verification | `test_install_preserves_existing_cross_attention_processor_identity`, `test_restore_default_attention_restores_exact_mapping`, and sampling/padding provenance explicitly recorded the state fix. Attention-on/off outputs were not identical in 3-sample diagnostics. |
| Evidence | `raven_repro/raven/attention.py`; `raven_repro/raven/pipeline_raven.py`; `raven_repro/tests/test_attention_shapes.py`; `outputs/raven_sampling_padding_ablation/20260714T093603Z/provenance.json`; `outputs/raven_diagnostics/20260713T171942Z/attacks/attack_summary.json` |
| Status | Fixed and verified by unit tests and diagnostic outputs. |

### 2026-07-14 - RAVEN Warp Did Not Match NFPA Coordinate/Sampling Convention

| Field | Details |
| --- | --- |
| Problem | The RAVEN integer zero-padding warp is not NFPA's latent warp convention. NFPA builds a 512x512 image-coordinate flow, normalizes by `/W` and `/H`, resizes coordinates bilinearly to latent size, and samples latent with `nearest` plus `reflection` padding. |
| Impact | Tree-Ring suppression and quality could differ because normalization, sampling mode, and padding were changed together. Integer zero padding was not a faithful NFPA-compatible ablation. |
| Core logic changed | Added explicit NFPA-compatible warp utilities and metadata: `coords_grid`, image-coordinate flow, NFPA normalization, coordinate resize metadata, nearest/reflection sampling, inverse-sampling sign metadata, and separate `latent_grid`/`integer` ablation modes. |
| Verification | NFPA unit tests passed for coordinate grid shape, `/W` `/H` normalization, four-direction impulse movement, reflection padding, nearest sampling, effective displacement, no NaN/Inf, and CPU/GPU consistency. |
| Evidence | `raven_repro/raven/warp.py`; `raven_repro/scripts/nfpa_warp_ablation.py`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/unit_test_results.json`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/effective_displacement.json` |
| Status | Fixed as named modes; `nfpa_exact`, `latent_grid`, `integer`, and `direct_latent` remain separate ablations. |

### 2026-07-14 - Quality Metric Reference Was Ambiguous

| Field | Details |
| --- | --- |
| Problem | Quality output names could be read as generic `psnr`/`ssim` without a clear reference image. |
| Impact | Attack quality could be compared against clean input in one place and watermarked input in another without being obvious. |
| Core logic changed | Per-sample records and aggregate outputs now save `psnr_vs_watermarked`, `ssim_vs_watermarked`, `psnr_vs_clean`, and `ssim_vs_clean`; the primary quality reference for P1 attack results is watermarked input. |
| Verification | P1 full aggregate reports quality under `quality.primary_reference = watermarked_input`; sampling/padding ablation Markdown explicitly states the primary reference. |
| Evidence | `raven_repro/scripts/raven_p1_full.py`; `raven_repro/scripts/nfpa_sampling_padding_ablation.py`; `outputs/raven_p1_full/diffusiondb/20260714T095907Z/aggregate_results.json`; `outputs/raven_sampling_padding_ablation/20260714T093603Z/aggregate_results.md` |
| Status | Fixed for current diagnostics and P1 full outputs. |

### 2026-07-15 - RAVEN Paper/NFPA Gap-Fill Warp And Inverse-Overlap Quality

| Field | Details |
| --- | --- |
| Problem | The previous P1 full driver used a direct latent-grid `/8` displacement with nearest/reflection. That was useful for the P1 ablation but did not preserve the requested priority rule: use RAVEN paper settings for shift sampling and use NFPA only for underspecified coordinate-grid implementation details. Existing PSNR/SSIM also used visual-shift overlap rather than the explicit inverse-warp correspondence formula requested for the paper-comparable quality protocol. |
| Impact | Future full runs could silently reuse the old P1 transform as if it were the paper/NFPA gap-fill setting, and quality metrics could include the wrong correspondence crop if flow direction and visual direction were confused. |
| Core logic changed | Added `raven_paper_nfpa_gap_fill` mode, which passes RAVEN image-pixel `dx/dy` directly into an NFPA-style 512x512 coordinate flow, uses `/W` `/H` normalization, bilinear coordinate-grid resize, inverse `grid_sample`, `padding_mode=reflection`, `align_corners=False`, and main `mode=nearest`. Bilinear is retained only as a same-grid ablation. Added `crop_overlap_inverse_warp` and quality helpers that compare watermarked input against pre/post color-transfer outputs over valid inverse-warp overlap only. Future `raven_p1_full.py` runs now use the new mode and record a config hash including grid version, sampling, padding, normalization, align_corners, and shift. |
| Verification | Focused syntax/tests passed: `46 passed, 1 warning` for `test_warp.py` and `test_overlap_metrics.py`. Existing DiffusionDB 1001 P1 outputs were recomputed for inverse-overlap quality without rerunning attack. A 10-sample validation compared old P1, new nearest main, and bilinear ablation with complex L1 scoring. |
| Evidence | `raven_repro/raven/warp.py`; `raven_repro/raven/metrics.py`; `raven_repro/raven/pipeline_raven.py`; `raven_repro/scripts/raven_p1_full.py`; `raven_repro/scripts/raven_paper_nfpa_gap_fill_eval.py`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md`; `outputs/raven_paper_nfpa_gap_fill/diffusiondb_quality_recompute_20260715T034545Z/quality_summary.json`; `outputs/raven_paper_nfpa_gap_fill/diffusiondb_validation_20260715T035849Z/aggregate_results.json` |
| Status | Implemented and verified on focused tests plus 10-sample validation. Full 1001 new-transform attack has not been run. |

### 2026-07-15 - Color Transfer Used Direct Generated-Luminance Statistics Instead Of Paper Two-Stage Formula

| Field | Details |
| --- | --- |
| Problem | The previous CIELAB transfer matched generated-image `L_opt` mean/std directly to the original watermarked luminance and then inserted original `a/b`. The requested RAVEN formula first builds `x_c_lab = (L_opt, a_w, b_w)`, converts LAB -> RGB -> LAB, computes statistics from the realized `L_c`, then matches `L_c` to `L_w`. |
| Impact | In gamut-clipped cases, the realized luminance after combining generated L with original chroma can differ from raw `L_opt`. Direct statistics can therefore produce slightly different luminance matching and diagnostics from the paper formula. |
| Core logic changed | `color_contrast_transfer` now defaults to `paper_exact_two_stage`; old behavior remains available as `direct_stats`. Diagnostics now include `L_opt`, `L_c`, `L_w`, pre/post clip `L_final` ranges, final output L mean/std, saturated pixel ratio, and luminance mean/std errors. Pipeline debug metadata records `color_transfer_mode=paper_exact_two_stage`. |
| Verification | `test_color_transfer.py` covers output shape/dtype/range, deterministic output, constant luminance with no NaN/Inf, final L mean/std closeness to original, and a gamut-clipping synthetic case where `paper_exact_two_stage` differs from `direct_stats`. 10-sample color-only validation reused existing `view_guided_output.png` files and did not rerun DDIM/denoising. |
| Evidence | `raven_repro/raven/color_transfer.py`; `raven_repro/tests/test_color_transfer.py`; `raven_repro/scripts/raven_color_transfer_validation.py`; `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.json` |
| Status | Fixed and verified for focused tests plus 10-sample color-only validation. Existing full attacked images generated before this change remain legacy color-transfer outputs. |

### 2026-07-14 - NFPA-Style Tree-Ring Complex L1 Metric Was Missing

| Field | Details |
| --- | --- |
| Problem | Existing Tree-Ring evaluation used `-log10(p)` fixed/calibrated threshold logic, while NFPA evaluates Tree-Ring with complex L1 distance `torch.abs(decoded_watermark - target_watermark).mean(-1)` where lower score indicates watermark. |
| Impact | The existing P1 result is useful as separate `-log10(p)` analysis but is not the requested NFPA-style Tree-Ring `TPR@1%FPR`. After-attack NFPA calibration also requires attacked-clean images, not only attacked-watermarked images. |
| Core logic changed | Added an independent `raven_nfpa_tr_eval.py` flow that copies existing P1 attacked-watermarked records, generates only attacked-clean images with the same shift plan/settings, scores original clean/watermarked/attacked-clean/attacked-watermarked with complex L1, and calibrates before/after thresholds separately with NFPA's strict `< threshold` rule. |
| Verification | DiffusionDB completed with 1001 rows. NFPA-style before threshold 76.23775482177734, before actual FPR 0.008991008991008992, before TPR 1.0; after threshold 79.72408294677734, after actual FPR 0.008991008991008992, after TPR 0.48451548451548454; attack success 0.5154845154845155. |
| Evidence | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/nfpa_l1_scores.jsonl`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/aggregate_results.json` |
| Status | Completed for DiffusionDB. This is separate from legacy `-log10(p)` fixed-threshold analysis. |

### 2026-07-15 - Attacked-Clean Evaluation Still Used Legacy `latent_grid` Config

| Field | Details |
| --- | --- |
| Problem | `raven_nfpa_tr_eval.py attack-clean` still called the pipeline with `warp_mode="latent_grid"`, while the new formal attacked-watermarked driver uses `raven_paper_nfpa_gap_fill` with nearest/reflection and paper-exact two-stage color transfer. |
| Impact | Post-attack NFPA-style L1 calibration could compare attacked-clean negatives produced by a different transform from attacked-watermarked positives, invalidating the after-attack `TPR@1%FPR`. |
| Core logic changed | `attack-clean` now uses `raven_paper_nfpa_gap_fill`, `padding_mode="reflection"`, `latent_sampling_mode="nearest"`, empty prompts, DDIM, strength 0.15, guidance 2.5, and `paper_exact_two_stage` color transfer. Both attacked-clean and attacked-watermarked records now save config fields and `transform_config_hash`; L1 scoring stops if run ID, seed, dx/dy, timestep, scheduler/prompt/warp/sampling/padding/color-transfer settings, or transform hash differ. Scoring output is standardized as `l1_scores.jsonl` plus `per_sample_results.csv`, with before/after thresholds calibrated separately from original-clean and attacked-clean scores. |
| Verification | `py_compile` passed for `raven_nfpa_tr_eval.py` and `raven_p1_full.py`. Focused tests passed: `58 passed, 8 warnings` for `test_warp.py`, `test_overlap_metrics.py`, `test_metrics.py`, and `test_color_transfer.py`. 2-sample smoke completed with finite L1 and no NaN/Inf. 10-sample validation completed with after TPR 0.700000, attack success 0.300000, config/hash audit passing for all 10 records, and no duplicate rounded/exact L1 groups. |
| Evidence | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `raven_repro/scripts/raven_p1_full.py`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/smoke2_eval/aggregate_results.json`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.json` |
| Status | Fixed and gate-verified. Full DiffusionDB 1001 rerun is the next stage and must use new timestamped outputs. |

## Confirmed Non-Bugs

### 2026-07-13 - Tree-Ring High TPR Was Not Only A Legacy Threshold Artifact

| Field | Details |
| --- | --- |
| Finding | The legacy fixed-threshold detect rate and clean-negative calibrated TPR were close for the old DiffusionDB Tree-Ring result. |
| Evidence | `outputs/verification_v2/metrics/TR_diffusiondb_1001_20260713T074340Z.json` reports calibrated TPR@1%FPR 0.7442557443, legacy detect rate 0.7512487512, actual clean FPR 0.0099900100, attacked ROC-AUC 0.9584182052. |
| Conclusion | The main gap was not explained by accidentally using the legacy fixed threshold. Attack pipeline and detector metric interpretation needed separate audit. |
| Status | Confirmed not the sole bug. |

### 2026-07-13 - DDIM Inversion Round Trip Was Not Obviously Broken

| Field | Details |
| --- | --- |
| Finding | A true DDIM inversion/denoise round trip reconstructed the input at reasonable quality. |
| Evidence | `outputs/raven_diagnostics/20260713T171750Z/roundtrip/roundtrip.json`: PSNR 30.3395, SSIM 0.9634, exact timestep 121, inverse scheduler `DDIMInverseScheduler`, denoise scheduler `DDIMScheduler`, eta 0.0. |
| Conclusion | DDIM inversion still required exact timestep/provenance fixes, but the round-trip diagnostic did not indicate a catastrophic inversion failure. |
| Status | Confirmed not a blocking bug for subsequent ablations. |

### 2026-07-13 - View-Guided Attention Was Not A No-Op

| Field | Details |
| --- | --- |
| Finding | Attention-on and attention-off outputs were not identical in the 3-sample diagnostic. |
| Evidence | `outputs/raven_diagnostics/20260713T171942Z/attacks/attack_summary.json`: max absolute pixel differences 108, 105, and 73 for run IDs 0-2. |
| Conclusion | Attention integration had state-restoration issues, but the hook path was not completely inert. |
| Status | Confirmed not a no-op after diagnostics. |

## Ablations And Comparisons

### 2026-07-13 - 3-Sample Attack Pipeline Diagnostic

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| DDIM + attention + independent 24 px | True DDIM, attention on, independent axes | DiffusionDB Tree-Ring, 3 samples, fixed threshold 1.6372738343 | Mean score after 4.1723; detect after 1.000; PSNR 23.888; SSIM 0.8409 | Shifted attack strongly lowered score but did not drop below threshold on 3 samples. |
| DDIM + no attention + independent 24 px | Attention off | Same | Mean score after 4.5206; detect after 1.000; PSNR 23.710; SSIM 0.8367 | Attention on changed output and was slightly stronger on this tiny cohort. |
| DDIM + attention + coupled 24 px | Coupled diagonal signs | Same | Mean score after 3.5942; detect after 1.000; PSNR 23.841; SSIM 0.8403 | Coupled diagonal was competitive in 3 samples, not enough evidence for final setting. |
| forward_noise + attention + independent 24 px | Random forward noising | Same | Mean score after 3.6579; detect after 1.000; PSNR 21.380; SSIM 0.7383 | forward_noise remained an ablation; quality was worse. |
| DDIM + no shift | Shift disabled | Same | Mean score after 29.6314; detect after 1.000; PSNR 33.806; SSIM 0.9676 | Shift itself caused the main suppression. |
| DDIM + attention + independent 32 px | Larger shift | Same | Mean score after 3.8718; detect after 1.000; PSNR 23.241; SSIM 0.8295 | Larger shift did not clearly dominate 24 px in 3 samples. |

Source: `outputs/raven_diagnostics/20260713T172532Z/tree_ring_scores/diagnostic_summary.md`.

### 2026-07-14 - Diagonal Shift Interpretation, 30 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A | image pixels, common sign, independent x/y magnitudes | DiffusionDB Tree-Ring, 30 samples, fixed threshold | Detect rate 0.6333; mean score 2.6072; PSNR 22.101; SSIM 0.6648 | Better than no-shift; sign binding not clearly best. |
| B | image pixels, independent x/y signs and magnitudes | Same | Detect rate 0.7000; mean score 2.8610; PSNR 22.127; SSIM 0.6673 | Main paper-interpretation candidate, but not strongest suppression in this cohort. |
| C | image pixels, strict `dx=dy` | Same | Detect rate 0.7667; mean score 2.6783; PSNR 22.109; SSIM 0.6753 | Strict diagonal did not improve detection rate. |
| D | direct latent cells, common sign | Same | Detect rate 0.7000; mean score 2.3560; PSNR 17.545; SSIM 0.5519; overlap 0.3166 | Suppression came with severe quality/overlap loss. |
| E | direct latent cells, independent signs | Same | Detect rate 0.7000; mean score 2.4713; PSNR 17.706; SSIM 0.5488; overlap 0.3166 | Direct-latent remained an ambiguity ablation, not a good formal candidate. |
| G | integer latent 3/4 cells, independent signs | Same | Detect rate 0.7000; mean score 2.6252; PSNR 24.071; SSIM 0.8213 | Higher quality than fractional image-pixel modes, similar detect rate. |
| I | no shift | Same | Detect rate 1.0000; mean score 33.8595; PSNR 37.109; SSIM 0.9521 | Confirms shift is necessary for Tree-Ring suppression. |

Source: `outputs/raven_diagonal_interpretation/20260714T071247Z/aggregate_results.md`.

### 2026-07-14 - NFPA Warp Convention, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| nfpa_independent | NFPA image-coordinate flow, `/W` `/H`, bilinear coordinate resize, nearest/reflection latent sampling | DiffusionDB Tree-Ring, 10 samples | Detect rate 0.6000; mean score 2.0998; PSNR 19.210; SSIM 0.5560 | Strong suppression, lower quality. |
| nfpa_sign_bound | Same NFPA warp, common sign | Same | Detect rate 0.6000; mean score 2.1879; PSNR 19.161; SSIM 0.5548 | Similar to independent signs on 10 samples. |
| nfpa_strict_diagonal | Same NFPA warp, `dx=dy` | Same | Detect rate 0.6000; mean score 2.2152; PSNR 19.584; SSIM 0.5982 | Similar suppression with slightly better quality. |
| integer_zero_pad | Image pixels rounded to latent cells, slicing, zero padding | Same | Detect rate 0.8000; mean score 3.4765; PSNR 22.964; SSIM 0.8014 | Better quality but weaker suppression. |
| direct_latent | 24-32 direct latent cells | Same | Detect rate 0.8000; mean score 2.8523; PSNR 16.345; SSIM 0.5057; overlap 0.3202 | Not a reasonable formal candidate due to quality/overlap loss. |
| no_shift | No shift | Same | Detect rate 1.0000; mean score 35.9116; PSNR 34.592; SSIM 0.9492 | Confirms shift effect. |

Sources: `outputs/raven_nfpa_warp_ablation/20260714T081940Z/aggregate_results.md`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/unit_test_results.json`.

### 2026-07-14 - NFPA Normalization, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| N1_nfpa_exact | `x_norm = 2*(x+dx)/W - 1` | Same samples, nearest/reflection, align_corners False | Detect rate 0.6000; mean score 2.1328; PSNR vs WM 19.210; SSIM 0.5560 | NFPA exact had slightly lower score but lower quality. |
| N2_pixel_center | Adds `+0.5` pixel-center offset | Same | Detect rate 0.6000; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Same detect rate; slightly better quality. |
| N3_latent_div8 | Direct `/8` latent grid displacement with same sampling/padding | Same | Detect rate 0.6000; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Matched N2 in this run, so prior quality gap was not normalization alone. |

Source: `outputs/raven_nfpa_normalization_ablation/20260714T090146Z/aggregate_results.md`.

### 2026-07-14 - Sampling/Padding, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| P1_nearest_reflection | nearest sampling + reflection padding | Fixed `/8` latent displacement, align_corners False | Detect rate 0.6000; 4 below threshold; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Best suppression among this 10-sample set; selected for P1 full. |
| P2_nearest_zeros | nearest + zero padding | Same | Detect rate 0.6000; 4 below threshold; mean score 2.4530; PSNR 19.235; SSIM 0.5667 | Reflection improved score and quality over zeros at nearest. |
| P3_bilinear_reflection | bilinear + reflection | Same | Detect rate 0.7000; 3 below threshold; mean score 2.6979; PSNR 20.606; SSIM 0.6117 | Better quality but weaker suppression. |
| P4_bilinear_zeros | bilinear + zeros | Same | Detect rate 0.7000; 3 below threshold; mean score 2.9293; PSNR 20.428; SSIM 0.6049 | Weakest suppression among the four. |

Source: `outputs/raven_sampling_padding_ablation/20260714T093603Z/aggregate_results.md`.

### 2026-07-15 - RAVEN-paper / NFPA-gap-fill Validation, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A_old_P1_latent_grid_nearest_reflection | Legacy direct latent-grid `/8` nearest/reflection output reused from old P1 | DiffusionDB first 10 samples; NFPA-style complex L1; post-color inverse-overlap quality vs watermarked input | Mean L1 before 54.087430; mean L1 after 81.962275; mean delta 27.874845; PSNR 19.732; SSIM 0.5911 | Old output remains a valid legacy reference but is not the new paper/NFPA gap-fill transform. |
| B_RAVEN_paper_NFPA_gap_fill_nearest | RAVEN image-pixel shift plan passed through NFPA image-coordinate grid, nearest/reflection main mode | Same | Mean L1 before 54.087430; mean L1 after 81.677802; mean delta 27.590371; PSNR 19.608; SSIM 0.5861 | New main mode executes end-to-end; suppression/quality are close to old P1 on this tiny cohort. |
| C_RAVEN_paper_NFPA_gap_fill_bilinear | Same grid/padding/shift as B; only latent value sampling changed to bilinear | Same | Mean L1 before 54.087430; mean L1 after 81.472040; mean delta 27.384610; PSNR 20.700; SSIM 0.6221 | Bilinear quality is higher but this is an ablation; main remains nearest because NFPA uses nearest. |

Sources: `outputs/raven_paper_nfpa_gap_fill/diffusiondb_validation_20260715T035849Z/aggregate_results.md`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md`.

### 2026-07-15 - Existing DiffusionDB P1 Inverse-Overlap Quality Recompute

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| Existing P1 outputs, post-color overlap | No attack rerun; recomputed valid inverse-warp overlap against watermarked input | DiffusionDB 1001 existing P1 attacked-watermarked outputs | Mean PSNR 20.097515; median 20.104606; mean SSIM 0.564739; median 0.571788; NaN/Inf 0 | Existing records had enough path and flow metadata to correct quality metrics without rerunning attack. |
| Existing P1 outputs, raw full image | Same images, no overlap crop | Same | Mean PSNR 14.897020; mean SSIM 0.439846 | Full-image metrics are lower because shifted non-corresponding regions are included; paper-comparable local protocol should use overlap fields. |

Source: `outputs/raven_paper_nfpa_gap_fill/diffusiondb_quality_recompute_20260715T034545Z/quality_summary.md`.

### 2026-07-15 - Color Transfer Formula Comparison, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| no_color_transfer | Reused pre-color `view_guided_output.png` directly | Existing 10 DiffusionDB validation outputs; no inversion/denoising rerun; NFPA-style complex L1 scoring | Mean L1 76.687901; overlap PSNR 23.090; overlap SSIM 0.6654; saturated ratio 0.015703; L mean/std errors 0.715407/0.385719 | Highest image similarity because no chroma/luminance correction is applied, but luminance mean error is worse than color-transfer modes. |
| direct_stats | Old local formula: match raw generated `L_opt` stats to original `L_w` | Same | Mean L1 81.677802; overlap PSNR 19.608; overlap SSIM 0.5861; saturated ratio 0.103858; L mean/std errors 0.081025/0.206632 | Legacy ablation retained; good luminance matching, but not the requested paper two-stage formula. |
| paper_exact_two_stage | Build `(L_opt,a_w,b_w)`, LAB->RGB->LAB, match realized `L_c` stats to `L_w` | Same | Mean L1 81.570660; overlap PSNR 19.579; overlap SSIM 0.5844; saturated ratio 0.109624; L mean/std errors 0.057105/0.146216 | New default follows requested paper formula and improves luminance mean/std matching versus direct_stats on this cohort. |

Source: `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.md`.

### 2026-07-14 - P1 Full Fixed `-log10(p)` Evaluation

| Dataset | N | Clean FPR | Before TPR | Attacked TPR | Attack success | ROC-AUC | PSNR vs WM | SSIM vs WM | Conclusion |
| --- | -: | -: | -: | -: | -: | -: | -: | -: | --- |
| DiffusionDB | 1001 | 0.009990 | 1.000000 | 0.688312 | 0.311688 | 0.943715 | 20.098 | 0.5647 | P1 lowered old DiffusionDB attacked TPR by 0.055944 versus the prior old-pipeline result, but many Tree-Ring marks remained detectable. |
| MS-COCO | 1000 | 0.012000 | 1.000000 | 0.634000 | 0.366000 | 0.934490 | 19.743 | 0.6063 | Fixed threshold clean FPR differed from exactly 1%; do not call this COCO result dataset-calibrated TPR@1%FPR. |

Sources: `outputs/raven_p1_full/combined_summary.json`; `outputs/raven_p1_full/diffusiondb/20260714T095907Z/aggregate_results.json`; `outputs/raven_p1_full/mscoco/20260714T095907Z/aggregate_results.json`.


### 2026-07-15 - Latest Formal DiffusionDB Rerun Gates

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| 2-sample smoke | New attacked-watermarked plus new attacked-clean, both `raven_paper_nfpa_gap_fill`, nearest/reflection, paper-exact color transfer | DiffusionDB first 2 samples; NFPA-style complex L1; separate before/after clean calibration | Before TPR 1.000000; after TPR 1.000000; NaN/Inf 0; duplicate exact/rounded groups 0 | Integration path works but N=2 is not statistical. |
| 10-sample validation | Same formal settings with deterministic shift plan | DiffusionDB first 10 samples; NFPA-style complex L1 | Before TPR 1.000000; after TPR 0.700000; attack success 0.300000; mean attacked-WM L1 81.251940; overlap PSNR vs WM 20.005228; overlap SSIM vs WM 0.594788; NaN/Inf 0 | Gate passed; safe to start full 1001 DiffusionDB run using the same scripts/config. |

Sources: `outputs/raven_tr_full_diffusiondb/20260715T060017Z/smoke2_eval/aggregate_results.md`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.md`.

## Open Or In-Progress Items

| Date | Item | Current evidence | Next verification |
| --- | --- | --- | --- |
| 2026-07-15 | Whether to expand `RAVEN-paper / NFPA-gap-fill` to 100-200 or full 1001 | 10-sample validation is complete, but full new-transform attack has not been run. | If requested, first run 100-200 samples in a new output directory; do not overwrite old P1 outputs. |
| 2026-07-15 | Paper PSNR/SSIM provenance | RAVEN arXiv HTML inspected; local report treats overlap PSNR/SSIM as requested paper-comparable protocol, while the inspected paper quality table emphasizes FID/CLIP. | If exact PSNR/SSIM numbers are required for a table, cite the user-defined overlap protocol separately from paper-reported FID/CLIP. |
