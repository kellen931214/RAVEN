# Debug Changelog

This file records implementation bugs, validated non-bugs, ablations, and the evidence used to verify each change. Large logs and generated outputs are not copied here; paths point to the source artifacts.

## Current Status

| Date | Area | Status | Evidence |
| --- | --- | --- | --- |
| 2026-07-14 | NFPA-style Tree-Ring complex L1 evaluation | In progress for DiffusionDB only. Existing attacked-watermarked images are reused; only missing attacked-clean images are being generated before L1 scoring. MS-COCO was not started after the scope was corrected. | `outputs/raven_nfpa_tr_eval/logs/run_nfpa_tr_diffusiondb_only_20260714T162739Z.log`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/attacked_clean_records.jsonl` |

## Confirmed Issues And Fixes

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

### 2026-07-14 - NFPA-Style Tree-Ring Complex L1 Metric Was Missing

| Field | Details |
| --- | --- |
| Problem | Existing Tree-Ring evaluation used `-log10(p)` fixed/calibrated threshold logic, while NFPA evaluates Tree-Ring with complex L1 distance `torch.abs(decoded_watermark - target_watermark).mean(-1)` where lower score indicates watermark. |
| Impact | The existing P1 result is useful as separate `-log10(p)` analysis but is not the requested NFPA-style Tree-Ring `TPR@1%FPR`. After-attack NFPA calibration also requires attacked-clean images, not only attacked-watermarked images. |
| Core logic changed | Added an independent `raven_nfpa_tr_eval.py` flow that copies existing P1 attacked-watermarked records, generates only attacked-clean images with the same shift plan/settings, scores original clean/watermarked/attacked-clean/attacked-watermarked with complex L1, and calibrates before/after thresholds separately with NFPA's strict `< threshold` rule. |
| Verification | `prepare` validated 1001 DiffusionDB manifest, shift plan, and attacked-watermarked hashes. DiffusionDB attacked-clean generation is in progress; scoring/aggregate are not complete yet. |
| Evidence | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/source_counts.json`; `outputs/raven_nfpa_tr_eval/logs/run_nfpa_tr_diffusiondb_only_20260714T162739Z.log` |
| Status | In progress. Do not report final NFPA-style DiffusionDB TPR until `nfpa_l1_scores.jsonl` and `aggregate_results.json` are complete. |

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

### 2026-07-14 - P1 Full Fixed `-log10(p)` Evaluation

| Dataset | N | Clean FPR | Before TPR | Attacked TPR | Attack success | ROC-AUC | PSNR vs WM | SSIM vs WM | Conclusion |
| --- | -: | -: | -: | -: | -: | -: | -: | -: | --- |
| DiffusionDB | 1001 | 0.009990 | 1.000000 | 0.688312 | 0.311688 | 0.943715 | 20.098 | 0.5647 | P1 lowered old DiffusionDB attacked TPR by 0.055944 versus the prior old-pipeline result, but many Tree-Ring marks remained detectable. |
| MS-COCO | 1000 | 0.012000 | 1.000000 | 0.634000 | 0.366000 | 0.934490 | 19.743 | 0.6063 | Fixed threshold clean FPR differed from exactly 1%; do not call this COCO result dataset-calibrated TPR@1%FPR. |

Sources: `outputs/raven_p1_full/combined_summary.json`; `outputs/raven_p1_full/diffusiondb/20260714T095907Z/aggregate_results.json`; `outputs/raven_p1_full/mscoco/20260714T095907Z/aggregate_results.json`.

## Open Or In-Progress Items

| Date | Item | Current evidence | Next verification |
| --- | --- | --- | --- |
| 2026-07-14 | NFPA-style complex L1 DiffusionDB result | `prepare` completed and validated source counts; attacked-clean generation is in progress. | Wait for `nfpa_l1_scores.jsonl` to reach 1001 rows and `aggregate_results.json` to be produced, then record before/after thresholds, actual FPR, TPR, attack success, and mean scores. |
| 2026-07-14 | Whether NFPA-style complex L1 changes the conclusion versus `-log10(p)` | 尚未確認. The metric and required attacked-clean calibration are still running. | Compare `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/aggregate_results.json` against P1 fixed `-log10(p)` outputs after completion. |
