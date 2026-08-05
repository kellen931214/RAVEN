# Issue #29 — real-artifact unified validation report

Branch: `feature/issue29-real-validation`
Final HEAD: `<d5cf9ee>` (before merge from feature/worktree)

All runs: `python3 experiments/eval.py --output-dir outputs/real_validation/<method>/diffusiondb/n2 --device cuda --stages detector`

Dataset: `diffusiondb_shared_tr` (GS, GM, T2S, RID, HSTR, HSQR) / `diffusiondb` (TR)
Clean images: `/workspace/RAVEN/data/clean/diffusiondb/`
Model: `RedbeardNZ/stable-diffusion-2-1-base` @ `c6a5e9bab8d874d081de76fa270ae0aefa5410ff`, DDIM 50 steps, 512×512
Test evidence: 135/135 TR targeted tests passed; T2S contract regression tests passed (2/2)

---

## GS — pass

- Cohort classification: formal
- Dataset: `diffusiondb_shared_tr`
- Run root: `outputs/real_validation/gs/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/gs/diffusiondb_shared_tr/GS/metadata.csv`
- Source metadata SHA256: `f77aaa7c003690b8a9e6c5f90c5e448cfe36211cd0573af0f231abda5d83d9db`
- Requested: 8, Scored: 4, Failed: 4 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image` (attacked cohorts absent; all available rows scored)
- Raw score: `bit_accuracy` (= `raw_score` = `canonical_score`)
- Score direction: `higher_is_watermarked`
- Provider identity: GS per-sample secret (index=run_id), secret_bundle_sha256 verified
- Provider config hash: `e11ff3985d9fb86ab0047b35005ab23d422561135bbc36bc9005ce50c9f559eb` (derived from real CSV fields)
- Source target SHA256: `5a14020949a306a0b0891c128869a19a6a5baa82921b5f0e9edfd9d6bd8cc8df` (verified)
- Source mask SHA256: `f80e7f814ec3c12e1fde467c29f60eb2de0efc69edbae02396f60106368f7ece` (verified)
- Threshold source: `official_beta_tail_tau_onebit = 0.6484375`, fpr 1e-06
- Comparison operator: `>=`
- Detection mode: `official_onebit`
- Method-specific metrics: bit_accuracy 1.0 (wm), 0.531/0.457 (clean)
- Trusted reference: `_table/experiment_results.md`, N=1001, tau_onebit 0.6484375, before bit_accuracy 1.0
- Absolute metric differences: tau 0.0 (exact); before-score 0.0 (exact)
- Parity classification: **exact agreement** on tau values and before-score

| run_id | cohort | bit_accuracy | tau_onebit | decision |
|---|---|---|---|---|
| 0 | original_watermarked | 1.0 | 0.648 | detected |
| 0 | original_clean | 0.531 | 0.648 | not detected |
| 1 | original_watermarked | 1.0 | 0.648 | detected |
| 1 | original_clean | 0.457 | 0.648 | not detected |

---

## GM — pass

- Cohort classification: formal
- Dataset: `diffusiondb_shared_tr`
- Run root: `outputs/real_validation/gm/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/gm/diffusiondb_shared_tr/GM/metadata.csv`
- Source metadata SHA256: `ffb2a3d2a52826e157db...`
- Requested: 8, Scored: 4, Failed: 4 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image`
- Raw score: `gm_raw_bit_accuracy` (= `raw_score` = `canonical_score`)
- Score direction: `higher_is_watermarked`
- Bundle dir: `/workspace/RAVEN/data/gm/diffusiondb_shared_tr/bundle/`
- Bundle manifest SHA256: `dabfb33edfb5440b572c7b392a4ec25ba399107a9ecdba832191b9e7e46ccdd0`
- Bundle config SHA256: `4b3713a0e35685d70864784d864c9d1b64e39b154788958e4add03240c7aaf0b`
- w1 SHA256: `5ba5da1de57a343f26c1193ff70890b8cb0cbbc6dac2c8313b8ccb08593ecf70`
- w2 SHA256: `41996186643b46158e81ab677837ac90ff05862e0a62e6ada934caa421701be5`
- Source target SHA256: `9ac25b60be6b2bddd4140d778afadfb2da271647d1de3fdbd4c28e366889fca5` (verified)
- Source mask SHA256: `85e3ea9c33b8e60b33567d55c9b225671631360b5eaa859535c95f9b298dab8f` (verified)
- Profile: `legacy`, `profile_is_official = false`, `gm_state_source = bundle`
- GNR: not used; Classifier: not used
- Method-specific metrics: gm_raw_bit_accuracy 1.0 (wm), 0.551/0.465 (clean)
- Trusted reference: `_table/experiment_results.md`, N=1001, before gm_raw_bit_accuracy ≈ 0.999996
- Absolute metric differences: before-score 3.90e-06 (2-sample 1.0 vs 1001-sample 0.999996)
- Parity classification: exact agreement within numerical precision

| run_id | cohort | gm_raw_bit_accuracy | gm_raw_ring_l1 |
|---|---|---|---|
| 0 | original_watermarked | 1.0 | 52.51 |
| 0 | original_clean | 0.551 | 76.30 |
| 1 | original_watermarked | 1.0 | 46.27 |
| 1 | original_clean | 0.465 | 72.14 |

---

## T2S — pass (after production adapter fix)

- Cohort classification: formal
- Dataset: `diffusiondb_shared_tr`
- Run root: `outputs/real_validation/t2s/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/t2s/diffusiondb_shared_tr/T2S/metadata.csv`
- Source metadata SHA256: `2fcb07c2c77383b15216...`
- Requested: 4, Scored: 2, Failed: 2 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image`
- Raw score: `t2s_score_true_key`
- Score direction: `higher_is_watermarked`
- Decision rule: `paired_key_comparison (score_true_key > score_control_key)`
- State artifacts: `/workspace/RAVEN/data/t2s/diffusiondb_shared_tr/T2S/watermark_state/00000{0,1}.json`
- State SHA256 (run 0): `e361759f4b5b6d2363279049ca632d0afcfb185e1db5505cc8e0a6a44351e686`
- Provider config SHA256: `e39d4dd5cc0b7a0f771513b7a3959055d7a68aae47aed3cbbb2ca61780a18c5d`
- Protocol mode: `official_encoder_shared_tr_clean`
- Inversion: `t2s_official`, RNG: `official_compatible`, steps: 10
- Method-specific metrics: key_accuracy 1.0, message_accuracy 1.0, bit_accuracy 1.0
- Trusted reference: `_table/experiment_results.md`, N=1001, before t2s_score_true_key 1.0/0.999996
- Absolute metric differences: 2-sample true_key vs 1001-sample; cohort difference only
- Parity classification: **tolerated numerical drift** (2-sample vs 1001-sample, same method semantics)
- Production defect: adapter wrapped tensor in `{"zT_torch": zT}` dict; scorer expected bare tensor. Fixed in `1345680` with `isinstance(zT, torch.Tensor)` guard.
- Regression test: `TestInversionTensorContract` — 2 tests

| run_id | true_key | control_key | margin | detection | key_acc | msg_acc | bit_acc |
|---|---|---|---|---|---|---|---|
| 0 | 1510.93 | 110.48 | 1400.45 | true | 1.0 | 1.0 | 1.0 |
| 1 | 1824.11 | 169.67 | 1654.44 | true | 1.0 | 1.0 | 1.0 |

---

## RID — pass

- Cohort classification: formal
- Dataset: `diffusiondb_shared_tr`
- Run root: `outputs/real_validation/rid/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/rid/diffusiondb_shared_tr/RID/metadata.csv`
- Source metadata SHA256: `0aeb89c09bc0f6915b9b...`
- Requested: 8, Scored: 4, Failed: 4 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image`
- Raw score: `rid_neg_channel_min_complex_l1` (raw = raw_l1, canonical = -raw)
- Raw direction: `lower_is_watermarked`, Canonical direction: `higher_is_watermarked`
- Comparison operator: `>=`
- Bundle dir: `/workspace/RAVEN/data/rid/diffusiondb_shared_tr/bundle/`
- Bundle config SHA256: `456cf9e5a402eb115583d3df334a5e4b517b36d3db3f6660321f4d9c0d9f7bcd`
- Selected pattern SHA256: `3cdc758baed14965c9fef928ac4e7940a5481fa6b8b02036e99149f09f56f56a`
- Mask SHA256: `5b39405f4828d60c5a55d6ef01a190f1c4ce5e08da910c173c66c7ae94d88b05`
- Key index: 628, protocol: `official_math_shared_tr_clean`
- Method-specific metrics: wm canonical -37/-32, clean canonical -76/-80
- Trusted reference: `_table/experiment_results.md`, N=1001, before canonical score -26.19
- Absolute metric differences: 2-sample wm mean ≈ -34 vs 1001-sample mean -26.19
- Parity classification: **cohort difference** (2-sample vs 1001-sample mean)

| run_id | cohort | raw (L1) | canonical (-raw) |
|---|---|---|---|
| 0 | original_watermarked | 37.27 | -37.27 |
| 0 | original_clean | 75.63 | -75.63 |
| 1 | original_watermarked | 31.55 | -31.55 |
| 1 | original_clean | 79.60 | -79.60 |

---

## HSTR — pass

- Cohort classification: formal
- Dataset: `diffusiondb_shared_tr`
- Run root: `outputs/real_validation/hstr/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/hstr/diffusiondb_shared_tr/HSTR/metadata.csv`
- Source metadata SHA256: `6ca4a3e43884dd757720...`
- Requested: 8, Scored: 4, Failed: 4 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image`
- Raw score: `hstr_score=-min(channel_0_l1, channel_3_l1)` (raw = raw_l1, canonical = -raw)
- Raw direction: `lower_is_watermarked`, Canonical direction: `higher_is_watermarked`
- Comparison operator: `>=`
- Bundle dir: `/workspace/RAVEN/data/hstr/diffusiondb_shared_tr/bundle/`
- Bundle config SHA256: `1b8ad4f7e32976fea770e69434107e66978a9fd405a3f0912c35c4f6de684f1f`
- Selected pattern SHA256: `1f8f5193b0b3968885bf1b440af4ab844e0e9a6acec5f1aea4a2eeec8119464e`
- Protocol: `official_math_shared_tr_clean`, state_source: `bundle_created`
- Method-specific metrics: wm canonical -25/-22, clean canonical -45/-48
- Trusted reference: `_table/experiment_results.md`, N=1001, before canonical score -17.31
- Absolute metric differences: 2-sample vs 1001-sample mean
- Parity classification: **cohort difference**

| run_id | cohort | raw (L1) | canonical (-raw) |
|---|---|---|---|
| 0 | original_watermarked | 25.34 | -25.34 |
| 0 | original_clean | 45.47 | -45.47 |
| 1 | original_watermarked | 21.75 | -21.75 |
| 1 | original_clean | 47.72 | -47.72 |

---

## TR — pass (exact deterministic parity)

- Cohort classification: formal
- Dataset: `diffusiondb`
- Run root: `outputs/real_validation/tr/diffusiondb/n2/`
- Source metadata: `/workspace/RAVEN/data/tr/diffusiondb/metadata.csv` (derived wpc variant)
- Integration commit: `943c380` (cherry-picked from issue #28 `9cb7871`)
- Requested: 8, Scored: 4, Failed: 4 (attacked missing)
- Attacked-image count: 0
- Detector stage status: `failed_missing_image`
- Raw score: `complex_l1_mean` (= `torch.abs(decoded - target).mean()`)
- Raw direction: `lower_is_watermarked`
- Canonical score: `-raw`, direction: `higher_is_watermarked`
- Comparison operator: `>=`
- Score protocol: `complex_l1_mean` (package-local `tr_scoring.py`, no legacy import)
- Provider config hash: `07496dc8e113bddc8ffed671b394adbcd3c83970e01f79d22d028adf1dc74c9e` (verified)
- Source target SHA256: `087e4198bb56d0d907ec502eb7ff35e7369ea72d5c136950b05d3021830502a3` (verified)
- Source mask SHA256: `6636fc4a74bdeb1a7c80b362e70e429acbf9e4f459fc0593b3c2c49b8659a6d6` (verified)
- Model: `RedbeardNZ/stable-diffusion-2-1-base`, revision `c6a5e9bab8d874d081de76fa270ae0aefa5410ff`, DDIM, 512, float32
- Targeted tests: 135/135 passed
- Trusted reference: issue #28 n2 run (`9cb7871`)
- Parity classification: **exact deterministic parity** — all 4 canonical scores diff = 0.00e+00

| run_id | cohort | raw (L1) | canonical (-raw) | decoded_abs_mean | target_abs_mean |
|---|---|---|---|---|---|
| 0 | original_watermarked | 57.66 | -57.66 | 45.23 | 64.68 |
| 0 | original_clean | 80.54 | -80.54 | 50.72 | 64.68 |
| 1 | original_watermarked | 55.33 | -55.33 | 45.38 | 64.68 |
| 1 | original_clean | 82.21 | -82.21 | 51.59 | 64.68 |

---

## HSQR — blocked

- Classification: **provenance mismatch / original generation-time mask artifact unavailable**
- Dataset: `diffusiondb_shared_tr`
- Source metadata: `/workspace/RAVEN/data/hsqr/diffusiondb_shared_tr/HSQR/metadata.csv`
- Source metadata SHA256: `446ca4b69c8dd1b11bc8d631d6ecf4fd20f8d7d3b405f487c167032365c1a81a`
- Bundle dir: `/workspace/RAVEN/data/hsqr/diffusiondb_shared_tr/bundle/`
- Manifest SHA256: `e55f41d19b669d1c8f2dae0bd2aa379f94764f7ed7f1bca49444ef962c72fb46`
- Requested: 8, Scored: 0, Failed: 8
- Error: `DetectorStateValidationError: detector/source mask SHA mismatch`
- Source `hsqr_mask_sha256`: `0416a022147a87305ec8ce0f3e5fec496cda82f55dab5302f0470f47ad2d6d5e`
- Detector-computed mask SHA256: `83f2e3e87e1760def35c60a479321c90d26e415c0166debb6675d5506de8af5a`
- Evidence:
  * Bundle: `manifest.json` + `selected_pattern.pt` — no `watermark_mask.pt`
  * Manifest has `selected_pattern_sha256` but no `mask_sha256` or `mask_file_sha256`
  * CSV `hsqr_mask_sha256 = 0416a022...` constant across all 1001 rows
  * Value `0416a022...` not in any persisted artifact
  * Cannot prove whether metadata stale, bundle drift, or derivation changed
- No metadata, bundle, or validation policy modified.

---

## Key commits

| Commit | Description |
|---|---|
| `1345680` | fix(t2s): pass inverted latent tensor directly to state scorer |
| `b841245` | test(t2s): lock inverted-latent scorer contract |
| `943c380` | feat(tr): integrate package-local complex-L1 scoring |
| `d5cf9ee` | docs(issue29): promote TR to pass |

T2S regression tests: 2/2 passed.
TR targeted tests: 135/135 passed.

---

## Legacy runtime dependency disposition (for issue #27)

| Artifact | Status | Evidence |
|---|---|---|
| `raven_repro/raven/detectors/tr_detector.py` | **safe to remove legacy** | Uses package-local `tr_scoring.py`; exact parity verified |
| `raven_repro/raven/detectors/gm_detector.py` | **still runtime-required** | Loads `extract_verification_scores.py` for bundle manifest / provider kwargs / scoring |
| `raven_repro/raven/detectors/gs_detector.py` | **still runtime-required** | Imports `extract_verification_scores.provider_kwargs` at lines 502, 1135 |
| `raven_repro/raven/detectors/fourier_detector.py` | **still runtime-required** | Loads `extract_verification_scores.py` for RID/HSTR/HSQR bundle/provider/scoring |
| `raven_repro/raven/detectors/t2s_detector.py` | **safe to remove legacy** | No reference to legacy scripts |
| `experiments/run_raven_formal_eval.py` | **reference-only** | References legacy scripts for legacy eval path |
| `experiments/run_raven_no_color_eval.py` | **reference-only** | References legacy scripts for legacy eval path |
| `raven_repro/scripts/extract_verification_scores.py` | **still runtime-required** | Required by GM, GS, Fourier detectors |
| `raven_repro/scripts/raven_nfpa_tr_eval.py` | **reference-only** | Legacy TR eval entrypoint; TR now uses `tr_scoring.py` |
