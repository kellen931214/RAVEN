# Formal Evaluation Protocol

## Authoritative Attack

`raven_repro/raven/eval_protocol.py` is authoritative: model
`RedbeardNZ/stable-diffusion-2-1-base` revision
`c6a5e9bab8d874d081de76fa270ae0aefa5410ff`; DDIM, 50 steps, strength 0.15,
guidance 2.5; empty prompts; image-pixel `raven_paper_nfpa_gap_fill`; nearest latent
sampling, reflection padding, `align_corners=False`; view-guided attention; paper-exact
two-stage color transfer. Formal calls pass every value explicitly. `debug_info.json`
must match all formal fields and its canonical transform hash before a sample is committed.

## Records And Metrics

Snapshots are complete-line, decoded, hashed, unique, fsynced batches. Attack cache
records bind run ID, dataset/method, clean/watermarked/attacked/debug paths and SHA,
snapshot/source hashes, seed, planned and effective flows, exact timestep, model/revision,
full config, transform/config hashes, metric protocol version, and Git HEAD. Resume drift
raises rather than skips or reruns.

Effective source displacement is sampled from the exact NFPA grid using a source-index
map. Quality passes source flow, not the opposite-sign visual shift, to inverse-warp
overlap. Primary names are `post_color_vs_watermarked_overlap_psnr/ssim`.

Detector calibration reports target/actual FPR, threshold, FP count, before TPR,
attacked TPR at the original threshold, ROC-AUC, and attack success. TR additionally
reports attacked-clean recalibration and separate full-precision/rounded2 protocols.
Provider config and target hashes are per-row and uniform per cohort. FID staging is
fresh and config-scoped. CLIP provenance is prompt-image bigG/LAION, OpenCLIP version,
preprocess identifier, original prompt, and L2 normalization. The RAVEN primary paper
specifies `OpenCLIP-ViT/G` and CLIP-Text against prompts but not a pretrained tag.
OpenCLIP's official registry distinguishes lowercase `ViT-g-14/laion2b_s12b_b42k`
from `ViT-bigG-14/laion2b_s39b_b160k`; the current experiment configuration's
uppercase-G interpretation is pinned as bigG and never mixed with historical lowercase-g
records.

## Verification Status

- Static compilation: passed.
- Requested targeted existing plus new protocol tests: `122 passed, 51 warnings`.
- Complete `raven_repro/tests` suite: `148 passed, 51 warnings`.
- Synthetic effective-flow crop: PSNR infinite and SSIM 1 by design; warnings are expected.
- 2-sample new-runner gate: not run.
- 10-sample new-runner gate: not run.
- 30-sample new-runner gate: not run.
- Full 1000/1001/8192 runs: not run.

GPU discovery currently fails with `nvidia-smi: Failed to initialize NVML: Unknown Error`.
Historical gate results are not substituted for new-runner gates. Status is therefore
`NOT SAFE TO RUN FULL EVAL`.

- Color alignment flow source: effective source flow from actual warp grid; planned flow fallback is forbidden.


## Shift-plan provenance clarification (2026-07-21)

The paper states that translations along each axis are randomly sampled from the
full integer ranges [24,32] or [-32,-24]. Formal future runs therefore use
`shift_plan_mode=paper_random_independent_axes`: each axis independently draws a magnitude from every integer 24
through 32 and a sign, using a deterministic RNG keyed by run ID and the
recorded attack seed. Planned flow remains the attack input; quality overlap
continues to use actual-grid effective source flow.

The earlier `formal_deterministic` five-magnitude/four-quadrant schedule is now classified as
`balanced_deterministic_schedule`. Existing outputs are retained as historical ablations and are not
relabeled or modified, but that schedule must not be described as paper-exact
random sampling.

## Shared-Clean Method Registration (2026-07-29)

The formal runner now registers RID, HSTR and HSQR `shared_tr_clean_v2` cohorts
beside TR, GS, GM and T2S. They use the existing snapshot -> attack-watermarked
-> verify -> quality -> FID -> CLIP -> aggregate -> validate stages and the same
attack cache. They do not run attacked-clean recalibration; attacked-clean remains
TR-only.

RID/HSTR/HSQR detector verification is bound to persisted bundles and source
metadata. The verification manifest carries method provenance from the snapshot
and attack record, and validation requires exactly one successful score row per
input row in input order. Missing or duplicate `run_id`, source metadata drift,
clean/watermarked/attacked SHA drift, provider/bundle/target/mask mismatch,
generation-config drift and attack-cache provenance drift are fatal.

Score reporting uses canonical score space for these methods: RID
`rid_neg_channel_min_complex_l1`, HSTR `hstr_score`, and HSQR `hsqr_score`; all
are `higher_is_watermarked` because canonical score is `-raw_l1`. Thresholds are
recorded as `empirical_clean_1pct_fpr` only in canonical-score space, with raw L1
direction recorded separately as `lower_is_watermarked`.
