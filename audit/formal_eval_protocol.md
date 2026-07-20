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
