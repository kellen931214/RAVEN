# Evaluation Benchmark for In-generation Watermarking in Diffusion Models

This repository provides an evaluation benchmark for diffusion-model watermarking methods, with a focus on in-generation watermark schemes and their robustness under watermark removal and forgery attacks.

<!-- ## Updates -->

<!-- **Code Release:** The code, instructions, and experiment scripts will be released soon. Please stay tuned. -->

## Benchmark Coverage

The benchmark is organized around three evaluation targets:

1. Watermark generation and detection
2. Watermark removal attacks
3. Watermark forgery attacks

## Supported Watermark Schemes

Select the watermark scheme via the `--wm_type` argument. All core implementations are located in `eval_bench_wm/utils/wm/`.

| Method | `--wm_type` | Implementation | Venue |
| :--- | :--- | :--- | :--- |
| [Tree-Ring](https://arxiv.org/abs/2305.20030) | `TR` | `tr_provider.py` | NeurIPS 2023 |
| [Gaussian Shading](https://arxiv.org/abs/2404.04956) | `GS` | `gs_provider.py` | CVPR 2024 |
| [RingID](http://arxiv.org/abs/2404.14055) | `RID` | `ringid_provider.py` | ECCV 2024 |
| [PRC](https://arxiv.org/abs/2410.07369) | `PRC` | `prc_provider.py` | ICLR 2025 |
| [TAG](https://arxiv.org/abs/2506.23484) | `TAG` | `tag_provider.py` | ICCV 2025 |
| [HSTR](https://arxiv.org/abs/2509.07647) / [HSQR](https://arxiv.org/abs/2509.07647) | `HSTR` / `HSQR` | `hs{tr,qr}_provider.py` | ICCV 2025 |
| [GaussMarker](https://arxiv.org/abs/2506.11444) | `GM` | `gm_provider.py` | ICML 2025 |
| [MaXsive](https://arxiv.org/abs/2507.21195) | `MAXSIVE` | `maxsive_provider.py` | ACM MM 2025 |
| [Shallow Diffuse](https://arxiv.org/abs/2410.21088) | `SHALLOW` | `shallow_provider.py` | NeurIPS 2025 |
| [T2S](https://arxiv.org/abs/2510.22366) | `T2S` | `t2s_provider.py` | NeurIPS 2025 |
| [SPH](https://openreview.net/pdf?id=2eAGrunxVz) | `SPH` | `sph_provider.py` | ICLR 2026 |

## Supported Removal Attacks

Execute removal experiments using `eval_bench_wm/run_removal.py` with the corresponding `--rm_method` option.

| Attack | `--rm_method` | Venue |
| :--- | :---: | :--- |
| [NFPA](https://openreview.net/pdf?id=yO2zE1yIYZ) | `nfpa` | NeurIPS 2025 |

---

## Supported Forgery Attacks

All forgery execution scripts are located under the `eval_bench_wm/` directory.

| Attack | Execution Script | Venue |
| :--- | :--- | :--- |
| [Reprompting forgery](https://arxiv.org/abs/2412.03283) | `run_reprompting.py` | CVPR 2025 |
| [Imprint forgery](https://arxiv.org/abs/2412.03283) | `run_imprint_forgery.py` | CVPR 2025 |

## Main Scripts

| Script | Purpose |
| :--- | :--- |
| `run_watermark.py` | Generate and evaluate watermarked images. |
| `run_verify_watermark.py` | Standalone GaussMarker verification, threshold calibration and paper evaluation from a saved bundle. |
| `run_no_watermark.py` | Generate clean, non-watermarked images. |
| `run_removal.py` | Run watermark removal attacks and evaluate detection performance. |
| `run_reprompting.py` | Run reprompting-based forgery evaluation. |
| `run_imprint_forgery.py` | Run imprinting-based forgery evaluation. |

## Supported Diffusion Backbones

The benchmark scripts include support for multiple text-to-image diffusion backbones, including:

<table>
<tr>
<td valign="top" width="50%">

- [x] SD 1.4
- [x] SD 1.5
- [x] SD 2.1 Base
- [x] SDXL

</td>
<td valign="top" width="50%">

- [x] SD3 / 3.5
- [x] PixArt-Sigma
- [x] FLUX.1-dev
- [ ] Sana
</td>
</tr>
</table>

## Quick Start
### 1. Environment Setup

```bash
cd eval_bench_wm
conda create -n eval_bench_wm python=3.11
conda activate eval_bench_wm
pip install -r requirements.txt
```

### 2. Running Experiments

```bash
## Generate and evaluate watermarked images:
python run_watermark.py --wm_type GS --num 1

## Run NFPA removal:
python run_removal.py --wm_type GS --rm_method nfpa --num 1 --save_images

## Run reprompting forgery:
python run_reprompting.py --wm_type GS --num 1

## Run imprint forgery:
python run_imprint_forgery.py --wm_type GS --cover_image_path images/stalin.jpg
```

Note that some watermarking methods (e.g., GM, SHALLOW, MaXsive) are tested for 4-channel latent diffusion models only. We therefore recommend evaluating them on SDXL or SD2.1. We have not yet extended these methods to 16-channel latent models.

---

## GaussMarker (`--wm_type GM`)

The GaussMarker path is implemented for parity with the official code at

```text
https://github.com/SunnierLee/GaussMarker
commit 4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155
```

That commit is recorded in every bundle manifest, threshold artifact, result row
and test. All GaussMarker algorithms live in `utils/wm/gm_provider.py`; the
runners only parse CLI arguments, enumerate images and serialize results.
`utils/wm/gm_bundle.py` holds the bundle schema/hashing and `utils/wm/gm_runtime.py`
holds shared IO/ROC bookkeeping. There is no second GaussMarker implementation
(`tests/gm_official_reference.py` is a test-only transcription of the official
code used for parity assertions and is never imported by runtime code).

### Operating modes

The implementation distinguishes two modes and never conflates them:

| Mode | What it is | Label |
| :--- | :--- | :--- |
| `paper_eval` | The official paper protocol: **verifiably paired** watermarked positives + non-watermarked negatives, same inversion/detector path, ensemble probability, ROC, TPR at the requested FPR and the resulting *experiment-specific* threshold. | `official_paper_evaluation` |
| `verify` | A RAVEN **deployment extension**: fixed-threshold decisions for individual suspect images from a bundle plus a compatible pre-calibrated threshold. | `deployment_verification_extension` / `calibrated_deployment_verification` / `user_supplied_threshold` |

Every result carries one of the labels
`official_paper_evaluation`, `official_profile_raw_scores`,
`calibrated_deployment_verification`, `deployment_verification_extension`,
`user_supplied_threshold`, `legacy_or_ablation_mode`.
A single-image fixed-threshold decision is never labelled as paper evaluation.

### The `official_sd21` profile

`--gm_profile official_sd21` (the default) sets and validates the official
values below. Generic RAVEN-wide parser defaults never override them. Any value
given explicitly on the command line is recorded in `gm_profile_overrides`, and
the run is then reported as an ablation (`gm_profile_is_official=false`).

| Generation | Value | | Detection | Value |
| :--- | ---: | --- | :--- | ---: |
| model | `stabilityai/stable-diffusion-2-1-base` | | inversion prompt | `""` |
| revision / dtype | `fp16` / `float16` | | inversion guidance | `1.0` |
| scheduler | `DPMSolverMultistepScheduler` | | inversion steps | `50` |
| resolution | `512` | | target FPR | `0.01` |
| steps / guidance | `50` / `7.5` | | `classifier_type` | `0` |
| `channel_copy`/`w_copy`/`h_copy` | `1`/`8`/`8` | | `model_nf` | `128` |
| generation FPR / `user_number` | `1e-6` / `1000000` | | VAE | posterior **sampling**, scale `0.18215` |
| `w_seed`/`w_channel`/`w_radius` | `999999`/`3`/`4` | | | |
| `w_pattern`/`w_mask_shape` | `ring`/`circle` | | | |
| `w_measurement`/`w_injection` | `l1_complex`/`complex` | | | |

### Bundle layout

```text
<gm_bundle>/
├── manifest.json
├── w1.pth
├── w2.pth
└── threshold.json   # only after calibration or explicit import
```

`w1.pth` is written in the exact official representation
(`{"w": Tensor, "m": ndarray[16384], "key": bytes[32], "nonce": bytes[12]}`) and
`w2.pth` is the official complex `(1, 4, 64, 64)` tensor, so both files are
directly interchangeable with `gaussmarker_gen.py` / `gaussmarker_det.py`.
The manifest binds the bundle to the model/revision/dtype, scheduler, latent
shape, ring configuration, GNR and classifier artifact hashes, inversion
configuration, code commit and the official reference commit. The key and nonce
are secrets: only their SHA256 appears in the manifest, and they never appear in
logs, CSV files or console output. Existing bundles are validated and are never
silently overwritten or regenerated.

Reusing an existing bundle **fails closed** unless the manifest and the current
run agree on every field in `gm_bundle.REQUIRED_BUNDLE_COMPAT_FIELDS`: model
ID/revision, torch dtype, scheduler, resolution, inversion prompt/guidance/steps,
VAE sampling and scaling factor, GNR SHA, classifier SHA and type, `model_nf`,
and the copy/ring configuration. A required field that is *absent* from the
manifest is also a rejection — an old or hand-edited manifest can never relax the
gate. Because the GNR and classifier hashes are part of that gate, adding a GNR
checkpoint after a bundle was created means building a new bundle that imports
the same `w1`/`w2` (`--gm_w1_path` / `--gm_w2_path`), which keeps the watermark
identity while recording the new detector configuration.

Officialness can only ever be *downgraded* by a bundle: a bundle created under an
ablation or with profile overrides keeps `profile_is_official=false`, and results
produced from it stay `legacy_or_ablation_mode` even when the current command
line selects the official profile. Runs record both `gm_cli_profile_is_official`
and `gm_bundle_profile_is_official` alongside the effective
`gm_profile_is_official`.

### Commands

```bash
# 1) Generate N watermarked images. Creates the bundle on first use and reuses it after.
python run_watermark.py \
    --wm_type GM --num 10 --seed 0 \
    --out_dir out/gm_generation --gm_bundle_dir out/gm_bundle

# 2) Standalone verification in a fresh process (no prompt, no original image needed)
python run_verify_watermark.py \
    --wm_type GM --gm_bundle_dir out/gm_bundle \
    --suspect_path images/or_directory --out_dir out/gm_verification

# 3) Calibrate a threshold from a clean and a same-bundle watermarked cohort
python run_verify_watermark.py --mode calibrate \
    --gm_bundle_dir out/gm_bundle \
    --positive_path out/gm_generation/images/watermarked \
    --negative_path out/clean_images \
    --out_dir out/gm_calibration

# 4) Official paper evaluation over paired cohorts (ROC, TPR@FPR, cohort threshold)
python run_verify_watermark.py --mode paper_eval \
    --gm_bundle_dir out/gm_bundle \
    --positive_path out/gm_generation/images/watermarked \
    --negative_path out/clean_images \
    --out_dir out/gm_paper_eval
```

Generation writes:

```text
<out_dir>/
├── images/watermarked/000000.png …
├── prompts/000000.txt …
├── sample_metadata/000000.json …
├── results.jsonl
└── run_manifest.json
```

`run_manifest.json` is the resume gate for the whole output directory and is
validated **before anything is written**: a `run_config_sha256` mismatch stops the
run with the directory byte-for-byte untouched, and a matching manifest is kept
verbatim (so `created_utc` and the recorded provenance stay those of the run that
actually produced the existing samples). A new manifest is only ever created when
none exists. Because that gate runs before any bundle may be created, a run whose
`run_manifest.json` already exists must point `--gm_bundle_dir` at the complete
bundle that produced it; the bundle is then loaded read-only. A new bundle is only
created for a genuinely new output run.

### Paired cohorts for `paper_eval`

Equal cohort sizes are **not** pairing. `paper_eval` only reports
`official_paper_evaluation` when every positive is bound to exactly one negative
through verified pairing provenance:

* an explicit `--pair_manifest`, or
* matching per-sample metadata sidecars, looked up as
  `<cohort>/../sample_metadata/<stem>.json` or `<cohort>/<stem>.json`.

A pair manifest is a **standalone** source of pairing provenance — no sidecar is
required when it declares the fields itself:

```json
{"protocol": "official_paper_eval",
 "pairs": [{"positive": "000000.png", "negative": "000000.png",
            "sample_id": 0, "prompt_sha256": "…",
            "positive_sample_seed": 0, "negative_sample_seed": 1000,
            "distortion_config_sha256": "…", "distortion_seed": 7}]}
```

Each field may be declared once for the pair (`sample_seed`) or per side
(`positive_sample_seed`). When sidecars *are* present they are cross-validated
against the manifest and any disagreement is a rejection.

The pairing must be a bijection: every positive and every negative image takes
part in exactly one pair, and the images the manifest references must be exactly
the two input cohorts. Referencing an image twice, or leaving one unpaired, fails
closed even when the manifest declares the right number of entries.

Each pair must agree on `sample_id` and `prompt_sha256`; the generation
`sample_seed` must match as well unless an explicit pair manifest declares the
pairing. When a distortion/attack was applied, both sides must carry identical
`distortion_config_sha256` and `distortion_seed`, and the declared `protocol` is
recorded. A pair missing any required field — from both the manifest and the
sidecar — fails closed. The verified pairing is hashed into `pairing_sha256` in the summary.
Without complete pairing provenance the run fails closed; `--allow_unmatched_cohorts`
downgrades it to `legacy_or_ablation_mode` instead and never to an official
paper evaluation.

The bundle identity (bits, encrypted message, key/nonce, ring target) is fixed
for the run, while **every sample independently samples its complete initial
latent** from the deterministic seed `--seed + sample_id`, exactly as official
`gaussmarker_gen.py` does. Reruns resume only when the sample seed, prompt hash,
run-configuration hash and bundle hash all match; otherwise the run fails closed
and no image is ever overwritten.

### Detector data flow and thresholds

```text
recovered latent zT_hat
├── raw_m = (zT_hat > 0) → GNR → ChaCha20 decrypt → copy-dimension voting → restored bit accuracy
└── FFT(zT_hat) → masked complex L1 to w2 → ring feature = -0.01 × raw ring L1
ensemble features = [restored_bit_accuracy, -0.01 × raw_ring_l1]
score = classifier.predict_proba(features)[:, 1]
```

The ring distance is always computed from the original continuous recovered
latent, never from the GNR output, the thresholded sign map or the restored
binary map. Raw detector scores (`raw_bit_accuracy`, `raw_ring_l1`,
`ring_classifier_feature`) are emitted even when no threshold is available.

Binary decisions use `score >= threshold` and require one of:

1. a compatible calibrated `threshold.json` stored with the bundle;
2. an explicit calibration run (mode `calibrate`);
3. an explicit `--gm_threshold`, labelled `user_supplied_threshold`.

A threshold artifact records the score definition, direction, comparison
operator, target and empirical FPR, TPR at the target FPR, ROC-AUC, cohort sizes
and hashes, model/inversion configuration, GNR/classifier/bundle hashes and the
calibration commit. Verification rejects an incompatible threshold artifact.

### Required artifacts

* `GM_utils/sd21_cls2.pkl` — the official ensemble classifier (shipped, byte-identical
  to the official repository; built with scikit-learn 1.5.2, so `scikit-learn` and
  `joblib` must be installed).
* `GM_utils/GNR_bits256/model_final.pth` — the GNR checkpoint. It is **not**
  shipped; train it with the official `train_GNR.py` or pass `--gm_gnr_path`.
  Without it, restored bit accuracy and the ensemble probability are unavailable:
  raw scores are still emitted, no binary decision is fabricated, and the cohort
  modes fail closed.

### Backward compatibility

* Outputs that reused a single complete initial latent across samples are
  **invalid for formal evaluation**.
* Outputs claimed as official GaussMarker without key/nonce-compatible ChaCha20
  state are **invalid**; `w1` files without `key`/`nonce` are now rejected rather
  than silently relabelled as official-compatible.
* Outputs without threshold provenance may keep their raw scores, but their
  binary decisions are **legacy**.
* Outputs without sufficient artifact/configuration hashes are **not
  independently auditable**.

## Acknowledgement

This repository heavily relies on and builds upon the [semantic-forgery](https://github.com/and-mill/semantic-forgery). We sincerely thank the authors for open-sourcing their code, which served as a foundational component for our evaluation.