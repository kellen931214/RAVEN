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
| `run_verify_watermark.py` | Standalone GM/HSTR verification, threshold calibration and paper evaluation from a saved bundle. |
| `run_no_watermark.py` | Generate clean, non-watermarked images. |
| `run_removal.py` | Run watermark removal attacks and evaluate detection performance. |
| `run_reprompting.py` | Run reprompting-based forgery evaluation. |
| `run_imprint_forgery.py` | Run imprinting-based forgery evaluation. |
| `run_verification.py` | Verify suspect images against a saved portable watermark state, in a standalone process. |

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


## HSTR / SFWMark (`--wm_type HSTR`)

The official HSTR path is aligned to the frozen SFWMark source:

```text
https://github.com/thomas11809/SFWMark
commit 78666128b44614a0cc471993649e3132d5dddfcb
```

Use `--hstr_profile official_sfwmark_sd21` for the official SD2.1 profile:
`stabilityai/stable-diffusion-2-1-base`, DDIM, float32, 512x512, 50 steps,
guidance 7.5, key seed base 7433, center slice `10:54`, radius 14 with cutoff 3,
and score `hstr_score=-min(channel_0_l1,channel_3_l1)`. HSTR algorithm code
stays in `utils/wm/hstr_provider.py`; `utils/wm/sfw_bundle.py` handles bundle
serialization and `utils/wm/sfw_runtime.py` handles IO, provenance and ROC.
HSQR is not changed by this path.

Official generation writes one persistent key bundle and indexed paired outputs:

```bash
python run_watermark.py \
    --wm_type HSTR --hstr_profile official_sfwmark_sd21 \
    --hstr_key_index 0 --num 10 --seed 0 \
    --out_dir out/hstr_generation
```

The default bundle path is `out/hstr_generation/hstr_bundle/` and contains
`manifest.json`, `selected_pattern.pt`, optional `pattern_list-2048.pt` when
`--hstr_save_full_keybook` is used, and `threshold.json` only after calibration.
Images are saved as `images/watermarked/000000.png` and
`images/no_watermark/000000.png`; each sample records its own base seed and
pre/post-injection latent hashes in `sample_metadata/000000.json`.

Standalone verification is a fresh-process deployment extension. It loads only
the bundle and suspect image(s), emits raw per-image scores, and emits a binary
decision only when a compatible calibrated threshold exists or `--hstr_threshold`
is explicitly supplied:

```bash
python run_verify_watermark.py \
    --wm_type HSTR --mode deployment_verify \
    --hstr_bundle_dir out/hstr_generation/hstr_bundle \
    --suspect_path out/hstr_generation/images/watermarked \
    --out_dir out/hstr_verify
```

Paper-style cohort evaluation and calibration use the same detector path,
`sklearn.metrics.roc_curve`, and the official strict operating point `FPR < 0.01`:

```bash
python run_verify_watermark.py \
    --wm_type HSTR --mode paper_eval \
    --hstr_bundle_dir out/hstr_generation/hstr_bundle \
    --positive_path out/hstr_generation/images/watermarked \
    --negative_path out/hstr_generation/images/no_watermark \
    --out_dir out/hstr_paper_eval

python run_verify_watermark.py \
    --wm_type HSTR --mode calibrate --overwrite_threshold \
    --hstr_bundle_dir out/hstr_generation/hstr_bundle \
    --positive_path out/hstr_generation/images/watermarked \
    --negative_path out/hstr_generation/images/no_watermark \
    --out_dir out/hstr_calibration
```

Do not report HSTR as TR/GS metrics. HSTR reports its own score direction
(`higher_is_watermarked`) and threshold family (`empirical_clean_1pct_fpr` when
calibrated from clean negatives).

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

---

## RingID (`--wm_type RID`)

The RingID path is implemented for parity with the official code at

```text
https://github.com/showlab/RingID
commit 45631a59aecd7d63ccdb640aaaf3e616fdb89fb9
```

(paper: *RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key
Identification*, ECCV 2024). That commit is recorded in every bundle manifest,
threshold artifact, result row and test. All RingID algorithms live in
`utils/wm/ringid_provider.py`; `utils/wm/rid_bundle.py` holds the key-artifact
schema/hashing and `utils/wm/rid_runtime.py` holds thin IO/scoring bookkeeping
(reusing the shared GPU preflight, image enumeration, cohort pairing and ROC
helpers). The runners only parse a CLI, enumerate inputs and serialize results.
`tests/rid_official_reference.py` is a test-only transcription of the official
code used for parity assertions and is never imported by runtime code.

### Verification vs. identification

RingID is not "one fixed key". Official `verify.py` uses candidate **628** as a
single verification *example*; official `identify.py` traverses the whole
candidate keybook. RAVEN keeps the two apart:

| Mode | What it is | Label |
| :--- | :--- | :--- |
| `paper_eval_verification` | Official verification protocol: **verifiably paired** watermarked positives + non-watermarked negatives for ONE key, same inversion/detector path, canonical score, ROC, AUC and TPR at the requested FPR plus the resulting *experiment-specific* threshold. The paper uses 1000 + 1000 images; anything smaller is a smoke/debug run. | `official_paper_verification` |
| `identify` / `paper_eval_identification` | Official multi-key identification: the channel-min distance against every candidate in the declared keybook, then `argmin`. Needs no true key; when per-image metadata declares one, identification accuracy is reported. | `official_paper_identification` |
| `verify` | A RAVEN **deployment extension**: fixed-threshold decisions for individual suspect images from a bundle plus a compatible pre-calibrated threshold (or `--rid_threshold`). Not the paper protocol. | `deployment_verification_extension` / `calibrated_deployment_verification` / `user_supplied_threshold` |

### Detector semantics (channel-min, not channel-average)

```text
recovered continuous latent -> FFT
  -> masked complex L1 for channel 0 and channel 3 independently
  -> rid_channel_min_l1 = min(channel_0_l1, channel_3_l1)      # official channel_min=1
  -> rid_score = -rid_channel_min_l1                            # higher_is_watermarked
  -> detected when rid_score >= threshold
```

Every result row carries `rid_channel_0_l1`, `rid_channel_3_l1`,
`rid_channel_min_l1`, `rid_score`, `score_direction` and the comparison
operator, so the raw positive distance always remains available. Batched input
produces **one record per image**; a per-image failure is `status="error"`,
never a negative detection.

### The `official_sd21` profile

`--rid_profile official_sd21` (the default) sets and validates the official
values below. Generic RAVEN-wide parser defaults never override them. Any value
given explicitly on the command line is recorded in `rid_profile_overrides` and
the run is reported as an ablation (`rid_profile_is_official=false`).

| Generation | Value | | Detection / key | Value |
| :--- | ---: | --- | :--- | ---: |
| model | `stabilityai/stable-diffusion-2-1-base` | | inversion prompt | `""` |
| revision / dtype | `fp16` / `float16` | | inversion guidance | `1.0` |
| scheduler | `DPMSolverMultistepScheduler` | | inversion steps | `50` |
| resolution | `512` | | VAE | posterior **mode**, scale `0.18215` |
| steps / guidance | `50` / `7.5` | | target FPR | `0.01` |
| radius / cutoff | `14` / `3` | | `channel_min` | `1` |
| ring / heterogeneous channel | `3` / `0` | | key RNG seed | `42` (official `--general_seed`) |
| `ring_width` | `1` (anything else fails) | | key RNG device / dtype | `cpu` / `float32` |
| quantization | `2` levels, `-64` / `+64` | | capacity | `2^(14-3) = 2048` |
| `fix_gt` / `time_shift` | `1` / `1` | | shift semantics | `official_code_exact` |

Inversion reuses the shared `official_forward_diffusion` transcription of the
official `InversableStableDiffusionPipeline.forward_diffusion`: the DPM
scheduler supplies only the timestep grid, `init_noise_sigma`,
`scale_model_input` and `alphas_cumprod`, while the state update is the manual
DDIM equation. `DPMSolverMultistepInverseScheduler` implements a *different*
update rule and is deliberately not used on the RingID path.

### Released code vs. paper-described shift

| `--rid_shift_semantics` | Meaning |
| :--- | :--- |
| `official_code_exact` (default) | The released spatial shift. In the frozen commit the `* args.time_shift_factor` multiplication is **commented out**, so `--time_shift_factor` is an upstream **no-op**; passing anything but `1.0` in this mode fails with an explicit error. |
| `paper_described_shift` | The paper's described scaling by eta (~0.8-0.9), applied explicitly. Available as `--rid_profile paper_shift_ablation`; it is never labelled released-code parity. |

Upstream `--watermark_seed` (default 5) is parsed by both official scripts but
never read; it is recorded in `upstream_unused_args` so it cannot be presented
as an effective knob. RAVEN's one canonical key-generation seed is
`--rid_key_seed`. The deprecated `--rid_seed` is accepted only with
`--rid_profile legacy` and is never remapped onto an official key.

### Key RNG and key identity

Official `verify.py`/`identify.py` call `set_random_seed(general_seed)`, draw a
base latent (whose *values* never reach the pattern — it starts from
`zeros_like` — but whose *draw* advances the RNG), then draw one Gaussian
tensor per candidate for the heterogeneous channel. RAVEN reproduces that call
sequence exactly. The draws happen on the declared key-RNG device: the canonical
default is a portable **CPU float32** draw so a key id means the same tensor on
every machine; `--rid_key_rng_device cuda --rid_key_rng_dtype float16`
reproduces the official runtime draw instead. The choice is always recorded in
the bundle manifest (`rng_algorithm`, `rng_seed`, `rng_device`, `rng_dtype`).

### Bundle layout

```text
<rid_bundle>/
├── manifest.json          # schema, official commit, profile, key identity, RNG recipe, hashes
├── selected_pattern.pt    # official complex pattern, shape (1, 4, 64, 64)
├── watermark_mask.pt      # official bool mask, shape (2, 64, 64)
└── threshold.json         # only after `calibrate` or an explicit import
```

The bundle stores the selected key verbatim plus the complete recipe needed to
regenerate the candidate keybook for identification; a regenerated keybook is
checked against `keybook_sha256`/`candidate_order_sha256` before it is used.
Bundles are never silently overwritten, an edited manifest or tampered tensor is
rejected on load, and a bundle whose key/detector configuration disagrees with
the current run fails closed. An explicit `--rid_key_index` that disagrees with
the bundle's key is rejected rather than silently substituted; omitting it
adopts the bundle's own key.

### Commands

```bash
# 1) Generate N watermarked images (+ their matched non-watermarked pair) for one key.
#    Creates the bundle on first use and reuses it afterwards.
python eval_bench_wm/run_watermark.py \
  --wm_type RID --rid_profile official_sd21 --rid_key_index 628 \
  --num 10 --seed 0 \
  --rid_bundle_dir out/rid_bundle --out_dir out/rid_generation

# 2) Official paper verification over paired cohorts (ROC, AUC, TPR@FPR)
python eval_bench_wm/run_verify_watermark.py \
  --mode paper_eval_verification --wm_type RID --rid_profile official_sd21 \
  --rid_bundle_dir out/rid_bundle \
  --clean_path data/rid/clean --watermarked_path data/rid/watermarked \
  --target_fpr 0.01 --out_dir out/rid_verification

# 3) Multi-key identification in a fresh process (no true key required)
python eval_bench_wm/run_verify_watermark.py \
  --mode identify --wm_type RID --rid_bundle_dir out/rid_bundle \
  --suspect_path images/or_directory --out_dir out/rid_identification

# 4) Deployment verification against a calibrated threshold
python eval_bench_wm/run_verify_watermark.py \
  --mode calibrate --wm_type RID --rid_bundle_dir out/rid_bundle \
  --watermarked_path data/rid/watermarked --clean_path data/rid/clean \
  --out_dir out/rid_calibration
python eval_bench_wm/run_verify_watermark.py \
  --mode verify --wm_type RID --rid_bundle_dir out/rid_bundle \
  --suspect_path images/or_directory --out_dir out/rid_deployment
```

Generation output layout (indexed, never overwritten):

```text
<out_dir>/
├── images/watermarked/000000.png ...
├── images/no_watermark/000000.png ...   # matched pair, disable with --rid_no_clean_pair
├── prompts/000000.txt ...
├── sample_metadata/000000.json ...
├── results.jsonl
└── run_manifest.json
```

Each sample draws its own **complete** initial latent from an explicit
`torch.Generator` seeded with `base_seed + sample_id`, so samples never share a
latent, never depend on iteration order and reproduce exactly after a restart.
Resume compares sample seed, prompt hash, run-config hash, bundle-config hash,
selected-pattern hash and the on-disk image hash before skipping anything.

### Paper distortion protocol

The official evaluation conditions are `Clean`, `Rotation 75°`, `JPEG 25`,
`Crop & Scale 0.75/0.75`, `Gaussian blur radius 8`, `Gaussian noise std 0.1`,
`Brightness 6`. Paired verification samples must receive identical stochastic
distortion parameters; the pairing gate rejects a cohort whose two sides declare
different `distortion_config_sha256`/`distortion_seed`.

### Backward compatibility

Existing RID outputs are classified as follows:

* Outputs produced with the plain circle-difference ring mask instead of the
  official rounder ring: **invalid for official-parity claims** (a different
  frequency region was scored).
* Outputs whose detector averaged channels 0 and 3 into one number while
  claiming official RingID: **invalid detector semantics**.
* Outputs that reused one complete initial latent across prompts: **invalid for
  formal multi-sample evaluation**.
* Fixed-key-only outputs: they may support verification, but they are **not**
  RingID identification results.
* Outputs scored through the generic `DPMSolverMultistepInverseScheduler`
  inversion without parity evidence: **legacy/approximate**.
* The legacy fixed `RID_THRESHOLD` in `utils/utils.py` (nominal FPR `1e-3`) is a
  **deployment legacy** operating point, not the paper's cohort-derived
  TPR@1%FPR.
* Outputs without key/pattern/mask/config hashes: **not independently
  auditable**.
* Results produced with a scaled spatial shift: **paper-described/ablation**,
  never exact released-code parity.

---

## HSQR (SFWMark) (`--wm_type HSQR`)

Official reference: <https://github.com/thomas11809/SFWMark>, pinned comparison
commit `78666128b44614a0cc471993649e3132d5dddfcb`
(`src/generate.py`, `src/detect.py`, `src/utils.py`).

The single authoritative HSQR implementation is
`utils/wm/hsqr_provider.py`. Runners only enumerate files, save/load artifacts
and call provider methods; the artifact schema lives in `utils/wm/sfw_bundle.py`,
the official detection front-end in `utils/wm/sfw_inversion.py` and the shared
runner plumbing in `utils/wm/sfw_runtime.py` / `utils/wm/runner_common.py`.

### Profiles

| Profile | Base key seed | Key selection | Model / scheduler | Official? |
| --- | --- | --- | --- | --- |
| `official_sfwmark_sd21` | **7433** (key `i` -> `7433 + i`) | explicit `--hsqr_key_index` | `stabilityai/stable-diffusion-2-1-base`, DDIM, float32, 512px, 50 steps, CFG 7.5 | yes |
| `legacy_raven` (parser default) | 999999 | `fix_gt` index into a process-global-RNG mapping | whatever the caller passes | **no** |

`legacy_raven` is the parser default so the pre-existing formal generators
(`experiments/generate_watermarked_images.py`, `run_removal.py`) keep producing
byte-identical cohorts. `run_watermark.py --wm_type HSQR` and
`run_verify_watermark.py --wm_type HSQR` are standalone runners: the former
selects `official_sfwmark_sd21` unless `--hsqr_profile` is given explicitly.

Any value set explicitly on the command line overrides the profile, is recorded
in `hsqr_profile_overrides`, and makes the run an **ablation**, never an official
run. The official profile fails closed on an incompatible latent shape, center
slice, capacity, base key seed or key-selection mode.

Frozen official facts enforced and tested: QR version 1, box size 2, border 0,
error correction `H` (42x42), payload `HSQR{key_seed % 10000}`, latent channel
`[3]`, center slice `10:54` (44x44), two 42x21 halves driving the sign of the
real / imaginary RFFT coefficients, `delta=0`, detector target `±45` as a complex
42x21 tensor, capacity `2^(14-3) = 2048`.

### Score direction and threshold semantics

```text
hsqr_l1_distance  raw mean complex L1 distance  (LOWER  = more watermarked)
hsqr_score        = -hsqr_l1_distance           (HIGHER = more watermarked)
decision          score >= threshold            (equivalently distance <= distance_threshold)
```

Both values are emitted on every record, and threshold artifacts store the score
threshold *and* the equivalent `distance_threshold` so a positive distance can
never be compared against a negative score threshold. The official SFWMark
release does **not** define a universal deployment threshold for arbitrary
suspect images:

* `paper_eval` reproduces the official cohort protocol (ROC on `score`,
  operating point at the last FPR strictly below `--target_fpr`) and records
  target FPR, **actual empirical FPR**, TPR, ROC-AUC and cohort counts.
* `deployment_verify` is a RAVEN deployment extension. Without a compatible
  bundled or supplied threshold it emits raw distances and scores and reports the
  binary decision as **undecided** — never as "not embedded".
* The legacy operating point (`-65.86233520507812`, nominal FPR `1e-3`) requires
  `--hsqr_allow_legacy_threshold`, is labelled `legacy_threshold` and is **not**
  a TPR@1%FPR result.

### Bundle layout

```text
<out_dir>/hsqr_bundle/
├── manifest.json          # schema, profile, geometry, key identity, hashes, git provenance
├── selected_pattern.pt    # boolean (c_wm, 42, 42) — sufficient for one-key verification
├── keybook.pt             # optional (2048, c_wm, 42, 42) — --hsqr_save_keybook
├── key_mapping.json       # optional per-sample key mapping (--hsqr_key_policy per_sample)
└── threshold.json         # only after `calibrate`
```

The manifest records `schema`, `method`, `official_reference_commit`,
`profile_name`, `profile_overrides`, `profile_is_official`, `model_id`,
`model_revision`, `scheduler_type`, `torch_dtype`, `resolution`, `latent_shape`,
`center_slice`, `watermark_channels`, `qr_version`, `box_size`, `border`,
`error_correction`, `delta`, `wm_capacity`, `base_key_seed`,
`selected_key_index`, `selected_key_seed`, `payload_text`,
`selected_pattern_sha256`, the optional keybook/mapping files with their hashes,
the inversion configuration, `bundle_config_sha256` and the creating git
branch/commit.

Loading a bundle uses the **persisted** pattern; it is never regenerated.
Incompatible profile, model, geometry, QR configuration, channel, delta, key
identity or hash fails closed, extra undeclared artifacts are rejected, and
creating a bundle over existing artifacts fails instead of overwriting.

### Commands

```bash
# 1) Direct generation: N indexed, non-overwriting watermarked images.
#    Each sample draws an independent complete base latent from seed+sample_id
#    and injects the pattern into a clone of it. --hsqr_paired also writes the
#    matched clean image from the same pre-injection latent.
python eval_bench_wm/run_watermark.py \
  --wm_type HSQR \
  --hsqr_profile official_sfwmark_sd21 \
  --hsqr_key_index 0 \
  --num 10 --seed 42 \
  --out_dir out/hsqr_generation

# 2) Standalone verification in a fresh process.
#    Needs only the bundle and the suspect image(s) — no prompt, no original
#    image, no live generation process, no keybook regeneration.
python eval_bench_wm/run_verify_watermark.py \
  --wm_type HSQR \
  --mode deployment_verify \
  --hsqr_bundle_dir out/hsqr_generation/hsqr_bundle \
  --suspect_path image.png_or_directory \
  --out_dir out/hsqr_verification

# 3) Paper-style calibration / evaluation over positive and negative cohorts.
python eval_bench_wm/run_verify_watermark.py \
  --wm_type HSQR \
  --mode paper_eval \
  --hsqr_bundle_dir out/hsqr_generation/hsqr_bundle \
  --positive_path out/hsqr_generation/images/watermarked \
  --negative_path out/hsqr_generation/images/no_watermark \
  --target_fpr 0.01 \
  --out_dir out/hsqr_calibration

# 4) Reuse that threshold for a binary decision (or --mode calibrate to store it
#    inside the bundle as threshold.json).
python eval_bench_wm/run_verify_watermark.py \
  --wm_type HSQR --mode deployment_verify \
  --hsqr_bundle_dir out/hsqr_generation/hsqr_bundle \
  --threshold_artifact out/hsqr_calibration/threshold.json \
  --suspect_path suspects/ --out_dir out/hsqr_decisions
```

### Output layout

```text
<out_dir>/
├── images/watermarked/000000.png ...
├── images/no_watermark/000000.png ...   # --hsqr_paired
├── prompts/000000.txt
├── sample_metadata/000000.json
├── hsqr_bundle/
├── results.jsonl
└── run_manifest.json
```

Every sample record carries `sample_id`, prompt and `prompt_sha256`,
`sample_seed`, `base_latent_sha256`, `clean_base_latent_sha256`,
`watermark_pre_injection_base_latent_sha256`, `watermarked_latent_sha256`, the
key index / seed / payload / pattern hash, image paths and hashes, the
model/scheduler/dtype/steps/guidance/resolution, `hsqr_bundle_config_sha256`,
`run_config_sha256`, the entrypoint and its SHA, and the git branch/commit.
The clean-vs-pre-injection pairing invariant is asserted per sample, and a run
whose samples repeat a complete base latent is rejected.

Resume is hash-gated: an existing `run_manifest.json` must match the current
`run_config_sha256`, and each existing sample must match its seed, prompt hash,
run config, bundle config, pattern hash and on-disk image hash. Anything else
stops the run with the output directory untouched.

### Batch and directory scoring

`HSQRProvider.l1_distances` / `detect_from_latent` return one value per batch
item in input order. The pre-Issue-#5 detector indexed the target at batch item
`0`, so a batch or a directory of suspects could silently report the first
image's result for every image; this is now covered by a regression test.
`HSQRProvider.identify()` is a separate API for the paper identification
experiment and does not change single-key verification.

Every scored image reports `status`/`error`; NaN or Inf in the recovered latent,
a shape mismatch, an inversion failure or a bundle/threshold mismatch is an
**error**, never a negative detection. Duplicate suspect paths are rejected and
directories are walked in deterministic name order.

### Inversion parity status

`utils/wm/sfw_inversion.py` transcribes the documented official `detect.py`
front-end — fixed `Resize((512, 512))`, `ToTensor`, `[-1, 1]`, VAE posterior
**mode**, VAE scaling factor,
`DDIMInverseScheduler.from_config(<current scheduler>.config)`, empty prompt,
guidance 0, 50 steps — and is deliberately *not* the generic
`PipeProvider.invert_images` path (different resize, `guidance_scale=1.0`, full
pipeline preprocessing) nor the GaussMarker loop (which never calls
`scheduler.step`).

It is used by `validate()` only for **official-profile** providers, so existing
legacy HSQR cohorts keep the exact inversion path they were produced and scored
with.

Two independent claims are tracked separately, and both are recorded on every
record:

| Field | Value | Meaning |
| --- | --- | --- |
| `inversion_parity_status` | `official_code_parity_verified_bitwise` | The **code** was compared element by element against the frozen official implementation and agreed exactly. |
| `inversion_weights_parity` | `official_weights_unavailable_not_verified` | The **weights** used were not the official ones, so published numbers are not reproduced. |

`tools/hsqr_inversion_parity.py` executes the official `transform_img`,
`pil2latent` and `ddim_invert` from the hash-pinned official `src/utils.py` and
compares them against `utils/wm/sfw_inversion` on the *same* loaded pipeline,
capturing intermediates with a UNet forward hook so neither path is modified.
All compared artifacts — preprocessed input tensor, VAE latent, all 50 inverse
scheduler timesteps, sampled intermediate latents, final recovered latent, HSQR
L1 distance and canonical score — were **bitwise identical** (`max_abs_diff == 0`).
Evidence: `tests/fixtures/hsqr_inversion_parity_evidence.json`, guarded by
`tests/test_hsqr_inversion_parity.py`.

The official `stabilityai/stable-diffusion-2-1-base` is currently **delisted from
the Hugging Face Hub** (HTTP 404 for the whole `stabilityai` SD-2 family with a
valid token), so the parity run used mirror weights. The tool refuses any
non-official model unless `--allow-non-official-model` is passed, and the
evidence file always records `official_model_used`, so a mirror run cannot be
mistaken for an official one.

### Official static fixtures

`tests/fixtures/hsqr_official_fixtures.json` is generated by **executing the
frozen official code**, not by re-reading the paper:
`tests/official_sfwmark_source.py` hash-pins `src/utils.py`
(`d3deb279…`) at commit `78666128b44614a0cc471993649e3132d5dddfcb` and runs it.
Regenerate or re-verify with:

```bash
export SFWMARK_OFFICIAL_SRC=/path/to/SFWMark   # frozen commit
python eval_bench_wm/tools/generate_hsqr_official_fixtures.py --check
```

Keys **0, 1, 1024 and 2047** are verified element by element at zero tolerance:
QR pattern (all 1 764 booleans), injected latent (bitwise), per-item L1 distance
and canonical score. `tests/test_hsqr_official_fixtures.py` enforces this in a
normal clone without network access; the live re-derivation is skipped when the
official checkout is absent. The official checkout is not vendored (separate
licence).

`tests/hsqr_official_reference.py` is kept as an independent second derivation
from the written specification; its digests were confirmed to agree bitwise with
the official execution.

### Backward compatibility

* Pre-Issue-#5 HSQR outputs have no persisted pattern/key identity, no
  independent per-sample base latent and no immutable configuration. They are
  **legacy / not independently auditable** and must be excluded from formal
  claims unless their provenance can be proven separately. Nothing is deleted.
* They must not be resumed under the new bundle/profile schema; the resume gates
  reject them because they carry no `run_config_sha256`.
* Detection rates computed with the legacy `-65.86233520507812` threshold remain
  **legacy** results at nominal FPR `1e-3` and are not TPR@1%FPR.
* `HSQRProvider.get_accuracies()` now returns one distance per batch item
  instead of a single-element list; batch-size-1 callers are unaffected.

---

## T2S (T2SMark)

Upstream reference: <https://github.com/0xD009/T2SMark>, pinned comparison commit
`0c1fbfd50fcd1fba135477a2c016e284d5d7914d`.

### Generation

Each sample gets its own base latent, session key and watermarked latent; images
and states are written per sample instead of overwriting a single file.

`--t2s_fix_key` follows upstream `run.py`: it fixes **both the master key and the
message** across samples to simulate a single account, and only the session key
and base latent stay per-sample.

`--t2s_rng_mode` selects the generation RNG lifecycle:

| Mode | Behaviour |
| :--- | :--- |
| `official_compatible` (default) | Reproduces upstream's whole per-sample RNG lifecycle: reseeds the process-global RNG with `set_random_seed(seed + index)` and draws in upstream's order (master key, message, session key, then each `encode`'s `randn` and noise-sign `randint`). |
| `raven_deterministic` | Uses one explicit CPU generator per sample and never touches global RNG state. A RAVEN provenance adaptation; it does **not** reproduce upstream's global-RNG side effects, so an end-to-end run diverges from upstream. Never describe it as upstream-exact. |

```bash
python run_watermark.py --wm_type T2S --num 2 \
    --modelid_target stabilityai/stable-diffusion-2-1-base \
    --scheduler_target DDIM --out_dir out/t2s
# -> out/t2s/images/00000.png ...    out/t2s/t2s_state/<watermark_id>.json
```

### Standalone verification

`run_verification.py` runs in a fresh process. It needs only the saved state,
the suspect image, and the model/inversion configuration. It does not need the
generating provider instance, a prior `get_wm_latents()` call, or the original
latent, and it never infers pairing from filename or row order.

```bash
# Explicit pairs
python run_verification.py --wm_type T2S \
    --pair out/t2s/t2s_state/sd21-t2s-00000.json=out/t2s/images/00000.png \
    --modelid_target stabilityai/stable-diffusion-2-1-base --strict_image_sha

# Or content-addressed, matching each state's recorded image_sha256
python run_verification.py --wm_type T2S \
    --state_dir out/t2s/t2s_state --image_dir out/t2s/images \
    --modelid_target stabilityai/stable-diffusion-2-1-base
```

Reported fields: `score_true_key`, `score_control_key`, `score_margin`,
`score_direction`, `decision_rule`, `detection_success`, the recovered session
key and message, plus `key_accuracy` / `message_accuracy` when the state carries
the corresponding expectation (otherwise `null` / `N/A`, never a false 0%).

Fail-closed gates: a state without a valid `state_sha256` is rejected outright;
states whose model, revision, scheduler, resolution, latent shape, channel
layout, RNG mode, inversion mode/steps or provider-config SHA disagree cannot be
verified in one run; and a CLI flag that contradicts the state aborts unless
`--allow_config_override` is passed. The first state's configuration is never
silently applied to the rest. The loaded model's latent shape is also checked
against the state.

### Detection semantics

Upstream's **formal evaluation is a cohort ROC**: `run.py` pools `norm1_no_w` as
negatives and `norm1_w` as positives across the whole run and reports AUC plus
TPR at FPR < 1e-6. Upstream defines **no per-image binary decision rule**.

The per-image test reported here is therefore labelled as a RAVEN deployment
extension, `paired_key_comparison`:

```
decision_rule            = paired_key_comparison
decision_rule_expression = score_true_key > score_control_key   (higher is watermarked)
decision_rule_provenance = raven_deployment_extension
official_evaluation      = cohort_roc_auc_and_tpr_at_fpr_1e-6
```

where the control key is `1 - master_key`. It is **not** an upstream rule and
**not** `TPR@1%FPR`. It has no calibrated margin threshold, so a mismatched state
can occasionally land marginally above zero; the discriminating signals are the
margin magnitude and the message accuracy. Report TPR/FPR only after running and
recording an actual negative cohort and threshold-calibration protocol.

### Inversion modes

The official T2S inversion and the benchmark's generic DDIM inversion are not
equivalent and are never silently substituted:

| `--t2s_inversion_mode` | Protocol |
| :--- | :--- |
| `t2s_official` (default) | Upstream `naive_forward_diffusion`: walks `reversed(scheduler.timesteps)` with a null prompt at `guidance_scale=1.0`, default 10 steps (`--t2s_num_inversion_steps`). SD3/SD3.5 use the flow-matching sigma update. |
| `benchmark_ddim` | The benchmark's `PipeProvider.invert_images` -> diffusers `DDIMInverseScheduler`, driven at the generation step count. |

### Channel layouts

| Latents | Key channels | Message channels |
| :--- | :--- | :--- |
| 4 (SD2.1) | `[--t2s_key_channel_idx]` (default `[0]`) | the other three |
| 16 (SD3/SD3.5) | `[0, 1, 2, 3]` | the other twelve |

### Cross-watermark shared-clean cohorts

`T2SProvider.new_sample(base_latent=...)` accepts the canonical shared TR base
latent and uses it as the tail-truncated source, so the pre-injection latent is
byte-identical to the shared one. A shape mismatch fails closed rather than
silently sampling a method-specific replacement. No formal shared-clean T2S
cohort is produced by this change.

## Acknowledgement

This repository heavily relies on and builds upon the [semantic-forgery](https://github.com/and-mill/semantic-forgery). We sincerely thank the authors for open-sourcing their code, which served as a foundational component for our evaluation.