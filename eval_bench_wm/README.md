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

## T2S (T2SMark)

Upstream reference: <https://github.com/0xD009/T2SMark>, pinned comparison commit
`0c1fbfd50fcd1fba135477a2c016e284d5d7914d`.

### Generation

Each sample gets its own base latent, session key, message and watermarked
latent; images and states are written per sample instead of overwriting a single
file. `--t2s_fix_key` fixes only the account-level master key.

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

### Detection semantics

The default rule is upstream's **true-key versus control-key** comparison on the
same image:

```
detection_success := score_true_key > score_control_key      (higher is watermarked)
```

where the control key is `1 - master_key`. This is **not** `TPR@1%FPR` and must
not be reported as such. It has no calibrated margin threshold, so a mismatched
state can occasionally land marginally above zero; the discriminating signals
are the margin magnitude and the message accuracy. In a measured 2-sample smoke
run, correct pairing gave margins of 1350.8 / 1828.0 with 100% message accuracy,
while deliberately mispaired states gave 2.3 / −15.2 with ~50% (chance) message
accuracy. Report TPR/FPR only after running and recording an actual negative
cohort and threshold-calibration protocol.

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