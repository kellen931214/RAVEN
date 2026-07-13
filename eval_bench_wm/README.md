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

## Acknowledgement

This repository heavily relies on and builds upon the [semantic-forgery](https://github.com/and-mill/semantic-forgery). We sincerely thank the authors for open-sourcing their code, which served as a foundational component for our evaluation.