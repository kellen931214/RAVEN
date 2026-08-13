# RAVEN Evaluation & Experiment Handover Document

## 1. Executive Summary & Discoveries

During the evaluation of Pixel Shift attacks on Tree-Ring Watermarked images (1001 samples, SD 2.1 Base on DiffusionDB), key discoveries were made regarding the shift implementations:

1. **Circular Shift (Roll / Toroidal Shift)**:
   - Previously labeled as `pixel_shift_raven` (Black Padding), this method actually executed `np.roll(img, (dx, dy))`.
   - **Characteristics**: Right edge wraps to left, bottom edge wraps to top. No black border (`RGB(0,0,0)`).
   - **Metrics**: Low FID (`24.70`) and high CLIP (`0.463`) due to continuous pixel texture, but creates visual tiling seams.

2. **Reflection Padding Shift**:
   - Fills shift gaps by mirroring edge pixels (`padding_mode='reflection'`).
   - Maintains continuous edge textures without zero-boundary artifacts during DDIM Inversion & REVAN latent reconstruction.

3. **True Zero Padding Shift (True Black Fill)**:
   - Fills shifted boundary gaps with true black `RGB(0,0,0)` (or zero latents).
   - **Pending Task**: Needs clean execution and evaluation for both Pure Shift and REVAN Inpainting cohorts.

---

## 2. Experimental Data Summary (1001 Samples)

### 2.1 Paper Baseline (Tree-Ring Table 14 Reference)
| Shift Magnitude | FID | LPIPS (AlexNet) | CLIP Score |
| :--- | :---: | :---: | :---: |
| **16px** | 45.18 | 0.4076 | 0.4484 |
| **24px** | 48.94 | 0.5055 | 0.4464 |
| **32px** | 52.08 | 0.5588 | 0.4454 |

### 2.2 Circular Shift (`np.roll` / Wrap-around) - Baseline
| Shift Magnitude | FID | LPIPS | OpenCLIP Score | TR Detector TPR@1% FPR |
| :--- | :---: | :---: | :---: | :---: |
| **16px** | 24.70 | 0.4083 | 0.4632 | **94.41%** |
| **24px** | 26.26 | 0.5123 | 0.4629 | **12.39%** |
| **32px** | 29.02 | 0.5674 | 0.4622 | **6.69%** |

### 2.3 Pixel Shift (Reflection Padding) + REVAN Inpainting
| Shift Magnitude | FID | LPIPS | OpenCLIP Score | TR Detector TPR@1% FPR |
| :--- | :---: | :---: | :---: | :---: |
| **16px** | 47.97 | 0.5579 | 0.4023 | **95.10%** |
| **24px** | 48.72 | 0.5986 | 0.4016 | **18.58%** |
| **32px** | 49.65 | 0.6240 | 0.4008 | **11.49%** |

### 2.4 Pure Pixel Shift (Reflection Padding) - No REVAN
| Shift Magnitude | FID | LPIPS | OpenCLIP Score | TR Detector TPR@1% FPR |
| :--- | :---: | :---: | :---: | :---: |
| **16px** | 47.61 | 0.3806 | 0.4026 | **98.50%** |
| **24px** | 48.26 | 0.4856 | 0.4019 | **24.98%** |
| **32px** | 49.38 | 0.5398 | 0.4010 | **17.38%** |

---

## 3. Outstanding Tasks for Next Agent / AI

### Task 1: True Zero-Padding Pure Pixel Shift (True Black Fill `RGB(0,0,0)`)
1. **Generation**:
   - Apply true zero-padding pixel shift (top-left gap filled with `RGB(0,0,0)`) on 1001 watermarked images from `/workspace/RAVEN/data/tr/diffusiondb/{id}/watermarked.png`.
   - Magnitudes: 16px, 24px, 32px.
   - Save output samples in `/workspace/RAVEN/outputs/true_zero_pixel_shift_{16,24,32}px/samples/watermarked/`.

2. **Evaluation**:
   - Compute **FID**, **LPIPS (AlexNet)**, **OpenCLIP (ViT-bigG-14)**, and **Tree-Ring Detector TPR@1% FPR**.
   - Output summary JSON to `/workspace/RAVEN/outputs/true_zero_pixel_shift_{16,24,32}px/results_summary.json`.

### Task 2: True Zero-Padding Pixel Shift + REVAN Inpainting
1. **Generation**:
   - Pass true zero-padded shifted latents through DDIM Inversion + REVAN inpainting.
   - Magnitudes: 16px, 24px, 32px.
   - Save output samples in `/workspace/RAVEN/outputs/true_zero_pixel_shift_revan_{16,24,32}px/samples/watermarked/`.

2. **Evaluation**:
   - Compute **FID**, **LPIPS**, **OpenCLIP**, and **Tree-Ring Detector TPR@1% FPR**.
   - Output summary JSON to `/workspace/RAVEN/outputs/true_zero_pixel_shift_revan_{16,24,32}px/results_summary.json`.

---

## 4. Hardware & Environment Guidelines

1. **GPU Selection Rules**:
   - Use free GPUs (e.g. GPU 5 / `CUDA_VISIBLE_DEVICES=0` when isolated, or RTX 6000 Ada Generation with 48GB VRAM).
   - **DO NOT USE**: GPU 0, 1, 3, 4, 8 (in use) or GPU 2 (`Blackwell Server Edition`, PyTorch `sm_120` CUDA kernel incompatible).

2. **PyTorch GPU Isolation Rule**:
   - Always set `export CUDA_VISIBLE_DEVICES=<gpu_id>` in the shell **BEFORE** running Python scripts to prevent PyTorch from initializing GPU 0.

3. **Background Persistence**:
   - Launch long-running jobs in detached `screen` sessions (e.g. `screen -dmS <name> bash -c "..."`) or background tasks so execution survives SSH disconnects.

---

## 5. Key File Locations

- **Clean Watermarked Source Images**: `/workspace/RAVEN/data/tr/diffusiondb/{id}/watermarked.png`
- **Clean Unwatermarked Source Images**: `/workspace/RAVEN/data/clean/diffusiondb/{id}.png`
- **Existing Worktree Branch**: `experiment/pure-pixel-shift-reflection` in `/workspace/RAVEN-worktrees/experiment-pure-pixel-shift-reflection`
- **Core Pipeline Code**: `/workspace/RAVEN/raven_repro/raven/pipeline_raven.py`
- **Warp Code**: `/workspace/RAVEN/raven_repro/raven/warp.py`
- **Evaluation Code**: `/workspace/RAVEN/raven_repro/eval.py`
