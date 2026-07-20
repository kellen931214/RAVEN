# RAVEN Reproduction

This is a clean reproduction scaffold for **RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**. It is intended for academic robustness evaluation on images you own or generated yourself.

The implementation follows the local `PLAN.md` and uses the NFPA repository as an implementation reference for frame-guided self-attention and the optional `grid_sample` warp ablation. It does not modify NFPA.

## Setup

```bash
cd /workspace/RAVEN/raven_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional evaluation packages:

```bash
pip install open_clip_torch torchmetrics clean-fid
```

The reproduction model is `RedbeardNZ/stable-diffusion-2-1-base` at revision `c6a5e9bab8d874d081de76fa270ae0aefa5410ff`.

## Diagnostic Single Image

`scripts/run_raven.py` is **ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT**.

```bash
python scripts/run_raven.py \
  --input path/to/watermarked.png \
  --output_dir outputs/raven_test \
  --model_id RedbeardNZ/stable-diffusion-2-1-base \
  --steps 50 \
  --strength 0.15 \
  --inversion_mode ddim \
  --guidance_scale 2.5 \
  --shift_min 24 \
  --shift_max 32 \
  --shift_sign random \
  --shift_sampling independent_axes \
  --shift_space image_pixels \
  --warp_mode integer \
  --padding_mode zeros \
  --view_guided_attention true \
  --color_transfer true \
  --seed 42 \
  --device cuda \
  --debug true
```

## Diagnostic Folder Attack

`scripts/attack_folder.py` is **ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT**.

```bash
python scripts/attack_folder.py \
  --input_dir data/watermarked \
  --output_dir outputs/raven_folder \
  --model_id RedbeardNZ/stable-diffusion-2-1-base \
  --steps 50 \
  --strength 0.15 \
  --guidance_scale 2.5 \
  --shift_min 24 \
  --shift_max 32 \
  --shift_sign random \
  --shift_space image_pixels \
  --warp_mode integer \
  --padding_mode zeros \
  --view_guided_attention true \
  --color_transfer true \
  --seed 42 \
  --device cuda
```

Each image is saved into its own subdirectory. Failed files are listed in `failed.txt`.

## Evaluation

The only formal evaluation entrypoint is `experiments/run_raven_formal_eval.py`.
Run stages separately against one immutable timestamped output root:

```bash
python experiments/run_raven_formal_eval.py \
  --dataset diffusiondb --method TR \
  --source-metadata /absolute/path/to/metadata.csv \
  --output-root outputs/raven_formal_eval/diffusiondb/TR/$(date -u +%Y%m%dT%H%M%SZ) \
  --expected-count 30 --batch-size 10 --device cuda --gpu 0 --stage snapshot
```

Then run `attack-watermarked`, `attack-clean` for TR, `verify`, `quality`, `fid`,
`clip`, `aggregate`, and `validate` with the identical arguments plus `--resume`.
Do not run a full cohort until the 2/10/30 gates pass in fresh output roots.

## Key Files

- `raven/pipeline_raven.py`: two-stream partial-noising and denoising pipeline.
- `raven/inversion.py`: VAE encoding plus partial DDIM inversion or Equation-(4) forward noising.
- `raven/warp.py`: integer zero-padded latent translation plus an explicit `grid_sample` ablation.
- `raven/attention.py`: view-guided self-attention processor. Text cross-attention is left unchanged.
- `raven/color_transfer.py`: LAB luminance matching and original-image chroma transfer.
- `scripts/run_raven.py`: single-image CLI.
- `scripts/attack_folder.py`: folder CLI with failure logging.
- `scripts/eval_quality.py`: PSNR/SSIM helper.
- `scripts/audit_dataset.py`: metadata, hash, image-format, and pairing audit.
- `../experiments/run_raven_formal_eval.py`: the single formal stage orchestrator.
- `raven/eval_protocol.py`: centralized formal attack, detector, FID, resume, provider, and CLIP provenance.
- `scripts/raven_nfpa_tr_eval.py`: Tree-Ring complex-L1 detector helper used by the formal runner.
- `scripts/raven_p1_full.py`, `scripts/quality_decomposition_experiment.py`, and `scripts/run_diffusiondb_chain_after_clean.py`: preserved ablation/evidence scripts, never formal entrypoints.
- `scripts/paired_generation_shards.py`: migrates committed rows into shard metadata, quarantines crash-written images without provenance, and merges shards only when run-ID coverage, latent uniqueness, image hashes, target/config hashes, and model revision all pass.
- `raven/pairing_provenance.py`: fail-closed latent, image, target, pairing, and attack-config audits.
- `scripts/build_verification_manifest.py` and `scripts/evaluate_verification.py`: strict pairing and calibrated verification utilities.
- Archived legacy scripts live under `archive/legacy_scripts_20260716/`.

## Outputs

For a normal run, the output directory contains:

```text
input.png
latent_shift_only.png
view_guided_output.png
final_color_corrected.png
debug_info.json
```

`latent_shift_only.png` is emitted only with debug mode enabled; it is not a formal attack output or quality input.

## Ablations

All commands in this section are **ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT**.

```bash
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/no_vga --view_guided_attention false --color_transfer true
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/no_color --view_guided_attention true --color_transfer false
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/strength_005 --strength 0.05
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/shift_latent --shift_space latent_pixels
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/forward_noise --inversion_mode forward_noise
python scripts/run_raven.py --input path/to/watermarked.png --output_dir outputs/legacy_shift --shift_sampling coupled_diagonal
```

## Reproduction Audit Workflow

Audit metadata and image pairing before loading a model:

```bash
python scripts/audit_dataset.py \
  --metadata /workspace/data/watermarked/mscoco/TR/metadata.csv \
  --workspace-root /workspace \
  --output outputs/audit/mscoco_TR.json
```

Use only `experiments/run_raven_formal_eval.py` for current formal evaluation.
Historical scripts remain solely as reproducibility evidence.

## Approximations Relative To The Paper

- The official RAVEN code is unavailable, so this is a faithful reproduction from the paper description and the local plan rather than an exact code release.
- The paper is ambiguous: Equation (4) specifies stochastic forward noising,
  while Implementation Details specifies DDIM inversion. The primary mode is
  now partial DDIM inversion; `--inversion_mode forward_noise` preserves the
  Equation-(4) implementation as a labeled ablation.
- Formal evaluation explicitly pins `raven_paper_nfpa_gap_fill`, image-pixel
  flow, nearest latent sampling, reflection padding, and `align_corners=False`.
  Integer, zero-padded, bilinear, and forward-noise modes are ablations only.
- Latent viewpoint modulation uses a global diagonal translation, matching the requested reproduction scope, not a learned or depth-aware camera transform.
- View-guided correspondence attention is implemented only for UNet self-attention. Cross-attention to text remains the default Diffusers behavior.
- The implementation assumes paired latent ordering `[reference, view]`; with classifier-free guidance this becomes `[uncond reference, uncond view, cond reference, cond view]`. Debug mode records attention call counts and Q/K/V checksums.
- No watermark detector feedback, query optimization, training, fine-tuning, or white-box watermark-specific logic is included.
- Diffusers attention processor APIs can change. This code targets the `AttnProcessor`-style API used by recent Diffusers releases and includes shape tests for the local processor logic.

## Troubleshooting

- If CUDA is unavailable, run with `--device cpu --dtype float32`; this will be slow.
- If model loading fails, verify the pinned RedbeardNZ revision in the local Hugging Face cache.
- If attention shape errors occur after upgrading Diffusers, first run `pytest tests/test_attention_shapes.py`, then inspect `raven/attention.py`.
- If outputs drift too much, lower `--strength` or use `--view_guided_attention true --color_transfer true`.
