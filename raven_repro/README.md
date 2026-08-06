# RAVEN Reproduction

This is a clean reproduction scaffold for **RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**. It is intended for academic robustness evaluation on images you own or generated yourself.

The implementation uses the NFPA repository as an implementation reference for frame-guided self-attention and the optional `grid_sample` warp ablation. It does not modify NFPA.

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

## Running Attacks

Use the unified entrypoint `experiments/main.py` for attacks:

```bash
python experiments/main.py \
  --dataset my_dataset --method TR \
  --metadata /path/to/metadata.csv \
  --output-dir /tmp/raven-attack \
  --roles watermarked
```

`main.py` requires a metadata CSV (no single-image `--input` flag, no batch
`--input_dir` flag). Per-sample failure is recorded in `records.jsonl` and
reflected in the exit code; the old per-file `error.txt` / `failed.txt` pattern
is not reproduced.

## Evaluation

Use `experiments/eval.py` to run quality metrics, detector evaluation, FID, and
CLIP against outputs produced by `experiments/main.py`.

## Key Files

- `raven/pipeline_raven.py`: two-stream partial-noising and denoising pipeline.
- `raven/inversion.py`: VAE encoding plus partial DDIM inversion or Equation-(4) forward noising.
- `raven/warp.py`: integer zero-padded latent translation plus an explicit `grid_sample` ablation.
- `raven/attention.py`: view-guided self-attention processor. Text cross-attention is left unchanged.
- `raven/color_transfer.py`: effective-source-flow aligned LAB luminance/chroma transfer; unaligned modes are unsupported.
- `../experiments/main.py`: unified attack runner.
- `../experiments/eval.py`: unified evaluation runner.
- `raven/eval_protocol.py`: centralized formal attack, detector, FID, resume, provider, and CLIP provenance.
- `raven/pairing_provenance.py`: fail-closed latent, image, target, pairing, and attack-config audits.
- `raven/evaluation/scoring.py`: canonical detector scoring helpers shared across all 7 methods.

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

Commands in this section are **ABLATION ONLY — NOT FORMAL EVALUATION**.
Use `experiments/main.py` with appropriate flags.

Several historical ablations (`forward_noise` + DDIM denoise,
`coupled_diagonal` shift sampling, fixed `positive`/`negative` shift sign,
single-image `--input`, folder `--input_dir`) have no `main.py` equivalent
and are intentionally removed. See the compatibility table above.

```bash
# No view-guided attention
python experiments/main.py --dataset ablation --method TR \
  --metadata /tmp/one_sample.csv --output-dir outputs/no_vga \
  --view-guided-attention false --color-transfer aligned

# Reduced strength
python experiments/main.py --dataset ablation --method TR \
  --metadata /tmp/one_sample.csv --output-dir outputs/strength_005 \
  --strength 0.05
```

## Reproduction Audit Workflow

Audit metadata and image pairing before loading a model:

```bash
python scripts/audit_dataset.py \
  --metadata /workspace/RAVEN/data/tr/diffusiondb/metadata.csv \
  --workspace-root /workspace \
  --output outputs/tr/diffusiondb/_audit/diffusiondb_TR.json
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

## Watermark Cohort Generation

`experiments/generate_*.py` scripts produce watermarked cohorts from shared
Tree-Ring clean images. Each method injects only its own watermark through
the authoritative provider and writes method-specific `watermarked.png` outputs.
