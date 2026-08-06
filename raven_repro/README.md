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

### Removed legacy scripts

`scripts/run_raven.py` and `scripts/attack_folder.py` were deleted in commit
`0dc34d7`. Both were ablation-only CLI wrappers around `RavenPipeline` with zero
production callers.

### Compatibility: old CLI → `experiments/main.py`

| Old `run_raven.py` / `attack_folder.py` | `experiments/main.py` equivalent | Status |
|---|---|---|
| `--inversion_mode ddim` | `--diffusion-mode ddim` | renamed — exact equivalent |
| `--inversion_mode forward_noise` | *none* | **unsupported** — `forward_noise` inversion + DDIM scheduler denoising. `--diffusion-mode ddpm` uses DDPM scheduler (stochastic denoising, different noise schedule). `--diffusion-mode ddim-ddpm` uses DDIM inversion (unet loop, not `add_noise`). The discarded combo `forward_noise + DDIM denoise` is the exact `FORBIDDEN_PAIR = ("ddpm", "ddim")` rejected by `experiment_config.py:37`. This ablation is intentionally removed. |
| `--shift_sampling independent_axes` | `--shift-mode random` (default) | renamed — `plan_shift()` hardcodes random independent axes per sample |
| `--shift_sampling coupled_diagonal` | *none* | **unsupported** — `plan_shift()` never produces this; pipeline path only reachable when `shift_x/shift_y` absent, which `main.py` always provides |
| `--shift_sign random` | (default) | renamed — always random per axis |
| `--shift_sign positive` / `negative` | *none* | **unsupported** — not reachable via `plan_shift()` |
| `--shift_space image_pixels` / `latent_pixels` | `--shift-space` | renamed — passes through to pipeline unchanged |
| `--warp_mode integer` / `grid_sample` | `--warp-mode` | renamed — passes through (default changed to `raven_paper_nfpa_gap_fill`) |
| `--padding_mode zeros` / `reflection` / `border` | `--padding-mode` | renamed — passes through. Note: formal warp modes (`raven_paper_nfpa_gap_fill`) reject `zeros` at pipeline level |
| `--shift_min` / `--shift_max` | `--shift-magnitude-min` / `--shift-magnitude-max` | renamed |
| `--seed N` | `--base-seed N` | renamed — per-sample seed = base_seed + run_id |
| `--view_guided_attention` | `--view-guided-attention` | renamed |
| `--color_transfer` | `--color-transfer` (`aligned`/`none`) | renamed — bool→choice |
| `--input <path>` (single image) | *none* | **unsupported** — `main.py` always reads input paths from metadata CSV |
| `--input_dir <path>` (batch folder) | *none* | **unsupported** — no directory-walk mode |
| `error.txt` / `failed.txt` | *none* | **unsupported** — `main.py` uses `records.jsonl` + exit code |

## Evaluation

The only formal evaluation entrypoint is `experiments/run_raven_formal_eval.py`.
Run stages separately against one output root. Omit `--output-root` and it resolves to
the canonical, content-addressed root
`outputs/<tr|gs>/<dataset>/<variant>/<source-manifest-sha>_<attack-config-hash>` — not a
timestamp — so an identical re-run resumes the same root instead of creating another
directory. An explicit `--output-root` must still live under the canonical root for
`--method`.

```bash
python experiments/run_raven_formal_eval.py \
  --dataset diffusiondb --method TR \
  --source-metadata data/tr/diffusiondb/metadata.csv \
  --expected-count 30 --batch-size 10 --device cuda --gpu 0 --stage snapshot
```

Then run `attack-watermarked`, `attack-clean` for TR, `verify`, `quality`, `fid`,
`clip`, `aggregate`, and `validate` with the identical arguments plus `--resume`.
`attack-clean` belongs to the TR protocol only; GS and the other methods never run it
and never write a per-sample `input.png`.
Formal color transfer is exclusively `paper_exact_two_stage_aligned` and consumes
`effective_source_flow_dx_image_px` / `effective_source_flow_dy_image_px`.
Do not run a full cohort until the 2/10/30 gates pass in fresh output roots.

## Key Files

- `raven/pipeline_raven.py`: two-stream partial-noising and denoising pipeline.
- `raven/inversion.py`: VAE encoding plus partial DDIM inversion or Equation-(4) forward noising.
- `raven/warp.py`: integer zero-padded latent translation plus an explicit `grid_sample` ablation.
- `raven/attention.py`: view-guided self-attention processor. Text cross-attention is left unchanged.
- `raven/color_transfer.py`: effective-source-flow aligned LAB luminance/chroma transfer; unaligned modes are unsupported.
- `../experiments/main.py`: unified attack runner (single-image and batch).
- `scripts/eval_quality.py`: PSNR/SSIM helper.
- `scripts/audit_dataset.py`: metadata, hash, image-format, and pairing audit.
- `../experiments/run_raven_formal_eval.py`: the single formal stage orchestrator.
- `raven/eval_protocol.py`: centralized formal attack, detector, FID, resume, provider, and CLIP provenance.
- `scripts/raven_nfpa_tr_eval.py`: Tree-Ring complex-L1 detector helper used by the formal runner.
- `scripts/run_diffusiondb_chain_after_clean.py`: disabled historical chain retained only for helper-level research evidence.
- `../experiments/run_raven_aligned_color_eval.py`: the sole effective-flow aligned postprocessing evaluator.
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

## Shared TR Clean V2

`shared_tr_clean_v2` is a formal RAVEN comparison profile, separate from each
method's official reproduction profile. GM, T2S, RID, HSTR, and HSQR runners
under `experiments/generate_*_from_tr_shared_clean.py` consume canonical
Tree-Ring metadata and clean artifacts, inject only the method-specific
watermark through the authoritative provider, and write only method-specific
`watermarked.png` outputs. A cohort is `formal_shared_tr_clean` only after the
cross-method audit validates it by `run_id` against `data/tr/diffusiondb/metadata.csv`.
See `docs/SHARED_TR_CLEAN_V2.md` for commands and schema details.
