# RAVEN: Erasing Invisible Watermarks via Novel View Synthesis

Reproduction and evaluation of the RAVEN attack against
seven watermarking methods.

## Supported Methods

| Method | Key |
|---|---|
| Tree-Ring | TR |
| Gaussian Shading | GS |
| GaussMarker | GM |
| T2SMark | T2S |
| RingID | RID |
| HSTR | HSTR |
| HSQR | HSQR |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires `eval_bench_wm/` sibling directory with watermark providers
and a local copy of `RedbeardNZ/stable-diffusion-2-1-base`.

## Attack

```bash
python raven_repro/main.py \
  --dataset my_dataset --method TR \
  --metadata /path/to/metadata.csv \
  --output-dir /tmp/raven-attack \
  --roles watermarked
```

`--metadata` CSV columns: `run_id`, `watermarked_path`, `clean_path`,
`prompt`, `prompt_id`.  Method-specific fields (TR parameters, GS
secrets, GM/T2S bundle paths) pass through unchanged.

See `python raven_repro/main.py --help` for all options.

## Evaluate

```bash
python raven_repro/eval.py \
  --output-dir /tmp/raven-attack \
  --device cuda
```

Reads `config.json` + `records.jsonl` produced by `main.py`.
Runs quality metrics, detector evaluation, FID, and CLIP
without re-initializing the attack pipeline.

## Output Structure

```
<output-dir>/
  config.json          attack configuration
  records.jsonl        per-sample attack records
  samples/
    watermarked/<id>/
      output.png       attacked image
      record.json      per-sample provenance
  evaluation/
    quality/           PSNR/SSIM
    detectors/         per-method scores
    fid/               FID results
    clip/              CLIP scores
```

## Data Sources

This repository does not include watermark datasets.
Create a metadata CSV with paths to your own images.
Generation scripts in `raven_repro/generate/`
can produce watermarked cohorts from shared TR clean images.

## License

Research use only.  See the paper for details.
