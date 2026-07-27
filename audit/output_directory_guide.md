# RAVEN data and output directory guide

Created UTC: 2026-07-22
Rewritten UTC: 2026-07-26 (canonical layout migration)

## Canonical roots

The repository has exactly five data/output roots. Nothing else may be created at the
top of `data/` or `outputs/`.

```
data/clean/   original clean images + the metadata/prompts that define them
data/tr/      Tree-Ring watermarked images + metadata
data/gs/      Gaussian Shading watermarked images + metadata
outputs/tr/   every Tree-Ring run artifact
outputs/gs/   every Gaussian Shading run artifact
```

Resolve them in code with `raven_repro/raven/eval_protocol.py` rather than hard-coding:
`method_data_root`, `method_output_root`, `cohort_dir`, `source_metadata_path`,
`watermarked_image_path`, `clean_data_dir`, `formal_output_root`, `formal_run_key`,
`scratch_run_root`, `assert_canonical_output_root`.

Cohort layout is per method (`FLAT_COHORT_METHODS`). Tree-Ring uses the flat form
since 2026-07-27; every other method keeps the nested per-method form:

```
data/tr/<dataset>/metadata.csv             data/gs/<dataset>/GS/metadata.csv
data/tr/<dataset>/<run_id>/watermarked.png data/gs/<dataset>/GS/<run_id>/watermarked.png
```

The old→new prefix table for pre-migration paths is in
[`path_migration_20260726.md`](path_migration_20260726.md).

## Where things live now

| Content | Location |
|---|---|
| TR clean images (1001) | `data/clean/diffusiondb/` |
| TR prompts + manifest | `data/clean/diffusiondb/inputs/` |
| TR watermarked + metadata | `data/tr/diffusiondb/` (flat) |
| GS clean images | none of its own — GS shares `data/clean/diffusiondb/` |
| GS watermarked + metadata | `data/gs/diffusiondb_shared_tr/GS/` |
| TR runs | `outputs/tr/diffusiondb/<experiment-parameter-slug>/` |
| GS runs | `outputs/gs/<dataset>/<experiment-parameter-slug>/` |
| Per-method/dataset result table | `outputs/<method>/<dataset>/_table/experiment_results.md` |
| TR result tables (legacy) | `outputs/tr/_tables/` |
| TR readable aliases | `outputs/tr/_readable/` |
| Quarantined invalid runs | `outputs/tr/diffusiondb/_invalid/` |

TR-paired and GS-paired clean images are **different image sets** (different base
latents) that share filenames. They live under distinct dataset directories and must
never be flattened together.

## Output policy

- `--output-root` is optional on `experiments/run_raven_formal_eval.py`. Omitted, it
  resolves to `outputs/<tr|gs>/<dataset>/<variant>/<run-key>`. An explicit value must
  still sit under the canonical root for `--method`.
- `run-key` = `<source-manifest-short-sha>_<attack-config-short-hash>`, content
  addressed rather than a timestamp, so an identical re-run **resumes** the same root
  instead of creating another directory.
- Everything except TR never runs attack-clean and never writes the redundant
  per-sample `input.png`; that is the complete protocol for those methods.
- Gates, smoke tests and dry-runs use `scratch_run_root()` (`/tmp/raven-<method>-…`),
  removed on success and kept + reported on failure. They never write into `outputs/`.

## Readable aliases

`outputs/tr/_readable/` holds symlinks into the authoritative run directories.

`ACTIVE_main_1001_four_attacks_five_evaluations` maps to
`outputs/tr/diffusiondb/formal_protocol_rerun/run_20260722T020154Z_3090`, the latest TR
suite. It contains four attack generations and five evaluated outputs:

1. DDIM, no shift, no color transfer.
2. DDPM, nearest/reflection shift, aligned color transfer.
3. DDIM, bilinear/reflection shift, aligned color transfer.
4. DDIM, nearest/reflection shift, aligned color transfer.
5. The same DDIM nearest/reflection pre-color attack evaluated without color transfer.

Outputs 4 and 5 intentionally reuse one attack generation; they are different
postprocessing/evaluation variants rather than two attacks.

`paper_exact_color_comparison/` records the passed smoke and the full1001 paper-exact
runs; both read the same immutable pre-color cohort but write separate roots.

Historical aliases keep the observed terminal/stale state in their names (GPU
architecture failure, idle-GPU wait, detector smoke failure, stopped partial smoke).
Run directories remain the authoritative provenance locations; aliases are navigation
aids only.
