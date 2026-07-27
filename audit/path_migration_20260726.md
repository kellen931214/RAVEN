# Canonical data/output layout migration — 2026-07-26

The repository now has exactly five data/output roots. Nothing else may be created at
the top of `data/` or `outputs/`.

```
data/
├── clean/   original clean images + the metadata/prompts that define them
├── tr/      Tree-Ring watermarked images + metadata
└── gs/      Gaussian Shading watermarked images + metadata

outputs/
├── tr/      every Tree-Ring run artifact (attack, eval, metrics, verification,
│            snapshots, provenance, logs, tables, quarantined invalid runs)
└── gs/      every Gaussian Shading run artifact
```

Canonical helpers live in `raven_repro/raven/eval_protocol.py`
(`method_data_root`, `method_output_root`, `cohort_dir`, `source_metadata_path`,
`watermarked_image_path`, `clean_data_dir`, `formal_output_root`, `formal_run_key`,
`scratch_run_root`, `assert_canonical_output_root`).

## Current cohorts

| Cohort | Clean | Watermarked | Runs |
|---|---|---|---|
| TR diffusiondb 1001 | `data/clean/diffusiondb/` | `data/tr/diffusiondb/` | `outputs/tr/diffusiondb/` |
| GS diffusiondb 1001 (TR-matched) | `data/clean/gs_diffusiondb_1001_match_tr/GS/` | `data/gs/gs_diffusiondb_1001_match_tr/GS/` | `outputs/gs/gs_diffusiondb_1001_match_tr/` |
| GS 10-pair gates | `data/clean/gs_*_10_50step/GS/` | `data/gs/gs_*_10_50step/GS/` | `outputs/gs/gs_*_10_50step/` |

TR-paired and GS-paired clean images are **different image sets** (different base
latents) that share filenames, so they are kept under distinct dataset directories and
must never be flattened together.

## Path prefix map (old → new)

Historical run records store an absolute path next to a content SHA-256 for every
artifact. The images and artifacts themselves are unchanged (SHA-verified), only their
location moved. Use this table to resolve any pre-migration absolute path.

| Old prefix | New prefix |
|---|---|
| `outputs/raven_paired_formal/diffusiondb/20260717T014700Z/data/generated/diffusiondb` | `data/clean/diffusiondb` |
| `outputs/raven_paired_formal/diffusiondb/20260717T014700Z/data/watermarked/diffusiondb` | `data/tr/diffusiondb` |
| `outputs/raven_paired_formal/diffusiondb/20260717T014700Z/inputs` | `data/clean/diffusiondb/inputs` |
| `outputs/raven_paired_formal/diffusiondb/20260717T014700Z` | `outputs/tr/diffusiondb/paired_formal_generation` |
| `outputs/gs_diffusiondb_1001_match_tr/clean/gs_diffusiondb_1001_match_tr/GS` | `data/clean/gs_diffusiondb_1001_match_tr/GS` |
| `outputs/gs_diffusiondb_1001_match_tr/watermarked/gs_diffusiondb_1001_match_tr` | `data/gs/gs_diffusiondb_1001_match_tr` |
| `outputs/gs_<gate>/clean/gs_<gate>/GS` | `data/clean/gs_<gate>/GS` |
| `outputs/gs_<gate>/watermarked/gs_<gate>` | `data/gs/gs_<gate>` |
| `outputs/raven_formal_eval/diffusiondb/TR` | `outputs/tr/diffusiondb/formal_eval` |
| `outputs/raven_formal_protocol_rerun/diffusiondb/TR` | `outputs/tr/diffusiondb/formal_protocol_rerun` |
| `outputs/raven_aligned_color_eval/diffusiondb/TR` | `outputs/tr/diffusiondb/aligned_color_eval` |
| `outputs/raven_color_transfer_comparison/diffusiondb/TR` | `outputs/tr/diffusiondb/color_transfer_comparison` |
| `outputs/raven_formal_gates/diffusiondb/TR` | `outputs/tr/diffusiondb/formal_gates` |
| `outputs/raven_ablation_eval/diffusiondb/TR` | `outputs/tr/diffusiondb/ablation_eval` |
| `outputs/raven_ablation_eval/gs_diffusiondb_1001_match_tr/GS` | `outputs/gs/gs_diffusiondb_1001_match_tr/ablation_eval` |
| `outputs/gs_formal_dryrun_cleanup` | `outputs/gs/gs_gate_cleanup_10_50step/formal_dryrun` |
| `outputs/gs_formal_dryrun_recheck` | `outputs/gs/gs_gate_recheck_10_50step/formal_dryrun` |
| `outputs/invalid/shared_latent_20260716` | `outputs/tr/diffusiondb/_invalid/shared_latent_20260716` |
| `outputs/legacy_invalid/20260718T072817Z` | `outputs/tr/diffusiondb/_invalid/legacy_20260718T072817Z` |
| `outputs/RAVEN_EVALS_READABLE` | `outputs/tr/_readable` |
| `RAVEN_TABLES` | `outputs/tr/_tables` |
| `logs/gs_*` | `outputs/gs/<cohort>/logs/` |

## What was rewritten, and what was deliberately not

**Rewritten (live, must resolve today):**
- `clean_path`, `watermarked_path`, `watermarked_image_path` in every source
  `metadata*.csv` under `data/`. Every rewritten path was verified to exist and to
  match its recorded `clean_sha256` / `watermarked_sha256`, and `pairing_sha256` was
  recomputed and confirmed **byte-identical** (paths are not in `PAIRING_HASH_FIELDS`).
- Generation `results*.json` / `summary*.json` / `shard_merge_audit.json` under `data/`.
- Code defaults, tests, and documentation.

**Deliberately not rewritten — completed run provenance:**
24,725 files inside already-VALIDATED run directories (mostly per-sample `record.json`
and `debug_info.json`, plus `attack_records_*.jsonl`, `snapshot_index.jsonl`,
`manifest.csv`, `run_config.json`) still contain pre-migration absolute paths.

They were left byte-identical **on purpose**:
- Each recorded path is stored *next to the content SHA-256 of that artifact*
  (`clean_path`+`clean_sha256`, `attacked_path`+`attacked_sha256`, …). Identity is
  SHA-anchored, not path-anchored, so the records remain content-valid.
- Those files are themselves hash-chained: `attack_records_sha256`,
  `snapshot_index_sha256`, `manifest_sha256` and `source_metadata_sha256` are recorded
  in the verification manifest and in `VALIDATED.json`. Rewriting the path strings would
  silently invalidate the recorded integrity chain of results that are already validated.

Consequence: a completed pre-migration run can still be **audited** (all SHAs verify),
but cannot be `--resume`d in place, because its recorded `source_metadata_sha256` refers
to the pre-migration CSV text. New runs use the canonical layout from the start.

## Output policy (enforced going forward)

- `--output-root` on `experiments/run_raven_formal_eval.py` is optional; omitted, it
  resolves to `outputs/<tr|gs>/<dataset>/<variant>/<run-key>`. An explicit value must
  still sit under the canonical root for `--method` (`assert_canonical_output_root`).
- `run-key` = `<source-manifest-short-sha>_<attack-config-short-hash>` — content
  addressed, **not** a timestamp, so an identical re-run resumes the same root instead
  of creating a new directory.
- Methods outside `ATTACK_CLEAN_METHODS` (i.e. everything except TR) never run
  attack-clean and never write the redundant per-sample `input.png`; that is their
  *complete* protocol, not a reduced storage-light variant.
- Gates / smoke tests / dry-runs use `scratch_run_root()` under `/tmp/raven-<method>-…`,
  deleted on success and kept + reported on failure. They must not be written to
  `outputs/`.
