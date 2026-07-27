# Debug Changelog

This file records implementation bugs, validated non-bugs, ablations, and the evidence used to verify each change. Large logs and generated outputs are not copied here; paths point to the source artifacts.

## Current Status

| Date | Area | Status | Evidence |
| --- | --- | --- | --- |
| 2026-07-27 | TensorFlow FID protocol as the default | Primary FID is now the TensorFlow FID protocol (clean-fid `legacy_tensorflow`: TF Inception-2015-12-05 features + TF-compatible bilinear resize); clean-fid `clean` is still computed and recorded as a secondary value so earlier TR numbers stay comparable. The formal quality-config hash now carries the FID protocol descriptor, and the runner accepts `scratch_run_root()` gate roots so gates stay out of `outputs/`. | `raven_repro/raven/quality.py`; `experiments/run_raven_formal_eval.py`; `raven_repro/tests/test_fid_staging.py`; `raven_repro/tests/test_canonical_layout.py` |
| 2026-07-27 | TR flat cohort canonical path migration | Canonical TR source metadata is `data/tr/diffusiondb/metadata.csv`; all 1001 TR watermarked image paths point at `data/tr/diffusiondb/<run_id>/watermarked.png`; existing GS shared-clean V2 rows were repointed at the new TR metadata SHA and pairing hashes recomputed with the formal helper. | `experiments/migrate_tr_flat_layout.py`; `experiments/migrate_gs_shared_clean_source.py`; `data/gs/diffusiondb_shared_tr/GS/cross_method_shared_clean_audit.json` |
| 2026-07-21 | Formal provenance and validation hardening | Source HEAD self-reference removed; runtime manifest copy, clean-tree/current-commit binding, pairing re-audit, detector tensor target/mask checks, explicit pre-color hashes, runtime scheduler/device/dtype/package provenance, threshold-specific table fields, and accurate no-color FID definitions added. CPU regression suite evidence is recorded below. | audit/formal_eval_protocol.md; formal runner and protocol regression tests |
| 2026-07-18 | Formal evaluation protocol audit | Implemented immutable snapshots, explicit formal attack/debug assertions, effective-grid quality flow, strict resume/FID/provider/CLIP provenance, full/rounded TR reporting, formal waiter/table, and quarantined pre-audit derived outputs. Complete CPU suite: 148 passed; new 2/10/30 GPU gates remain blocked by unavailable NVML, so full eval is not safe. | `audit/formal_eval_protocol.md`; `audit/current_eval_processes.md`; `outputs/legacy_invalid/20260718T072817Z/DO_NOT_USE.md` |
| 2026-07-17 | Tree-Ring paired generation and formal provenance | Shared-latent source and every derived `TPR=0.177822` result rejected. Per-sample paired latent generation, two-GPU pair sharding, orphan quarantine, fail-closed provenance gates, paired attack config hashes, and two aligned-color-only variants implemented; formal rerun in progress. | `raven_repro/raven/pairing_provenance.py`; `raven_repro/scripts/paired_generation_shards.py`; `raven_repro/scripts/run_diffusiondb_chain_after_clean.py`; `outputs/raven_paired_formal_smoke/diffusiondb/20260717T033000Z/data/watermarked/diffusiondb/TR/shard_merge_audit.json` |
| 2026-07-15 | DiffusionDB latest RAVEN-paper/NFPA-gap-fill Tree-Ring L1 rerun preparation | Fixed attacked-clean config drift, verified 2-sample smoke and 10-sample validation; full 1001 run prepared for nohup. | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `raven_repro/scripts/raven_p1_full.py`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.json` |
| 2026-07-15 | RAVEN-paper / NFPA-gap-fill warp and inverse-overlap quality | Implemented and verified on focused tests plus 10-sample DiffusionDB validation. Existing 1001 DiffusionDB P1 outputs were quality-recomputed without rerunning attack. | `raven_repro/raven/warp.py`; `raven_repro/raven/metrics.py`; `raven_repro/scripts/raven_paper_nfpa_gap_fill_eval.py`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md` |
| 2026-07-15 | RAVEN exact two-stage color transfer | Implemented and verified. Existing 10 validation pre-color outputs were reused; no DDIM inversion or denoising was rerun. | `raven_repro/raven/color_transfer.py`; `raven_repro/scripts/raven_color_transfer_validation.py`; `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.md` |
| 2026-07-14 | NFPA-style Tree-Ring complex L1 evaluation | Completed for DiffusionDB only. MS-COCO was not run after scope was corrected. | `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/aggregate_results.json` |




## 2026-07-27 — GS detector rejected every V2 shared-clean cohort (`missing gs_sampling_seed`)

### Problem
The GS verify stage failed for the whole cohort with
`run_id=<id>: missing gs_sampling_seed` (10/10 rows errored in the 10-sample
gate), so no Gaussian Shading RAVEN evaluation could reach `verify`, let alone
`aggregate`/`validate`.

### Root cause
Two stale V1-only assumptions in the reachable GS detector path:
1. `extract_verification_scores.py` hard-required the fixed pair
   (`gs_sampling_seed`, `gs_sampling_uniform_sha256`). `gs_sampling_seed` exists
   only in the V1 GS pairing protocol; the V2 shared-TR-clean protocol derives
   the uniforms deterministically from the shared Tree-Ring latent and therefore
   defines no sampling seed (see `pairing_provenance.GS_CORE_FIELDS`, which
   removes it deliberately). The authoritative per-protocol field tuple existed
   and was simply not used here.
2. `provider_kwargs()` passed `gs_sampling_seed=integer(row, …, 0)`, i.e. it
   would have silently constructed the detector with a fabricated seed 0 for any
   cohort without one. Detection never uses the sampling seed (the target payload
   comes from the secret), so the value was pure fake provenance.

The verification manifest also did not carry the cohort's `protocol` field, so
the detector had no way to know which field set applies.

### Affected files
- `raven_repro/scripts/extract_verification_scores.py:gs_sampling_provenance`
  (new), `:provider_kwargs`, GS branch of `main`.
- `raven_repro/scripts/build_verification_manifest.py:main` (GS row block).
- `raven_repro/tests/test_verification_pipeline.py`.

### Affected outputs
No formal output was produced by the failing path — the 10-sample gate root in
`/tmp` was the only artifact and it was preserved for diagnosis, then removed.
No existing result is invalidated: the GS detector never ran to completion
before this fix, so there is no earlier GS evaluation carrying a fabricated
sampling seed.

### Fix
`gs_sampling_provenance(row, identifier)` resolves the required sampling fields
from `gs_fields_for_protocol(row["protocol"])`, requires
`gs_sampling_uniform_sha256` for every supported protocol, requires
`gs_sampling_seed` only for V1, and fails closed on an unknown protocol.
`provider_kwargs("GS", …)` includes `gs_sampling_seed` only when the row
actually has one. `build_verification_manifest.py` now writes the cohort's
`protocol` into GS manifest rows so the protocol identity travels with the data.

### Reused code
`raven.pairing_provenance.gs_fields_for_protocol` / `gs_fields_for_rows` — the
same authoritative V1/V2 field-set resolver already used by
`run_raven_formal_eval.py:snapshot_stage` and `build_verification_manifest.py`.
No second field list was introduced.

### Historical bug coverage
Reviewed the 2026-07-27 GS shared-clean V2 and TR flat-layout entries, which
introduced the V2 field set. Searched the repository for `gs_sampling_seed`:
remaining occurrences are the CLI flag and generation-time uniform draw in
`eval_bench_wm/utils/wm/gs_provider.py` (correct — V1 generation), the column
name in the score schema (kept, empty for V2), and the migration/audit history.
No other reachable path still requires the V1-only field.

### Regression prevention
New tests assert that a V2 row resolves to `gs_sampling_uniform_sha256` alone,
that a V1 row still requires both fields, that a V1 row missing its seed raises,
that an unknown protocol fails closed, that GS detector kwargs never fabricate a
sampling seed, and that the verification manifest records `protocol`.

### Validation
- `python -m pytest -q raven_repro/tests/test_verification_pipeline.py`: 5 passed.
- Full detector probe on the preserved 10-sample gate attack records
  (`build_verification_manifest.py` + `extract_verification_scores.py --method GS`,
  `CUDA_VISIBLE_DEVICES=0`): 10 manifest rows, **0 errors**; before-attack
  bit accuracy `1.0` on every row, attacked bit accuracy ≈0.47–0.55, clean-image
  bit accuracy ≈0.45–0.55, `gs_official_tau_onebit=0.6484375`.

### Watermark integrity
- Source-data validity: `data/gs/diffusiondb_shared_tr/GS/metadata.csv`, 1001
  rows, protocol `gaussian_shading_shared_tr_clean_v2`, unchanged.
- Clean/watermarked pairing status: pairing audit passes in the snapshot stage.
- Base-latent uniqueness status: unchanged (per-run unique).
- Watermark target and mask status: detector target/mask SHAs are still verified
  against the source row; the probe passed those checks for all 10 rows.
- Attack-pairing status: GS attacks the watermarked role only;
  `attacked_clean_count` stays 0.
- Detector score definition: GS `bit_accuracy`, higher_is_watermarked.
- Threshold calibration source: official beta-tail `tau_onebit` (0.6484375);
  no empirical clean-negative 1%-FPR calibration was performed.
- Actual empirical FPR: not measured (no clean-negative cohort evaluation).
- Quality metric reference / CLIP / FID staging: unchanged by this fix.
- Outputs requiring regeneration: none.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: pending at time of entry
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Push status: pending
- Entry point: `experiments/run_raven_formal_eval.py` (verify stage)
- Formal output eligibility: bug fix only; the GS runs it unblocks are recorded
  by `experiments/update_experiment_table.py` when they finish.


## 2026-07-27 — TensorFlow FID protocol becomes the default FID

### Problem
FID was computed only with clean-fid's own `clean` mode (anti-aliased bicubic
resizing). The watermarking literature reports FID under the original
TensorFlow protocol (Inception-2015-12-05 pool3 features with
TensorFlow-compatible bilinear resizing), so the reported numbers were not the
FID definition the comparison targets. The FID protocol was also written as a
free-text string inside the formal quality-config hash, so a protocol change
would not have been visible in provenance.

### Root cause
`raven_repro/raven/quality.py:clean_fid` called `cleanfid.fid.compute_fid`
without a `mode` argument, silently taking the library default, and
`run_raven_formal_eval.py:fid_stage` then stamped `"mode": "clean"` on the
result. No caller could choose or record another FID protocol.

### Affected files
- `raven_repro/raven/quality.py:clean_fid`, `:fid_protocol_descriptor`,
  `:require_fid_mode` (new).
- `experiments/run_raven_formal_eval.py:run_config` (quality-config hash),
  `:fid_stage`.
- `raven_repro/raven/eval_protocol.py:is_scratch_run_root` (new),
  `:assert_canonical_output_root`.
- `raven_repro/tests/test_fid_staging.py`,
  `raven_repro/tests/test_canonical_layout.py`.

### Affected outputs
No existing output was modified, deleted or recomputed. Every FID value already
on disk under `outputs/tr/` was produced with clean-fid `clean` mode and stays
valid **as a clean-fid `clean` result** — it is not a TensorFlow-protocol FID
and must not be relabelled as one. Those runs are complete and validated, so
they need no resume; a `--resume` of one of them would now fail closed on the
changed `quality_config_hash`, which is the intended behaviour.

### Fix
`clean_fid()` computes the primary value with `mode="legacy_tensorflow"` (the
TensorFlow FID protocol) and additionally records clean-fid `clean` as a
secondary value in the same `fid_result.json`
(`mode_values`, `secondary_modes`, `mode_feature_extractors`, `protocol`).
`value`/`mode` are the TensorFlow-protocol primary, so the experiment table's
`FID` column reports the TF value. Unknown FID modes fail closed via
`require_fid_mode()`. The formal `quality_config_hash` now embeds
`fid_protocol_descriptor()` instead of a hand-written string.
`assert_canonical_output_root()` additionally accepts a root created by
`scratch_run_root(method, …)`, so the required 10-sample gate can run in `/tmp`
instead of polluting `outputs/` — an arbitrary `/tmp` path is still rejected.

### Reused code
Existing `cleanfid` package, the existing `clean_fid()` call sites (all four
runners take the default and therefore switch together), the existing
`stage_fid_records()` staging/manifest gate, and the existing
`scratch_run_root()` helper. No second FID implementation was added; TensorFlow
itself is not installed next to the torch/diffusers attack pipeline because
clean-fid's `legacy_tensorflow` mode is that protocol.

### Historical bug coverage
Reviewed the 2026-07-21 provenance-hardening entry (which introduced the FID
definitions and no-color FID staging) and the 2026-07-27 layout entries.
Searched for every `clean_fid` / `clean-fid` call site: `run_raven_formal_eval.py`,
`run_raven_no_color_eval.py`, `run_raven_aligned_color_eval.py`,
`run_raven_color_transfer_eval.py` — all four call the shared helper with the
default mode, so no stale per-runner FID copy remains. The no-color/aligned
variant hashes already include `source_code_manifest_sha256`, so the protocol
change is covered there transitively.

### Regression prevention
New tests assert: the primary mode is `legacy_tensorflow` with `clean` recorded
as secondary; both modes are actually computed and reported under
`mode_values`; an unregistered mode raises; the formal quality-config hash uses
the shared descriptor rather than the old literal; and the canonical-root guard
accepts only `scratch_run_root()` gate roots in `/tmp`.

### Validation
- `python -m pytest -q raven_repro/tests/test_fid_staging.py raven_repro/tests/test_canonical_layout.py`: 22 passed.
- `python -m pytest -q tests` from `raven_repro/`: 342 passed, 61 warnings
  (339 before + 3 new FID protocol tests, plus the extended layout test).
- Real-image FID probe on 12 clean vs 12 GS watermarked images with
  `CUDA_VISIBLE_DEVICES=0`: `legacy_tensorflow`, `legacy_pytorch` (275.36) and
  `clean` (271.68) all execute; clean-fid downloads the TF Inception graph to
  `/tmp/inception-2015-12-05.pt`. With several GPUs visible clean-fid wraps the
  feature extractor in `nn.DataParallel` and fails with an NCCL error, so FID
  must run with a single visible GPU — the formal runner's `--gpu` already
  pins `CUDA_VISIBLE_DEVICES`.

### Watermark integrity
- Source-data validity: unchanged; GS cohort
  `data/gs/diffusiondb_shared_tr/GS/metadata.csv` has 1001 rows, all
  watermarked images present.
- Clean/watermarked pairing status: untouched by this change.
- Base-latent uniqueness status: untouched.
- Watermark target and mask status: untouched.
- Attack-pairing status: untouched; no attack was rerun.
- Detector score definition: unchanged (GS `bit_accuracy`, TR `l1_complex`).
- Threshold calibration source: unchanged.
- Actual empirical FPR: not affected by this change.
- Quality metric reference: unchanged (watermarked input vs final
  post-color-transfer attacked image, effective-flow overlap).
- CLIP input definition: unchanged (attacked-watermarked image vs source prompt).
- FID staging status: unchanged strict fresh staging; only the feature
  extractor/resize protocol behind the value changed.
- Outputs requiring regeneration: none. Existing `outputs/tr/` FID values remain
  valid clean-fid `clean` results and are not comparable to the new primary
  TensorFlow-protocol value; the secondary `clean` value recorded in new runs is
  what should be compared against them.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: pending at time of entry
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Push status: pending
- Entry point: `experiments/run_raven_formal_eval.py`
- Formal output eligibility: metric-protocol change only; the GS runs that use
  it are recorded separately below when they finish.


## 2026-07-27 — TR flat cohort canonical path migration and GS shared-clean source refresh

### Problem
The Tree-Ring cohort was intentionally moved to the flat layout
`data/tr/diffusiondb/metadata.csv` and
`data/tr/diffusiondb/<run_id>/watermarked.png`, but reachable code, tests and
shared-clean metadata still assumed the nested
`data/tr/diffusiondb/TR/metadata.csv` source path. GS shared-clean V2 rows also
recorded the old TR metadata path and SHA, which are part of the V2 pairing
identity.

### Root cause
Canonical source layout was treated as a global nested method layout instead of
a per-method cohort fact. TR was moved on disk, while GS was not; therefore only
TR belongs to the flat cohort set.

### Affected files
- `raven_repro/raven/eval_protocol.py:cohort_dir`,
  `:source_metadata_path`, `:watermarked_image_path`.
- `experiments/generate_gs_from_tr_shared_clean.py:DEFAULT_TR_METADATA`.
- `experiments/wait_and_run_raven_eval_all_datasets.py:process_cohort`.
- `experiments/migrate_tr_flat_layout.py`.
- `experiments/migrate_gs_shared_clean_source.py`.
- `.agents/skills/raven-shared-clean/SKILL.md`.
- `.agents/skills/raven-experiment-naming/SKILL.md`.
- `raven_repro/tests/test_canonical_layout.py`.
- `raven_repro/tests/test_gaussian_shading_shared_tr_clean.py`.
- `audit/output_directory_guide.md`.
- `audit/path_migration_20260726.md`.

### Affected outputs
TR image bytes were not regenerated, moved or re-encoded during this migration.
`data/tr/diffusiondb/metadata.csv` was already in the flat path form when
checked here, with SHA
`b359a5104f93d54580914f152f074a72f7aae59e8ab6ef4a6a05ab91662aa66c`.
The partial GS shared-clean V2 cohort contained 475 generated rows; only
`shared_clean_source_metadata_path`, `shared_clean_source_metadata_sha256` and
`pairing_sha256` were rewritten in
`data/gs/diffusiondb_shared_tr/GS/metadata.csv`. Its post-migration SHA is
`176edf49d2128533862a29965f92e5ff515e17a015afbd305e9a1ff4d033829a`.

### Fix
`source_metadata_path()` now resolves through `cohort_dir()` and
`FLAT_COHORT_METHODS`, where only TR is flat. `watermarked_image_path()` exposes
the same canonical per-method rule for image references. The formal waiter now
uses `source_metadata_path(method, dataset)` by default instead of a nested
string template. GS path rules remain nested because the GS cohort on disk was
not moved.

### Reused code
The migration scripts reuse `sha256_path`, `audit_pairing_rows`,
`audit_tr_gs_shared_clean` and the formal `build_pairing_sha256()` instead of
duplicating pairing-hash logic.

### Historical bug coverage
Reviewed the 2026-07-26 canonical layout and 2026-07-27 GS shared-clean V2
entries. Searched code, tests, skills and docs for nested TR path patterns. Old
paths that appear only in historical changelog/path-migration tables remain as
historical evidence, not live canonical definitions.

### Regression prevention
Canonical-path tests now assert TR is flat, GS remains nested, and real TR
metadata rows point to `watermarked_image_path("TR", "diffusiondb", run_id)`.
The GS shared-clean tests fail closed when the canonical TR metadata file is not
present at `data/tr/diffusiondb/metadata.csv`.

### Validation
- GS generation was active and was stopped with SIGINT before metadata writes;
  the final log ended with `KeyboardInterrupt` and no updater process remained.
- `python experiments/migrate_tr_flat_layout.py --dry-run data/tr/diffusiondb/metadata.csv`
  verified 1001/1001 TR watermarked SHA-256 values, with `paths_rewritten=0`.
- `python experiments/migrate_gs_shared_clean_source.py --tr-metadata data/tr/diffusiondb/metadata.csv data/gs/diffusiondb_shared_tr/GS/metadata.csv`
  migrated 475 GS rows and passed GS pairing plus TR-GS shared-clean audit.
- Independent TR flat-path SHA check: `rows=1001`, `missing_paths=0`,
  `sha_mismatches=0`.
- `python -m py_compile experiments/wait_and_run_raven_eval_all_datasets.py experiments/generate_gs_from_tr_shared_clean.py experiments/migrate_tr_flat_layout.py experiments/migrate_gs_shared_clean_source.py raven_repro/raven/eval_protocol.py`: passed.
- `python -m pytest -q raven_repro/tests/test_canonical_layout.py raven_repro/tests/test_gaussian_shading_shared_tr_clean.py`: 69 passed, 14 warnings.
- `python -m pytest -q tests` from `raven_repro/`: 339 passed, 61 warnings.
- `git diff --check`: passed.

### Watermark integrity
- Source-data validity: TR metadata has 1001 rows and all recorded
  watermarked SHA-256 values match files at the flat layout.
- Clean/watermarked pairing status: TR pairing audit passes; GS generated rows
  pass pairing audit after recomputing V2 pairing hashes.
- Base-latent uniqueness status: TR has 1001 unique base latent hashes; GS
  generated rows have 475 unique base latent hashes.
- Watermark target and mask status: unchanged; migration does not touch target,
  mask or image content hashes.
- Attack pairing: not rerun; formal GS attacks were not launched.
- Detector score definition: unchanged (`bit_accuracy` for GS, method-specific
  TR detector definitions unchanged).
- Threshold calibration source / actual empirical FPR / quality / CLIP / FID:
  not evaluated by this metadata migration.
- Outputs requiring regeneration: none from this path migration; interrupted GS
  generation remains partial and resumable only after this migration commit is
  validated.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: pending at time of entry
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Push status: pending
- Entry point: `experiments/generate_gs_from_tr_shared_clean.py`
- Formal output eligibility: metadata migration/audit only; no formal attack
  launched.


## 2026-07-21 - Idle-GPU strict protocol rerun dispatcher

### Dispatcher worker-log startup correction

The compatible-GPU relaunch reached worker assignment but failed before spawning
a worker because the parent opened the per-worker log before creating its
directory. The parent now creates the log directory first. A regression test
locks the required ordering. No attack process or output was created by either
failed dispatcher root.

### Compatible-GPU allowlist correction

The first dispatcher launch stopped before creating any worker because physical
GPU 6 is a Blackwell sm_120 device while the installed PyTorch supports only
through sm_90. The relaunch interface now accepts an explicit physical-GPU
allowlist. This run is pinned to GPUs 4, 5, and 8, which previously completed
the same formal workload successfully; GPU 6 is never considered. This is an
explicit user-directed relaunch, not automatic fallback after the failure.


Added experiments/wait_and_run_raven_protocol_variants.py and
experiments/run_raven_no_color_eval.py. The dispatcher never signals existing
processes. It waits for GPUs with no compute PID, low utilization, and at least
18 GiB free memory; it also waits for at least 64 GiB available CPU RAM and
performs a CUDA allocation/kernel probe before launching a worker.

Four attack workflows produce five complete evaluations. DDIM nearest shift
aligned-color is shared with DDIM nearest shift no-color, whose detector, FID,
CLIP, PSNR, and SSIM consume only the SHA-bound pre-color record. Every worker
must pass a fresh 10-sample cohort before its 1001-sample cohort. Existing roots
are reused only when their validation status, sample count, current commit,
source manifest, and attack config agree; the old balanced-schedule roots do not
qualify as strict paper-random results.

## 2026-07-21 - Formal provenance and validation hardening

### Confirmed root causes

- The committed source manifest recorded its build commit and the runner required
  that value to equal current HEAD. Committing the manifest necessarily advanced
  HEAD, creating a permanent self-reference failure despite unchanged source
  files.
- Formal snapshots did not invoke the existing paired-base-latent audit, attack
  records omitted part of the generation pairing provenance, and final
  validation did not repeat a full-cohort pairing audit.
- The formal TR detector hashed raw target bytes, while generation used the
  canonical tensor hash containing shape, dtype, and bytes. The detector did
  not verify its target or mask against generation before scoring.
- Pre-color images were inferred from a debug directory rather than bound by
  path and SHA in the attack record. Runtime device, dtype, resolved scheduler,
  torch, and diffusers provenance were also absent from transform hashes.
- The table collapsed two empirical FPRs and two threshold bases into ambiguous
  columns. FID staging hard-coded post-color wording even for no-color inputs.
- The prior five-magnitude/four-quadrant deterministic plan did not sample every
  integer in the paper's independent per-axis ranges.

### Fix and validation behavior

Formal execution now requires a clean working tree, validates every source
manifest file SHA and size, copies the exact manifest and companion SHA into the
output root, and binds the run/resume to that runtime manifest SHA plus current
commit. The manifest's historical build HEAD is informational only.

TR snapshot and final validation both run the complete pairing audit. Attack and
verification records retain pairing/base-latent/target/mask/generation/watermark
hashes; attacked-clean and attacked-watermarked must share pairing SHA. The TR
provider uses generation's canonical tensor hash function and checks target and
mask before creating its score directory.

Pre-color path/SHA, CUDA device class, float16 dtype, scheduler selector and
resolved config/hash, torch version, and diffusers version are hashed and
resume-validated. No-color binding accepts only the explicit pre-color fields.
FID callers provide their reference/attacked definitions, and the formal table
reports both actual FPRs and both attack-success threshold bases.

New future paper-protocol shifts use deterministic per-sample RNG over every
integer magnitude 24 through 32 with independently selected signs per axis. The
old five-magnitude balanced schedule remains historical evidence under the
explicit name balanced_deterministic_schedule; running outputs are not modified.

### Affected existing outputs

The following in-progress roots were launched from commit 9a8a6c7 and were left
running without source changes:

- outputs/raven_ablation_eval/diffusiondb/TR/nfpa_bilinear_reflection_ddim_aligned/1001_20260721T110735Z
- outputs/raven_ablation_eval/diffusiondb/TR/nfpa_nearest_reflection_ddpm_aligned/1001_20260721T110735Z
- outputs/raven_ablation_eval/diffusiondb/TR/nfpa_nearest_reflection_ddim_no_shift_aligned/1001_20260721T110735Z

Their images, immutable snapshots, attack records, manifests, logs, and metrics
remain preserved historical artifacts. They used the earlier balanced shift
schedule and lack the new runtime/pre-color/detector tensor provenance, so they
must not receive a new strict VALIDATED.json. Attack images need not be deleted
or silently rerun. Formal detector score records/aggregates must be regenerated
with target/mask checks; FID must be freshly staged with the correct definition;
CLIP and quality values may remain historical but require fresh
provenance-bound records and validation before inclusion in a new formal table.

### Regression coverage

Negative tests cover source-manifest companion/source drift, dirty-tree
preflight, duplicate/shared base latents, clean/watermarked latent mismatch,
target hash mismatch, mask hash mismatch, replaced pre-color images,
dtype/scheduler drift, both actual-FPR table columns, and no-color FID wording.
No GPU evaluation was started for this change.


## 2026-07-21 — Formal NFPA/DDPM/no-shift ablation variants

### Scope
The formal evaluator now accepts a complete, validated attack-config file. The
only supported ablation deltas are `latent_sampling_mode=bilinear`,
`inversion_mode=scheduler_mode=ddpm`, and `shift_plan_mode=zero`; all retain
the pinned model revision, 512px inputs, 50 steps, strength 0.15, CFG 2.5,
reflection padding, view-guided attention, aligned color transfer, detector,
CLIP, FID, and effective-flow overlap metrics.

### Root cause
The baseline runner used a module-level configuration directly. That made a
controlled sampler/inversion/shift ablation impossible without duplicating
or weakening formal attack, resume, debug, and manifest checks.

### Fix
`eval_protocol.py` centrally validates and hashes each complete variant config;
the existing formal runner, pipeline, manifest builder, and resume validation
consume that config. DDPM uses the existing forward-noise inversion primitive
and a `DDPMScheduler`; DDIM is unchanged. The zero-shift plan keeps the same
per-run seed but records `(0, 0)`. The runner can also consume the existing
1001-row snapshot index, verifies every historical batch hash and the original
metadata SHA before selecting a gate cohort, and records that index SHA in each
new snapshot. Source manifests include tracked variant configs, and result
tables identify the exact variant.

### Gate-exposed bilinear fix
The first bilinear smoke failed before an attack record was committed because
its actual-grid effective source flow was fractional (`23.5 px`) while aligned
color transfer and overlap quality rejected non-integers. The shared metrics
module now samples the reference/chroma at continuous effective source
coordinates with bilinear interpolation over the real valid rectangle; it does
not round to planned flow or include reflected padding. Integer flows still
reduce to direct correspondence. Focused regression tests cover fractional
alignment, valid bounds, exact synthetic crop correspondence, and quality
metadata provenance.

### Regression prevention
`test_formal_variant_config.py` checks config/debug hashing, zero-shift seed
stability, DDPM validation, and the DDPM inversion path. Full formal tests and
10-sample GPU gates must pass before any 1001-sample variant launch.

## Confirmed Issues And Fixes

### 2026-07-17 - Corrected Paired DDIM-Shift No-Color Evaluation

| Field | Details |
| --- | --- |
| Problem | The corrected paired 1001-sample run intentionally evaluated only aligned-color and blended-aligned-color variants. The only existing `ddim_shift_no_color` aggregate came from the rejected shared-latent/unpaired source and could not be reused. |
| Source validity | The new no-color evaluation is restricted to `outputs/raven_paired_formal/diffusiondb/20260717T014700Z`, whose pairing audit has 1001 unique base-latent seeds/hashes, zero duplicate base latents, paired clean/watermarked base hashes, matching target/mask/config hashes, and complete image SHA provenance. |
| Fix | `quality_decomposition_experiment.py` now supports explicit `--variants ddim_shift_no_color`. It directly reuses each verified clean/WM `view_guided_output.png` before color transfer, records `color_transfer_mode=none`, and keeps the existing alignment+blend default unchanged. Detector, threshold, CLIP-vs-prompt, FID, inverse-overlap PSNR/SSIM, pairing, and attack-config gates remain shared with the authoritative evaluator. Fresh FID staging now records a canonical manifest SHA and rejects duplicate paths, missing files, stale targets, broken links, and staging file-set drift. |
| Attack pairing | Attacked-clean and attacked-watermarked inputs must match run ID, attack seed, dx/dy, exact DDIM timestep, strength, guidance, inversion mode/prompts, warp, sampling, padding, normalization, model revision, pairing hash, and canonical attack-config hash before scoring. |
| Evaluation definitions | Tree-Ring score is official complex L1 with lower-is-watermarked and strict `< threshold`; the threshold is calibrated from no-color attacked-clean scores at target FPR 1%. CLIP is attacked-watermarked no-color output versus the source prompt. FID compares original watermarked images with no-color attacked-watermarked images. PSNR/SSIM compare the same pair over inverse-warp valid overlap. |
| Memory controls | No DDIM/UNet attack rerun is needed. Images are processed one at a time with one CPU thread, no DataLoader or dataset cache, RAM guards, incremental records, and fresh temporary FID staging. |
| Legacy handling | Every old shared-latent no-color result remains `INVALID_SHARED_LATENT`; no old detector or quality value is reused. |
| Validation | Focused tests passed (`19 passed`) and the complete suite passed (`101 passed`, 12 expected warnings). The first smoke was quarantined under `invalid/incomplete_no_color_smoke_gpu_mapping_20260717T174140Z` after logical GPU index 0 mapped to a full 48 GiB device; no metric output was produced. The UUID-bound 2-sample smoke completed with two unique base latents, pairing/config/hash audits passing, direct pre-color paths, matching detector target, finite metrics, zero NaN/Inf, and fresh FID staging manifest SHA `99405e947f1db8c2c636e0d42a5c6bdceb29441d0f49d6dfb0e8ac6a8d97182a`. Evidence: `quality_decomposition_no_color_smoke2_20260717T174412Z/run.log` and `aggregate_results.json`. The corrected 1001-sample run is active at `quality_decomposition_no_color_1001_20260717T174846Z/` with PID recorded in `pid` and progress in `run.log`. |
| Git provenance | Branch `agent/cleanup-quality-decomposition`, base HEAD `3330e67c0a9538268691ac32fe00ecca79abef50`; working tree dirty; not committed or pushed; results remain non-release until publication. |

### 2026-07-17 - Tree-Ring Generator Reused One Full Latent And Clean Images Were Not Paired

| Field | Details |
| --- | --- |
| Problem | `experiments/generate_watermarked_images.py` previously called `get_wm_latents()` once outside the sample loop and reused the returned complete `wm_zT` for all 1001 prompts. Clean images were generated by a separate sequential RNG workflow, so a clean/watermarked pair did not share a base latent. The old manifest then wrote `generation_seed=42+run_id` without any latent hash proving that claim. |
| Impact | The source cohort was non-independent and unpaired. Every detector or quality result derived from `outputs/raven_color_alignment_ablation/diffusiondb/20260716T082019Z`, including aligned-color `TPR=0.177822`, is invalid and must not be reported. |
| Core logic changed | Each sample now creates a unique base latent from its own seed, injects Tree-Ring into a clone, and generates clean/watermarked images with identical prompt/model/scheduler/steps/guidance and the same exact base latent. Generation is sharded by run ID across two GPUs; each worker owns complete pairs and writes an independent shard metadata/log/results file. A merge gate requires exact run-ID coverage and audits every row before producing formal `metadata.csv`. Per-row provenance records base seed/hash, clean and watermarked base hashes, watermarked latent hash, target/mask hashes, image SHA-256 values, generation/watermark config hashes, and a canonical pairing hash. P1, attacked-clean, detector, and quality entry points reject missing provenance, duplicate latent hashes, file drift, target drift, or clean/watermarked attack-config mismatch. Crash-written images without a committed metadata row are moved to `invalid/orphaned_unrecorded/` and regenerated; they are never resumed as valid data. |
| Formal variants | The corrected formal rerun evaluates only `alignment_color` (`paper_exact_two_stage_aligned`, alpha 1.0) and `blend_alignment_color` (`paper_exact_two_stage_aligned_blend`, alpha 0.5). No no-shift or no-color metric branch is run. CLIP is image-text cosine between `wm_attack` and the source prompt. |
| Memory controls | Each of the two workers remains sample-at-a-time with no DataLoader or dataset cache, one CPU thread per process, incremental disk writes, RAM guards, and a waiter that requires two idle compatible GPUs before launch. The two workers duplicate only the diffusion pipeline, not the dataset in CPU memory. |
| Verification | The complete test suite passed (`93 passed`, 12 expected warnings). A two-sample/two-GPU smoke assigned run 0 to GPU 0 and run 1 to GPU 2, produced one audited row per shard, merged with `count=2`, two unique latent seeds/hashes and image hashes, no duplicate latent, and matching target/mask/config/revision. The merged manifest independently passed the same pairing audit. Evidence: `outputs/raven_paired_formal_smoke/diffusiondb/20260717T033000Z/run.log`, `logs/paired_generation_shard_000.log`, `logs/paired_generation_shard_001.log`, and `data/watermarked/diffusiondb/TR/shard_merge_audit.json`. The interrupted unrecorded run 1 was retained under `outputs/raven_paired_formal/diffusiondb/20260717T014700Z/invalid/orphaned_unrecorded/20260717T032839Z/`. |
| Legacy handling | After the corrected formal run succeeds, logs/provenance and invalid aggregate files are archived under `outputs/invalid/shared_latent_20260716/`; contaminated images and derived data are deleted by `invalidate_shared_latent_outputs.py`. |
| Status | Code fix, two-GPU pair sharding, root `run.log`, orphan recovery, and fail-closed merge gate implemented. Two-GPU generation/manifest smoke passed; corrected 1001-sample rerun is pending/resumable. |

### 2026-07-13 - Partial Inversion Was Treated Ambiguously As DDIM

| Field | Details |
| --- | --- |
| Problem | The original partial inversion path used random forward noising via `scheduler.add_noise(clean_latents, noise, timestep)` for the Equation (4) interpretation. That is not true DDIM inversion and should not be the formal reproduction setting. |
| Impact | RAVEN attack diagnostics mixed a random forward-noising ablation with the formal DDIM-inversion reproduction path, making Tree-Ring suppression results hard to interpret. |
| Core logic changed | `ddim` became the primary reproduction mode. `forward_noise` is retained as a labeled ablation. DDIM inversion records denoise scheduler, inverse scheduler, prediction type, target timestep, inverse timestep sequence, and `eta=0.0`. |
| Verification | Round-trip audit reconstructed a real image through VAE encode -> DDIM inversion to timestep 121 -> DDIM denoise -> VAE decode with PSNR 30.3395, SSIM 0.9634, latent MAE 0.02739. |
| Evidence | `raven_repro/raven/inversion.py`; `raven_repro/scripts/audit_ddim_roundtrip.py`; `outputs/raven_diagnostics/20260713T171750Z/roundtrip/roundtrip.json` |
| Status | Fixed and verified for the diagnostic setting. |

### 2026-07-13 - Shift Sign And Overlap Crop Convention Were Wrong/Ambiguous

| Field | Details |
| --- | --- |
| Problem | The local shift convention and overlap crop were ambiguous and previously matched inverse-sampling wording rather than the requested visual convention: positive x should move content right and positive y should move content down for the integer RAVEN translation mode. |
| Impact | PSNR/SSIM overlap comparisons and direction-based diagnostics could compare the wrong regions or report misleading direction metadata. |
| Core logic changed | `crop_overlap` was updated to the right/down visual convention. Integer translation uses explicit slicing with zero padding and no wrap-around. Tests now validate impulse movement for `(+x,0)`, `(-x,0)`, `(0,+y)`, `(0,-y)` and no circular wrap. |
| Verification | Unit tests in `raven_repro/tests/test_warp.py` and `raven_repro/tests/test_overlap_metrics.py`; diagonal interpretation ablation recorded valid overlap and per-direction outcomes. |
| Evidence | `raven_repro/raven/metrics.py`; `raven_repro/raven/warp.py`; `outputs/raven_diagonal_interpretation/20260714T071247Z/aggregate_results.md` |
| Status | Fixed for integer/latent-grid RAVEN modes; NFPA exact mode separately records flow direction and visual direction because NFPA uses inverse sampling. |

### 2026-07-13 - Attention Processor State Could Leak Across Runs

| Field | Details |
| --- | --- |
| Problem | Attention processors were not restored as an exact original mapping after a run; cross-attention processors could be replaced by generic defaults and debug state could leak across repeated ablations. |
| Impact | Attention-on/off and sampling/padding comparisons risked cross-run contamination, making output differences unreliable. |
| Core logic changed | The pipeline stores the original UNet attention processor mapping, installs view-guided processors only for self-attention, preserves existing cross-attention processors, and restores the exact mapping before/after runs. Debug metadata records processor counts and call counts. |
| Verification | `test_install_preserves_existing_cross_attention_processor_identity`, `test_restore_default_attention_restores_exact_mapping`, and sampling/padding provenance explicitly recorded the state fix. Attention-on/off outputs were not identical in 3-sample diagnostics. |
| Evidence | `raven_repro/raven/attention.py`; `raven_repro/raven/pipeline_raven.py`; `raven_repro/tests/test_attention_shapes.py`; `outputs/raven_sampling_padding_ablation/20260714T093603Z/provenance.json`; `outputs/raven_diagnostics/20260713T171942Z/attacks/attack_summary.json` |
| Status | Fixed and verified by unit tests and diagnostic outputs. |

### 2026-07-14 - RAVEN Warp Did Not Match NFPA Coordinate/Sampling Convention

| Field | Details |
| --- | --- |
| Problem | The RAVEN integer zero-padding warp is not NFPA's latent warp convention. NFPA builds a 512x512 image-coordinate flow, normalizes by `/W` and `/H`, resizes coordinates bilinearly to latent size, and samples latent with `nearest` plus `reflection` padding. |
| Impact | Tree-Ring suppression and quality could differ because normalization, sampling mode, and padding were changed together. Integer zero padding was not a faithful NFPA-compatible ablation. |
| Core logic changed | Added explicit NFPA-compatible warp utilities and metadata: `coords_grid`, image-coordinate flow, NFPA normalization, coordinate resize metadata, nearest/reflection sampling, inverse-sampling sign metadata, and separate `latent_grid`/`integer` ablation modes. |
| Verification | NFPA unit tests passed for coordinate grid shape, `/W` `/H` normalization, four-direction impulse movement, reflection padding, nearest sampling, effective displacement, no NaN/Inf, and CPU/GPU consistency. |
| Evidence | `raven_repro/raven/warp.py`; `raven_repro/scripts/nfpa_warp_ablation.py`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/unit_test_results.json`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/effective_displacement.json` |
| Status | Fixed as named modes; `nfpa_exact`, `latent_grid`, `integer`, and `direct_latent` remain separate ablations. |

### 2026-07-14 - Quality Metric Reference Was Ambiguous

| Field | Details |
| --- | --- |
| Problem | Quality output names could be read as generic `psnr`/`ssim` without a clear reference image. |
| Impact | Attack quality could be compared against clean input in one place and watermarked input in another without being obvious. |
| Core logic changed | Per-sample records and aggregate outputs now save `psnr_vs_watermarked`, `ssim_vs_watermarked`, `psnr_vs_clean`, and `ssim_vs_clean`; the primary quality reference for P1 attack results is watermarked input. |
| Verification | P1 full aggregate reports quality under `quality.primary_reference = watermarked_input`; sampling/padding ablation Markdown explicitly states the primary reference. |
| Evidence | `raven_repro/scripts/raven_p1_full.py`; `raven_repro/scripts/nfpa_sampling_padding_ablation.py`; `outputs/raven_p1_full/diffusiondb/20260714T095907Z/aggregate_results.json`; `outputs/raven_sampling_padding_ablation/20260714T093603Z/aggregate_results.md` |
| Status | Fixed for current diagnostics and P1 full outputs. |

### 2026-07-15 - RAVEN Paper/NFPA Gap-Fill Warp And Inverse-Overlap Quality

| Field | Details |
| --- | --- |
| Problem | The previous P1 full driver used a direct latent-grid `/8` displacement with nearest/reflection. That was useful for the P1 ablation but did not preserve the requested priority rule: use RAVEN paper settings for shift sampling and use NFPA only for underspecified coordinate-grid implementation details. Existing PSNR/SSIM also used visual-shift overlap rather than the explicit inverse-warp correspondence formula requested for the paper-comparable quality protocol. |
| Impact | Future full runs could silently reuse the old P1 transform as if it were the paper/NFPA gap-fill setting, and quality metrics could include the wrong correspondence crop if flow direction and visual direction were confused. |
| Core logic changed | Added `raven_paper_nfpa_gap_fill` mode, which passes RAVEN image-pixel `dx/dy` directly into an NFPA-style 512x512 coordinate flow, uses `/W` `/H` normalization, bilinear coordinate-grid resize, inverse `grid_sample`, `padding_mode=reflection`, `align_corners=False`, and main `mode=nearest`. Bilinear is retained only as a same-grid ablation. Added `crop_overlap_inverse_warp` and quality helpers that compare watermarked input against pre/post color-transfer outputs over valid inverse-warp overlap only. Future `raven_p1_full.py` runs now use the new mode and record a config hash including grid version, sampling, padding, normalization, align_corners, and shift. |
| Verification | Focused syntax/tests passed: `46 passed, 1 warning` for `test_warp.py` and `test_overlap_metrics.py`. Existing DiffusionDB 1001 P1 outputs were recomputed for inverse-overlap quality without rerunning attack. A 10-sample validation compared old P1, new nearest main, and bilinear ablation with complex L1 scoring. |
| Evidence | `raven_repro/raven/warp.py`; `raven_repro/raven/metrics.py`; `raven_repro/raven/pipeline_raven.py`; `raven_repro/scripts/raven_p1_full.py`; `raven_repro/scripts/raven_paper_nfpa_gap_fill_eval.py`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md`; `outputs/raven_paper_nfpa_gap_fill/diffusiondb_quality_recompute_20260715T034545Z/quality_summary.json`; `outputs/raven_paper_nfpa_gap_fill/diffusiondb_validation_20260715T035849Z/aggregate_results.json` |
| Status | Implemented and verified on focused tests plus 10-sample validation. Full 1001 new-transform attack has not been run. |

### 2026-07-15 - Color Transfer Used Direct Generated-Luminance Statistics Instead Of Paper Two-Stage Formula

| Field | Details |
| --- | --- |
| Problem | The previous CIELAB transfer matched generated-image `L_opt` mean/std directly to the original watermarked luminance and then inserted original `a/b`. The requested RAVEN formula first builds `x_c_lab = (L_opt, a_w, b_w)`, converts LAB -> RGB -> LAB, computes statistics from the realized `L_c`, then matches `L_c` to `L_w`. |
| Impact | In gamut-clipped cases, the realized luminance after combining generated L with original chroma can differ from raw `L_opt`. Direct statistics can therefore produce slightly different luminance matching and diagnostics from the paper formula. |
| Core logic changed | `color_contrast_transfer` now defaults to `paper_exact_two_stage`; old behavior remains available as `direct_stats`. Diagnostics now include `L_opt`, `L_c`, `L_w`, pre/post clip `L_final` ranges, final output L mean/std, saturated pixel ratio, and luminance mean/std errors. Pipeline debug metadata records `color_transfer_mode=paper_exact_two_stage`. |
| Verification | `test_color_transfer.py` covers output shape/dtype/range, deterministic output, constant luminance with no NaN/Inf, final L mean/std closeness to original, and a gamut-clipping synthetic case where `paper_exact_two_stage` differs from `direct_stats`. 10-sample color-only validation reused existing `view_guided_output.png` files and did not rerun DDIM/denoising. |
| Evidence | `raven_repro/raven/color_transfer.py`; `raven_repro/tests/test_color_transfer.py`; `raven_repro/scripts/raven_color_transfer_validation.py`; `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.json` |
| Status | Fixed and verified for focused tests plus 10-sample color-only validation. Existing full attacked images generated before this change remain legacy color-transfer outputs. |

### 2026-07-14 - NFPA-Style Tree-Ring Complex L1 Metric Was Missing

| Field | Details |
| --- | --- |
| Problem | Existing Tree-Ring evaluation used `-log10(p)` fixed/calibrated threshold logic, while NFPA evaluates Tree-Ring with complex L1 distance `torch.abs(decoded_watermark - target_watermark).mean(-1)` where lower score indicates watermark. |
| Impact | The existing P1 result is useful as separate `-log10(p)` analysis but is not the requested NFPA-style Tree-Ring `TPR@1%FPR`. After-attack NFPA calibration also requires attacked-clean images, not only attacked-watermarked images. |
| Core logic changed | Added an independent `raven_nfpa_tr_eval.py` flow that copies existing P1 attacked-watermarked records, generates only attacked-clean images with the same shift plan/settings, scores original clean/watermarked/attacked-clean/attacked-watermarked with complex L1, and calibrates before/after thresholds separately with NFPA's strict `< threshold` rule. |
| Verification | DiffusionDB completed with 1001 rows. NFPA-style before threshold 76.23775482177734, before actual FPR 0.008991008991008992, before TPR 1.0; after threshold 79.72408294677734, after actual FPR 0.008991008991008992, after TPR 0.48451548451548454; attack success 0.5154845154845155. |
| Evidence | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/nfpa_l1_scores.jsonl`; `outputs/raven_nfpa_tr_eval/diffusiondb/20260714T161952Z/aggregate_results.json` |
| Status | Completed for DiffusionDB. This is separate from legacy `-log10(p)` fixed-threshold analysis. |

### 2026-07-15 - Attacked-Clean Evaluation Still Used Legacy `latent_grid` Config

| Field | Details |
| --- | --- |
| Problem | `raven_nfpa_tr_eval.py attack-clean` still called the pipeline with `warp_mode="latent_grid"`, while the new formal attacked-watermarked driver uses `raven_paper_nfpa_gap_fill` with nearest/reflection and paper-exact two-stage color transfer. |
| Impact | Post-attack NFPA-style L1 calibration could compare attacked-clean negatives produced by a different transform from attacked-watermarked positives, invalidating the after-attack `TPR@1%FPR`. |
| Core logic changed | `attack-clean` now uses `raven_paper_nfpa_gap_fill`, `padding_mode="reflection"`, `latent_sampling_mode="nearest"`, empty prompts, DDIM, strength 0.15, guidance 2.5, and `paper_exact_two_stage` color transfer. Both attacked-clean and attacked-watermarked records now save config fields and `transform_config_hash`; L1 scoring stops if run ID, seed, dx/dy, timestep, scheduler/prompt/warp/sampling/padding/color-transfer settings, or transform hash differ. Scoring output is standardized as `l1_scores.jsonl` plus `per_sample_results.csv`, with before/after thresholds calibrated separately from original-clean and attacked-clean scores. |
| Verification | `py_compile` passed for `raven_nfpa_tr_eval.py` and `raven_p1_full.py`. Focused tests passed: `58 passed, 8 warnings` for `test_warp.py`, `test_overlap_metrics.py`, `test_metrics.py`, and `test_color_transfer.py`. 2-sample smoke completed with finite L1 and no NaN/Inf. 10-sample validation completed with after TPR 0.700000, attack success 0.300000, config/hash audit passing for all 10 records, and no duplicate rounded/exact L1 groups. |
| Evidence | `raven_repro/scripts/raven_nfpa_tr_eval.py`; `raven_repro/scripts/raven_p1_full.py`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/smoke2_eval/aggregate_results.json`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.json` |
| Status | Fixed and gate-verified. Full DiffusionDB 1001 rerun is the next stage and must use new timestamped outputs. |

## Confirmed Non-Bugs

### 2026-07-13 - Tree-Ring High TPR Was Not Only A Legacy Threshold Artifact

| Field | Details |
| --- | --- |
| Finding | The legacy fixed-threshold detect rate and clean-negative calibrated TPR were close for the old DiffusionDB Tree-Ring result. |
| Evidence | `outputs/verification_v2/metrics/TR_diffusiondb_1001_20260713T074340Z.json` reports calibrated TPR@1%FPR 0.7442557443, legacy detect rate 0.7512487512, actual clean FPR 0.0099900100, attacked ROC-AUC 0.9584182052. |
| Conclusion | The main gap was not explained by accidentally using the legacy fixed threshold. Attack pipeline and detector metric interpretation needed separate audit. |
| Status | Confirmed not the sole bug. |

### 2026-07-13 - DDIM Inversion Round Trip Was Not Obviously Broken

| Field | Details |
| --- | --- |
| Finding | A true DDIM inversion/denoise round trip reconstructed the input at reasonable quality. |
| Evidence | `outputs/raven_diagnostics/20260713T171750Z/roundtrip/roundtrip.json`: PSNR 30.3395, SSIM 0.9634, exact timestep 121, inverse scheduler `DDIMInverseScheduler`, denoise scheduler `DDIMScheduler`, eta 0.0. |
| Conclusion | DDIM inversion still required exact timestep/provenance fixes, but the round-trip diagnostic did not indicate a catastrophic inversion failure. |
| Status | Confirmed not a blocking bug for subsequent ablations. |

### 2026-07-13 - View-Guided Attention Was Not A No-Op

| Field | Details |
| --- | --- |
| Finding | Attention-on and attention-off outputs were not identical in the 3-sample diagnostic. |
| Evidence | `outputs/raven_diagnostics/20260713T171942Z/attacks/attack_summary.json`: max absolute pixel differences 108, 105, and 73 for run IDs 0-2. |
| Conclusion | Attention integration had state-restoration issues, but the hook path was not completely inert. |
| Status | Confirmed not a no-op after diagnostics. |

## Ablations And Comparisons

### 2026-07-13 - 3-Sample Attack Pipeline Diagnostic

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| DDIM + attention + independent 24 px | True DDIM, attention on, independent axes | DiffusionDB Tree-Ring, 3 samples, fixed threshold 1.6372738343 | Mean score after 4.1723; detect after 1.000; PSNR 23.888; SSIM 0.8409 | Shifted attack strongly lowered score but did not drop below threshold on 3 samples. |
| DDIM + no attention + independent 24 px | Attention off | Same | Mean score after 4.5206; detect after 1.000; PSNR 23.710; SSIM 0.8367 | Attention on changed output and was slightly stronger on this tiny cohort. |
| DDIM + attention + coupled 24 px | Coupled diagonal signs | Same | Mean score after 3.5942; detect after 1.000; PSNR 23.841; SSIM 0.8403 | Coupled diagonal was competitive in 3 samples, not enough evidence for final setting. |
| forward_noise + attention + independent 24 px | Random forward noising | Same | Mean score after 3.6579; detect after 1.000; PSNR 21.380; SSIM 0.7383 | forward_noise remained an ablation; quality was worse. |
| DDIM + no shift | Shift disabled | Same | Mean score after 29.6314; detect after 1.000; PSNR 33.806; SSIM 0.9676 | Shift itself caused the main suppression. |
| DDIM + attention + independent 32 px | Larger shift | Same | Mean score after 3.8718; detect after 1.000; PSNR 23.241; SSIM 0.8295 | Larger shift did not clearly dominate 24 px in 3 samples. |

Source: `outputs/raven_diagnostics/20260713T172532Z/tree_ring_scores/diagnostic_summary.md`.

### 2026-07-14 - Diagonal Shift Interpretation, 30 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A | image pixels, common sign, independent x/y magnitudes | DiffusionDB Tree-Ring, 30 samples, fixed threshold | Detect rate 0.6333; mean score 2.6072; PSNR 22.101; SSIM 0.6648 | Better than no-shift; sign binding not clearly best. |
| B | image pixels, independent x/y signs and magnitudes | Same | Detect rate 0.7000; mean score 2.8610; PSNR 22.127; SSIM 0.6673 | Main paper-interpretation candidate, but not strongest suppression in this cohort. |
| C | image pixels, strict `dx=dy` | Same | Detect rate 0.7667; mean score 2.6783; PSNR 22.109; SSIM 0.6753 | Strict diagonal did not improve detection rate. |
| D | direct latent cells, common sign | Same | Detect rate 0.7000; mean score 2.3560; PSNR 17.545; SSIM 0.5519; overlap 0.3166 | Suppression came with severe quality/overlap loss. |
| E | direct latent cells, independent signs | Same | Detect rate 0.7000; mean score 2.4713; PSNR 17.706; SSIM 0.5488; overlap 0.3166 | Direct-latent remained an ambiguity ablation, not a good formal candidate. |
| G | integer latent 3/4 cells, independent signs | Same | Detect rate 0.7000; mean score 2.6252; PSNR 24.071; SSIM 0.8213 | Higher quality than fractional image-pixel modes, similar detect rate. |
| I | no shift | Same | Detect rate 1.0000; mean score 33.8595; PSNR 37.109; SSIM 0.9521 | Confirms shift is necessary for Tree-Ring suppression. |

Source: `outputs/raven_diagonal_interpretation/20260714T071247Z/aggregate_results.md`.

### 2026-07-14 - NFPA Warp Convention, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| nfpa_independent | NFPA image-coordinate flow, `/W` `/H`, bilinear coordinate resize, nearest/reflection latent sampling | DiffusionDB Tree-Ring, 10 samples | Detect rate 0.6000; mean score 2.0998; PSNR 19.210; SSIM 0.5560 | Strong suppression, lower quality. |
| nfpa_sign_bound | Same NFPA warp, common sign | Same | Detect rate 0.6000; mean score 2.1879; PSNR 19.161; SSIM 0.5548 | Similar to independent signs on 10 samples. |
| nfpa_strict_diagonal | Same NFPA warp, `dx=dy` | Same | Detect rate 0.6000; mean score 2.2152; PSNR 19.584; SSIM 0.5982 | Similar suppression with slightly better quality. |
| integer_zero_pad | Image pixels rounded to latent cells, slicing, zero padding | Same | Detect rate 0.8000; mean score 3.4765; PSNR 22.964; SSIM 0.8014 | Better quality but weaker suppression. |
| direct_latent | 24-32 direct latent cells | Same | Detect rate 0.8000; mean score 2.8523; PSNR 16.345; SSIM 0.5057; overlap 0.3202 | Not a reasonable formal candidate due to quality/overlap loss. |
| no_shift | No shift | Same | Detect rate 1.0000; mean score 35.9116; PSNR 34.592; SSIM 0.9492 | Confirms shift effect. |

Sources: `outputs/raven_nfpa_warp_ablation/20260714T081940Z/aggregate_results.md`; `outputs/raven_nfpa_warp_ablation/20260714T081940Z/unit_test_results.json`.

### 2026-07-14 - NFPA Normalization, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| N1_nfpa_exact | `x_norm = 2*(x+dx)/W - 1` | Same samples, nearest/reflection, align_corners False | Detect rate 0.6000; mean score 2.1328; PSNR vs WM 19.210; SSIM 0.5560 | NFPA exact had slightly lower score but lower quality. |
| N2_pixel_center | Adds `+0.5` pixel-center offset | Same | Detect rate 0.6000; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Same detect rate; slightly better quality. |
| N3_latent_div8 | Direct `/8` latent grid displacement with same sampling/padding | Same | Detect rate 0.6000; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Matched N2 in this run, so prior quality gap was not normalization alone. |

Source: `outputs/raven_nfpa_normalization_ablation/20260714T090146Z/aggregate_results.md`.

### 2026-07-14 - Sampling/Padding, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| P1_nearest_reflection | nearest sampling + reflection padding | Fixed `/8` latent displacement, align_corners False | Detect rate 0.6000; 4 below threshold; mean score 2.2591; PSNR 19.424; SSIM 0.5725 | Best suppression among this 10-sample set; selected for P1 full. |
| P2_nearest_zeros | nearest + zero padding | Same | Detect rate 0.6000; 4 below threshold; mean score 2.4530; PSNR 19.235; SSIM 0.5667 | Reflection improved score and quality over zeros at nearest. |
| P3_bilinear_reflection | bilinear + reflection | Same | Detect rate 0.7000; 3 below threshold; mean score 2.6979; PSNR 20.606; SSIM 0.6117 | Better quality but weaker suppression. |
| P4_bilinear_zeros | bilinear + zeros | Same | Detect rate 0.7000; 3 below threshold; mean score 2.9293; PSNR 20.428; SSIM 0.6049 | Weakest suppression among the four. |

Source: `outputs/raven_sampling_padding_ablation/20260714T093603Z/aggregate_results.md`.

### 2026-07-15 - RAVEN-paper / NFPA-gap-fill Validation, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A_old_P1_latent_grid_nearest_reflection | Legacy direct latent-grid `/8` nearest/reflection output reused from old P1 | DiffusionDB first 10 samples; NFPA-style complex L1; post-color inverse-overlap quality vs watermarked input | Mean L1 before 54.087430; mean L1 after 81.962275; mean delta 27.874845; PSNR 19.732; SSIM 0.5911 | Old output remains a valid legacy reference but is not the new paper/NFPA gap-fill transform. |
| B_RAVEN_paper_NFPA_gap_fill_nearest | RAVEN image-pixel shift plan passed through NFPA image-coordinate grid, nearest/reflection main mode | Same | Mean L1 before 54.087430; mean L1 after 81.677802; mean delta 27.590371; PSNR 19.608; SSIM 0.5861 | New main mode executes end-to-end; suppression/quality are close to old P1 on this tiny cohort. |
| C_RAVEN_paper_NFPA_gap_fill_bilinear | Same grid/padding/shift as B; only latent value sampling changed to bilinear | Same | Mean L1 before 54.087430; mean L1 after 81.472040; mean delta 27.384610; PSNR 20.700; SSIM 0.6221 | Bilinear quality is higher but this is an ablation; main remains nearest because NFPA uses nearest. |

Sources: `outputs/raven_paper_nfpa_gap_fill/diffusiondb_validation_20260715T035849Z/aggregate_results.md`; `outputs/raven_paper_nfpa_gap_fill/audit_report_20260715T040535Z.md`.

### 2026-07-15 - Existing DiffusionDB P1 Inverse-Overlap Quality Recompute

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| Existing P1 outputs, post-color overlap | No attack rerun; recomputed valid inverse-warp overlap against watermarked input | DiffusionDB 1001 existing P1 attacked-watermarked outputs | Mean PSNR 20.097515; median 20.104606; mean SSIM 0.564739; median 0.571788; NaN/Inf 0 | Existing records had enough path and flow metadata to correct quality metrics without rerunning attack. |
| Existing P1 outputs, raw full image | Same images, no overlap crop | Same | Mean PSNR 14.897020; mean SSIM 0.439846 | Full-image metrics are lower because shifted non-corresponding regions are included; paper-comparable local protocol should use overlap fields. |

Source: `outputs/raven_paper_nfpa_gap_fill/diffusiondb_quality_recompute_20260715T034545Z/quality_summary.md`.

### 2026-07-15 - Color Transfer Formula Comparison, 10 Samples

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| no_color_transfer | Reused pre-color `view_guided_output.png` directly | Existing 10 DiffusionDB validation outputs; no inversion/denoising rerun; NFPA-style complex L1 scoring | Mean L1 76.687901; overlap PSNR 23.090; overlap SSIM 0.6654; saturated ratio 0.015703; L mean/std errors 0.715407/0.385719 | Highest image similarity because no chroma/luminance correction is applied, but luminance mean error is worse than color-transfer modes. |
| direct_stats | Old local formula: match raw generated `L_opt` stats to original `L_w` | Same | Mean L1 81.677802; overlap PSNR 19.608; overlap SSIM 0.5861; saturated ratio 0.103858; L mean/std errors 0.081025/0.206632 | Legacy ablation retained; good luminance matching, but not the requested paper two-stage formula. |
| paper_exact_two_stage | Build `(L_opt,a_w,b_w)`, LAB->RGB->LAB, match realized `L_c` stats to `L_w` | Same | Mean L1 81.570660; overlap PSNR 19.579; overlap SSIM 0.5844; saturated ratio 0.109624; L mean/std errors 0.057105/0.146216 | New default follows requested paper formula and improves luminance mean/std matching versus direct_stats on this cohort. |

Source: `outputs/raven_color_transfer_validation/diffusiondb_20260715T042018Z/aggregate_results.md`.

### 2026-07-14 - P1 Full Fixed `-log10(p)` Evaluation

| Dataset | N | Clean FPR | Before TPR | Attacked TPR | Attack success | ROC-AUC | PSNR vs WM | SSIM vs WM | Conclusion |
| --- | -: | -: | -: | -: | -: | -: | -: | -: | --- |
| DiffusionDB | 1001 | 0.009990 | 1.000000 | 0.688312 | 0.311688 | 0.943715 | 20.098 | 0.5647 | P1 lowered old DiffusionDB attacked TPR by 0.055944 versus the prior old-pipeline result, but many Tree-Ring marks remained detectable. |
| MS-COCO | 1000 | 0.012000 | 1.000000 | 0.634000 | 0.366000 | 0.934490 | 19.743 | 0.6063 | Fixed threshold clean FPR differed from exactly 1%; do not call this COCO result dataset-calibrated TPR@1%FPR. |

Sources: `outputs/raven_p1_full/combined_summary.json`; `outputs/raven_p1_full/diffusiondb/20260714T095907Z/aggregate_results.json`; `outputs/raven_p1_full/mscoco/20260714T095907Z/aggregate_results.json`.


### 2026-07-15 - Latest Formal DiffusionDB Rerun Gates

| Implementation | Main difference | Evaluation setting | Result | Conclusion |
| --- | --- | --- | --- | --- |
| 2-sample smoke | New attacked-watermarked plus new attacked-clean, both `raven_paper_nfpa_gap_fill`, nearest/reflection, paper-exact color transfer | DiffusionDB first 2 samples; NFPA-style complex L1; separate before/after clean calibration | Before TPR 1.000000; after TPR 1.000000; NaN/Inf 0; duplicate exact/rounded groups 0 | Integration path works but N=2 is not statistical. |
| 10-sample validation | Same formal settings with deterministic shift plan | DiffusionDB first 10 samples; NFPA-style complex L1 | Before TPR 1.000000; after TPR 0.700000; attack success 0.300000; mean attacked-WM L1 81.251940; overlap PSNR vs WM 20.005228; overlap SSIM vs WM 0.594788; NaN/Inf 0 | Gate passed; safe to start full 1001 DiffusionDB run using the same scripts/config. |

Sources: `outputs/raven_tr_full_diffusiondb/20260715T060017Z/smoke2_eval/aggregate_results.md`; `outputs/raven_tr_full_diffusiondb/20260715T060017Z/validation10_eval/aggregate_results.md`.

## Open Or In-Progress Items

| Date | Item | Current evidence | Next verification |
| --- | --- | --- | --- |
| 2026-07-15 | Whether to expand `RAVEN-paper / NFPA-gap-fill` to 100-200 or full 1001 | 10-sample validation is complete, but full new-transform attack has not been run. | If requested, first run 100-200 samples in a new output directory; do not overwrite old P1 outputs. |
| 2026-07-15 | Paper PSNR/SSIM provenance | RAVEN arXiv HTML inspected; local report treats overlap PSNR/SSIM as requested paper-comparable protocol, while the inspected paper quality table emphasizes FID/CLIP. | If exact PSNR/SSIM numbers are required for a table, cite the user-defined overlap protocol separately from paper-reported FID/CLIP. |

## 2026-07-20 — Effective-flow aligned color transfer only

### Problem
The formal pipeline still executed unaligned `paper_exact_two_stage`, while old
decomposition launchers exposed no-color, blended, direct-stat, and legacy-flow
paths. The unaligned result had poor quality and could not be treated as the
final color-transfer protocol.

### Root cause
`RavenPipeline.run()` hard-coded the unaligned mode and did not pass warp-derived
effective source flow into color transfer. Historical decomposition scripts read
`flow_dx_image_px` / `flow_dy_image_px`, which represented planned legacy flow.

### Affected files
- `raven_repro/raven/color_transfer.py`
- `raven_repro/raven/pipeline_raven.py`
- `raven_repro/raven/eval_protocol.py`
- `experiments/run_raven_aligned_color_eval.py`

### Affected outputs
`outputs/raven_formal_eval/diffusiondb/TR/1001_20260718T090947Z` remains immutable
but its unaligned post-color metrics are legacy. The previous no-color output is
ablation-only. Neither may be merged into the new aligned result.

### Fix
Only `paper_exact_two_stage_aligned` remains executable. Alignment and quality
overlap use `effective_source_flow_dx_image_px` and
`effective_source_flow_dy_image_px`; missing actual-grid flow fails closed. The
aligned evaluator reuses immutable pre-color views and rebuilds attacked-clean
and attacked-watermarked postprocessing under a new config/source manifest.

### Reused code
The formal manifest builder, Tree-Ring `score-formal`, NFPA rounded2 threshold,
FID staging, OpenCLIP helper, and overlap metrics remain unchanged.

### Historical bug coverage
All reachable unaligned/direct/blended color modes and old decomposition/color
alignment launchers were removed or disabled. Archived evidence is not a formal
entrypoint.

### Regression prevention
Tests reject legacy modes and legacy flow keywords, require actual-grid effective
flow, prove planned flow is ignored by the alignment selector, and reject
attacked-clean/watermarked effective-flow drift.

### Validation
Targeted aligned/effective-flow tests: 75 passed. Full repository suite: 143
passed after the gate-exposed snapshot fix. The replacement 2-sample and
1001-sample aligned evaluation results remain pending execution.

### Watermark integrity
- Source data: unchanged immutable 1001-sample paired cohort.
- Pairing/base latents: unchanged; no generation rerun.
- Attack pairing: seed, planned flow, effective flow, timestep, model revision,
  and transform provenance checked per run ID.
- Detector: unchanged complex L1, lower-is-watermark, strict `<` threshold.
- Threshold: original-clean and attacked-clean calibration reported separately.
- Quality: watermarked input versus aligned attacked-watermarked valid overlap.
- CLIP: aligned attacked-watermarked image versus original prompt.
- FID: fresh watermarked-versus-aligned-attacked staging.

### Gate-exposed snapshot fix
The first 2-sample gate at
`outputs/raven_aligned_color_eval/diffusiondb/TR/2_20260720T081355Z` failed
before detector scoring because the strict manifest builder received the full
1001-row source snapshot index with a 2-row attack record set. The failed root
and log are preserved. The runner now creates an immutable exact-ID cohort
snapshot and index, preserves the source snapshot SHA separately, and retains
strict equality between snapshot and attack run-ID sets.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: pending
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Push status: pending
- Entry point: `experiments/run_raven_aligned_color_eval.py`
- Formal output eligibility: pending tests, gates, and aligned full evaluation

## 2026-07-22 — Deterministic scheduler provenance hashing

### Gate failure
The formal protocol dispatcher root
`outputs/raven_formal_protocol_rerun/diffusiondb/TR/run_20260721T185822Z`
failed both active 10-sample smoke workflows before detector scores. The strict
TR verifier reported `run_id=0: attacked pair transform_config_hash mismatch`.
The two pending variants never launched and no 1001-sample workflow started.
All failed logs and partial attack outputs remain preserved.

### Root cause and impact
Diffusers stores `_use_default_values` as private scheduler metadata derived from
an unordered set. Identical clean and watermarked `DDPMScheduler` instances
serialized that list in different orders, producing different scheduler and
transform hashes even though every inference-relevant scheduler parameter,
seed, shift, dtype, package version, and attack configuration matched. This was
a provenance false positive; no detector or aggregate result was produced.

### Fix and regression coverage
`canonical_scheduler_config()` now excludes Diffusers private underscore-prefixed
metadata before scheduler hashing. Scheduler class and Diffusers version remain
separate required provenance fields. All public inference parameters remain in
the hash, so a real change such as `variance_type` still changes the hash and is
rejected by pairing/resume validation. `RavenPipeline` applies this canonicalizer
before writing debug and attack records. Regression tests cover private metadata
order/version drift and public scheduler parameter drift. Focused tests: 15
passed. Full CPU suite: 163 passed with 44 existing warnings. `py_compile` and
`git diff --check` passed. Replacement GPU smoke/full runs require a new source
manifest and new timestamped output root; the failed root is not resumable.

## 2026-07-22 - Human-readable output aliases

The timestamped formal output roots were difficult to interpret, but the active
waiter and evaluators reference them by absolute path. To avoid interrupting or
invalidating those processes, no output was renamed or moved. Added
`outputs/RAVEN_EVALS_READABLE/` with descriptive symlink aliases and an output
README, plus `audit/output_directory_guide.md`. The main timestamped suite is
identified as four attack generations producing five evaluation variants.
Historical failed/stale roots and the two concurrently active paper-exact
full1001 roots are labeled separately. No process was stopped.


## 2026-07-22 - Centered NFPA bilinear warp ablation

### Problem
`nfpa_warp_single_latent()` uses `grid_sample(..., align_corners=False)` but the
legacy NFPA image-grid normalization maps integer pixel indices with
`2*x/W - 1`. For bilinear sampling this is half a pixel off; zero flow is not an
identity warp. Nearest sampling can hide the issue through quantization.

### Root cause
The existing `raven_paper_nfpa_gap_fill` mode intentionally preserved the
historical NFPA `/W` and `/H` coordinate convention. That mode did not expose a
formal RAVEN-named centered variant, so bilinear ablations could not select the
PyTorch pixel-center convention without using lower-level diagnostic modes.

### Affected files
- `raven_repro/raven/warp.py:nfpa_warp_single_latent`
- `raven_repro/raven/warp.py:raven_paper_nfpa_gap_fill_centered_warp`
- `raven_repro/raven/warp.py:translate_latent`
- `raven_repro/raven/pipeline_raven.py:RavenPipeline.run`
- `raven_repro/raven/eval_protocol.py:normalize_formal_attack_config`
- `experiments/raven_ablation_configs/nfpa_centered_bilinear_reflection.json`

### Affected outputs
Existing `raven_paper_nfpa_gap_fill` outputs remain immutable and are still valid
for the legacy coordinate convention they recorded. They are not corrected in
place and must not be relabeled as centered. Any centered-bilinear comparison
requires a new output root generated with
`warp_mode=raven_paper_nfpa_gap_fill_centered`.

### Fix
Added `raven_paper_nfpa_gap_fill_centered` as a separate warp mode. It reuses the
same NFPA image-coordinate flow, reflection padding, inverse `grid_sample`, and
sampling-mode plumbing, but calls `nfpa_warp_single_latent(...,
pixel_center_offset=0.5)`. Debug metadata and transform hashes now record the
pixel-center offset, coordinate convention, and grid implementation version.
Formal ablation config validation now permits only the legacy and centered RAVEN
NFPA gap-fill modes.

### Reused code
The existing NFPA flow builder, `nfpa_warp_single_latent()`, direct
`latent_grid_warp()` reference helper, RavenPipeline attack body, detector,
quality, FID, and CLIP workflows were reused. No DDIM/DDPM, shift, attention,
detector, or metric protocol was changed.

### Historical bug coverage
Reviewed the 2026-07-15 NFPA gap-fill entries and the 2026-07-20 effective-flow
aligned color-transfer entry. Searched reachable `warp_mode` validation and
formal ablation config paths. The legacy `raven_paper_nfpa_gap_fill` reference
test remains unchanged to prevent accidental result drift.

### Regression prevention
New tests cover centered zero-flow identity for nearest and bilinear sampling,
centered RAVEN/NFPA equivalence to `latent_grid_warp`, bilinear effective source
flow matching the planned flow, centered formal ablation config hashing, and
legacy reference behavior preservation.

### Validation
- `python -m py_compile raven_repro/raven/warp.py raven_repro/raven/pipeline_raven.py raven_repro/raven/eval_protocol.py`: passed.
- `PYTHONPATH=raven_repro pytest -q raven_repro/tests/test_warp.py`: 41 passed, 1 warning.
- `PYTHONPATH=raven_repro pytest -q raven_repro/tests/test_formal_variant_config.py raven_repro/tests/test_formal_resume.py`: 8 passed.
- `PYTHONPATH=raven_repro pytest -q raven_repro/tests/test_effective_displacement.py raven_repro/tests/test_formal_variant_config.py raven_repro/tests/test_formal_resume.py`: 48 passed, 40 expected PSNR infinity warnings.
- `git diff --check`: passed.

### Watermark integrity
- Source data: unchanged; no dataset, clean image, or watermarked image touched.
- Attack pairing: not rerun; new mode only changes recorded warp implementation
  when explicitly selected.
- Detector: unchanged Tree-Ring complex L1 protocol.
- Threshold: unchanged original-clean and attacked-clean calibration reporting.
- Quality: unchanged use of actual-grid effective source flow.
- CLIP: unchanged prompt-image metric.
- FID: unchanged watermarked-versus-attacked staging protocol.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: pending
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Push status: pending
- Entry point: `experiments/run_raven_formal_eval.py`
- Formal output eligibility: code change validated; centered full evaluation not run.

## 2026-07-26 — Canonical data/output layout migration

All data and run outputs were reorganised into five canonical roots: `data/clean/`,
`data/tr/`, `data/gs/`, `outputs/tr/`, `outputs/gs/`. No image pixels were re-encoded
and nothing was regenerated; every moved file's SHA-256 was verified unchanged and the
TR and GS pairing audits pass at the new locations.

Earlier entries in this changelog reference pre-migration paths. They are kept as
written (they are a historical log); resolve them with the old→new prefix table in
`audit/path_migration_20260726.md`.

## 2026-07-27 — GS shared-clean V2: Gaussian Shading from the canonical TR clean latent

### Problem
The TR and GS cohorts were generated from different clean latents. TR sampled
`torch.randn(seed)`; GS (V1, `gaussian_shading_shared_uniform_v1`) drew its own
uniforms from a seeded numpy RNG and reconstructed a clean latent with
`norm.ppf(u)`. The two cohorts therefore had different clean images with the same
filenames, and no paired quality metric across methods could use one shared clean
reference.

### Root cause
`GsProvider` had no way to embed from an externally supplied clean latent: the only
official entrypoint, `_get_official_wm_latents()`, always sampled its own uniforms.

### Affected files
- `eval_bench_wm/utils/wm/gs_provider.py:get_wm_latents_from_uniforms` (new),
  `:_normalized_uniforms`, `:_normalized_clean_latent`, `:get_wm_latents` (fail
  closed in the new mode), plus `GS_SHARED_TR_CLEAN_MODE` /
  `OFFICIAL_MATH_PROTOCOL_MODES`.
- `raven_repro/raven/pairing_provenance.py`: `GS_SHARED_TR_CLEAN_PROTOCOL`,
  `GS_V2_REQUIRED_FIELDS`, `gs_fields_for_protocol`, protocol-aware
  `build_pairing_sha256` / `audit_pairing_rows`, new `audit_tr_gs_shared_clean`.
- `experiments/generate_gs_from_tr_shared_clean.py` (new generator).
- `experiments/run_raven_formal_eval.py`,
  `raven_repro/scripts/build_verification_manifest.py`: protocol-aware GS field set.
- `raven_repro/tests/test_gaussian_shading_shared_tr_clean.py` (new, 53 tests).

### Affected outputs
None invalidated. The V1 GS cohort keeps its own protocol name, metadata and
pairing hashes and is never relabelled. TR clean, TR watermarked and TR metadata
were not read-modified at all (verified: 0 files under `data/tr/` or `data/clean/`
modified during this work).

### Fix
New protocol `gaussian_shading_shared_tr_clean_v2`:
rebuild the TR base latent from `base_latent_seed`, verify its tensor SHA against
both `base_latent_sha256` and `clean_base_latent_sha256`, verify the existing TR
clean image against `clean_sha256`, derive `u = norm.cdf(float64(base))`, and embed
with the unchanged official quantile partition `norm.ppf((u + b) / 2)`. The GS row
points at the TR clean path and SHA. Only the GS watermarked image is generated.

### Reused code
`tensor_sha256`, `sha256_path`, `canonical_json_sha256`, `build_pairing_sha256`,
`audit_pairing_rows`, `configure_gpu`/`setup_run_logging`, `pipe_utils`, and the
existing official payload/cipher/threshold/detector helpers on `GsProvider`.

### Historical bug coverage
The 2026-07-25/26 GS entries were reviewed. The V1 re-derivation and shard-config
work is untouched: `GS_REQUIRED_FIELDS` keeps its exact field set and order, and
`migrate_gs_detection_metadata.py` still targets V1 only.

### Regression prevention
- `zT_clean_torch` is the supplied tensor's own storage, asserted via `data_ptr()`,
  so the shared-latent SHA cannot drift through a float round trip.
- `get_wm_latents()` raises in the shared-clean mode: the mode has no RNG at all,
  and a test monkeypatches `np.random.default_rng`/`uniform` to prove it.
- Uniforms must be float64 and strictly inside (0, 1); never clamped or resampled.
- V2 rows must satisfy `tr_base_latent_sha256 == base_latent_sha256`,
  `tr_clean_sha256 == clean_sha256`, `tr_clean_path == clean_path`, and
  `shared_clean_sample_sha256 == base_latent_sha256`.
- `audit_tr_gs_shared_clean` re-checks those against the TR rows themselves, so a
  hand-edited GS row cannot self-certify.
- The generator fails closed if the pipeline dtype is not float32, if the latent
  shape differs from the TR cohort, or if its `generation_config_sha256` differs
  from the TR cohort's.

### Validation
- `python3 -m pytest -q` in `raven_repro/`: 352 passed (299 before + 53 new).
- N=1 GPU gate in `/tmp` (deleted on success), run_id=0:
  TR base latent SHA == GS base latent SHA == `bea48052…825a`;
  TR clean SHA == GS clean SHA == `c60db047…bebac`;
  `clean_path` identical (`data/clean/diffusiondb/000000.png`);
  before-attack `bit_accuracy = 1.0`, detected under
  `official_beta_tail_tau_onebit = 0.6484375` with `>=`;
  cross-method shared-clean audit passed against all 1001 TR rows.
- Resume gate: second run wrote 0 rows, verified and skipped 1, left the image
  byte-identical and the metadata at 1 row.

### Watermark integrity
- Source data: TR clean/watermarked/metadata unchanged and byte-verified.
- Clean/watermarked pairing: TR and GS now share one clean image and one
  pre-watermark latent per run_id.
- Base-latent uniqueness: one distinct latent per run_id (audited).
- Watermark target: one GS target per run, as V1.
- Attack pairing: not rerun; no attack artifacts produced.
- Detector: unchanged official GS bit-accuracy detector.
- Threshold: official beta-tail `tau_onebit`; this is NOT a 1%-FPR calibrated
  threshold and the detection rate must not be reported as TPR@1%FPR.
- Empirical clean FPR: not measured (no clean-negative cohort evaluated).
- Quality/CLIP/FID: unchanged; not run.
- Outputs requiring regeneration: none. The full 1001-sample V2 cohort has not
  been generated yet — only the N=1 gate.

### Git provenance
- Repository: `kellen931214/RAVEN`
- Branch: `agent/cleanup-quality-decomposition`
- Commit: see below
- Remote branch: `origin/agent/cleanup-quality-decomposition`
- Entry point: `experiments/generate_gs_from_tr_shared_clean.py`
- Formal output eligibility: code validated; N=1 gate only, full cohort not run.

## 2026-07-27 — GS V1 data and migration tool removed

### Problem
After GS moved to the shared-clean protocol, the V1 cohorts and their attack
outputs described a protocol the project no longer uses, and the V1-only metadata
migration tool had no remaining input.

### Root cause
Not a bug — a deliberate, user-authorised protocol change.

### Affected files
- Deleted: `experiments/migrate_gs_detection_metadata.py` and its 19 tests in
  `raven_repro/tests/test_gaussian_shading_official.py`.
- `raven_repro/scripts/build_formal_source_manifest.py`: dropped from `CORE_FILES`.
- `raven_repro/tests/test_canonical_layout.py`: the two tests that asserted the
  deleted V1 cohort existed on disk now assert the shared-clean invariant instead
  (GS has no clean directory of its own; `data/clean` holds no `gs_*` cohort).
- `raven_repro/tests/test_gaussian_shading_shared_tr_clean.py`: the real-V1-cohort
  audit test was replaced with a synthetic V1 cohort so V1 protocol coverage
  survives the data deletion.

### Affected outputs
Deleted, irreversibly, on explicit user instruction:
`data/gs/*` (445 MB, 4 cohorts), `data/clean/gs_*` (440 MB),
`outputs/gs/*` (2.7 GB, including the validated V1 1001 attack/eval results).
`data/gs/` and `outputs/gs/` remain as empty canonical roots.

**Not touched, verified:** `data/clean/diffusiondb/` (1004 files before and after)
and `data/tr/` (1020 files before and after); `data/tr/diffusiondb/TR/metadata.csv`
SHA `ade3792423850bc519ac8383ac332e4c013cf0c72f258f11693b9381a34d04bc` unchanged.

### Fix
V1 *code* is retained deliberately — `GS_PAIRING_PROTOCOL`, `GS_REQUIRED_FIELDS`
and the `official_compatible` seeded-sampling path are still reachable from the
standalone reproduction runners, so only the V1-specific migration tool was
removed.

### Regression prevention
`audit_pairing_rows` still accepts and validates V1 rows; a synthetic V1 cohort
test asserts the V1 field set, protocol name and sampling-seed uniqueness so the
retained V1 code cannot silently rot now that no V1 data exists.

### Validation
`python3 -m pytest -q` in `raven_repro/`: 333 passed (352 − 19 removed).

### Watermark integrity
- Source data: TR clean/watermarked/metadata verified unchanged.
- Outputs requiring regeneration: the GS cohort itself — regenerated with
  `experiments/generate_gs_from_tr_shared_clean.py` into
  `data/gs/diffusiondb_shared_tr/GS/`.
- Previous V1 GS results: no longer usable and no longer present.

### Git provenance
- Branch: `agent/cleanup-quality-decomposition`
- Entry point: `experiments/generate_gs_from_tr_shared_clean.py`
- Formal output eligibility: V1 results withdrawn; V2 cohort generation in progress.
