# Legacy Evaluation Removal

Tracked legacy entrypoints were deleted in commit `3330e67` before this audit:

- `experiments/run_raven_eval_from_watermarked.py`
- `experiments/run_raven_eval_mscoco.sh`
- `experiments/run_raven_eval_mscoco_smoke.sh`
- `experiments/wait_and_run_raven_eval_mscoco.py`
- `experiments/build_raven_eval_table.py`

The all-datasets waiter was absent and has now been recreated as a formal-only waiter.
Post-change source search contains no old executable references. An ignored orphan
`experiments/__pycache__/run_raven_eval_from_watermarked.cpython-310.pyc` remains because
the required 30-sample gate has not passed and deletion was explicitly gated on all
validation. It is not importable by any current source entrypoint.

Protocol-invalid derived outputs were moved, not deleted, to
`outputs/legacy_invalid/20260718T072817Z/`; see its `DO_NOT_USE.md`. Clean/watermarked
datasets, inputs, and root logs were preserved. These outputs cannot be resumed or merged
by the formal runner because its cache layout, immutable snapshots, config hashes, Git
SHA, input/output/debug SHA, and debug assertions are all required.

Preserved detector providers and formal helpers include all GS/TR/RID/HSTR/HSQR provider
files plus `raven_p1_full.py`, `raven_nfpa_tr_eval.py`,
`extract_verification_scores.py`, and `evaluate_verification.py`. Research scripts are
labeled `ABLATION ONLY - NOT A FORMAL EVALUATION ENTRYPOINT`.

## 2026-07-20 aligned color migration

- `paper_exact_two_stage`, `paper_exact_two_stage_aligned_blend`, and
  `direct_stats` are no longer executable color-transfer modes.
- Historical unaligned formal output
  `outputs/raven_formal_eval/diffusiondb/TR/1001_20260718T090947Z` is retained
  as legacy evidence and must not be merged with aligned results.
- Removed obsolete no-color/decomposition/alignment launchers. The sole
  reprocessing entrypoint is `experiments/run_raven_aligned_color_eval.py`.
