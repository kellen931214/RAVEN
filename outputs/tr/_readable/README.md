# RAVEN evaluation output guide

This directory contains symlink aliases only. Original output directories are
not renamed, moved, deleted, or modified because active processes use their
absolute paths.

## Current formal protocol suite

`ACTIVE_main_1001_four_attacks_five_evaluations` points to the timestamped
`run_20260722T020154Z_3090` directory.

It contains four attack generations and five evaluated output variants:

1. `ddim_no_shift_no_color`: DDIM, zero shift, evaluated without color transfer. Completed 1001.
2. `ddpm_nearest_shift_aligned_color`: DDPM, nearest/reflection shift, aligned color. Completed 1001.
3. `ddim_bilinear_shift_aligned_color`: DDIM, bilinear/reflection shift, aligned color. Completed 1001.
4. `ddim_nearest_shift_aligned_and_no_color`: one DDIM nearest/reflection shifted attack cohort, evaluated both with aligned color and without color. The aligned formal attack is validated; the full no-color detector evaluation is currently active.

The existing `easy_view/` directory inside that run provides short links to
the four job outputs, logs, and status JSON files.

## Paper-exact color comparison

`paper_exact_color_comparison/` separates the passed smoke, failed/partial
attempts, and the two currently active full jobs. The two entries named
`ACTIVE_full1001_paper_exact_duplicate_A` and `_B` are separate processes that
read the same immutable pre-color cohort and write different output roots.
Neither process was stopped or modified while creating this guide.

## Historical protocol runs

`formal_protocol_history/` classifies old timestamped roots by observed state:

- `FAILED_gpu6_unsupported_*`: stopped because the installed PyTorch build did not support that GPU architecture.
- `INCOMPLETE_waited_for_gpu_no_jobs_*`: dispatcher state only; no experiment output was produced.
- `FAILED_smoke_detector_*`: early 10-sample detector/manifest workflow failed.
- `STALE_ddim_noshift_smoke_gpu8_*`: old state says running, but no matching live process exists.
- `STOPPED_partial_ddim_noshift_smoke_*`: partial smoke with an explicit stop-request record.

Timestamped source directories remain authoritative provenance. These aliases
are navigation aids and must not be used to merge or resume incompatible runs.
