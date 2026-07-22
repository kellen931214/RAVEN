# Human-readable RAVEN output aliases

Created UTC: 2026-07-22

The navigation root is:

`outputs/RAVEN_EVALS_READABLE/`

Only symbolic links were added. No active or historical output directory was
renamed, moved, deleted, or modified because running processes retain absolute
paths to the timestamped directories.

## Main 1001-sample suite

`ACTIVE_main_1001_four_attacks_five_evaluations` maps to
`outputs/raven_formal_protocol_rerun/diffusiondb/TR/run_20260722T020154Z_3090`.

The suite contains four attack generations and five evaluated outputs:

1. DDIM, no shift, no color transfer.
2. DDPM, nearest/reflection shift, aligned color transfer.
3. DDIM, bilinear/reflection shift, aligned color transfer.
4. DDIM, nearest/reflection shift, aligned color transfer.
5. The same DDIM nearest/reflection pre-color attack evaluated without color transfer.

The fourth and fifth outputs intentionally reuse one attack generation; they
are different postprocessing/evaluation variants rather than two attacks.

## Paper-exact comparison

The readable root records one passed smoke, failed/partial attempts, and two
active full1001 paper-exact processes as separate aliases. Both active jobs
read the same immutable pre-color cohort but write separate output roots. They
were not stopped or modified by this naming change.

## Historical roots

Historical aliases include the observed terminal/stale state in their names:
GPU-architecture failure, idle-GPU wait without output, detector smoke failure,
stale GPU8 smoke, or explicitly stopped partial smoke. Timestamped directories
remain the authoritative provenance locations; aliases are navigation aids.
