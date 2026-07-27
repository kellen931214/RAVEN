# RAVEN Experiment Table Recording Policy

## Purpose

Use this skill after a watermark generation, attack or evaluation experiment
finishes.

Experiment results must be written by a deterministic program.

Do not manually copy metric values into Markdown.

Do not ask a language model to infer metric values from logs.

Do not label method-specific detector results using another detector's metric
definition.

## Authoritative updater

Always use:

python experiments/update_experiment_table.py \
  --run-root <completed-run-root>

## Table location

The table is written to:

outputs/<method>/<dataset>/_table/experiment_results.md

The updater resolves this from the run's own structured provenance (its recorded
method and dataset), not from the run-root path string. Passing --table
overrides it; do not override it merely to collect unrelated runs together.

One table per method and one per dataset:

- Do not merge two methods into one table.
- Do not merge two datasets into one table.
- Do not keep a second copy of the same table at another path.
- `_table` is reserved at that level and never holds run artifacts.

All runs of the same method and dataset upsert into that one table, so two
sampling-mode variants of one cohort appear as two rows sharing a table, each
row named by its experiment-parameter slug. See
`.agents/skills/raven-experiment-naming/SKILL.md` for the slug rules.

Add the table directory to .gitignore if runtime result records would otherwise
make the formal source worktree dirty during a run.

## The table is written by a program, never by hand

After an experiment finishes, the row must be produced by running the updater.

- Do not hand-write, hand-edit or hand-transcribe a metric value into the table.
- Do not copy a number out of a log, a console line or a chat message.
- Do not fill a cell from memory or from a previous run's value.
- Do not add a row for a run that has not finished.

If the updater cannot produce a value, the cell stays as the missing marker. A
missing metric is reported as missing; it is never supplied by a human.

## No autonomous monitoring

Do not monitor a detached process using:

- polling loops
- watch
- repeated ps
- repeated nvidia-smi
- sleep loops
- repeated log tailing
- conversational background monitoring

The updater must run only:

1. after a synchronous experiment command exits successfully
2. as a completion hook already attached to the experiment command
3. when the user explicitly asks to record an already completed run

For a synchronous run:

python <experiment-runner> ... &&
python experiments/update_experiment_table.py --run-root "$OUT"

For a detached run:

nohup bash -lc '
  set -euo pipefail
  python <experiment-runner> ...
  python experiments/update_experiment_table.py --run-root "$OUT"
' > "$OUT/launcher.log" 2>&1 &

This is a completion hook, not monitoring.

## Required table columns

The Markdown table must contain:

| Finished UTC | Method | Dataset | Experiment | Stage | N | Attack | Detector Metric | Score Direction | Threshold Type | Threshold | Nominal FPR | Before Score | After Score | Before Detection Rate | After Detection Rate | Attack Success | Empirical Clean FPR | ROC-AUC | FID | CLIP | PSNR | SSIM | Status | Run Root |

Definitions:

- Stage:
  watermark_generation, attack_only, evaluation or formal_evaluation.

- Detector Metric:
  the actual raw detector score, such as bit_accuracy or l1_complex.

- Score Direction:
  higher_is_watermarked or lower_is_watermarked.

- Threshold Type:
  the actual threshold family used by the detector.

- Before Score:
  aggregate raw detector score before attack.

- After Score:
  aggregate raw detector score after attack.

- Before Detection Rate:
  fraction of original watermarked images classified as watermarked using the
  recorded threshold.

- After Detection Rate:
  fraction of attacked watermarked images classified as watermarked using the
  recorded threshold.

- Empirical Clean FPR:
  actual false-positive rate measured from the clean-negative cohort when such
  calibration was performed.

## Detector-specific reporting rules

### Gaussian Shading

For GS:

- Detector Metric must be bit_accuracy.
- Score Direction must be higher_is_watermarked.
- Before Score must be the structured aggregate of before-attack bit accuracy.
- After Score must be the structured aggregate of attacked bit accuracy.
- Threshold Type must report the actual official threshold family, such as:
  - official_beta_tail_tau_onebit
  - official_beta_tail_tau_bits
- Threshold must be the actual recorded GS threshold.
- Nominal FPR may contain the configured theoretical GS FPR when explicitly
  stored by the detector.
- Before Detection Rate and After Detection Rate must be computed using the
  stored threshold and stored comparison operator.
- Empirical Clean FPR must be — unless a clean-negative cohort was actually
  evaluated.
- Do not label a GS detection rate as TPR@1%FPR.
- Do not convert the official GS beta-tail threshold into an empirical
  clean-negative FPR threshold.
- Do not report bit accuracy under a TPR column.

An official GS threshold and an empirical 1%-FPR threshold are different
protocols and must never be presented as the same quantity.

### Tree-Ring

For TR:

- Detector Metric must report the actual detector score, such as l1_complex.
- Score Direction must match the detector implementation.
- Threshold Type may be empirical_clean_1pct_fpr only when the threshold was
  actually calibrated from the corresponding clean-negative cohort.
- Empirical Clean FPR must contain the measured clean FPR when available.
- Before Detection Rate and After Detection Rate may correspond to TPR values,
  but the generic table column names must remain Detection Rate so different
  watermark methods can share one table.
- Do not invent TPR@1%FPR when clean calibration was not performed.

### Every other watermark method

The GS bit-accuracy family and the TR TPR family are examples, not the complete
set. Every additional watermark method must follow the same discipline:

- Report the method's own raw detector metric name.
- Report the method's own score direction.
- Report the method's own threshold family under Threshold Type, using that
  method's real name for it.
- Report detection rates computed with that method's own stored threshold and
  stored comparison operator.
- Report Empirical Clean FPR only when that method's clean-negative cohort was
  actually evaluated.
- Never reuse another method's metric name, threshold family, detection-rate
  definition or attack-success definition.
- Never coerce a new method into the GS schema or into the TR schema.

A method without a registered method-specific extractor must fail closed. Add an
extractor for it; do not guess its detector fields.

## General metric rules

- Use actual structured values only.
- Use — when a metric is absent or not applicable.
- Do not replace missing values with zero.
- Do not replace failed computations with zero.
- Do not write NaN or Inf.
- Preserve sufficient precision from the authoritative result.
- Do not automatically multiply values by 100.
- Only treat a number as a percentage when the source field explicitly records
  percentage units.
- Run Root must be repository-relative when possible.
- Never call an incomplete result validated.

## Attack Success

Use the attack-success value and definition from the authoritative structured
aggregate.

Do not independently redefine or recompute attack success when the aggregate
already provides it.

When no authoritative attack-success value exists, write —.

Do not assume:

attack_success = 1 - after_detection_rate

unless that exact definition is explicitly used by the stored experiment
protocol.

## Generation-only and attack-only runs

A completed generation-only or attack-only experiment may be recorded even when
detector and quality metrics are unavailable.

In that case:

- Stage must describe the actual completed stage.
- unavailable metric columns must contain —
- Status must not be validated_formal_result
- an explicit structured completion record must exist

Do not treat missing evaluation metrics as an error for a deliberately
generation-only experiment.

For a full evaluation or formal evaluation, required aggregate and validation
files must be present.

## Idempotent update

The updater must upsert, not blindly append duplicate rows.

Use this identity:

method + dataset + experiment-slug + run-key + stage

Running the updater twice for the same completed run must update the existing row
rather than create a second row.

## Authoritative files

Read structured results in this priority:

1. VALIDATED.json
2. formal_aggregate.json
3. method-specific verification aggregate JSON
4. structured per-sample verification JSON or JSONL
5. quality aggregate JSON
6. FID aggregate JSON
7. CLIP aggregate JSON
8. run_config.json
9. explicit structured generation or attack completion record

Do not parse human-formatted console output when structured files exist.

Do not parse launcher.log, run.log or terminal output for metric values.

If structured files conflict, fail closed and report the conflicting paths and
fields.

## Completion behavior

At the end of every completed experiment workflow:

- run the updater once
- print the table path
- print whether the row was inserted or updated
- print the experiment identity
- do not print the complete Markdown table unless requested
