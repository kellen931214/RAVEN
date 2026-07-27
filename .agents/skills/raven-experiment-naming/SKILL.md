# RAVEN Experiment Naming Policy

## Purpose

Use this skill whenever creating or selecting a data-generation, attack,
evaluation, ablation, gate or formal experiment output directory.

Directory names must describe the experiment actually being run.

## Canonical formal output layout

Use:

outputs/<method>/<dataset>/<experiment-parameter-slug>/

optionally with a content-addressed run key below it:

outputs/<method>/<dataset>/<experiment-parameter-slug>/<run-key>/

The experiment-parameter-slug level is mandatory. The run-key level is optional:
include it when several immutable configurations share one slug, and omit it when
the slug already identifies the experiment completely.

Examples:

outputs/gs/diffusiondb/ddim_inverse_ddpm_forward_nearest/

outputs/gs/diffusiondb/ddim_inverse_ddpm_forward_bilinear/

outputs/gs/diffusiondb/shared_clean_gs_from_tr/<run-key>/

outputs/tr/diffusiondb/ddim_nearest_reflection_aligned_color/<run-key>/

outputs/tr/diffusiondb/ddim_nearest_reflection_no_color/<run-key>/

## Cohort data layout

Cohort layout is a per-method fact, not a global rule. It is declared by
FLAT_COHORT_METHODS in raven_repro/raven/eval_protocol.py.

Tree-Ring uses the flat layout (since 2026-07-27):

data/tr/<dataset>/metadata.csv

data/tr/<dataset>/<run_id>/watermarked.png

Every other method keeps the nested per-method layout:

data/<method>/<dataset>/<METHOD>/metadata.csv

data/<method>/<dataset>/<METHOD>/<run_id>/watermarked.png

Do not change one method's layout because another method's layout changed. A
method is added to FLAT_COHORT_METHODS only after its cohort on disk has
actually been moved.

Resolve these paths with cohort_dir(), source_metadata_path() and
watermarked_image_path(). Do not hard-code a cohort path string.

## Results table location

The summary table for one method and one dataset lives at:

outputs/<method>/<dataset>/_table/experiment_results.md

`_table` is a reserved directory name at that level. Never use it as an
experiment slug, and never place run artifacts inside it.

One table per method and dataset. Do not merge methods or datasets into one
table, and do not keep a second copy of the same table elsewhere.

## The folder name must state the experiment parameters

The slug must name the primary experiment parameters directly, so the directory is
readable without opening any config. State, in this order where applicable:

1. inversion scheduler (for example `ddim_inverse`)
2. reconstruction / reverse-sampling scheduler (for example `ddpm_forward`)
3. latent sampling mode (`nearest` / `bilinear`)
4. warp / padding mode when it is the variable under test
5. color-transfer mode when it is the variable under test
6. the ablation variable when the run exists to isolate one

Two runs that differ in any of these parameters must have different slugs, and
the slug must show which parameter differs. `ddim_inverse_ddpm_forward_nearest`
and `ddim_inverse_ddpm_forward_bilinear` are correct: everything else about the
two runs is identical and the name says exactly what changed.

Do not encode a parameter in the slug that the run does not actually use. The
slug is provenance, not decoration: it must agree with the recorded attack
config.

## Method component

Use lowercase method names:

- tr
- gs

Do not place TR results under outputs/gs.

Do not place GS results under outputs/tr.

## Experiment slug

The experiment slug must identify the experiment actually being run.

Resolve it using this priority:

1. explicit --experiment-name
2. explicit --variant
3. experiment_name or variant stored in the config
4. attack-config filename stem
5. deterministic slug derived from meaningful experiment settings

Meaningful settings may include:

- watermark method
- shared-clean source
- inversion scheduler
- reconstruction scheduler
- shift or no-shift
- latent sampling mode
- padding mode
- color or no-color
- aligned-color mode
- attack family
- ablation variable
- detector variant
- generation protocol

Do not use generic experiment slugs such as:

- formal
- experiment
- test
- run
- latest
- final
- new
- temp

unless the word is part of a more specific semantic experiment name.

For example:

- shared_clean_gs_from_tr
- ddim_inverse_ddpm_nearest_reflection_aligned_color
- ddim_inverse_ddpm_nearest_reflection_no_color
- zero_shift_ablation
- bilinear_sampling_ablation

## Run key

Keep the existing content-addressed run-key policy.

The run key must be derived from immutable configuration and provenance such as:

- source manifest SHA
- attack config SHA

Do not use timestamps as formal run keys.

Identical experiment configurations must resolve to the same output root and
continue using --resume.

Different semantic experiments must not share the same experiment slug, even
when they use the same source cohort.

## Temporary runs

Gate, smoke and dry-run outputs must use the existing scratch_run_root() under
/tmp.

Do not write temporary runs into outputs/.

Delete successful scratch runs.

Preserve and report failed scratch paths.
