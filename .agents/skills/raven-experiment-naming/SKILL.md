# RAVEN Experiment Naming Policy

## Purpose

Use this skill whenever creating or selecting a data-generation, attack,
evaluation, ablation, gate or formal experiment output directory.

Directory names must describe the experiment actually being run.

## Canonical formal output layout

Use:

outputs/<method>/<dataset>/<experiment-slug>/<run-key>/

Examples:

outputs/gs/diffusiondb/shared_clean_gs_from_tr/<run-key>/

outputs/gs/diffusiondb/ddim_inverse_ddpm_reflection/<run-key>/

outputs/tr/diffusiondb/ddim_nearest_reflection_aligned_color/<run-key>/

outputs/tr/diffusiondb/ddim_nearest_reflection_no_color/<run-key>/

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
