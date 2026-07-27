# RAVEN Attack Artifact Policy

## Purpose

Use this skill whenever launching, modifying, resuming, or evaluating a RAVEN
watermark-removal attack.

This skill controls the per-sample attack artifact layout. It does not change
the attack mathematics, evaluation definitions, detector definitions or formal
validation policy.

## Required per-sample layout

Every newly generated attacked sample must use:

attack_cache/<attack-config-hash>/<run-id>/<role>/
├── output/
│   ├── view_guided_output.png
│   ├── final_color_corrected.png
│   └── debug_info.json
└── record.json

Definitions:

- view_guided_output.png:
  the attacked image before aligned color transfer; this is the canonical
  no-color/pre-color output.

- final_color_corrected.png:
  the final attacked image after the configured color-transfer stage.

- debug_info.json:
  deterministic attack runtime and transform provenance.

- record.json:
  input references, hashes, attack configuration, output references and hashes.

## Forbidden redundant outputs

Do not create:

- input.png
- final.png aliases
- copied clean images
- copied watermarked source images
- files ending in _1, copy, backup, latest or similar duplicate names
- duplicate encodings of the same source or attacked image

Source images must remain in data/ and be referenced from record.json using
their exact path and SHA-256.

Do not duplicate source images inside an attack output directory.

## Method rules

For GS:

- Only attack the watermarked role.
- Never run attack-clean.
- Never create an attacked-clean cache.
- attacked_clean_count must remain 0.
- Do not save input.png.

For TR:

- Attack-watermarked uses the same four-artifact layout.
- Attack-clean may exist only when required by the full formal TR/NFPA
  recalibration protocol.
- Even when attack-clean is required, do not create redundant input.png copies
  in newly implemented workflows.
- Do not disable attacked-clean merely to save storage when the requested
  experiment is the full TR protocol.

This skill is a policy for newly created or modified workflows. Do not modify
existing validated historical results merely to make their file layout match
this policy.

## Required behavior

Before accepting a completed sample, verify that:

- view_guided_output.png exists
- final_color_corrected.png exists
- debug_info.json exists
- record.json exists
- record.json points to the exact three output files
- the recorded SHA values match the actual files
- no forbidden redundant output was produced

Do not silently accept legacy aliases in a new run.
