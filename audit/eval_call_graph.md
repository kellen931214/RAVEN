# Evaluation Call Graph Audit

## Removed Historical Legacy Chain

Git history at the pre-removal revision proves the chain:

```text
experiments/wait_and_run_raven_eval_all_datasets.py
  -> experiments/run_raven_eval_from_watermarked.py
  -> experiments/build_raven_eval_table.py
```

The runner omitted protocol-sensitive `RavenPipeline.run()` arguments and therefore used
the pipeline's integer warp default. It exposed uncalibrated `paper_1pct`, resumed from a
CSV/output existence check, cropped quality by planned flow, and reused FID staging files.
Its detector/table outputs are legacy only. Commit `3330e67` removed the tracked files.

## Current Entries

| Entry | Next layer | Attack config | Detector / threshold | Quality / FID / CLIP | Resume / output | Class |
| --- | --- | --- | --- | --- | --- | --- |
| `experiments/run_raven_formal_eval.py` | `RavenPipeline`; strict manifest; score extractor/evaluator; TR NFPA helper | centralized pinned model revision, DDIM 50, strength .15, guidance 2.5, image-pixel NFPA-gap-fill, nearest/reflection, VGA, paper two-stage color | per-method raw scores; same-cohort clean calibration; TR full/rounded complex-L1 with original and attacked-clean thresholds | watermarked vs final post-color effective-flow overlap; immutable per-method watermarked-vs-attacked FID; centralized prompt-image bigG CLIP | immutable snapshots; per-run config/hash/SHA cache; `<output-root>` | **formal** |
| `experiments/wait_and_run_raven_eval_all_datasets.py` | only the formal runner, stage by stage | hashes the centralized formal attack config into cohort lock | final metrics only after exact expected snapshot count | invokes all formal stages | dataset/method/config lock; completed-stage markers | **formal waiter** |
| `experiments/build_raven_formal_eval_table.py` | validated aggregate files only | reports hash | explicit calibrated columns; GS bit accuracy separate | explicit reference/provenance columns | refuses overwrite; requires `VALIDATED.json` | **formal report** |
| `raven_repro/scripts/raven_nfpa_tr_eval.py score-formal` | detector provider/inversion | consumes matched formal attacked clean/watermarked records | strict `<`; full precision and rounded2 separately | none | fresh output only | **formal helper** |
| `raven_repro/scripts/extract_verification_scores.py` + `evaluate_verification.py` | provider/inversion then pure aggregation | consumes strict formal manifest | one provider config; actual clean FPR; legacy fixed threshold separately named | none | fsynced per-row records | **formal helper** |
| `raven_p1_full.py`, color/quality experiments, `run_raven.py`, `attack_folder.py`, historical chain | independent research code | varies by experiment | varies | varies | historical roots | **ablation/diagnostic only** |

Formal detector config hash covers method, target FPR, and strict manifest score source.
Quality config hash covers the primary reference, effective-overlap rule, FID definition,
and CLIP provenance. Combined-scheme FID is not produced; per-method FID is never averaged
and presented as a paper combined FID.
