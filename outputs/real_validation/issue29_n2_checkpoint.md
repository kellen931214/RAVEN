# Issue #29 — real n2 validation checkpoint

Branch: `feature/issue29-real-validation`
HEAD: `5fb3dbb` (merge: sync latest feature/worktree before TR parity fix push)

All runs: `python3 experiments/eval.py --output-dir outputs/real_validation/<method>/diffusiondb/n2 --device cuda --stages detector`

Exit code 2 for every method: 4 `failed_missing_image` rows (attacked cohorts
absent — no attacked images exist for any real cohort).  This is expected and
must not be interpreted as a detector failure.  The 4 scored rows per method
are clean + watermarked original-cohort samples.

## Pass

| Method | scored/failed | raw score | canonical | direction |
|---|---|---|---|---|
| GS | 4/4 | bit_accuracy | = raw | higher_is_watermarked |
| GM | 4/4 | gm_raw_bit_accuracy | = raw | — |
| RID | 4/4 | rid_neg_channel_min_complex_l1 | = -raw | higher_is_watermarked |
| HSTR | 4/4 | hstr_score=-min(ch0,ch3) | = -raw | higher_is_watermarked |

All four methods: 2/2 watermarked bit_accuracy = 1.0; clean scores below
thresholds.  Provider/target/mask identity verified.  Bundle provenance
verified (GM/RID/HSTR).  GS threshold = official_beta_tail_tau_onebit
(0.6484375, source `bsmhmmlf/Gaussian-Shading watermark.py@09c678f`).

## Blocked

**T2S** — `DetectorScoringError: 'dict' object has no attribute 'dim'`.
`t2s_inversion_mod.invert_image()` return type incompatible with
`T2SProvider.accuracies_for_state()` tensor expectation.  No scored rows.
Production return-contract defect — not an evaluator-orchestrator issue.
Fix proposal documented in PR description; no production code changed.

**HSQR** — `DetectorStateValidationError: detector/source mask SHA mismatch`.
Source CSV records `hsqr_mask_sha256=0416a022...`; detector computes
`83f2e3e8...` from bundle.  The real generation metadata disagrees with the
HSQR bundle artifact.  No scored rows.

## Deferred

**TR** — pending issue #18 smoke validation and issue #28 cherry-pick.
