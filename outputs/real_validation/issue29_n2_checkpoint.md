# Issue #29 — real n2 validation checkpoint

Checkpoint commit: `58ccf89` — test(issue29): add real n2 validation checkpoint
Fix commit: `1345680` — fix(t2s): pass inverted latent tensor to state scorer

Branch: `feature/issue29-real-validation`

All runs: `python3 experiments/eval.py --output-dir outputs/real_validation/<method>/diffusiondb/n2 --device cuda --stages detector`

## Pass — scored rows

| Method | scored | failed (missing image) | failed (detector) | raw score | direction |
|---|---|---|---|---|---|
| GS | 4 (2 wm, 2 clean) | 4 attacked | 0 | bit_accuracy | higher_is_watermarked |
| GM | 4 (2 wm, 2 clean) | 4 attacked | 0 | gm_raw_bit_accuracy | higher_is_watermarked |
| T2S | 2 (2 wm) | 2 attacked | 0 | t2s_score_true_key | higher_is_watermarked |
| RID | 4 (2 wm, 2 clean) | 4 attacked | 0 | rid_neg_channel_min_complex_l1 (canonical=-raw) | higher_is_watermarked |
| HSTR | 4 (2 wm, 2 clean) | 4 attacked | 0 | hstr_score=-min(ch0,ch3) (canonical=-raw) | higher_is_watermarked |

### Per-method detail

**GS**: 2/2 wm bit_accuracy=1.0, clean 0.53/0.46 below official_beta_tail_tau_onebit=0.648. Provider/target/mask verified.

**GM**: 2/2 wm gm_raw_bit_accuracy=1.0, clean 0.55/0.46. Bundle provenance verified (w1, w2, config, target, mask). No GNR/classifier used.

**T2S**: 2/2 wm detection_success=true, true_key ≫ control_key (margin 1400/1654), key/message/bit accuracy all 1.0. state_sha256 verified, watermark_id=t2s-shared-tr-clean-00000[0|1], inversion=t2s_official.

**RID**: 2/2 wm raw_l1 37/32 (canonical -37/-32), clean raw_l1 76/80 (canonical -76/-80). lower raw = watermarked, correct. Bundle verified.

**HSTR**: 2/2 wm raw_l1 25/22 (canonical -25/-22), clean raw_l1 45/48 (canonical -45/-48). lower raw = watermarked, correct. Bundle verified.

### Exit code

All methods exit 2.  Attacked cohorts (`attacked_clean`, `attacked_watermarked`) are absent
for every real cohort — these produce explicit `failed_missing_image` rows.  GS/GM/RID/HSTR
each have 4 attacked-missing and 0 detector failures.  T2S has 2 attacked-missing (no clean
cohort).  Exit 2 reflects the missing attacked images, not a scoring defect.

## Blocked

**HSQR** — `DetectorStateValidationError: detector/source mask SHA mismatch`.
Source CSV records `hsqr_mask_sha256=0416a022...`; detector computes
`83f2e3e8...` from bundle artifact.  Source metadata disagrees with the
HSQR bundle.  No scored rows.  No metadata or bundle modified.

## Deferred

**TR** — pending issue #18 smoke validation and issue #28 cherry-pick.

## Fix record

T2S was initially blocked by `DetectorScoringError: 'dict' object has no attribute 'dim'`.
Root cause: adapter `t2s_detector.py` wrapped the inverted latent tensor `zT` in `{"zT_torch": zT}`
before passing to `T2SProvider.accuracies_for_state(state, zT)`, which expects a bare tensor.
`t2s_inversion_mod.invert_image()` properly returns `torch.Tensor`; the defect was adapter-side.
Fixed in `1345680` by passing `zT` directly with an `isinstance(zT, torch.Tensor)` guard.
No modification to inversion module or scorer contract.
