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

GS, GM, T2S, RID, HSTR exit 2 only because attacked cohorts (`attacked_clean`,
`attacked_watermarked`) are absent — every available detector row scored successfully
(0 detector failures).  T2S has 2 attacked-missing (no clean cohort).

HSQR exits 2 with compound failures: 4 attacked-missing rows AND 4 independent
`DetectorStateValidationError` rows from source-metadata / bundle mask SHA mismatch.
The exit code alone does not distinguish these; the `row_status_counts` and
`failure_cause_counts` in the aggregate output do.

## Blocked

**HSQR** — `DetectorStateValidationError: detector/source mask SHA mismatch`.

Source CSV records `hsqr_mask_sha256=0416a022...`; detector computes
`83f2e3e8...` from the bundle's `selected_pattern.pt`.  Classification:
**source metadata stale** — the generation pipeline wrote a mask hash that
the current detector code cannot reproduce from the persisted bundle.

Provenance evidence:

- CSV SHA256: `446ca4b6...c1a81a`
- Bundle dir: `/workspace/RAVEN/data/hsqr/diffusiondb_shared_tr/bundle/`
- Manifest SHA256: `e55f41d1...2fb46`, schema `sfw_bundle_v1`
- Manifest has `selected_pattern_sha256=4fb8b70e...` but NO `mask_sha256`
  or `mask_file_sha256` field
- Bundle contents: `manifest.json`, `selected_pattern.pt` — NO `watermark_mask.pt`
  (contrast RID which has `watermark_mask.pt`; HSTR also lacks it but
  derives a matching hash from its pattern)
- CSV `hsqr_mask_sha256` = `0416a022...` (constant across all 1001 rows)
- CSV `watermark_mask_sha256` = `0416a022...` (same value)
- CSV `hsqr_selected_pattern_sha256` = `4fb8b70e...` — matches manifest
- The value `0416a022...` does not appear in the bundle manifest or any
  artifact file on disk
- Generation was at git `8e9eb5ce` on branch `issue-6-shared-clean`, dirty=False

Root cause hypothesis: HSQR generation computed a mask tensor in memory,
hashed it for the CSV, but the bundle was saved without a dedicated mask
artifact.  The detector re-derives a mask from `selected_pattern.pt` but
arrives at `83f2e3e8...` — different from the generation-time value.  Either
the mask derivation changed between generation and detection, or the
generation ran a different path.

No metadata, bundle, or artifact modified.

## Pass — TR (after issue #28 integration)

Integration commit: `943c380` (cherry-picked from issue #28 `9cb7871`).
4/4 scored: 2 original_clean + 2 original_watermarked, 4 attacked-missing.
Exit 2 (attacked cohorts absent).

Score protocol: raw = complex_l1_mean, raw direction lower_is_watermarked,
canonical = -raw, direction higher_is_watermarked, comparison >=.
Target/mask/provider config all verified.  Detector computes mask
`6636fc4a...` / target `087e4198...` from metadata row; matches real
watermark config.

Exact deterministic parity with issue #28 n2 reference — zero diff on all
4 canonical scores.  No legacy extract_verification_scores.py import.

## Fix record

T2S was initially blocked by `DetectorScoringError: 'dict' object has no attribute 'dim'`.
Root cause: adapter `t2s_detector.py` wrapped the inverted latent tensor `zT` in `{"zT_torch": zT}`
before passing to `T2SProvider.accuracies_for_state(state, zT)`, which expects a bare tensor.
`t2s_inversion_mod.invert_image()` properly returns `torch.Tensor`; the defect was adapter-side.
Fixed in `1345680` by passing `zT` directly with an `isinstance(zT, torch.Tensor)` guard.
No modification to inversion module or scorer contract.
