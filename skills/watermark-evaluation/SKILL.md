# Watermark Evaluation Integrity Skill

## Purpose

Use this skill for watermark generation, removal attacks, detector evaluation, image-quality evaluation, and formal metric reporting.

This includes pipelines involving:

* clean images
* watermarked images
* attacked-clean images
* attacked-watermarked images
* Tree-Ring and other watermark providers
* RAVEN or other removal attacks
* CLIP, FID, PSNR, SSIM, ROC-AUC, TPR, and attack success rate

This skill extends:

`skills/experiment-integrity/SKILL.md`

Both skills must be followed.

---

## 1. Identify the Formal Watermark Pipeline

Before modifying or running an evaluation, trace:

```text
prompt or source sample
→ base latent or source image
→ clean generation
→ watermark injection
→ watermarked generation
→ clean attack
→ watermarked attack
→ detector inversion
→ detector score
→ threshold calibration
→ quality metrics
→ aggregation
```

Record the actual source file and function used at each stage.

Do not assume a script is authoritative because its name contains:

```text
formal
paper
NFPA
fixed
latest
final
```

Confirm the actual imports and call chain.

---

## 2. Search Existing Watermark Implementations

Before adding or changing code, search for existing implementations.

Example:

```bash
rg -n "get_wm_latents|wm_zT|base_latent"
rg -n "w_measurement|l1_complex|p_value|-log10"
rg -n "watermark_seed|w_seed|target_watermark"
rg -n "attacked_clean|attacked_watermarked"
rg -n "TPR|FPR|roc_auc|threshold"
rg -n "crop_overlap|inverse_warp"
rg -n "CLIP|FID|PSNR|SSIM"
```

Check:

* formal evaluators
* provider implementations
* generation scripts
* attack scripts
* legacy evaluation scripts
* ablation scripts
* result aggregation scripts

Do not implement a second detector, threshold function, or overlap function when a trusted shared implementation exists.

---

## 3. Clean and Watermarked Pairing

For paired quality evaluation, clean and watermarked samples must derive from the same per-sample base input.

Correct pattern:

```python
generator = torch.Generator(device=device).manual_seed(sample_seed)
base_latent = torch.randn(
    latent_shape,
    generator=generator,
    device=device,
)

clean_latent = base_latent.clone()
watermarked_latent = inject_watermark(base_latent.clone())
```

Required invariant:

```text
clean base-latent SHA
==
watermark pre-injection base-latent SHA
```

Different samples must use independent base latents:

```text
sample_i base-latent SHA
!=
sample_j base-latent SHA
```

It is acceptable for multiple samples to share the same watermark target or key when required by the method.

It is not acceptable for all samples to share the same complete base latent unless the protocol explicitly requires it.

Do not infer latent pairing from:

* the same prompt
* the same run ID
* filenames
* row order
* generation order

---

## 4. Required Watermark Provenance

Record applicable per-sample fields:

```text
run_id
prompt and prompt SHA
sample seed
base-latent SHA
clean base-latent SHA
watermark pre-injection base-latent SHA
watermarked latent SHA
watermark seed
watermark target SHA
watermark mask SHA
provider configuration SHA
clean image SHA
watermarked image SHA
attacked-clean image SHA
attacked-watermarked image SHA
attack configuration SHA
model ID and revision
detector implementation version
```

The evaluator must fail when required pairing or watermark provenance is missing.

---

## 5. Watermark Metadata Hashing

Watermark pairing and provider metadata must use the canonical hashing implementation from:

`skills/experiment-integrity/SKILL.md`

Normalize fields such as:

```text
run_id
sample seed
watermark seed
channel
radius
mask shape
pattern
measurement
injection mode
model revision
paths
booleans
```

Required test:

```text
pairing hash before writing metadata
==
pairing hash after metadata reload
```

If the hash changes after CSV or JSON serialization, stop and fix type normalization.

Do not bypass the pairing gate.

---

## 6. Attack Pairing

For TPR evaluation after attack, attacked-clean and attacked-watermarked cohorts must use the same attack distribution.

For corresponding samples, validate applicable settings:

```text
attack seed
shift dx and dy
DDIM timestep
active denoising steps
strength
guidance scale
inversion mode
warp mode
sampling mode
padding mode
normalization
color-transfer mode
prompt
negative prompt
model ID and revision
transform-config SHA
```

Required invariant:

```text
attacked-clean transform-config SHA
==
attacked-watermarked transform-config SHA
```

Do not calibrate the threshold from clean images attacked using a different pipeline than the watermarked images.

---

## 7. Detector Integrity

For every detector result, record:

* exact score definition
* score direction
* target or key
* mask
* inversion method
* negative distribution
* positive distribution
* threshold rule
* comparison operator
* target FPR
* actual empirical FPR
* sample count

Keep detector scores distinct.

Do not mix:

```text
p-value
-log10(p)
complex L1
real-valued L1
similarity
raw detector statistic
```

For NFPA-style Tree-Ring complex L1:

```text
score = mean(abs(decoded complex watermark - target complex watermark))
```

Lower score means more likely to contain the watermark.

Do not label a p-value result as complex L1.

Do not assume `w_measurement="l1_complex"` is respected unless the actual call path has been verified.

---

## 8. Threshold and TPR@1%FPR

A result may be reported as `TPR@1%FPR` only when:

1. The negative cohort is correct.
2. The positive cohort is correct.
3. Both cohorts use compatible preprocessing and attacks.
4. Score direction is correct.
5. Threshold calculation is recorded.
6. Comparison operator is recorded.
7. Actual empirical FPR is reported.
8. Sample count is reported.

For a lower-is-positive detector:

```text
detected when score < threshold
```

Report both:

```text
target FPR
actual empirical FPR
```

Finite sample sizes may prevent the actual FPR from being exactly 1%.

---

## 9. CLIP Evaluation

For watermark removal evaluation, use:

```text
attacked watermarked image
versus
original source prompt
```

Record:

* image path and SHA
* prompt and prompt SHA
* CLIP model
* pretrained weights
* preprocessing
* score aggregation
* sample count

Do not accidentally compute:

* watermarked versus attacked image similarity
* attacked image versus empty prompt
* clean image versus prompt

unless the metric is explicitly defined that way.

---

## 10. PSNR and SSIM

For spatial-shift attacks, calculate PSNR and SSIM only after applying the verified overlap or inverse-warp alignment rule.

Record:

* reference image
* evaluated image
* dx and dy semantics
* flow direction
* visual content direction
* inverse-sampling convention
* overlap crop implementation
* valid overlap size

Use one shared tested overlap helper.

Do not maintain separate formal and legacy crop implementations.

For the primary RAVEN quality evaluation, clearly define whether the reference is:

```text
watermarked input
or
paired clean image
```

Do not mix both definitions under the same metric name.

---

## 11. FID Evaluation

Before every FID calculation:

1. Create a fresh staging directory.
2. Populate it only from the current verified manifest.
3. Verify exact sample counts.
4. Verify matching run IDs.
5. Reject missing files.
6. Reject extra files.
7. Remove or reject stale symlinks.
8. Record the staging manifest SHA.

Do not reuse a previous FID staging directory merely because files already exist.

Clearly record which distributions are compared, for example:

```text
watermarked images versus attacked-watermarked images
```

---

## 12. Required Watermark Smoke Test

Before running the full dataset, use at least two samples.

Verify:

```text
sample seeds differ
base-latent hashes differ across samples
clean and watermark pre-injection hashes match within each pair
watermark target hashes match the intended protocol
metadata hashes survive reload
clean and watermarked images exist
attack configuration hashes match
detector target and mask hashes match
CLIP uses attacked-watermarked image versus prompt
no NaN or Inf appears
```

The smoke test must stop before evaluation if any required provenance check fails.

---

## 13. Required Negative Tests

Intentionally modify at least one of:

```text
base-latent SHA
sample seed
watermark seed
watermark target SHA
mask SHA
pairing SHA
attack-config SHA
model revision
detector version
```

The formal pipeline must reject the modified sample.

Also test that the evaluator rejects:

* shared base latent across supposedly independent samples
* mismatched clean and watermark base latents
* attacked-clean and attacked-watermarked configuration drift
* missing provenance fields
* stale attack outputs
* stale FID staging files

---

## 14. Resume Rules for Watermark Pipelines

Do not skip a generated or attacked image based only on:

```text
run ID
image path
metadata row
output directory
completed flag
```

Before reuse, verify:

```text
source image SHA
base-latent SHA
watermark target SHA
provider-config SHA
attack-config SHA
code commit
entrypoint SHA
model revision
output SHA
```

If source data are regenerated, existing attacked outputs must be rejected unless their recorded source SHA still matches.

A new source dataset must not silently reuse attacks or metrics produced from an older dataset.

---

## 15. Formal Output Eligibility

Classify every watermark result as one of:

```text
valid
invalid
legacy
incomplete
non-release
not independently auditable
```

Examples:

* Missing latent provenance: not independently auditable
* Shared complete base latent: invalid for independent-sample evaluation
* Detector metric mismatch: invalid for the claimed detector metric
* Attack-config drift: invalid for calibrated TPR
* Unpushed code: non-release
* Smoke test only: incomplete
* Old p-value analysis: legacy

Do not select a result because its value looks better.

---

## 16. Watermark `debug.md` Requirements

For watermark-related changes, the `debug.md` entry must additionally include:

```text
source-data validity
clean/watermarked pairing status
base-latent uniqueness status
watermark target and mask status
attack-pairing status
detector score definition
threshold calibration source
actual empirical FPR
quality metric reference
CLIP input definition
FID staging status
outputs requiring regeneration
```

State explicitly whether each previous result remains usable.

---

## 17. Final Watermark Report

After completing watermark work, report:

### Source data

* number of samples
* base-latent uniqueness
* clean/watermarked pairing
* provenance completeness

### Attack

* attacked-clean and attacked-watermarked compatibility
* transform-config verification
* stale output handling

### Detector

* score definition
* threshold rule
* target and actual FPR
* TPR
* ROC-AUC
* attack success rate

### Quality

* CLIP definition and result
* FID distributions and result
* PSNR reference and result
* SSIM reference and result
* overlap rule

### Outputs

* valid
* invalid
* legacy
* requiring regeneration

### GitHub

* branch
* commit SHA
* push status
* remote verification
* `debug.md` entry
