# Cross-Watermark Shared Clean and Latent Policy

## Purpose

Use this skill for every new watermark-generation cohort involving more than one
watermark method.

All watermark methods in the same comparison cohort must use the exact same
clean image and the exact same pre-watermark base latent for each run_id.

The canonical shared source is the TR clean-generation process.

This skill applies to TR, GS and every watermark method added later.

## Canonical base randomness

For each run_id, generate the shared base latent exactly once using the
canonical TR procedure:

sample_seed = base_seed + run_id

generator = torch.Generator(device="cpu").manual_seed(sample_seed)

base_cpu = torch.randn(
    latent_shape,
    generator=generator,
    dtype=torch.float32,
    device="cpu",
)

The canonical base_cpu tensor is the sole source of clean-image generation
randomness for that run_id.

Do not allow each watermark provider to independently sample another clean or
base latent.

## Shared latent invariant

Every watermark method must receive the exact same pre-watermark tensor bytes.

Required invariant:

TR base_latent_sha256
==
GS base_latent_sha256
==
all other watermark methods' base_latent_sha256

Passing the same numeric seed is not sufficient.

Using the same probability distribution is not sufficient.

Using different random-number generators with the same seed is not sufficient.

The exact tensor must be directly shared or reconstructed from the canonical TR
algorithm and verified equal.

## Shared clean image invariant

Generate the clean image once from the canonical shared latent.

Store it at:

data/clean/<dataset>/<run-id>.png

Every watermark metadata row for that run_id must reference the same:

- clean_path
- clean_sha256
- base_latent_seed
- base_latent_sha256
- clean_base_latent_sha256
- generation_config_sha256
- prompt_sha256

Do not regenerate, copy, rename or re-encode the clean image separately for each
watermark method.

## Watermark-specific behavior

Watermark-specific secrets, keys, messages, masks and targets may remain
method-specific.

Only the pre-watermark clean-generation randomness and clean latent must be
identical across methods.

A provider may transform the shared latent only as part of watermark injection.

A provider must not replace the shared clean latent with its own independently
sampled latent.

## GS from canonical TR latent

For GS, derive the quantile uniforms from the canonical TR float32 base latent:

uniforms = scipy.stats.norm.cdf(
    base_cpu.numpy().astype(np.float64)
)

Use the official Gaussian Shading encrypted-bit quantile-partition formula for
the GS watermarked latent.

The GS provider must:

- accept externally supplied uniforms
- accept the externally supplied shared clean latent
- return that exact shared clean latent as zT_clean_torch
- not call np.random.default_rng() for clean/base sampling in shared-clean mode
- preserve GS secret construction
- preserve GS payload replication
- preserve GS encryption
- preserve GS encrypted-bit layout
- preserve GS quantile-partition watermark formula
- preserve GS detector and detection threshold behavior

Do not reconstruct zT_clean_torch with norm.ppf(uniforms), because numerical
CDF/PPF round-trip differences may change the exact tensor bytes.

The externally supplied shared clean latent must be returned directly.

## Unsupported providers

When a watermark method cannot embed from the canonical shared clean latent:

- stop
- report the incompatibility
- do not silently generate a method-specific clean cohort
- do not label the resulting cohort shared-clean
- do not substitute a visually similar clean image
- do not rely only on matching seeds or filenames

## Metadata

Every new shared-clean watermark row must record:

- shared_clean_protocol
- shared_clean_source_method=TR
- shared_clean_sample_sha256
- base_latent_seed
- base_latent_sha256
- clean_base_latent_sha256
- generation_config_sha256
- prompt_sha256
- clean_path
- clean_sha256
- watermarked_path
- watermarked_sha256

Method-specific provenance must remain recorded separately.

For GS, also record:

- gs_uniform_derivation
- gs_sampling_uniform_sha256
- gs_secret_index
- gs_message_sha256
- gs_key_sha256
- gs_nonce_sha256
- gs_secret_bundle_sha256
- gs_protocol_mode
- GS threshold and detector configuration fields

## Required cross-method checks

For each run_id verify:

- same run_id
- same prompt SHA
- same generation config SHA
- same sample seed
- same base latent SHA
- same clean base latent SHA
- same clean path
- same clean image SHA

Do not infer equality from:

- matching filenames
- matching directory names
- matching run IDs
- matching numeric seeds
- matching distributions
- visually similar clean images

## Existing cohorts

Do not relabel legacy method-specific cohorts as shared-clean.

Existing TR results may remain the canonical shared-clean source.

To create a new shared-clean GS cohort:

1. read and verify the existing TR metadata
2. reconstruct the canonical TR base latent
3. verify its tensor SHA against the TR metadata
4. reuse the existing verified TR clean image
5. derive GS uniforms from the canonical TR base latent
6. generate only the new GS watermarked image
7. write new GS shared-clean metadata
8. point GS metadata to the exact existing TR clean path and SHA
9. run the cross-method shared-clean checks

Do not modify existing validated TR outputs.
