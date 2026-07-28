# `shared_tr_clean_v2` — the canonical Tree-Ring shared-clean protocol

Cross-method watermark comparison only means something when every method is
measured against *the same* clean image. `shared_tr_clean_v2` is the protocol
that makes that true: one canonical Tree-Ring source row per `run_id` supplies
the prompt, the base latent, the clean image and the generation configuration to
every other method, and each method produces **only** its own watermarked image.

Phase 1 (Issue #9) covers **GaussMarker (GM)** and **T2SMark (T2S)**, alongside
the already-merged **Gaussian Shading (GS)** cohort. RingID, HSTR and HSQR remain
open under Issue #6.

---

## 1. Official reproduction and `shared_tr_clean_v2` are distinct profiles

These are two different things and must never be conflated in a report:

| | official reproduction | `shared_tr_clean_v2` |
| --- | --- | --- |
| purpose | reproduce the upstream paper's own numbers | compare methods against one shared clean image |
| base latent | drawn by the method itself | the canonical Tree-Ring latent |
| clean image | the method's own | the canonical Tree-Ring clean image |
| model / scheduler / dtype | the method's official profile | the Tree-Ring cohort's configuration |
| claim | end-to-end upstream parity | shared-clean cross-method comparability |

Every row of a `shared_tr_clean_v2` cohort carries `shared_clean_profile` and a
`not_claimed` field in its `watermark_config.json` stating exactly what is *not*
being claimed. Legacy cohorts are never silently relabelled — GS V1
(`gaussian_shading_shared_uniform_v1`) keeps its own protocol name, metadata and
pairing hashes, and each method's V2 protocol is a separate, separately-audited
name.

### What each method adapts from official sampling

**GaussMarker** — `gaussmarker_shared_tr_clean_v2`, mode
`official_math_shared_tr_clean`.

The GaussMarker math is unchanged: the bundle's ChaCha20-encrypted message, the
per-element truncated-Gaussian partition, and the official complex Tree-Ring ring
injection with the bundle's ring target and circle mask. The one adaptation is
the *source of randomness*. Official `gaussmarker_gen.py` calls
`truncnorm.rvs` per element from the legacy global NumPy RNG; here the same
truncated normal is obtained deterministically from the canonical latent as

```
u = norm.cdf(tr_base_latent_float32)      # strictly inside (0, 1)
z = norm.ppf((u + encrypted_bit) / 2)     # the identical quantile partition
```

which is the same transform Gaussian Shading V2 uses. Additionally the cohort
runs the Tree-Ring configuration (RedbeardNZ SD 2.1-base, DDIM, 50 steps,
guidance 7.5, **float32**) rather than the official SD 2.1 fp16 + DPMSolver
profile, and the injected latent is therefore not cast to fp16. Because the run
does not use the official profile it is labelled `legacy` and
`profile_is_official` is false — it can never be reported as official GM.

**T2SMark** — `t2smark_shared_tr_clean_v2`, mode
`official_encoder_shared_tr_clean`.

The T2S encoder is unchanged: the same key pattern derivation, the same
master/session key and message lifecycle in upstream's draw order, the same
repeated-bit codeword, the same tail/central magnitude split, and the default
`official_compatible` RNG mode. The one adaptation is that the Gaussian source
whose order statistics the encoder reuses is the canonical Tree-Ring latent
instead of a fresh `torch.randn`. The cohort runs the Tree-Ring float32 DDIM
configuration.

---

## 2. What is generated, and what is read-only

Generated: `data/<method>/<dataset>/<METHOD>/<run_id>/watermarked.png`, the
cohort `metadata.csv`, the audit/config/summary JSONs, and (T2S only) a portable
per-sample watermark state under `watermark_state/`.

Read-only, never written, copied, re-encoded, renamed or deleted:

```
data/clean/<dataset>/<run_id>.png        the canonical clean images
data/tr/<dataset>/metadata.csv           the canonical source metadata
data/tr/<dataset>/<run_id>/watermarked.png
```

There are **no** per-method clean directories. Every method row points at the
Tree-Ring `clean_path` and `clean_sha256`. Each runner snapshots the size, mtime
and SHA-256 of every clean image it reads and re-checks all three after
generation; a single difference stops the run and is reported in
`clean_source_integrity.json`.

## 3. Mandatory identities

Every GM/T2S row must satisfy, for its `run_id`:

```
method.prompt_sha256                                  == tr.prompt_sha256
method.base_latent_seed                               == tr.base_latent_seed
method.base_latent_sha256                             == tr.base_latent_sha256
method.clean_base_latent_sha256                       == tr.clean_base_latent_sha256
method.watermark_pre_injection_base_latent_sha256     == tr.base_latent_sha256
method.clean_path                                     == tr.clean_path
method.clean_sha256                                   == tr.clean_sha256
method.generation_config_sha256                       == tr.generation_config_sha256
```

and the watermarked artifacts must differ from the clean/base ones.

## 4. Proving the provider consumed the shared latent

A matching SHA in a row is what an unsound run would also report, so each runner
proves consumption independently before writing anything:

* **GM** — `zT_clean_torch` must be the supplied tensor's own storage
  (`data_ptr()`), its SHA must still be canonical, and the uniforms implied by
  the provider's pre-injection latent must re-derive to `norm.cdf` of the
  supplied latent within `1e-6` (a float32 storage round trip; an unrelated
  latent is off by O(0.1)).
* **T2S** — `state.base_latent_sha256` must be the canonical SHA, and the
  multiset of `|z|` must be unchanged by encoding. T2S rebuilds its output purely
  by reordering and re-signing its source's magnitudes, so this is exact and a
  fresh `torch.randn` cannot reproduce it. The provider itself raises if the
  invariant breaks.

## 5. Commands

Authoritative algorithm code lives in `eval_bench_wm/utils/wm/gm_provider.py` and
`eval_bench_wm/utils/wm/t2s_provider.py`. The runners under `experiments/` do IO,
provenance and gating only; shared canonical-source plumbing is in
`experiments/shared_clean_tr.py`.

### Two-row GPU smoke (`smoke_only`, not eligible for formal reporting)

```bash
SMOKE=/tmp/raven-shared-clean-smoke

python3 experiments/generate_gm_from_tr_shared_clean.py \
  --tr-metadata data/tr/diffusiondb/metadata.csv \
  --output-dir "$SMOKE/gm" \
  --gm-bundle-dir "$SMOKE/gm_bundle" --create-bundle true \
  --gm-watermark-bits-seed 20260728 \
  --run-ids 0 1 --dataset-name diffusiondb_shared_tr_smoke \
  --device cuda --gpu 0 --smoke-only true

python3 experiments/generate_t2s_from_tr_shared_clean.py \
  --tr-metadata data/tr/diffusiondb/metadata.csv \
  --output-dir "$SMOKE/t2s" \
  --run-ids 0 1 --dataset-name diffusiondb_shared_tr_smoke \
  --device cuda --gpu 0 --smoke-only true
```

Smoke outputs go to a scratch directory, never to `data/` or `outputs/`, and
every row carries `smoke_only=True` / `formal_output_eligible=False`.

### Cross-method audit (TR + GS + GM + T2S)

```bash
python3 raven_repro/scripts/audit_shared_clean_cohorts.py \
  --tr-metadata  data/tr/diffusiondb/metadata.csv \
  --gs-metadata  data/gs/diffusiondb_shared_tr/GS/metadata.csv \
  --gm-metadata  "$SMOKE/gm/metadata.csv" \
  --t2s-metadata "$SMOKE/t2s/metadata.csv" \
  --output       audit/shared_clean_tr_gs_gm_t2s.json
```

The audit matches by `run_id` only. It rejects duplicate or missing rows,
re-hashes every referenced image, and fails closed on prompt, latent, clean,
generation-config, bundle/state or protocol drift. Where two methods cover the
same `run_id` they must agree on the clean artifacts and must have produced
different watermarked images.

### Formal cohort generation — gated

Do **not** run the full cohort from a feature branch. Required order:
implementation and focused tests → two-row GPU smoke → Draft PR review → merge to
`main` → check out a fixed clean `main` commit → run the formal cohorts → run the
cross-method audit → record output roots, manifests, row counts and the commit
SHA.

```bash
# only from a clean, fixed main commit
python3 experiments/generate_gm_from_tr_shared_clean.py \
  --tr-metadata data/tr/diffusiondb/metadata.csv \
  --gm-bundle-dir data/gm/diffusiondb_shared_tr/bundle --create-bundle true \
  --gm-watermark-bits-seed <recorded-seed> \
  --device cuda --gpu <idle-gpu>          # -> data/gm/diffusiondb_shared_tr/GM

python3 experiments/generate_t2s_from_tr_shared_clean.py \
  --tr-metadata data/tr/diffusiondb/metadata.csv \
  --device cuda --gpu <idle-gpu>          # -> data/t2s/diffusiondb_shared_tr/T2S
```

## 6. Resume

Resume is deterministic and fail closed. Passing `--resume` re-audits every
stored row, then per `run_id` re-derives the full identity — source row, bundle
or state identity, provider entrypoint SHA, watermark and generation config
hashes, and the recorded output's SHA — and refuses to skip on any mismatch.
Without `--resume`, a runner will not touch an existing cohort at all. A GM
bundle can only be created for a brand-new cohort: once a `metadata.csv` exists,
`--create-bundle` is ignored, so a rejected resume can never leave a stray bundle
behind.

## 7. GPU gate

CUDA, NVML, driver or container-runtime failure is a hard stop for the smoke and
for formal generation. There is no CPU fallback: a CPU run is not a GPU smoke and
must never be reported as one.
