# Gaussian Shading — Auditable Official-Compatible Implementation Memory

Single source of truth for the auditable, official-compatible Gaussian Shading (GS)
watermark added to RAVEN alongside the retained legacy provider. Do not create
additional GS memory files; update this one.

## Branch / HEAD / provenance
- Branch: `agent/cleanup-quality-decomposition`
- Original GS implementation commit reviewed: `a69e517fc99b7b4bfbcbfc1f687599de84cc9ca4`
- GS default flip to official implementation path: `02b669c` (2026-07-25).
- Detection-threshold / parser / injection cleanup + metadata migration script:
  this branch HEAD (2026-07-25). The authoritative source SHAs are recorded in
  `audit/formal_source_manifest.json` (rebuilt at that HEAD).
- Official reference asserted in code/metadata: `bsmhmmlf/Gaussian-Shading`,
  commit `09c678fadc7545acf7be12647ddf2a5e66f6a9dc` (`watermark.py`,
  `run_gaussian_shading.py`, `inverse_stable_diffusion.py`).
  NOTE: that repo is **not vendored** in `external/` (only tree-ring is). Parity is
  asserted against an **independent in-test reference implementation**
  (`test_gaussian_shading_official.py::independent_official_reference`), not against a
  checked-out upstream tree. Vendoring upstream for a byte-level diff is an open item.

## Root cause (what was wrong with the pre-existing GS path)
1. **Payload layout** — legacy uses byte-string replication (`message*num_replications`
   reshaped `(num_replications, bits)`), which is NOT the official
   `watermark.repeat(1, channel_copy, hw_copy, hw_copy)` channel/spatial repeat.
2. **Cipher** — legacy uses `cryptography` ChaCha20 (16-byte nonce, streaming API);
   official uses **PyCryptodome `ChaCha20.new`** with a **32-byte key + 12-byte nonce**.
3. **Shared latent in generic runners** — `run_watermark.py` / `run_removal.py` build one
   `wm_zT` before the prompt loop and reuse it for every prompt → identical GS latent
   across all images.
4. **Formal generator TR-only** — `generate_watermarked_images.py` hard-raised
   `"paired formal generation currently supports only Tree-Ring"`.
5. **Threshold mislabeling risk** — legacy GS `GS_THRESHOLDS` is a fixed default, not a
   TPR@1%FPR calibrated on the current cohort's clean negatives.
6. **TR FFT assertions** — `verify_tree_ring_injection` (frequency-domain injection-only
   check) is TR-specific and must never gate GS.

## Final architecture
Two explicit modes on `GsProvider` (`eval_bench_wm/utils/wm/gs_provider.py`), selected by
`--gs_protocol_mode {legacy, official_compatible}`; **default `official_compatible`**
(changed 2026-07-25 — GS now follows the official implementation path by default).
**Legacy is retained but must be requested explicitly** with `--gs_protocol_mode legacy`;
it is never selected implicitly, so old *legacy* results are only reproduced on demand.

### Default behavior summary (post-2026-07-25)
- `gs_protocol_mode` default = `official_compatible` (parser + `GsProvider.__init__` +
  `eval_protocol.PROVIDER_DEFAULTS["GS"]`).
- `gs_detection_mode` default = `official_onebit` (new arg) → detection success uses the
  official beta-tail `tau_onebit` with `>=`, **not** the legacy fixed `GS_THRESHOLDS`.
  `legacy_default` (legacy `GS_THRESHOLDS`, strict `>`) must be requested explicitly.
- Official GS **params** are already the defaults: `message_width_in_bytes=32`,
  `gs_channel_copy=1`, `gs_hw_copy=8`, `l=1`, `gs_fpr=1e-6`, `gs_user_number=1_000_000`.
- **Standalone reproduction runners** (`run_watermark.py`, `run_removal.py`) additionally
  adopt official upstream **generation** defaults for GS unless overridden:
  `modelid_target=stabilityai/stable-diffusion-2-1-base`, `scheduler_target=DPM`
  (DPMSolverMultistepScheduler), `revision=fp16` (weight variant only), and an
  **official-inspired inversion approximation** (empty prompt, `guidance_scale=1`,
  DPM-family inverse scheduler via `invert_z0`) — NOT a byte-level or numerical
  reproduction of upstream's manual DDIM-style inversion.
  The model/scheduler/revision injection only applies on the official reproduction
  model (see `apply_official_reproduction_defaults`): an explicit non-official
  model never gets `revision=fp16` or a forced DPM scheduler.
- The `inversion_note` / `dtype_note` returned by `apply_official_reproduction_defaults`
  are **parameter-aware**: the official-inspired DPM inversion note is emitted **only**
  when `using_official_model` is true **and** the resolved scheduler is DPM. For any
  other configuration (explicit non-official model, or an explicit non-DPM scheduler
  such as DDIM/Euler) the note records **only the actual scheduler** and makes **no
  official-reproduction claim** (and no fp16 weight-variant claim).
- **Formal generator** (`experiments/generate_watermarked_images.py`) deliberately keeps
  the shared matched-cohort generation pipeline (`RedbeardNZ/stable-diffusion-2-1-base` +
  DDIM + the fork dtype) so GS stays comparable to TR/RID/… — it does **not** switch GS
  model/scheduler. GS there still uses official-compatible encode/decode + official
  detection thresholds; only the generation pipeline is held common.

### Three protocol layers (recorded distinctly in metadata — never conflated)
1. **watermark implementation protocol** = `official_compatible` (payload/cipher/sampling/
   decode/majority-vote/thresholds).
2. **generation benchmark protocol** = `shared_formal_cohort_redbeardnz_ddim` (the formal
   generator's shared cohort pipeline).
3. **standalone upstream-inspired reproduction configuration** =
   `stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16` (the standalone
   runners' generation settings) — with an **official-inspired inversion approximation**
   (see below) and `revision=fp16` selecting the **fp16 weight variant only** (global
   compute dtype stays `torch.float32`, so this is not true fp16-compute parity).
Formal GS rows carry all three (`watermark_implementation_protocol`,
`generation_benchmark_protocol`, `upstream_official_reproduction_runner`) plus
`gs_detection_mode` and `detection_threshold_comparison_operator`.

### Legacy mode (unchanged, byte-for-byte behavior retained)
- Byte replication → `(num_replications, message_width_in_bits)` barcode.
- `cryptography` ChaCha20, window size `l`, `norm.ppf((u+y)/2**l)` sampling.
- Reverse: `norm.cdf`, decrypt, segment, per-column majority vote.

### Official-compatible mode (new)
- **Payload**: `_official_payload` → bits reshaped to
  `(1, C/channel_copy, H/hw_copy, W/hw_copy)` then `.repeat(1, channel_copy, hw_copy, hw_copy)`
  == official `watermark.repeat`.
- **Cipher**: `_official_encrypt_bits` / `_official_decrypt_diffused` use
  **PyCryptodome `ChaCha20.new(key=32B, nonce=12B)`** on packed bits.
- **Sampling**: `_official_latent_from_bits` = `norm.ppf((u + encrypted_bit) / 2)`,
  `u ~ Uniform[0,1)` from a per-run seeded `np.random.default_rng`.
  Payload layout、ChaCha20 cipher、sign decoding 與 majority vote 對齊官方。
  對 l=1，inverse-CDF sampling 與官方 positive/negative half-Gaussian
  truncnorm sampling 分布等價，但不宣稱與 upstream scipy RNG bit-exact。
- **Paired clean latent**: `zT_clean_torch = norm.ppf(u)` — the SAME uniforms as the
  watermarked latent, a distribution-preserving partition (this is the GS pairing, NOT a
  TR FFT injection). `pairing_relation = shared_sampling_uniforms_distribution_preserving_partition`.
- **Decode**: threshold latent at `>0`, ChaCha20-decrypt, `_official_majority_vote`
  (strict `votes > channel_copy*hw_copy*hw_copy//2`; **ties → 0**, matching official).
- **Constraints enforced in `__init__`**: `l==1`; `C % channel_copy == 0`;
  `H % hw_copy == 0`; `C*H*W/(channel_copy*hw_copy^2) == message_width_in_bits`;
  key len 32; nonce truncated to 12; message len == `message_width_in_bytes`.

## Mode & parameters (official-compatible defaults used for formal GS)
| Param | Value | Meaning |
|---|---|---|
| `gs_protocol_mode` | `official_compatible` | required for formal GS generation |
| `message_width_in_bytes` | 32 | 256-bit payload |
| `gs_channel_copy` | 1 | channel repeat factor `f_c` |
| `gs_hw_copy` | 8 | spatial repeat factor `f_h=f_w` |
| `l` | 1 | required (window size) |
| `gs_fpr` | 1e-6 | official beta-tail FPR target |
| `gs_user_number` | 1_000_000 | official traceability user count |
| latent shape | (1,4,64,64) | SD-2-1-base @512 |
Layout check: `4*64*64 / (1*8*8) = 256 bits = 32 bytes`. ✔

## run_id → secret mapping (auditable, reproducible, unique)
- `secret_index = run_id` → indexes into fixed pools
  `utils/wm/{messages,keys,nonces}.py` (10,000 entries each, all unique;
  keys 32B, nonces 16B→first 12B used, `message[:32]` unique across all 10,000).
- `gs_sampling_seed = deterministic_gs_sampling_seed(base_seed, run_id) = base_seed + run_id`
  (`experiments/generate_watermarked_images.py`) — matches the Tree-Ring per-row seed
  schedule (`base_latent_seed = base_seed + run_id`) so GS and TR consume the same numeric
  seed per `run_id` for apples-to-apples comparison. Seeds only the GS uniform draw; the
  payload/cipher/latent construction in `gs_provider.py` are unchanged.
- A **fresh `GsProvider` per run_id** (`offset=run_id`, `gs_secret_index=run_id`,
  `gs_sampling_seed=...`) → independent message/key/nonce/sampling/latent per image.
- Metadata stores **only** `gs_secret_index` + SHA-256 of message/key/nonce/bundle +
  sampling-uniform SHA-256. **No raw secrets** (`secret_provenance()` returns hashes only;
  extractor `FIELDNAMES` no longer has `key_hex`/`nonce_hex`/`ground_truth_bits`;
  decoded bits stored as `*_decoded_bits_sha256`).

## Threshold protocols (kept separate — never conflated)
Detection-success threshold is now selected by `--gs_detection_mode` (default
`official_onebit`). The active choice is resolved by `GsProvider.active_detection_threshold()`
and applied by `GsProvider.is_detection_successful(value)`; all three entry points (formal
generator, `run_watermark.py`, `run_removal.py`) route GS detection through it.
1. **Official beta-tail (DEFAULT)** — `GsProvider.official_thresholds()` via
   `scipy.special.betainc`: `tau_onebit` (single-user detection, `gs_detection_mode=
   official_onebit`) and `tau_bits` (traceability, ×user_number,
   `gs_detection_mode=official_traceability`), comparison `>=`. For the 256-bit/1e-6/1e6
   config: `tau_onebit=0.6484375`, `tau_bits=0.71484375`. Stored per-row as
   `gs_official_tau_onebit/gs_official_tau_bits`; `detection_threshold_type` =
   `official_beta_tail_tau_onebit` / `official_beta_tail_tau_bits`.
2. **Legacy default (explicit only)** — `gs_detection_mode=legacy_default` → fixed
   `GS_THRESHOLDS` (`0.70703125`), strict `>`, `threshold_type="legacy_default_threshold"`,
   `calibrated_from_current_clean_negatives=False`. Never selected implicitly. The legacy
   `get_detection_threshold("GS", ...)` / `describe_legacy_detection_threshold` helpers are
   retained unchanged for other callers but are no longer the GS detection default.
3. **1%-FPR calibrated** — MUST be computed downstream from THIS cohort's clean-negative
   scores (`raven_repro/scripts/eval_reproduction.py` protocol). NOT produced by the
   generator; not to be back-filled from (1) or (2).

## Related files (modified / added)
- `eval_bench_wm/utils/wm/gs_provider.py` — dual-mode provider, official payload/cipher/
  sampling/decode/majority-vote, `secret_provenance`, `official_thresholds`,
  `watermark_target_tensor`.
- `experiments/generate_watermarked_images.py` — TR **and** GS method-specific paths;
  `deterministic_gs_sampling_seed`, `validate_gs_resume_provenance`; per-run provider;
  GS pairing via shared uniforms; GS provenance rows; TR FFT check gated to `wm_type=="TR"`.
- `eval_bench_wm/run_watermark.py`, `run_removal.py` — **fail closed**: GS with `num!=1`
  raises (redirects to the auditable generator) so shared-`zT` cannot ship for GS. Now also
  inject the official upstream GS **generation** defaults (`apply_official_reproduction_defaults`:
  stabilityai SD2.1-base + DPM + `revision=fp16`) when not overridden, route GS detection
  through `GsProvider.is_detection_successful` (official `tau_onebit` by default), and
  `run_removal.py` records `gs_detection_mode` + official tau + comparison operator per row.
- `eval_bench_wm/utils/wm/gs_provider.py` — `--gs_detection_mode` arg,
  `active_detection_threshold()`, `is_detection_successful()`, and module-level
  `apply_official_reproduction_defaults()` (standalone runners only; GS-guarded).
- `raven_repro/raven/pairing_provenance.py` — `GS_PAIRING_PROTOCOL`,
  `gaussian_shading_shared_uniform_v1`, `GS_REQUIRED_FIELDS`, per-method audit with GS
  uniqueness (secret index/bundle/sampling seed/uniforms/target-per-run).
- `raven_repro/raven/eval_protocol.py` — GS `PROVIDER_FIELDS_BY_METHOD`/`DEFAULTS`
  (protocol mode, copies, fpr, user_number). `PROVIDER_DEFAULTS["GS"]["gs_protocol_mode"]`
  flipped `legacy → official_compatible` (fallback/type hint only; hash-safe since formal
  GS rows always carry the field explicitly). `gs_detection_mode` intentionally excluded
  from the embedding-config hash.
- `experiments/run_raven_formal_eval.py` — GS pairing audit + drift/resume/attack fields.
- `raven_repro/scripts/extract_verification_scores.py` — per-row GS provider rebuild +
  detector/source secret & target SHA parity; hashes-only; decoded-bits SHA.
- `raven_repro/scripts/evaluate_verification.py`, `build_verification_manifest.py` — GS wiring.
- `eval_bench_wm/utils/utils.py` — `describe_legacy_detection_threshold` (committed earlier).
- `raven_repro/tests/test_gaussian_shading_official.py` — **new** test suite.

## Success commands
The 25-step command below is a **fast smoke gate only**. Before the formal N=1000 run,
a **10-pair, 50-step generation** gate and a **50-step verification inversion** gate must
pass. 10-image integration gate (**do NOT raise `--num_pairs`; N=1000 is the formal run**):
```bash
cd /workspace/RAVEN
export TQDM_DISABLE=1
python experiments/generate_watermarked_images.py \
  --wm_types GS --dataset_name gs_gate \
  --prompts_csv <csv with >=10 'prompt' rows> \
  --output_dir <out>/watermarked --clean_output_dir <out>/clean \
  --num_pairs 10 --seed 42 \
  --gs_protocol_mode official_compatible \
  --message_width_in_bytes 32 --gs_channel_copy 1 --gs_hw_copy 8 \
  --num_inference_steps_target 25 \
  --modelid_target RedbeardNZ/stable-diffusion-2-1-base --require_free_gpu true
```
Re-run the identical command to exercise **resume** (skips all rows, revalidates GS
provenance, `rows_written_this_run=0`).

Tests:
```bash
cd /workspace/RAVEN && python -m pytest raven_repro/tests/test_gaussian_shading_official.py -q
cd /workspace/RAVEN/raven_repro && python -m pytest tests/ -q   # 247 passed (TR unaffected)
```

## Tests (all passing)
`test_gaussian_shading_official.py` (**43**): official layout/cipher/decode + inverse-CDF
reference (fixed SHAs against the in-repo reference; not upstream scipy RNG bit-exact);
determinism + unique-per-run (10 seeds→10 unique latents/secrets/uniforms); GS sampling
seed == TR schedule; direct-decode bit-accuracy=1 + random-clean baseline in [0.35,0.65];
majority-vote ties→0; metadata index+hash only; GS pairing-audit uniqueness; resume accepts
identical / rejects sampling-seed drift; **default protocol=official_compatible + detection
mode=official_onebit; official tau_onebit actually drives detection; traceability/legacy
modes explicit & distinct; legacy protocol still works explicitly; reproduction-default
injection semantics (official model→DPM+fp16, explicit non-official model→no fp16/no DPM,
explicit scheduler/revision preserved, non-GS untouched); parameter-aware inversion note
(official model+DPM→official-inspired DPM note; non-official+DDIM→no DPM/official claim;
explicit Euler→note matches actual scheduler); run_watermark/run_removal parsers accept
`--model_revision`; formal generator keeps RedbeardNZ+DDIM; detection fields excluded from
provenance hashes; migration script — full latent/sampling/secret/target/base + config
SHA/content re-derivation and formal-mapping (secret_index==run_id,
sampling_seed==base_latent_seed) checks: upgrade, idempotent, immutable/pairing preservation,
image/secret/target/watermarked-latent/sampling-uniform/seed-drift/secret-index/seed-vs-base
mismatch fail-closed, missing/SHA-mismatch/content-drift shard-config rejection, sharded
config resolution, partial-schema unify, PNG untouched, dry-run**. Full `raven_repro/tests`:
**247 passed** (TR unchanged).

## Verified acceptance (this session)
- Per-run secret & GS latent unique + reproducible: gate audit shows 10 unique
  secret indexes / bundles / sampling seeds / sampling uniforms / targets / latents. ✔
- Official mode == independent reference at fixed payload/key/nonce (SHA-pinned test). ✔
- No-attack direct decode bit-accuracy = 1.0 (gate `before_detection_rate=1.0`). ✔
- Clean random latent ≈ random baseline (test [0.35,0.65]). ✔
- TR/other methods do not regress (full `raven_repro/tests` = **247 passed**). ✔
- 10-image 25-step gate + resume + audit pass. ✔
- **10-pair 50-step generation gate + identical-command resume gate** (2026-07-25, HEAD
  `c648efa`): 10/10 generated, `before_detection_rate=1.0`, pairing audit all-unique;
  resume `rows_written_this_run=0` (all rows skipped + GS provenance revalidated). ✔
- **Full GS RAVEN end-to-end dry-run** (2026-07-25, N=10, fp16 attack pipeline):
  `snapshot → attack-watermarked → verify → quality → fid → clip → aggregate → validate`
  reached `result_classification=formal_complete`; `attacked_clean_count=0` and no
  attacked-clean artifacts on disk (attack-clean stayed TR-only, method-gated —
  no code change needed); `source_code_manifest_sha256=8a204ca1…` matches the rebuilt
  manifest. ✔
- **Migration full re-derivation** verified on the real 1001-image cohort and the fresh
  10-pair gate cohort: latent/uniform/target/secret/base SHAs + both config canonical
  SHAs recompute exactly; idempotent (`already_current`, no write). ✔

## Limitations / remaining deviations from official
- Upstream `bsmhmmlf/Gaussian-Shading` is **not vendored**; parity is vs. an independent
  in-repo reference, not a byte-diff of upstream. (Vendor + pin for a hard diff.)
- Nonces in the pool are 16B, truncated to 12B for official mode; uniqueness is preserved
  in practice (bundle SHA + sampling uniqueness audited) but is not the upstream 12B-native
  nonce generator.
- Downstream **RAVEN attack→extract→evaluate** GS path is covered by unit tests and the
  per-row provider-rebuild parity check, but a **full end-to-end RAVEN attack run** on GS
  images was NOT executed this session (only generation + resume + audit).
- Legacy GS DDIM inversion is retained unchanged and remains legacy-labeled; no
  official-vs-legacy image-quality comparison was run.
- **fp16 compute (residual difference):** the standalone reproduction runners set
  `revision=fp16` (fp16 *weight* variant), but the fork's global compute dtype is
  `torch.float32` (`pipe_provider.DTYPE`). True fp16-compute parity with upstream is NOT
  forced here (it would change every method + the formal cohort). This is a known,
  documented residual difference, not a silent divergence.
- **Inversion (official-inspired approximation, NOT equivalent):** the standalone GS
  official path uses an *official-inspired standalone inversion configuration* — empty
  prompt, `guidance_scale=1`, and a DPM-family inverse scheduler (Diffusers
  `DPMSolverMultistepInverseScheduler`, selected via `scheduler_target=DPM`, executed by
  the existing `invert_z0`). It is deliberately **not** claimed to be byte-identical,
  numerically equivalent, or upstream-behavior-equivalent: upstream
  `inverse_stable_diffusion.py` uses a **manual DDIM-style backward/forward-diffusion
  update** (reversed DPM timesteps, per-step UNet noise prediction, manual `alpha_prod_t`
  / `alpha_prod_t_prev` `backward_ddim`), which is a different implementation. A true
  GS-only port of that manual inversion was **not** done (it would need isolation +
  numerical tests and must not touch the shared `invert_z0` used by TR/RID/HSTR/HSQR).
  So the current standalone inversion remains an official-*inspired* approximation.
- Detection default flipped to the official beta-tail `tau_onebit`; the formal generator's
  `before_detection_successful` for GS is now computed against `tau_onebit` (`>=`) rather
  than the legacy fixed threshold. Provenance hashes are unaffected (detection fields are
  not part of `PAIRING_HASH_FIELDS`/`GS_REQUIRED_FIELDS`, and `generation_config`/
  `watermark_config` were not modified).

## N=1000 readiness checklist
- [x] Official mode matches independent reference (SHA-pinned).
- [x] Per-run unique+reproducible secret/seed/sampling/latent.
- [x] Secret pools ≥1000 unique (10,000; `message[:32]` unique across all 10,000 → the
      per-run unique-target audit holds at N=1000).
- [x] Metadata = hashes/indices only.
- [x] Legacy vs official-tau vs 1%-FPR thresholds recorded separately.
- [x] 10-image 25-step fast smoke gate + resume + audit green.
- [x] 10-pair 50-step generation + identical-command resume gate (2026-07-25, HEAD
      `c648efa`: 10/10, `before_detection_rate=1.0`, resume `rows_written_this_run=0`).
- [x] TR non-regression.
- [x] Full end-to-end RAVEN attack pipeline dry-run on a small GS cohort (2026-07-25,
      N=10: `formal_complete`, no attacked-clean artifacts).
- [ ] Optional: vendor upstream GS for a byte-level parity diff.
- [ ] Compute the 1%-FPR threshold from N=1000 clean negatives (downstream, not generator).

**Gate status (2026-07-25).** The 10-pair 50-step generation + identical-command resume
gate and the full GS RAVEN end-to-end dry-run (N=10) both **passed** at HEAD `c648efa`
(see Verified acceptance). These are wiring + provenance gates on a small cohort.

- The final 1%-FPR threshold must still be calibrated from the N=1000 clean-negative
  cohort (downstream, not the generator); N=10 is only a wiring and provenance gate and
  is NOT a statistically valid 1%-FPR result.

## GS end-to-end gate stages
Correct GS stage order: `snapshot → attack-watermarked → verify → quality → fid → clip →
aggregate → validate`. `attack-clean` belongs **only** to the TR flow; GS must **not** run
`attack-clean`. This is enforced method-specifically in `run_raven_formal_eval.py` and
does not need a code change:
- `attack_stage(role="clean")` raises `attack-clean is required only for the formal TR/NFPA
  protocol` for any non-TR method.
- `verify_stage`, `aggregate_stage`, and `validate_stage` gate every attacked-clean
  requirement behind `method == "TR"`, so the GS pipeline reaches `validate` without any
  attacked-clean records. TR keeps the full attack-clean recalibration protocol unchanged.
An orchestrator must therefore simply omit the `attack-clean` stage for GS (running it is a
fail-closed error, not a silent no-op).

## Metadata migration (removed 2026-07-27)
`experiments/migrate_gs_detection_metadata.py` upgraded pre-`02b669c` V1 GS metadata to
the official detection schema in place. It was deleted together with the V1 GS cohorts
it served (see "GS V1 data removal" below); no GS metadata on disk needs it any more.
Its 19 unit tests were removed with it. Historical detail remains in
`DEBUG_CHANGELOG.md` (2026-07-25 / 2026-07-26 entries).

## N=1000 readiness (explicit)
- **N=1000 generation**: the 50-step generation and resume gate **passed** (2026-07-25,
  HEAD `c648efa`).
- **N=1000 attack/evaluation**: the 10-pair full GS RAVEN end-to-end dry-run **passed**
  (2026-07-25, `formal_complete`).
- N=10 is **not** enough for a statistically valid 1% FPR result; it only verifies pipeline
  wiring and provenance.

## Shared-clean V2 cohort (`gaussian_shading_shared_tr_clean_v2`, 2026-07-27)

A second, separately-named GS protocol that embeds from the **canonical Tree-Ring
clean latent** instead of its own sampled one. The V1 cohort
(`gaussian_shading_shared_uniform_v1`) is untouched and is never relabelled.

| | V1 `shared_uniform` | V2 `shared_tr_clean` |
|---|---|---|
| `gs_protocol_mode` | `official_compatible` | `official_math_shared_tr_clean` |
| uniforms | `np.random.default_rng(base_seed + run_id)` | `norm.cdf(float64(TR base latent))` |
| clean latent | `norm.ppf(u)` | the TR latent itself (same storage) |
| clean image | own GS cohort | the existing TR clean image |
| `gs_sampling_seed` | recorded | absent (no RNG draw exists) |
| extra fields | — | `GS_SHARED_CLEAN_V2_FIELDS` |

Identical in both: message construction, payload replication
(`repeat(1, channel_copy, hw_copy, hw_copy)`), secret index = run_id, key, nonce,
PyCryptodome ChaCha20, encrypted-bit layout, `norm.ppf((u + b) / 2)`, majority
vote (ties -> 0), official beta-tail thresholds and the detector.

**Claim discipline.** V2 reproduces the official Gaussian *quantile-partition
embedding math*. It is **not** byte-identical to upstream sampling — upstream
draws its own uniforms. The generator records this verbatim in
`watermark_config.json` (`official_math_claim` / `not_claimed`).

Generator: `experiments/generate_gs_from_tr_shared_clean.py`
(default output `data/gs/diffusiondb_shared_tr/GS/`, shard-safe `--resume`,
`--run-ids` for gates). It generates **only** GS watermarked images; it has no
clean-image write path at all.

Cross-method proof: `audit_tr_gs_shared_clean(tr_rows, gs_rows)` writes
`cross_method_shared_clean_audit.json` and requires equality of `prompt_sha256`,
`generation_config_sha256`, `base_latent_seed`, `base_latent_sha256`,
`clean_base_latent_sha256`, `clean_path` and `clean_sha256`, plus on-disk SHA
verification of the clean and GS watermarked images.

Path fields are required metadata but deliberately excluded from the V2 pairing
hash, so a canonical-layout move cannot invalidate an audited cohort; the content
each path points at is still bound in through its SHA-256.

N=1 gate (2026-07-27, run_id=0, `/tmp`, deleted on success):
TR/GS base-latent SHA `bea48052…825a` equal; TR/GS clean SHA `c60db047…bebac`
equal; `clean_path` identical; before-attack `bit_accuracy = 1.0`, detected at
`official_beta_tail_tau_onebit = 0.6484375` (`>=`); cross-method audit passed
against all 1001 TR rows. Full 1001-sample cohort not yet generated.

## GS V1 data removal (2026-07-27)

All Gaussian Shading data and outputs from the V1 protocol were deleted at the
user's request when GS moved to the shared-clean protocol:

| Deleted | Size |
|---|---|
| `data/gs/{gs_diffusiondb_1001_match_tr, gs_formal_gate_10_50step, gs_gate_cleanup_10_50step, gs_gate_recheck_10_50step}` | 445 MB |
| `data/clean/gs_*` (the GS-specific clean cohorts) | 440 MB |
| `outputs/gs/*` (all GS attack / eval / verification output) | 2.7 GB |
| `experiments/migrate_gs_detection_metadata.py` (+19 tests) | — |

`data/clean/diffusiondb/` (the Tree-Ring clean cohort, 1004 files) and all of
`data/tr/` (1020 files) were **not** touched — verified by file count and by the
unchanged metadata SHA `ade37924…4d04bc`. `data/gs/` and `outputs/gs/` remain as
empty canonical roots.

V1 *code* is retained: `GS_PAIRING_PROTOCOL`, `GS_REQUIRED_FIELDS` and the
`official_compatible` seeded-sampling path still exist and are still tested
(synthetically, since the V1 data is gone), because the standalone reproduction
runners use that mode.

## Next steps
1. ~~Run the 10-pair 50-step generation and identical-command resume gate.~~ Done
   (2026-07-25, HEAD `c648efa`).
2. ~~Run the 10-pair full GS RAVEN end-to-end gate.~~ Done (2026-07-25, `formal_complete`).
3. Launch/complete N=1000 generation and attack/evaluation (small-cohort gates pass).
4. Calibrate 1%-FPR from N=1000 clean negatives; report TPR@1%FPR separately from legacy.
5. (Optional) Vendor `bsmhmmlf/Gaussian-Shading@09c678f` for a byte-level parity diff.
