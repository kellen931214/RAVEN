# Gaussian Shading — Auditable Official-Compatible Implementation Memory

Single source of truth for the auditable, official-compatible Gaussian Shading (GS)
watermark added to RAVEN alongside the retained legacy provider. Do not create
additional GS memory files; update this one.

## Branch / HEAD / provenance
- Branch: `agent/cleanup-quality-decomposition`
- GS implementation commit reviewed: `a69e517fc99b7b4bfbcbfc1f687599de84cc9ca4`
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
`--gs_protocol_mode {legacy, official_compatible}`; **default `legacy`** so old results
never change silently.

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
- `gs_sampling_seed = deterministic_gs_sampling_seed(base_seed, run_id)`
  `= int(sha256(f"gaussian-shading-official-sampling-v1:{base_seed}:{run_id}")[:8]) & (2^63-1)`
  (`experiments/generate_watermarked_images.py`).
- A **fresh `GsProvider` per run_id** (`offset=run_id`, `gs_secret_index=run_id`,
  `gs_sampling_seed=...`) → independent message/key/nonce/sampling/latent per image.
- Metadata stores **only** `gs_secret_index` + SHA-256 of message/key/nonce/bundle +
  sampling-uniform SHA-256. **No raw secrets** (`secret_provenance()` returns hashes only;
  extractor `FIELDNAMES` no longer has `key_hex`/`nonce_hex`/`ground_truth_bits`;
  decoded bits stored as `*_decoded_bits_sha256`).

## Threshold protocols (kept separate — never conflated)
1. **Legacy default** — `get_detection_threshold("GS", ...)` → fixed `GS_THRESHOLDS`.
   Labeled via `describe_legacy_detection_threshold`:
   `threshold_type="legacy_default_threshold"`,
   `calibrated_from_current_clean_negatives=False`. Rows carry
   `detection_threshold_type` + `threshold_calibrated_from_current_clean_negatives=False`.
2. **Official beta-tail** — `GsProvider.official_thresholds()` via `scipy.special.betainc`:
   `tau_onebit` (single-user detection) and `tau_bits` (traceability, ×user_number),
   comparison `>=`. For the 256-bit/1e-6/1e6 config: `tau_onebit=0.6484375`,
   `tau_bits=0.71484375`. Stored per-row as `gs_official_tau_onebit/gs_official_tau_bits`.
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
  raises (redirects to the auditable generator) so shared-`zT` cannot ship for GS.
- `raven_repro/raven/pairing_provenance.py` — `GS_PAIRING_PROTOCOL`,
  `gaussian_shading_shared_uniform_v1`, `GS_REQUIRED_FIELDS`, per-method audit with GS
  uniqueness (secret index/bundle/sampling seed/uniforms/target-per-run).
- `raven_repro/raven/eval_protocol.py` — GS `PROVIDER_FIELDS_BY_METHOD`/`DEFAULTS`
  (protocol mode, copies, fpr, user_number).
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
cd /workspace/RAVEN/raven_repro && python -m pytest tests/ -q   # 179 passed (TR unaffected)
```

## Tests (all passing)
`test_gaussian_shading_official.py` (7): official layout/cipher/decode + inverse-CDF
reference (fixed SHAs against the in-repo reference; not upstream scipy RNG bit-exact); determinism + unique-per-run (10 seeds→10 unique latents/secrets/uniforms);
direct-decode bit-accuracy=1 + random-clean baseline in [0.35,0.65]; majority-vote ties→0;
metadata index+hash only (no key_hex/nonce_hex/ground_truth_bits); GS pairing-audit
uniqueness (rejects duplicate sampling uniforms); resume accepts identical / rejects
sampling-seed drift. Full `raven_repro/tests`: **179 passed** (TR unchanged).

## Verified acceptance (this session)
- Per-run secret & GS latent unique + reproducible: gate audit shows 10 unique
  secret indexes / bundles / sampling seeds / sampling uniforms / targets / latents. ✔
- Official mode == independent reference at fixed payload/key/nonce (SHA-pinned test). ✔
- No-attack direct decode bit-accuracy = 1.0 (gate `before_detection_rate=1.0`). ✔
- Clean random latent ≈ random baseline (test [0.35,0.65]). ✔
- TR tests do not regress (179 passed). ✔
- 10-image gate + resume + audit pass. ✔

## Limitations / remaining deviations from official
- Upstream `bsmhmmlf/Gaussian-Shading` is **not vendored**; parity is vs. an independent
  in-repo reference, not a byte-diff of upstream. (Vendor + pin for a hard diff.)
- Nonces in the pool are 16B, truncated to 12B for official mode; uniqueness is preserved
  in practice (bundle SHA + sampling uniqueness audited) but is not the upstream 12B-native
  nonce generator.
- Downstream **RAVEN attack→extract→evaluate** GS path is covered by unit tests and the
  per-row provider-rebuild parity check, but a **full end-to-end RAVEN attack run** on GS
  images was NOT executed this session (only generation + resume + audit).
- Legacy GS DDIM inversion / detection thresholds are unchanged and remain
  legacy-labeled; no official-vs-legacy image-quality comparison was run.

## N=1000 readiness checklist
- [x] Official mode matches independent reference (SHA-pinned).
- [x] Per-run unique+reproducible secret/seed/sampling/latent.
- [x] Secret pools ≥1000 unique (10,000; `message[:32]` unique across all 10,000 → the
      per-run unique-target audit holds at N=1000).
- [x] Metadata = hashes/indices only.
- [x] Legacy vs official-tau vs 1%-FPR thresholds recorded separately.
- [x] 10-image 25-step fast smoke gate + resume + audit green.
- [ ] 10-pair 50-step generation + identical-command resume gate.
- [x] TR non-regression.
- [ ] Full end-to-end RAVEN attack pipeline dry-run on a small GS cohort (recommended
      before N=1000).
- [ ] Optional: vendor upstream GS for a byte-level parity diff.
- [ ] Compute the 1%-FPR threshold from N=1000 clean negatives (downstream, not generator).

**N=1000 safe to run?** Not yet. The existing 10-image 25-step fast smoke gate,
resume check, and provenance audit are green, but the required 10-pair 50-step
generation and identical-command resume gate have not yet been executed.

- N=1000 generation is not approved until the 50-step generation and resume gate passes.
- N=1000 attack/evaluation is not approved until the 10-pair full GS RAVEN
  end-to-end gate passes.
- The final 1%-FPR threshold must be calibrated from the N=1000 clean-negative
  cohort; N=10 is only a wiring and provenance gate.

## GS end-to-end gate stages
Correct GS stage order: `snapshot → attack-watermarked → verify → quality → fid → clip →
aggregate → validate`. `attack-clean` belongs **only** to the TR flow; GS must **not** run
`attack-clean`.

## N=1000 readiness (explicit)
- **N=1000 generation**: must first pass the 50-step generation and resume gate.
- **N=1000 attack/evaluation**: must then pass the 10-pair full GS RAVEN end-to-end gate.
- N=10 is **not** enough for a statistically valid 1% FPR result; it only verifies pipeline
  wiring and provenance.

## Next steps
1. Run the 10-pair 50-step generation and identical-command resume gate.
2. Run the 10-pair full GS RAVEN end-to-end gate.
3. Only after both gates pass, launch N=1000 generation and attack/evaluation.
4. Calibrate 1%-FPR from N=1000 clean negatives; report TPR@1%FPR separately from legacy.
5. (Optional) Vendor `bsmhmmlf/Gaussian-Shading@09c678f` for a byte-level parity diff.
