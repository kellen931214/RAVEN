"""T2S parity and state tests.

Upstream reference: https://github.com/0xD009/T2SMark @ 0c1fbfd50fcd1fba135477a2c016e284d5d7914d

``_OfficialT2SMark`` below is a verbatim transcription of upstream ``src/t2s.py``
with only the ``.cuda()`` placement calls dropped, so device placement cannot
mask a numerical or PRNG difference. It exists solely as a test oracle and is
never imported by production code.
"""

from __future__ import annotations

import json
import sys
from functools import reduce
from pathlib import Path

import pytest
import torch
from scipy.stats import norm

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from utils.canonical import canonical_json_dumps, canonical_json_sha256  # noqa: E402
from utils.wm.t2s_provider import (  # noqa: E402
    T2S_STATE_SCHEMA_VERSION,
    T2SMark,
    T2SWatermarkState,
    bits_to_str,
    channel_layout,
    detect_from_reversed_latents,
    str_to_bits,
)


class _OfficialT2SMark:
    """Verbatim upstream src/t2s.py (CPU placement)."""

    def __init__(self, m, tau, latent_shape):
        self.latent_shape = latent_shape
        self.n = reduce(lambda x, y: x * y, self.latent_shape, 1)
        self.m = m
        self.r = int(2 * norm.cdf(-tau) * self.n / m)
        self.k = self.m * self.r
        self.noise_size = self.n - self.k
        self.prng = torch.Generator()

    def binlist2int(self, binlist):
        res = reduce(lambda x, y: x * 2 + y, binlist)
        if isinstance(binlist, torch.Tensor):
            return res.item()
        return res

    def encode(self, b, K):
        z = torch.randn(self.latent_shape).flatten()
        self.prng.manual_seed(self.binlist2int(K))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng)[: self.k]

        b_r = (1 - 2 * b).repeat(self.r).float()
        codeword = b_r * v_value

        w = torch.zeros(self.n).bool()
        w[v_support] = True

        tail = torch.topk(z.abs(), k=self.k, dim=0, largest=True, sorted=False)
        central = torch.topk(z.abs(), k=self.noise_size, dim=0, largest=False, sorted=False)

        z_w = torch.zeros(self.n)
        z_w[w] = tail.values * codeword
        z_w[~w] = central.values * (torch.randint(0, 2, (self.noise_size,)).float() * 2 - 1)
        return z_w.reshape(self.latent_shape)

    def decode(self, reversed_noise, K, detection=False):
        self.prng.manual_seed(self.binlist2int(K))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng)[: self.k]

        w = torch.zeros(self.n).bool()
        w[v_support] = True

        watermarked_vec = reversed_noise.flatten()[w] * v_value
        p = watermarked_vec.reshape(self.r, self.m).sum(dim=0)
        b = (p < 0).int()
        if detection:
            return b, torch.norm(p.flatten(), p=1).item()
        return b


KEY_SHAPE_SD21 = (1, 64, 64)
MSG_SHAPE_SD21 = (3, 64, 64)
KEY_SHAPE_SD35 = (4, 64, 64)
MSG_SHAPE_SD35 = (12, 64, 64)


# --------------------------------------------------------------------------
# 1. Official parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m, shape",
    [
        (16, KEY_SHAPE_SD21),
        (256, MSG_SHAPE_SD21),
        (16, KEY_SHAPE_SD35),
        (256, MSG_SHAPE_SD35),
    ],
)
def test_parameters_match_official(m, shape):
    local = T2SMark(m=m, tau=0.674, latent_shape=shape)
    official = _OfficialT2SMark(m=m, tau=0.674, latent_shape=shape)
    assert (local.n, local.r, local.k, local.noise_size) == (
        official.n, official.r, official.k, official.noise_size
    )


@pytest.mark.parametrize("m, shape", [(16, KEY_SHAPE_SD21), (16, KEY_SHAPE_SD35)])
def test_key_derived_support_and_signs_match_official(m, shape):
    """A fixed key must produce the identical support/sign sequence."""
    local = T2SMark(m=m, tau=0.674, latent_shape=shape)
    official = _OfficialT2SMark(m=m, tau=0.674, latent_shape=shape)

    for seed in (0, 1, 12345, 65535):
        key = torch.tensor([int(bit) for bit in format(seed, "016b")], dtype=torch.int32)

        v_value, v_support = local.key_pattern(key)

        official.prng.manual_seed(official.binlist2int(key))
        exp_value = torch.randint(0, 2, (official.k,), generator=official.prng).float() * 2 - 1
        exp_support = torch.randperm(official.n, generator=official.prng)[: official.k]

        assert torch.equal(v_value, exp_value), f"v_value mismatch for key seed {seed}"
        assert torch.equal(v_support, exp_support), f"v_support mismatch for key seed {seed}"


def test_key_pattern_is_device_independent_seed():
    """Guards the historical bug: a CUDA generator changed the key mapping."""
    mark = T2SMark(m=16, tau=0.674, latent_shape=KEY_SHAPE_SD21)
    assert mark.prng.device.type == "cpu", "key PRNG must stay on CPU for official parity"

    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(7))
    first_value, first_support = mark.key_pattern(key)

    if torch.cuda.is_available():
        cuda_mark = T2SMark(m=16, tau=0.674, latent_shape=KEY_SHAPE_SD21,
                            device=torch.device("cuda"))
        cuda_value, cuda_support = cuda_mark.key_pattern(key)
        assert torch.equal(first_value.cpu(), cuda_value.cpu())
        assert torch.equal(first_support.cpu(), cuda_support.cpu())


@pytest.mark.parametrize(
    "m, shape", [(16, KEY_SHAPE_SD21), (256, MSG_SHAPE_SD21), (256, MSG_SHAPE_SD35)]
)
def test_encode_matches_official(m, shape):
    """Same global RNG state + same key/bits -> byte-identical latent."""
    local = T2SMark(m=m, tau=0.674, latent_shape=shape)
    official = _OfficialT2SMark(m=m, tau=0.674, latent_shape=shape)

    bits = torch.randint(0, 2, (m,), generator=torch.Generator().manual_seed(3))
    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(4))

    torch.manual_seed(99)
    got = local.encode(bits, key)
    torch.manual_seed(99)
    expected = official.encode(bits, key)

    assert torch.equal(got, expected)


@pytest.mark.parametrize("m, shape", [(16, KEY_SHAPE_SD21), (256, MSG_SHAPE_SD21)])
def test_decode_matches_official(m, shape):
    local = T2SMark(m=m, tau=0.674, latent_shape=shape)
    official = _OfficialT2SMark(m=m, tau=0.674, latent_shape=shape)

    bits = torch.randint(0, 2, (m,), generator=torch.Generator().manual_seed(5))
    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(6))

    torch.manual_seed(11)
    latent = official.encode(bits, key)
    noisy = latent + 0.05 * torch.randn(latent.shape, generator=torch.Generator().manual_seed(12))

    got_bits, got_norm = local.decode(noisy, key, detection=True)
    exp_bits, exp_norm = official.decode(noisy, key, detection=True)

    assert torch.equal(got_bits, exp_bits)
    assert got_norm == pytest.approx(exp_norm, rel=1e-6)


def test_encode_decode_round_trip_is_exact():
    mark = T2SMark(m=256, tau=0.674, latent_shape=MSG_SHAPE_SD21)
    bits = torch.randint(0, 2, (256,), generator=torch.Generator().manual_seed(21))
    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(22))
    latent = mark.encode(bits, key, generator=torch.Generator().manual_seed(23))
    assert torch.equal(mark.decode(latent, key), bits.int())


def test_explicit_generator_matches_global_rng():
    """The per-sample generator only makes upstream's randomness explicit."""
    mark = T2SMark(m=16, tau=0.674, latent_shape=KEY_SHAPE_SD21)
    bits = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(31))
    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(32))

    explicit = mark.encode(bits, key, generator=torch.Generator().manual_seed(77))
    torch.manual_seed(77)
    implicit = mark.encode(bits, key)
    assert torch.equal(explicit, implicit)


# --------------------------------------------------------------------------
# 2. Channel layouts
# --------------------------------------------------------------------------


def test_channel_layout_4ch_matches_official():
    for key_idx in range(4):
        key_channels, msg_channels = channel_layout(4, key_idx)
        assert key_channels == [key_idx]
        assert msg_channels == [i for i in range(4) if i != key_idx]


def test_channel_layout_16ch_matches_official():
    key_channels, msg_channels = channel_layout(16)
    assert key_channels == [0, 1, 2, 3]
    assert msg_channels == list(range(4, 16))
    assert len(msg_channels) == 12


def test_channel_layout_rejects_other_widths():
    with pytest.raises(ValueError):
        channel_layout(8)


# --------------------------------------------------------------------------
# 3. State round trip
# --------------------------------------------------------------------------


def _make_state(**overrides) -> T2SWatermarkState:
    base = dict(
        watermark_id="t2s-test-0001",
        latent_shape=[1, 4, 64, 64],
        key_channels=[0],
        msg_channels=[1, 2, 3],
        key_length=16,
        msg_length=256,
        tau=0.674,
        master_key_bits="0" * 8 + "1" * 8,
        expected_session_key_bits="01" * 8,
        expected_message_bits="10" * 128,
        inversion_mode="t2s_official",
        num_inversion_steps=10,
        provider_config_sha256="a" * 64,
        model_id="stabilityai/stable-diffusion-2-1-base",
        scheduler="DDIM",
        num_inference_steps=50,
        resolution=512,
        sample_seed=1,
    )
    base.update(overrides)
    return T2SWatermarkState(**base)


def test_state_round_trip_preserves_fields_and_sha(tmp_path):
    state = _make_state()
    digest = state.state_sha256()

    path = state.save(tmp_path / "state.json")
    reloaded = T2SWatermarkState.load(path)

    assert reloaded == state
    assert reloaded.state_sha256() == digest, "state SHA must survive write/read"

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["state_sha256"] == digest
    assert on_disk["schema_version"] == T2S_STATE_SCHEMA_VERSION


def test_state_rejects_tampered_sha(tmp_path):
    state = _make_state()
    path = state.save(tmp_path / "state.json")

    record = json.loads(path.read_text(encoding="utf-8"))
    record["tau"] = 0.5  # change a field but keep the old digest
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="state_sha256 mismatch"):
        T2SWatermarkState.load(path)


def test_state_rejects_unknown_schema_version(tmp_path):
    state = _make_state()
    record = state.to_dict()
    record["schema_version"] = T2S_STATE_SCHEMA_VERSION + 1
    path = tmp_path / "state.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        T2SWatermarkState.load(path)


def test_canonical_json_is_deterministic():
    payload = {"b": 1, "a": {"d": 2, "c": [1, 2, 3]}}
    assert canonical_json_dumps(payload) == canonical_json_dumps(dict(reversed(list(payload.items()))))


def test_canonical_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_sha256({"x": float("nan")})
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_sha256({"x": [1.0, float("inf")]})


def test_canonical_matches_raven_repro():
    """The eval_bench_wm copy must not drift from raven_repro's implementation."""
    raven_root = BENCH_ROOT.parent / "raven_repro"
    if not raven_root.exists():
        pytest.skip("raven_repro not present")
    if str(raven_root) not in sys.path:
        sys.path.insert(0, str(raven_root))
    from raven.pairing_provenance import canonical_json_sha256 as raven_hash

    payload = {"z": 1, "a": "x", "nested": {"k": [1, 2], "j": True}, "n": None, "f": 0.5}
    assert canonical_json_sha256(payload) == raven_hash(payload)


# --------------------------------------------------------------------------
# 4. Detection semantics
# --------------------------------------------------------------------------


def _encode_latent_for_state(state: T2SWatermarkState, seed: int = 100) -> torch.Tensor:
    t2s_key, t2s_msg = state.build_marks()
    gen = torch.Generator().manual_seed(seed)
    z_key = t2s_key.encode(str_to_bits(state.expected_session_key_bits), state.master_key(), generator=gen)
    z_msg = t2s_msg.encode(str_to_bits(state.expected_message_bits), str_to_bits(state.expected_session_key_bits), generator=gen)

    latents = torch.zeros(state.latent_shape)
    latents[:, state.key_channels, :, :] = z_key.unsqueeze(0)
    latents[:, state.msg_channels, :, :] = z_msg.unsqueeze(0)
    return latents


def test_detection_succeeds_on_matching_state():
    state = _make_state()
    latents = _encode_latent_for_state(state)

    result = detect_from_reversed_latents(state, latents)

    assert result["detection_success"] is True
    assert result["score_true_key"] > result["score_control_key"]
    assert result["score_margin"] == pytest.approx(
        result["score_true_key"] - result["score_control_key"]
    )
    assert result["key_accuracy"] == pytest.approx(1.0)
    assert result["message_accuracy"] == pytest.approx(1.0)
    assert result["decision_rule"] == "score_true_key > score_control_key"
    assert result["score_direction"] == "higher_is_watermarked"


def test_wrong_state_does_not_reproduce_correct_result():
    """A mismatched watermark state must not verify like the right one."""
    right = _make_state(master_key_bits="0" * 8 + "1" * 8)
    wrong = _make_state(
        watermark_id="t2s-test-wrong",
        master_key_bits="1" * 8 + "0" * 8,
        expected_session_key_bits="10" * 8,
        expected_message_bits="01" * 128,
    )
    assert right.master_key_bits != wrong.master_key_bits

    latents = _encode_latent_for_state(right)

    good = detect_from_reversed_latents(right, latents)
    bad = detect_from_reversed_latents(wrong, latents)

    assert good["detection_success"] is True
    assert bad["score_margin"] < good["score_margin"]
    assert bad["recovered_session_key_bits"] != good["recovered_session_key_bits"]
    assert bad["message_accuracy"] < 0.9
    assert bad["state_sha256"] != good["state_sha256"]


def test_missing_expectations_report_null_not_zero():
    """Absent expected key/message must be null, never a false 0%."""
    state = _make_state(expected_session_key_bits=None, expected_message_bits=None)
    latents = _encode_latent_for_state(_make_state())

    result = detect_from_reversed_latents(state, latents)

    assert result["key_accuracy"] is None
    assert result["message_accuracy"] is None
    assert result["key_accuracy_status"] == "N/A"
    assert result["message_accuracy_status"] == "N/A"
    # Detection must still be possible without the expectations.
    assert isinstance(result["detection_success"], bool)
    assert result["recovered_message_bits"]


def test_bit_accuracy_is_na_not_zero_when_message_unknown():
    from utils.bit_accuracy import extract_bit_accuracy
    from utils.wm.t2s_provider import T2SProvider

    state = _make_state(expected_session_key_bits=None, expected_message_bits=None)
    latents = _encode_latent_for_state(_make_state())

    payload = T2SProvider.accuracies_for_state(state, latents)
    metric = extract_bit_accuracy(payload)

    assert metric.status == "N/A"
    assert metric.value is None


def test_detection_rejects_channel_mismatch():
    state = _make_state()
    with pytest.raises(ValueError, match="channel mismatch"):
        detect_from_reversed_latents(state, torch.zeros(1, 16, 64, 64))


# --------------------------------------------------------------------------
# 5. Shared-clean compatibility
# --------------------------------------------------------------------------


def test_encode_accepts_supplied_base_latent():
    """The supplied latent's magnitudes are the ones actually embedded."""
    mark = T2SMark(m=16, tau=0.674, latent_shape=KEY_SHAPE_SD21)
    bits = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(41))
    key = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(42))

    shared = torch.randn(KEY_SHAPE_SD21, generator=torch.Generator().manual_seed(43))
    encoded = mark.encode(bits, key, generator=torch.Generator().manual_seed(44), z=shared)

    # Tail-truncated sampling reuses |z| exactly, only re-placing and re-signing.
    assert torch.allclose(
        encoded.abs().flatten().sort().values, shared.abs().flatten().sort().values
    )
    assert torch.equal(mark.decode(encoded, key), bits.int())


def test_encode_rejects_wrong_shaped_base_latent():
    mark = T2SMark(m=16, tau=0.674, latent_shape=KEY_SHAPE_SD21)
    bits = torch.zeros(16, dtype=torch.int32)
    key = torch.zeros(16, dtype=torch.int32)
    with pytest.raises(ValueError, match="expected"):
        mark.encode(bits, key, z=torch.randn(3, 64, 64))


def test_provider_rejects_mismatched_shared_latent():
    """Fail closed instead of silently drawing a method-specific latent."""
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(latent_shape=(1, 4, 64, 64), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="refusing to substitute"):
        provider.new_sample(sample_seed=1, base_latent=torch.randn(1, 16, 64, 64))


def test_provider_uses_shared_latent_and_records_its_sha():
    from utils.canonical import tensor_sha256
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(latent_shape=(1, 4, 64, 64), device=torch.device("cpu"))
    shared = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(51))

    latents, state = provider.new_sample(sample_seed=7, base_latent=shared)

    assert state.base_latent_sha256 == tensor_sha256(shared.float().cpu())
    # Pre-injection magnitudes are preserved per channel group.
    for group in (state.key_channels, state.msg_channels):
        assert torch.allclose(
            latents[0, group].abs().flatten().sort().values,
            shared[0, group].abs().flatten().sort().values,
            atol=1e-6,
        )
    assert detect_from_reversed_latents(state, latents)["detection_success"] is True


def test_samples_are_independent():
    """Different samples must not share a complete base or watermarked latent."""
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(latent_shape=(1, 4, 64, 64), device=torch.device("cpu"))
    first_latents, first_state = provider.new_sample(sample_seed=1)
    second_latents, second_state = provider.new_sample(sample_seed=2)

    assert first_state.watermarked_latent_sha256 != second_state.watermarked_latent_sha256
    assert not torch.equal(first_latents, second_latents)
    assert first_state.expected_session_key_bits != second_state.expected_session_key_bits
    assert first_state.expected_message_bits != second_state.expected_message_bits
    assert first_state.provider_config_sha256 == second_state.provider_config_sha256


def test_fix_key_keeps_master_key_but_varies_session_state():
    """--t2s_fix_key is an account-level key, not a shared per-sample latent."""
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(latent_shape=(1, 4, 64, 64), device=torch.device("cpu"),
                           t2s_fix_key=True, seed=5)
    _, first = provider.new_sample(sample_seed=1)
    _, second = provider.new_sample(sample_seed=2)

    assert first.master_key_bits == second.master_key_bits
    assert first.expected_session_key_bits != second.expected_session_key_bits
    assert first.watermarked_latent_sha256 != second.watermarked_latent_sha256


def test_get_wm_latents_produces_a_new_sample_each_call():
    """Guards the historical bug: one latent reused for every prompt."""
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(latent_shape=(1, 4, 64, 64), device=torch.device("cpu"))
    first = provider.get_wm_latents()
    second = provider.get_wm_latents()
    assert not torch.equal(first["zT_torch"], second["zT_torch"])


def test_get_accuracies_requires_no_prior_state_for_standalone_path():
    """Standalone verification must not need get_wm_latents() on the instance."""
    state = _make_state()
    latents = _encode_latent_for_state(state)
    result = detect_from_reversed_latents(state, latents)
    assert result["detection_success"] is True


def test_bits_str_round_trip():
    bits = torch.randint(0, 2, (32,), generator=torch.Generator().manual_seed(1))
    assert torch.equal(str_to_bits(bits_to_str(bits)), bits.int())
