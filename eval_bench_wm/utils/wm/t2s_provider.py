"""T2SMark provider.

Upstream reference: https://github.com/0xD009/T2SMark
Pinned comparison commit: 0c1fbfd50fcd1fba135477a2c016e284d5d7914d
Upstream files compared: ``src/t2s.py``, ``run.py``, ``run_sd35.py``,
``option.py``, ``src/utils.py``, ``src/inversion/inverse_stable_diffusion.py``.

This module holds the single authoritative implementation of the T2S encoder,
decoder, and detector score. ``run_watermark.py`` and ``run_verification.py``
both dispatch here; neither reimplements any part of it.

RNG modes
---------
``official_compatible`` (default) reproduces upstream's *whole* generation RNG
lifecycle, not merely the encoder given fixed inputs. Per sample it reseeds the
process-global RNG with ``set_random_seed(seed + sample_index)`` and then draws
from the global CPU stream in upstream's exact order:

    master_key, message   (only when --t2s_fix_key is off)
    session key
    z_k randn, z_k noise-sign randint
    z_b randn, z_b noise-sign randint

``raven_deterministic`` instead uses one explicit CPU ``torch.Generator`` per
sample and never touches process-global RNG state. It is a RAVEN provenance
adaptation, not an upstream behaviour.

The distinction is about global RNG side effects, and it is what the modes are
tested on. In a process where nothing else consumes the global CPU stream, both
modes happen to draw the same values for a given ``sample_seed``, because
``torch.manual_seed(s)`` and ``torch.Generator().manual_seed(s)`` seed the same
CPU MT19937 stream and the draw order is identical. That coincidence is not the
guarantee. Upstream *advances the process-global RNG*, so the diffusion
pipeline's own subsequent sampling inherits that state; ``raven_deterministic``
deliberately leaves the global RNG untouched, so an end-to-end run diverges from
upstream. Only ``official_compatible`` is claimed to reproduce upstream end to
end; ``raven_deterministic`` must never be described as upstream-exact.

Deliberate differences from upstream, and why each is necessary
---------------------------------------------------------------
1. Key-derived PRNG stays on CPU (``torch.Generator()``), exactly as upstream.
   Upstream is CPU-only here; an earlier local revision used
   ``torch.Generator(device=device)``, which silently changed the key -> pattern
   mapping because CUDA and CPU Philox produce different streams for the same
   seed. Restored to upstream. The derived tensors are moved to the target
   device afterwards, which is device portability and does not alter the
   mapping.
2. The watermark mask and the decode arithmetic are built on CPU. Upstream
   builds the mask on CPU and the arithmetic on CUDA. Doing both on CPU keeps a
   key's pattern and a detector's score bit-identical across devices, which
   standalone verification requires (the verifying process may not have the same
   GPU as the generating process). The values are mathematically identical.
3. ``raven_deterministic`` exists in addition to ``official_compatible``; see
   above. Only ``official_compatible`` is claimed to match upstream bit-for-bit.

Everything else -- ``r``, ``k``, ``noise_size``, tail/central magnitude
selection, repeated-bit layout, codeword signs, ``binlist2int``, the master /
session / fake key and message lifecycle, and the 4-channel and 16-channel
splits -- follows upstream.

Detection
---------
Upstream's *formal* evaluation is a cohort ROC: it pools ``norm1_no_w`` as
negatives and ``norm1_w`` as positives across the whole run and reports AUC plus
TPR at FPR < 1e-6 (``run.py`` lines 140-159). It has no per-image binary rule.
The per-image ``score_true_key > score_control_key`` test implemented here is a
RAVEN deployment extension named ``paired_key_comparison``; it is neither an
upstream rule nor a calibrated TPR at a target FPR.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from functools import reduce
from pathlib import Path

import torch
from scipy.stats import norm

from utils.canonical import canonical_json_dumps, canonical_json_sha256
from utils.utils import set_random_seed

from .wm_provider import WmProvider


T2S_STATE_SCHEMA_VERSION = 2

#: ``official_compatible`` reproduces upstream's full generation RNG lifecycle
#: bit-for-bit. ``raven_deterministic`` is a RAVEN provenance adaptation using an
#: explicit per-sample generator; it is NOT bit-exact with upstream.
T2S_RNG_MODES = ("official_compatible", "raven_deterministic")

#: Upstream inversion (``naive_forward_diffusion`` in
#: ``src/inversion/inverse_stable_diffusion.py``) versus the benchmark's generic
#: diffusers ``DDIMInverseScheduler`` path. These are NOT equivalent and are
#: never substituted for one another; see ``t2s_inversion.py``.
T2S_INVERSION_MODES = ("t2s_official", "benchmark_ddim")

#: RAVEN deployment extension, NOT an upstream rule. Upstream's formal
#: evaluation is a cohort ROC (AUC + TPR at FPR < 1e-6) over pooled control-key
#: and true-key scores; it defines no per-image binary decision. This per-image
#: comparison is therefore named explicitly and is not a calibrated TPR at a
#: target FPR.
T2S_SCORE_DIRECTION = "higher_is_watermarked"
T2S_DECISION_RULE = "paired_key_comparison"
T2S_DECISION_RULE_EXPRESSION = "score_true_key > score_control_key"
T2S_DECISION_RULE_PROVENANCE = "raven_deployment_extension"
T2S_OFFICIAL_EVALUATION = "cohort_roc_auc_and_tpr_at_fpr_1e-6"


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--t2s_channels", type=int, default=0, choices=[0, 4, 16], help="0 auto-detects from latent shape")
parser.add_argument("--t2s_key_channel_idx", type=int, default=0)
parser.add_argument("--t2s_key_length", type=int, default=16)
parser.add_argument("--t2s_msg_length", type=int, default=256)
parser.add_argument("--t2s_tau", type=float, default=0.674)
parser.add_argument("--t2s_fix_key", action="store_true", default=False,
                    help="Upstream --fix_key: fix BOTH the master key and the message across "
                         "samples to simulate a single account. Only the session key and the "
                         "base latent stay per-sample.")
parser.add_argument("--t2s_rng_mode", type=str, default="official_compatible",
                    choices=list(T2S_RNG_MODES),
                    help="official_compatible reproduces upstream's full generation RNG "
                         "lifecycle bit-for-bit; raven_deterministic uses an explicit "
                         "per-sample generator and is NOT bit-exact with upstream")
parser.add_argument("--t2s_inversion_mode", type=str, default="t2s_official",
                    choices=list(T2S_INVERSION_MODES),
                    help="t2s_official reproduces upstream naive_forward_diffusion; "
                         "benchmark_ddim uses the generic diffusers DDIMInverseScheduler path")
parser.add_argument("--t2s_num_inversion_steps", type=int, default=10,
                    help="Upstream option.py default is 10")
parser.add_argument("--t2s_state_dir", type=str, default=None,
                    help="Directory to write per-sample portable watermark state JSON")


def bits_to_str(bits: torch.Tensor) -> str:
    return "".join(str(int(bit)) for bit in bits.detach().cpu().flatten().tolist())


def str_to_bits(text: str) -> torch.Tensor:
    if any(char not in "01" for char in text):
        raise ValueError("bit string must contain only '0' and '1'")
    return torch.tensor([int(char) for char in text], dtype=torch.int32)


class T2SMark:
    """Tail-truncated sampling encoder/decoder. Mirrors upstream ``src/t2s.py``."""

    def __init__(self,
                 m: int,
                 tau: float,
                 latent_shape: typing.Tuple[int, int, int],
                 device: torch.device = torch.device("cpu")):
        self.latent_shape = tuple(latent_shape)
        self.n = reduce(lambda x, y: x * y, self.latent_shape, 1)
        self.m = m
        self.r = int(2 * norm.cdf(-tau) * self.n / m)
        self.k = self.m * self.r
        self.noise_size = self.n - self.k
        self.device = device
        # Upstream ``src/t2s.py`` line 13: torch.Generator() -- CPU, no device arg.
        self.prng = torch.Generator()

        if self.r <= 0 or self.k > self.n:
            raise ValueError(
                f"Invalid T2SMark parameters: m={m}, tau={tau}, latent_shape={latent_shape}, r={self.r}, k={self.k}"
            )

    def binlist2int(self, binlist):
        res = reduce(lambda x, y: x * 2 + y, binlist)
        if isinstance(binlist, torch.Tensor):
            return int(res.item())
        return int(res)

    def key_pattern(self, key: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Authoritative key -> (sign vector, support) mapping.

        Both encode and decode derive their pattern here so the two can never
        disagree. Upstream order is preserved: ``randint`` is drawn before
        ``randperm`` from the same freshly seeded CPU generator, so changing the
        order or the device would change every key's pattern.
        """
        self.prng.manual_seed(self.binlist2int(key))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng)[: self.k]
        return v_value, v_support

    def _watermark_mask(self, v_support: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.n, dtype=torch.bool)
        mask[v_support] = True
        return mask

    def encode(self,
               bits: torch.Tensor,
               key: torch.Tensor,
               generator: typing.Optional[torch.Generator] = None,
               z: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed ``bits`` under ``key`` into a tail-truncated latent.

        ``generator`` must be a CPU generator. ``None`` uses the global CPU RNG,
        which is upstream's behaviour.

        ``z`` optionally supplies the Gaussian source whose order statistics are
        reused. Upstream always draws it fresh; supplying it lets a
        cross-watermark cohort feed the canonical shared TR base latent instead
        of a method-specific replacement (see ``raven-shared-clean``). The
        distribution is unchanged because that latent is standard normal too.
        """
        if z is None:
            z = torch.randn(self.latent_shape, generator=generator)
        elif tuple(z.shape) != self.latent_shape:
            raise ValueError(
                f"supplied base latent has shape {tuple(z.shape)}, expected {self.latent_shape}"
            )
        z = z.detach().float().cpu().flatten()

        v_value, v_support = self.key_pattern(key)
        watermark_mask = self._watermark_mask(v_support)

        repeated_bits = (1 - 2 * bits.detach().cpu()).repeat(self.r).float()
        codeword = repeated_bits * v_value

        tail = torch.topk(z.abs(), k=self.k, dim=0, largest=True, sorted=False)
        central = torch.topk(z.abs(), k=self.noise_size, dim=0, largest=False, sorted=False)

        z_w = torch.zeros(self.n)
        z_w[watermark_mask] = tail.values * codeword
        noise_signs = torch.randint(0, 2, (self.noise_size,), generator=generator).float() * 2 - 1
        z_w[~watermark_mask] = central.values * noise_signs
        return z_w.reshape(self.latent_shape).to(self.device)

    def decode(self,
               reversed_noise: torch.Tensor,
               key: torch.Tensor,
               detection: bool = False):
        """Recover bits under ``key``; optionally also return the L1 vote norm."""
        v_value, v_support = self.key_pattern(key)
        watermark_mask = self._watermark_mask(v_support)

        flat = reversed_noise.detach().flatten().float().cpu()
        if flat.numel() != self.n:
            raise ValueError(
                f"T2SMark.decode expected {self.n} elements for latent_shape={self.latent_shape}, "
                f"got {flat.numel()}"
            )

        watermarked_vec = flat[watermark_mask] * v_value
        votes = watermarked_vec.reshape(self.r, self.m).sum(dim=0)
        bits = (votes < 0).int()
        if detection:
            return bits, torch.norm(votes.flatten(), p=1).item()
        return bits


def channel_layout(num_channels: int, key_channel_idx: int = 0) -> typing.Tuple[typing.List[int], typing.List[int]]:
    """Return (key_channels, msg_channels).

    4-channel (SD2.1, upstream ``run.py``): one key channel, three message
    channels. 16-channel (SD3/SD3.5, upstream ``run_sd35.py`` line 16-18):
    channels 0-3 carry the key, the remaining twelve carry the message.
    """
    if num_channels == 4:
        if not 0 <= key_channel_idx < 4:
            raise ValueError("--t2s_key_channel_idx must be in [0, 3] for 4-channel T2S")
        key_channels = [key_channel_idx]
        msg_channels = [i for i in range(4) if i != key_channel_idx]
    elif num_channels == 16:
        key_channels = [0, 1, 2, 3]
        msg_channels = [i for i in range(16) if i not in key_channels]
    else:
        raise ValueError(f"T2S supports only 4-channel or 16-channel latents, got {num_channels}")
    return key_channels, msg_channels


@dataclasses.dataclass(frozen=True)
class T2SWatermarkState:
    """Portable, canonical-JSON watermark state for standalone verification.

    Everything needed to verify a suspect image in a brand-new process lives
    here. Notably it does NOT reference the generating provider instance, the
    original latent, or any file ordering.
    """

    watermark_id: str
    latent_shape: typing.List[int]
    key_channels: typing.List[int]
    msg_channels: typing.List[int]
    key_length: int
    msg_length: int
    tau: float
    master_key_bits: str
    rng_mode: str
    inversion_mode: str
    num_inversion_steps: int
    provider_config_sha256: str
    schema_version: int = T2S_STATE_SCHEMA_VERSION
    # Optional expectations. Absent means "unknown", which must report
    # unavailable/null accuracy -- never a false 0%.
    expected_session_key_bits: typing.Optional[str] = None
    expected_message_bits: typing.Optional[str] = None
    # Generation provenance, required to reproduce the inversion faithfully.
    model_id: typing.Optional[str] = None
    model_revision: typing.Optional[str] = None
    scheduler: typing.Optional[str] = None
    num_inference_steps: typing.Optional[int] = None
    guidance_scale: typing.Optional[float] = None
    resolution: typing.Optional[int] = None
    sample_seed: typing.Optional[int] = None
    base_latent_sha256: typing.Optional[str] = None
    watermarked_latent_sha256: typing.Optional[str] = None
    prompt_sha256: typing.Optional[str] = None
    image_sha256: typing.Optional[str] = None

    def payload(self) -> typing.Dict[str, typing.Any]:
        """State fields without the self-referential digest."""
        return dataclasses.asdict(self)

    def state_sha256(self) -> str:
        return canonical_json_sha256(self.payload())

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        record = self.payload()
        record["state_sha256"] = self.state_sha256()
        return record

    @classmethod
    def from_dict(cls, record: typing.Mapping[str, typing.Any]) -> "T2SWatermarkState":
        record = dict(record)
        if "state_sha256" not in record:
            raise ValueError(
                "T2S state is missing state_sha256; an unsigned state cannot be "
                "verified and is rejected"
            )
        declared = record.pop("state_sha256")
        if not isinstance(declared, str) or len(declared) != 64:
            raise ValueError(f"T2S state_sha256 must be a 64-char hex digest, got {declared!r}")

        version = record.get("schema_version")
        if version != T2S_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported T2S state schema_version={version!r}, "
                f"this build reads {T2S_STATE_SCHEMA_VERSION}"
            )

        known = {field.name for field in dataclasses.fields(cls)}
        unexpected = set(record) - known
        if unexpected:
            raise ValueError(f"unexpected T2S state fields: {sorted(unexpected)}")
        missing = {
            field.name
            for field in dataclasses.fields(cls)
            if field.default is dataclasses.MISSING
        } - set(record)
        if missing:
            raise ValueError(f"missing required T2S state fields: {sorted(missing)}")

        state = cls(**record)
        # Fail closed: the digest must survive the write/read round trip.
        if declared != state.state_sha256():
            raise ValueError(
                "T2S state_sha256 mismatch after reload: "
                f"declared={declared} recomputed={state.state_sha256()}"
            )
        return state

    def save(self, path: typing.Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json_dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: typing.Union[str, Path]) -> "T2SWatermarkState":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- derived helpers -------------------------------------------------

    def master_key(self) -> torch.Tensor:
        return str_to_bits(self.master_key_bits)

    def control_key(self) -> torch.Tensor:
        """Upstream ``fake_key = 1 - master_key`` (run.py line 97)."""
        return 1 - self.master_key()

    def build_marks(self, device: torch.device = torch.device("cpu")
                    ) -> typing.Tuple[T2SMark, T2SMark]:
        _, _, height, width = self.latent_shape
        t2s_key = T2SMark(m=self.key_length, tau=self.tau,
                          latent_shape=(len(self.key_channels), height, width), device=device)
        t2s_msg = T2SMark(m=self.msg_length, tau=self.tau,
                          latent_shape=(len(self.msg_channels), height, width), device=device)
        return t2s_key, t2s_msg


def detect_from_reversed_latents(state: T2SWatermarkState,
                                 reversed_latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
    """Authoritative T2S detector scoring. The only implementation.

    ``score_true_key`` and ``score_control_key`` are upstream's ``norm1_w`` and
    ``norm1_no_w``. Upstream consumes them as a *cohort* ROC (AUC and TPR at
    FPR < 1e-6 over the pooled run) and defines no per-image decision. The
    per-image ``score_true_key > score_control_key`` test reported here is a
    RAVEN deployment extension (``paired_key_comparison``), not an upstream rule
    and not a calibrated TPR at a target FPR.
    """
    if reversed_latents.dim() == 3:
        reversed_latents = reversed_latents.unsqueeze(0)
    if reversed_latents.shape[0] != 1:
        raise ValueError(f"T2S detection expects batch size 1, got {reversed_latents.shape[0]}")

    expected_channels = state.latent_shape[1]
    if reversed_latents.shape[1] != expected_channels:
        raise ValueError(
            f"channel mismatch: state expects {expected_channels}, got {reversed_latents.shape[1]}"
        )

    t2s_key, t2s_msg = state.build_marks()
    master_key = state.master_key()
    control_key = state.control_key()

    key_plane = reversed_latents[0, state.key_channels, :, :]
    msg_plane = reversed_latents[0, state.msg_channels, :, :]

    _, score_control_key = t2s_key.decode(key_plane, control_key, detection=True)
    recovered_key, score_true_key = t2s_key.decode(key_plane, master_key, detection=True)
    recovered_msg = t2s_msg.decode(msg_plane, recovered_key)

    # Missing expectations must stay null rather than collapsing to 0%.
    key_accuracy = None
    if state.expected_session_key_bits is not None:
        expected_key = str_to_bits(state.expected_session_key_bits)
        key_accuracy = (recovered_key == expected_key).float().mean().item()

    message_accuracy = None
    if state.expected_message_bits is not None:
        expected_msg = str_to_bits(state.expected_message_bits)
        message_accuracy = (recovered_msg == expected_msg).float().mean().item()

    return {
        "watermark_id": state.watermark_id,
        "score_true_key": score_true_key,
        "score_control_key": score_control_key,
        "score_margin": score_true_key - score_control_key,
        "score_direction": T2S_SCORE_DIRECTION,
        "decision_rule": T2S_DECISION_RULE,
        "decision_rule_expression": T2S_DECISION_RULE_EXPRESSION,
        "decision_rule_provenance": T2S_DECISION_RULE_PROVENANCE,
        "official_evaluation": T2S_OFFICIAL_EVALUATION,
        "detection_success": bool(score_true_key > score_control_key),
        "recovered_session_key_bits": bits_to_str(recovered_key),
        "recovered_message_bits": bits_to_str(recovered_msg),
        "key_accuracy": key_accuracy,
        "message_accuracy": message_accuracy,
        "key_accuracy_status": "OK" if key_accuracy is not None else "N/A",
        "message_accuracy_status": "OK" if message_accuracy is not None else "N/A",
        "state_sha256": state.state_sha256(),
    }


class T2SProvider(WmProvider):
    def __init__(
        self,
        t2s_channels: int = 0,
        t2s_key_channel_idx: int = 0,
        t2s_key_length: int = 16,
        t2s_msg_length: int = 256,
        t2s_tau: float = 0.674,
        t2s_fix_key: bool = False,
        t2s_rng_mode: str = "official_compatible",
        t2s_inversion_mode: str = "t2s_official",
        t2s_num_inversion_steps: int = 10,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.t2s_channels = t2s_channels or self.num_channels
        if self.t2s_channels not in (4, 16):
            raise ValueError(f"T2S supports only 4-channel or 16-channel latents, got {self.t2s_channels}")
        if self.t2s_channels != self.num_channels:
            raise ValueError(
                f"--t2s_channels={self.t2s_channels} does not match latent channels={self.num_channels}"
            )
        if self.batch_size != 1:
            raise ValueError("T2SProvider currently supports batch_size=1")
        if t2s_inversion_mode not in T2S_INVERSION_MODES:
            raise ValueError(
                f"unknown --t2s_inversion_mode={t2s_inversion_mode!r}, expected one of {T2S_INVERSION_MODES}"
            )
        if t2s_rng_mode not in T2S_RNG_MODES:
            raise ValueError(
                f"unknown --t2s_rng_mode={t2s_rng_mode!r}, expected one of {T2S_RNG_MODES}"
            )

        self.key_channel_idx = t2s_key_channel_idx
        self.key_length = t2s_key_length
        self.msg_length = t2s_msg_length
        self.tau = t2s_tau
        self.fix_key = t2s_fix_key
        self.rng_mode = t2s_rng_mode
        self.inversion_mode = t2s_inversion_mode
        self.num_inversion_steps = t2s_num_inversion_steps
        self.seed = seed
        self.sample_counter = 0

        _, _, height, width = self.latent_shape
        self.key_channels, self.msg_channels = channel_layout(self.t2s_channels, self.key_channel_idx)

        self.t2s_key = T2SMark(m=self.key_length, tau=self.tau,
                               latent_shape=(len(self.key_channels), height, width), device=self.device)
        self.t2s_msg = T2SMark(m=self.msg_length, tau=self.tau,
                               latent_shape=(len(self.msg_channels), height, width), device=self.device)

        # Account-level secrets. Upstream ``run.py`` lines 57-60 draws BOTH the
        # master key and the message once, before the loop, when --fix_key is
        # set; only the session key is redrawn per sample.
        self.fixed_master_key = None
        self.fixed_msg = None
        if self.fix_key:
            if self.rng_mode == "official_compatible":
                set_random_seed(self.seed)
                self.fixed_master_key = torch.randint(0, 2, (self.key_length,))
                self.fixed_msg = torch.randint(0, 2, (self.msg_length,))
            else:
                gen = torch.Generator()
                gen.manual_seed(self.seed)
                self.fixed_master_key = torch.randint(0, 2, (self.key_length,), generator=gen)
                self.fixed_msg = torch.randint(0, 2, (self.msg_length,), generator=gen)

        self.state: typing.Optional[T2SWatermarkState] = None
        self.wm_latents: typing.Optional[torch.Tensor] = None

    def get_wm_type(self) -> str:
        return "T2S"

    def provider_config(self) -> typing.Dict[str, typing.Any]:
        return {
            "wm_type": "T2S",
            "channels": self.t2s_channels,
            "key_channel_idx": self.key_channel_idx,
            "key_channels": list(self.key_channels),
            "msg_channels": list(self.msg_channels),
            "key_length": self.key_length,
            "msg_length": self.msg_length,
            "tau": self.tau,
            "fix_key": bool(self.fix_key),
            "rng_mode": self.rng_mode,
            "latent_shape": list(self.latent_shape),
            "inversion_mode": self.inversion_mode,
            "num_inversion_steps": self.num_inversion_steps,
            "upstream_commit": "0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
            "state_schema_version": T2S_STATE_SCHEMA_VERSION,
        }

    def provider_config_sha256(self) -> str:
        return canonical_json_sha256(self.provider_config())

    def new_sample(self,
                   sample_seed: typing.Optional[int] = None,
                   watermark_id: typing.Optional[str] = None,
                   base_latent: typing.Optional[torch.Tensor] = None,
                   **provenance) -> typing.Tuple[torch.Tensor, T2SWatermarkState]:
        """Build one independent watermarked latent plus its portable state.

        Every call draws a fresh base latent, session key and message, so no two
        samples share a complete base or watermarked latent. Only the master key
        is reused when ``--t2s_fix_key`` is set, which is the account-level key.

        ``base_latent`` optionally supplies the canonical shared TR base latent
        for a cross-watermark cohort. It is split across the key and message
        channel groups and used as the tail-truncated source, so the
        pre-injection latent is byte-identical to the shared one. It must have
        exactly this provider's latent shape; anything else fails closed rather
        than silently sampling a method-specific replacement.
        """
        if sample_seed is None:
            sample_seed = self.seed + self.sample_counter
        if watermark_id is None:
            watermark_id = f"t2s-{sample_seed:08d}"

        if self.rng_mode == "official_compatible":
            # Upstream run.py line 89: reseed the process-global RNG per sample,
            # then draw from the global CPU stream in upstream's exact order.
            set_random_seed(int(sample_seed))
            gen = None
        else:
            # RAVEN provenance mode: one explicit CPU generator per sample, no
            # process-global RNG side effects. NOT bit-exact with upstream.
            gen = torch.Generator()
            gen.manual_seed(int(sample_seed))

        # Upstream order (run.py lines 91-94): master_key, msg, then session key.
        if self.fix_key:
            master_key = self.fixed_master_key.clone()
            msg = self.fixed_msg.clone()
        else:
            master_key = torch.randint(0, 2, (self.key_length,), generator=gen)
            msg = torch.randint(0, 2, (self.msg_length,), generator=gen)

        session_key = torch.randint(0, 2, (self.key_length,), generator=gen)

        from utils.canonical import tensor_sha256

        base_latent_sha256 = None
        z_key_src = z_msg_src = None
        if base_latent is not None:
            if tuple(base_latent.shape) != tuple(self.latent_shape):
                raise ValueError(
                    f"shared base latent has shape {tuple(base_latent.shape)}, "
                    f"expected {tuple(self.latent_shape)}; refusing to substitute a "
                    "method-specific latent"
                )
            base_cpu = base_latent.detach().float().cpu()
            base_latent_sha256 = tensor_sha256(base_cpu)
            z_key_src = base_cpu[0, self.key_channels, :, :]
            z_msg_src = base_cpu[0, self.msg_channels, :, :]

        z_key = self.t2s_key.encode(session_key, master_key, generator=gen, z=z_key_src)
        z_msg = self.t2s_msg.encode(msg, session_key, generator=gen, z=z_msg_src)

        latents = torch.zeros(self.latent_shape, device=self.device, dtype=self.dtype)
        latents[:, self.key_channels, :, :] = z_key.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        latents[:, self.msg_channels, :, :] = z_msg.to(device=self.device, dtype=self.dtype).unsqueeze(0)

        state = T2SWatermarkState(
            watermark_id=watermark_id,
            latent_shape=list(self.latent_shape),
            key_channels=list(self.key_channels),
            msg_channels=list(self.msg_channels),
            key_length=self.key_length,
            msg_length=self.msg_length,
            tau=self.tau,
            master_key_bits=bits_to_str(master_key),
            expected_session_key_bits=bits_to_str(session_key),
            expected_message_bits=bits_to_str(msg),
            rng_mode=self.rng_mode,
            inversion_mode=self.inversion_mode,
            num_inversion_steps=self.num_inversion_steps,
            provider_config_sha256=self.provider_config_sha256(),
            sample_seed=int(sample_seed),
            base_latent_sha256=base_latent_sha256,
            watermarked_latent_sha256=tensor_sha256(latents),
            **provenance,
        )

        self.sample_counter += 1
        self.state = state
        self.wm_latents = latents
        self.master_key = master_key
        self.session_key = session_key
        self.msg = msg
        return latents, state

    def invert_images(self,
                      images,
                      pipe_provider_target=None,
                      num_inference_steps: typing.Optional[int] = None,
                      **kwargs) -> typing.Dict[str, typing.Any]:
        """Provider-owned inversion so the configured T2S protocol is honoured.

        ``imprint_utils.validate`` prefers this over the generic pipe inversion,
        which keeps ``t2s_official`` and ``benchmark_ddim`` from being mixed.
        """
        from .t2s_inversion import invert_image

        if pipe_provider_target is None:
            raise ValueError("T2SProvider.invert_images requires pipe_provider_target")
        zT = invert_image(
            pipe_provider_target,
            images,
            inversion_mode=self.inversion_mode,
            num_inversion_steps=self.num_inversion_steps,
            benchmark_num_inference_steps=num_inference_steps,
        )
        return {"zT_torch": zT}

    # ---- compatibility wrappers -----------------------------------------

    def get_wm_latents(self, sample_seed: typing.Optional[int] = None, **kwargs) -> typing.Dict[str, typing.Any]:
        """Compatibility wrapper over :meth:`new_sample`.

        Each call produces a NEW independent sample. Callers that need the
        portable state should use :meth:`new_sample` directly.
        """
        latents, state = self.new_sample(sample_seed=sample_seed)
        return {
            "zT_torch": latents,
            "message_bits_str_list": [state.expected_message_bits],
            "t2s_state": state,
        }

    def get_accuracies(self, reversed_latents_w: torch.Tensor) -> typing.Dict[str, typing.Any]:
        """Compatibility wrapper over :func:`detect_from_reversed_latents`."""
        if self.state is None:
            raise RuntimeError(
                "T2SProvider.get_accuracies() needs a sample state. Call new_sample()/"
                "get_wm_latents() first, or use detect_from_reversed_latents(state, latents) "
                "for standalone verification."
            )
        return self.accuracies_for_state(self.state, reversed_latents_w)

    @staticmethod
    def accuracies_for_state(state: T2SWatermarkState,
                             reversed_latents_w: torch.Tensor) -> typing.Dict[str, typing.Any]:
        """Detector result mapped onto the benchmark's result schema."""
        result = detect_from_reversed_latents(state, reversed_latents_w)

        msg_acc = result["message_accuracy"]
        key_acc = result["key_accuracy"]
        key_display = f"{key_acc:.5f}" if key_acc is not None else "N/A"
        msg_display = f"{msg_acc:.5f}" if msg_acc is not None else "N/A"

        payload = {
            # bit_accuracies stays None when no expected message exists so the
            # shared extractor reports N/A instead of a false 0%.
            "bit_accuracies": None if msg_acc is None else [msg_acc],
            "accuracies": None if msg_acc is None else [msg_acc],
            "message_bits_str_list": [result["recovered_message_bits"]],
            "value": result["score_true_key"],
            "detection_success": result["detection_success"],
            "log_message": (
                f"T2S key_acc: {key_display}; msg_acc: {msg_display}; "
                f"score_true_key: {result['score_true_key']:.5f}; "
                f"score_control_key: {result['score_control_key']:.5f}; "
                f"margin: {result['score_margin']:.5f} "
                f"(RAVEN paired_key_comparison deployment extension; "
                f"upstream evaluates a cohort ROC, not a per-image rule; not TPR@1%FPR)"
            ),
            "t2s_key_accuracy": key_acc,
            "t2s_score_true_key": result["score_true_key"],
            "t2s_score_control_key": result["score_control_key"],
            "t2s_score_margin": result["score_margin"],
            "t2s_score_direction": result["score_direction"],
            "t2s_decision_rule": result["decision_rule"],
        }
        payload.update({key: value for key, value in result.items() if key not in payload})
        return payload
