"""RingID provider — the single authoritative RingID implementation in RAVEN.

Official reference (frozen):
    https://github.com/showlab/RingID
    commit 45631a59aecd7d63ccdb640aaaf3e616fdb89fb9
    (``utils.py``, ``verify.py``, ``identify.py``, ``inverse_stable_diffusion.py``)
    Paper: *RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key
    Identification*, ECCV 2024.

Everything algorithmic lives here:

* rounder-ring / circle masks (official ``RounderRingMask`` + ``ring_mask``)
* candidate keybook construction in official ``itertools.product`` order
* lossless-imprinting correction ``fix_gt`` and the spatial (time) shift
* per-sample watermark injection into a *per-sample* complete initial latent
* official per-channel masked complex L1 and the channel-wise **minimum**
* the canonical verification score and multi-key argmin/top-k identification
* the RingID-specific inversion adapter (official ``forward_diffusion``)
* key/profile compatibility validation against a persisted bundle

Runners (``run_watermark.py``, ``run_verify_watermark.py``) only parse a CLI,
enumerate inputs, call provider methods and serialize results. They must never
re-implement any of the above.

Three workflows are supported and never conflated:

``paper_eval_verification``
    matched watermarked positives + non-watermarked negatives for ONE key →
    canonical score → cohort ROC → AUC and TPR at the requested FPR.
``identify``
    traverse the declared candidate keybook and return the argmin key,
    exactly as official ``identify.py`` does.
``verify`` (deployment extension)
    fixed, provenance-bound threshold applied to individual suspect images.
    This is a RAVEN extension, *not* the official paper protocol.

Two shift semantics are kept explicitly distinct and are never both called
"official parity":

``official_code_exact``
    the released code: the ``* args.time_shift_factor`` multiplication is
    commented out, so the released behaviour is an **unscaled** shift and
    ``--time_shift_factor`` is an upstream no-op.
``paper_described_shift``
    the paper's described scaling by eta (~0.8-0.9). Ablation only.
"""

from __future__ import annotations

import argparse
import itertools
import random
import typing

import numpy as np
import torch

from . import rid_bundle
from .ddim_inversion import official_forward_diffusion
from .rid_bundle import (
    OFFICIAL_RINGID_COMMIT,
    OFFICIAL_RINGID_REPO,
    RidBundle,
    RidBundleError,
)
from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL


# ---------------------------------------------------------------------------
# Frozen official constants (utils.py of the reference commit)
# ---------------------------------------------------------------------------

RADIUS = 14
RADIUS_CUTOFF = 3
ANCHOR_X_OFFSET = 0
ANCHOR_Y_OFFSET = 0     # 1 = not correct, 0 = correct
USE_ROUNDER_RING = True

HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [3]
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)

#: The canonical RingID detector score. Lower L1 distance means "more likely
#: watermarked", so the canonical score negates it and is higher-is-watermarked.
RID_SCORE_DEFINITION = "rid_neg_channel_min_complex_l1"

#: Upstream ``--watermark_seed`` (default 5) is parsed by both official scripts
#: but never read: the released verify.py/identify.py seed everything through
#: ``set_random_seed(args.general_seed)``. Recorded so nobody can present it as
#: an effective knob.
OFFICIAL_UNUSED_ARGS = ("watermark_seed",)

#: In the released code the spatial-shift scale factor multiplication is
#: commented out; the effective released behaviour is factor 1.0.
SHIFT_SEMANTICS = ("official_code_exact", "paper_described_shift")

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

KEY_RNG_DTYPES = {"float32": torch.float32, "float16": torch.float16}

#: ``utils.set_random_seed`` of the reference commit, recorded by name in the
#: bundle so the RNG lifecycle is auditable.
RNG_ALGORITHM = "official_ringid_set_random_seed_torch_manual_seed"

#: Official ``verify.py`` uses candidate 628 as its single verification example.
DEFAULT_KEY_INDEX = 628


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

#: Immutable official Stable Diffusion 2.1 profile (Issue #3 §4). Generic RAVEN
#: parser defaults must not silently override any of these values; an explicit
#: CLI override is recorded and downgrades the run to a labelled ablation.
RID_OFFICIAL_SD21_PROFILE: typing.Dict[str, typing.Any] = {
    # generation
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "model_revision": "fp16",
    "rid_torch_dtype": "float16",
    "scheduler_target": "DPM",
    "resolution": 512,
    "num_inference_steps_target": 50,
    "guidance_scale_target": 7.5,
    # watermark identity
    "ring_width": 1,
    "quantization_levels": 2,
    "ring_value_range": 64,
    "assigned_keys": -1,
    "fix_gt": 1,
    "time_shift": 1,
    "time_shift_factor": 1.0,
    "rid_shift_semantics": "official_code_exact",
    "rid_key_seed": 42,          # official ``--general_seed`` default
    "rid_key_rng_device": "cpu",
    "rid_key_rng_dtype": "float32",
    # detection
    "channel_min": 1,
    "rid_inversion_prompt": "",
    "rid_inversion_guidance": 1.0,
    "rid_inversion_steps": 50,
    "rid_vae_sample": False,     # official ``get_image_latents(..., sample=False)``
    "rid_vae_scaling_factor": 0.18215,
    "rid_target_fpr": 0.01,
}

#: Same official code path but with the paper-described scaled spatial shift.
#: Explicitly NOT released-code parity; every result is labelled as an ablation.
RID_PAPER_SHIFT_PROFILE: typing.Dict[str, typing.Any] = dict(
    RID_OFFICIAL_SD21_PROFILE, rid_shift_semantics="paper_described_shift"
)

RID_PROFILES = {
    "official_sd21": RID_OFFICIAL_SD21_PROFILE,
    "paper_shift_ablation": RID_PAPER_SHIFT_PROFILE,
    # "legacy" applies nothing and never claims official parity.
    "legacy": {},
}


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--rid_profile", default="official_sd21", type=str, choices=sorted(RID_PROFILES))
parser.add_argument("--rid_bundle_dir", default=None, type=str,
                    help="Reusable RingID key bundle (manifest.json + selected_pattern.pt + "
                         "watermark_mask.pt [+ threshold.json]).")
parser.add_argument("--rid_key_index", default=None, type=int,
                    help="Candidate index in the official keybook. Defaults to the bundle's own "
                         "key, or to 628 when a new bundle is created; 628 is the official "
                         "verify.py example key, not the whole RingID method.")
parser.add_argument("--rid_key_seed", default=42, type=int,
                    help="Canonical RAVEN key-generation seed (official --general_seed).")
parser.add_argument("--rid_key_rng_device", default="cpu", type=str, choices=("cpu", "cuda"))
parser.add_argument("--rid_key_rng_dtype", default="float32", type=str, choices=sorted(KEY_RNG_DTYPES))
parser.add_argument("--rid_seed", default=None, type=int,
                    help="DEPRECATED legacy RAVEN key seed. Only accepted with "
                         "--rid_profile legacy; it is never remapped to an official key.")
parser.add_argument("--ring_width", default=1, type=int)
parser.add_argument("--quantization_levels", default=2, type=int)
parser.add_argument("--ring_value_range", default=64, type=int)

parser.add_argument('--fix_gt', type=int, default=1, help='use watermark after discarding the imag part on space domain as gt.')
parser.add_argument('--time_shift', type=int, default=1, help='use time-shift')
parser.add_argument('--time_shift_factor', type=float, default=1.0,
                    help='Scale applied after the spatial shift. UPSTREAM NO-OP in the frozen '
                         'official commit (the multiplication is commented out); only effective '
                         'with --rid_shift_semantics paper_described_shift.')
parser.add_argument("--rid_shift_semantics", default="official_code_exact", type=str,
                    choices=SHIFT_SEMANTICS)
parser.add_argument('--assigned_keys', type=int, default=-1, help='number of assigned keys, -1 for all possible kyes')
parser.add_argument('--channel_min', type=int, default=1, help='only for heterogeous watermark, when match gt, take min among channels as the result')

parser.add_argument("--rid_torch_dtype", default="float16", type=str, choices=sorted(TORCH_DTYPES))
parser.add_argument("--rid_inversion_prompt", default="", type=str)
parser.add_argument("--rid_inversion_guidance", default=1.0, type=float)
parser.add_argument("--rid_inversion_steps", default=50, type=int)
parser.add_argument("--rid_vae_sample", action="store_true", default=False,
                    help="Sample the VAE posterior. Official RingID uses the posterior MODE.")
parser.add_argument("--rid_vae_scaling_factor", default=0.18215, type=float)
parser.add_argument("--rid_target_fpr", default=0.01, type=float)
parser.add_argument("--rid_threshold", default=None, type=float,
                    help="Explicit deployment threshold on the canonical score (-channel-min L1).")
parser.add_argument("--rid_top_k", default=5, type=int, help="top-k keys reported by identification.")
parser.add_argument("--rid_no_clean_pair", dest="rid_save_clean", action="store_false", default=True,
                    help="Do not generate the matched non-watermarked image. The official "
                         "verify.py protocol needs the pair, so this downgrades the cohort.")


#: Flags whose *negation* is expressed as a separate switch (none yet, but the
#: profile machinery mirrors gm_provider so store-true defaults stay overridable).
RID_NEGATION_FLAGS: typing.Dict[str, str] = {}


def apply_arg_defaults(args, argv) -> typing.Dict[str, typing.Any]:
    """Apply the selected RingID profile without letting generic defaults win.

    Any value explicitly present on the command line takes precedence and is
    recorded in ``rid_profile_overrides``; a run with overrides is *not* an
    official run and is labelled as an ablation.
    """
    profile_name = getattr(args, "rid_profile", "official_sd21")
    profile = RID_PROFILES.get(profile_name, {})
    argv = list(argv or [])

    def explicitly_set(name: str) -> bool:
        flag = f"--{name}"
        return any(token == flag or token.startswith(flag + "=") for token in argv)

    applied, overrides = {}, {}
    for name, value in profile.items():
        if explicitly_set(name):
            overrides[name] = getattr(args, name, None)
            continue
        negated = RID_NEGATION_FLAGS.get(name)
        if negated is not None and negated in argv:
            overrides[name] = getattr(args, name, None)
            continue
        setattr(args, name, value)
        applied[name] = value

    args.rid_profile_overrides = overrides
    args.rid_profile_is_official = (profile_name == "official_sd21" and not overrides)
    return {"profile": profile_name, "applied": applied, "overrides": overrides,
            "is_official": args.rid_profile_is_official}


# ---------------------------------------------------------------------------
# Official masks (utils.py)
# ---------------------------------------------------------------------------

def set_random_seed(seed: int = 0) -> None:
    """Official ``utils.set_random_seed`` (reference commit), verbatim."""
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def circle_mask(size=64, r=RADIUS, x_offset=ANCHOR_X_OFFSET, y_offset=ANCHOR_Y_OFFSET,
                mode="full") -> np.ndarray:
    """Official ``utils.circle_mask``. Returns a (size, size) bool array."""
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset - 1
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    if mode == "left":
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & ((x > x0) + ((x == x0) & (y > y0)))
    if mode == "right":
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & ((x < x0) + ((x == x0) & (y < y0)))
    if mode == "full":
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & (
            ((x > x0) + ((x == x0) & (y > y0))) + ((x < x0) + ((x == x0) & (y < y0)))
        )
    raise NotImplementedError(f'Circle mask "{mode}" not implemented.')


def plain_ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF, x_offset=ANCHOR_X_OFFSET,
                    y_offset=ANCHOR_Y_OFFSET, mode="full") -> np.ndarray:
    """Official ``utils.ring_mask`` *before* the rounder-ring override."""
    outer = circle_mask(size=size, r=r_out, x_offset=x_offset, y_offset=y_offset, mode=mode)
    inner = circle_mask(size=size, r=r_in, x_offset=x_offset, y_offset=y_offset, mode=mode)
    return outer & (~inner)


class RounderRingMask:
    """Official ``utils.RounderRingMask``, verbatim (torchvision rotation vote)."""

    def __init__(self, size=64, r_out=RADIUS, x_offset=ANCHOR_X_OFFSET,
                 y_offset=ANCHOR_Y_OFFSET, mode="full"):
        from torchvision import transforms

        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i * 4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y + num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            res[angle] = transforms.functional.rotate(zero_bg_freq, angle=angle)

        res = res.numpy()
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y], return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values != 0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values != 0][0]

    def get_ring_mask(self, r_out, r_in):
        assert r_out <= self.r_out
        right_end = 0 if r_in - 1 < 0 else r_in - 1
        cand_list = self.ring_vector_np[r_out - 1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)
        if self.size % 2:
            mask = mask[:self.size - 1, :self.size - 1]  # [64, 64]
        return mask


_ROUNDER_MASK_OBJ: typing.Optional[RounderRingMask] = None
_RING_MASK_CACHE: typing.Dict[typing.Tuple[int, int, int], np.ndarray] = {}


def rounder_mask_object() -> RounderRingMask:
    """Lazily build the official rounder-ring object (it costs ~8s to construct)."""
    global _ROUNDER_MASK_OBJ
    if _ROUNDER_MASK_OBJ is None:
        _ROUNDER_MASK_OBJ = RounderRingMask(
            size=65, r_out=RADIUS, x_offset=ANCHOR_X_OFFSET, y_offset=ANCHOR_Y_OFFSET
        )
    return _ROUNDER_MASK_OBJ


def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF, x_offset=ANCHOR_X_OFFSET,
              y_offset=ANCHOR_Y_OFFSET, mode="full") -> np.ndarray:
    """The *effective* official ``ring_mask``.

    In the reference commit ``USE_ROUNDER_RING = True`` rebinds ``ring_mask`` to
    the rounder-ring implementation at import time, so every official mask —
    the watermark region mask *and* every single-radius ring inside
    ``make_Fourier_ringid_pattern`` — is a rounder ring. The previous RAVEN
    provider defined the rounder-ring class but never called it and used the
    plain circle-difference mask instead (584 vs 556 selected coefficients).
    """
    if not USE_ROUNDER_RING:
        return plain_ring_mask(size=size, r_out=r_out, r_in=r_in, x_offset=x_offset,
                               y_offset=y_offset, mode=mode)
    assert size == 64
    assert mode == "full", f"not implemented mode {mode}"
    cache_key = (size, r_out, r_in)
    cached = _RING_MASK_CACHE.get(cache_key)
    if cached is None:
        cached = rounder_mask_object().get_ring_mask(r_out=r_out, r_in=r_in)
        _RING_MASK_CACHE[cache_key] = cached
    return cached


def fft(input_tensor: torch.Tensor) -> torch.Tensor:
    """Official ``utils.fft``."""
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))


def ifft(input_tensor: torch.Tensor) -> torch.Tensor:
    """Official ``utils.ifft``."""
    assert len(input_tensor.shape) == 4
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))


def make_Fourier_ringid_pattern(device, key_value_combination, no_watermark_latents,
                                radius, radius_cutoff, ring_watermark_channel,
                                heter_watermark_channel, heter_watermark_region_mask=None,
                                ring_width=1):
    """Official ``utils.make_Fourier_ringid_pattern``, verbatim.

    Note the base latent only supplies shape/dtype/device: the pattern starts
    from ``torch.zeros_like``. Its *draw* still matters because it advances the
    same global RNG stream the heterogeneous noise is taken from.
    """
    if ring_width != 1:
        raise NotImplementedError('Proposed watermark generation only implemented for ring width = 1.')

    if len(key_value_combination) != (radius - radius_cutoff):
        raise ValueError('Mismatch between #key values and #slots')

    shape = no_watermark_latents.shape
    if len(shape) != 4:
        raise ValueError(f'Invalid shape for initial latent: {shape}')

    latents_fft = fft(no_watermark_latents)
    watermarked_latents_fft = torch.zeros_like(latents_fft).to(device)

    radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

    # put ring
    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        mask = torch.tensor(
            ring_mask(size=shape[-1], r_out=this_r_out, r_in=this_r_in)
        ).to(device).to(torch.float64)

        for batch_index in range(shape[0]):
            for channel_index in range(len(ring_watermark_channel)):
                channel = ring_watermark_channel[channel_index]
                value = key_value_combination[radius_index][channel_index]
                watermarked_latents_fft[batch_index, channel].real = (
                    (1 - mask) * watermarked_latents_fft[batch_index, channel].real + mask * value
                )
                watermarked_latents_fft[batch_index, channel].imag = (
                    (1 - mask) * watermarked_latents_fft[batch_index, channel].imag + mask * value
                )

    # put noise
    if len(heter_watermark_channel) > 0:
        assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
        heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
        w_content = fft(torch.randn(*shape, device=device))  # [N, c, h, w]

        for batch_index in range(shape[0]):
            for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                watermarked_latents_fft[batch_index, channel_id].real = (
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real
                    + channel_mask * w_content[batch_index][channel_id].real
                )
                watermarked_latents_fft[batch_index, channel_id].imag = (
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag
                    + channel_mask * w_content[batch_index][channel_id].imag
                )

    return watermarked_latents_fft


def generate_Fourier_watermark_latents(device, radius, radius_cutoff, watermark_region_mask,
                                       watermark_channel, original_latents=None,
                                       watermark_pattern=None) -> torch.Tensor:
    """Official ``utils.generate_Fourier_watermark_latents``, verbatim."""
    if original_latents is None:
        raise NotImplementedError('Original latents should be provided.')
    if watermark_pattern is None:
        raise NotImplementedError('Fourier watermark pattern should be provided.')

    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim=(-1, -2))

    assert len(watermark_channel) == len(watermark_region_mask)
    for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
        watermarked_latents_fft[:, channel] = (
            watermarked_latents_fft[:, channel] * ~channel_mask
            + watermark_pattern[:, channel] * channel_mask
        )

    return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim=(-1, -2))).real


def official_channel_distances(pattern: torch.Tensor, recovered_fft: torch.Tensor,
                               mask: torch.Tensor,
                               channel: typing.Sequence[int] = WATERMARK_CHANNEL,
                               mode: str = "complex") -> typing.List[float]:
    """Official ``utils.get_distance(..., p=1, channel_min=True)`` per channel.

    Returns one masked mean-|difference| per entry of ``channel`` (so ``[ch0,
    ch3]`` for the official default). The official ``channel_min=1`` result is
    ``min`` of this list for the released 1 ring + 1 heterogeneous channel
    configuration; the caller keeps the raw per-channel values as well.
    """
    if pattern.shape != recovered_fft.shape:
        raise ValueError(f'Shape mismatch during eval: {pattern.shape} vs {recovered_fft.shape}')
    if mode not in ("complex", "real", "imag"):
        raise NotImplementedError(f'Eval mode not implemented: {mode}')

    a, b = pattern[0][channel], recovered_fft[0][channel]
    if mode == "complex":
        diff = torch.abs(a - b)
    elif mode == "real":
        diff = torch.abs(a.real - b.real)
    else:
        diff = torch.abs(a.imag - b.imag)

    distances = []
    for c_idx in range(len(mask)):
        mask_c = torch.zeros_like(mask)
        mask_c[c_idx] = mask[c_idx]
        distances.append(torch.mean(diff[mask_c]).item())
    return distances


class RingIDProvider(WmProvider):
    """Authoritative RingID implementation (generation, verification, identification)."""

    def __init__(self,
                 rid_profile: str = "official_sd21",
                 rid_bundle_dir: typing.Optional[str] = None,
                 rid_key_index: typing.Optional[int] = None,
                 rid_key_seed: int = 42,
                 rid_key_rng_device: str = "cpu",
                 rid_key_rng_dtype: str = "float32",
                 rid_seed: typing.Optional[int] = None,
                 channel_min: int = 1,
                 ring_value_range: int = 64,
                 quantization_levels: int = 2,
                 ring_width: int = 1,
                 assigned_keys: int = -1,
                 fix_gt: int = 1,
                 time_shift: int = 1,
                 time_shift_factor: float = 1.0,
                 rid_shift_semantics: str = "official_code_exact",
                 rid_torch_dtype: str = "float16",
                 rid_inversion_prompt: str = "",
                 rid_inversion_guidance: float = 1.0,
                 rid_inversion_steps: int = 50,
                 rid_vae_sample: bool = False,
                 rid_vae_scaling_factor: float = 0.18215,
                 rid_target_fpr: float = 0.01,
                 rid_threshold: typing.Optional[float] = None,
                 rid_top_k: int = 5,
                 rid_create_bundle: bool = False,
                 rid_profile_is_official: typing.Optional[bool] = None,
                 rid_profile_overrides: typing.Optional[typing.Mapping[str, typing.Any]] = None,
                 modelid_target: typing.Optional[str] = None,
                 model_revision: typing.Optional[str] = None,
                 scheduler_target: typing.Optional[str] = None,
                 resolution: int = 512,
                 **kwargs):
        super().__init__(**kwargs)

        self.profile = rid_profile
        self.profile_overrides = dict(rid_profile_overrides or {})
        self.profile_is_official = (
            bool(rid_profile_is_official)
            if rid_profile_is_official is not None
            else (rid_profile == "official_sd21" and not self.profile_overrides)
        )

        # Legacy --rid_seed is never silently remapped onto an official key.
        if rid_seed is not None:
            if rid_profile != "legacy":
                raise RidBundleError(
                    "--rid_seed is the deprecated legacy RAVEN key seed and does not denote an "
                    "official RingID key. Use --rid_key_seed with an official profile, or pass "
                    "--rid_profile legacy to run the labelled legacy configuration."
                )
            rid_key_seed = int(rid_seed)
        self.legacy_rid_seed = None if rid_seed is None else int(rid_seed)

        self.key_seed = int(rid_key_seed)
        self.key_rng_device = str(rid_key_rng_device)
        self.key_rng_dtype_name = str(rid_key_rng_dtype)
        self.key_rng_dtype = KEY_RNG_DTYPES[self.key_rng_dtype_name]
        # ``None`` means "adopt the bundle's key, or the official example key 628
        # when a new bundle is created". An explicit index that disagrees with the
        # bundle is a rejection, never a silent substitution.
        self.requested_key_index = None if rid_key_index is None else int(rid_key_index)
        self.key_index = DEFAULT_KEY_INDEX if rid_key_index is None else int(rid_key_index)

        self.channel_min = int(channel_min)
        if self.channel_min:
            assert len(HETER_WATERMARK_CHANNEL) > 0
        self.ring_value_range = int(ring_value_range)
        self.quantization_levels = int(quantization_levels)
        self.ring_width = int(ring_width)
        self.assigned_keys = int(assigned_keys)
        self.fix_gt = int(fix_gt)
        self.time_shift = int(time_shift)
        self.time_shift_factor = float(time_shift_factor)
        self.shift_semantics = str(rid_shift_semantics)

        if self.shift_semantics not in SHIFT_SEMANTICS:
            raise RidBundleError(f"unknown rid_shift_semantics {self.shift_semantics!r}")
        if self.shift_semantics == "official_code_exact" and self.time_shift_factor != 1.0:
            raise RidBundleError(
                "--time_shift_factor is an upstream NO-OP in the frozen official commit (the "
                "multiplication is commented out), so it cannot be represented as effective in "
                "official_code_exact mode. Use --rid_shift_semantics paper_described_shift to run "
                "the paper-described scaled shift as an explicit ablation."
            )
        if self.profile_is_official and self.ring_width != 1:
            raise RidBundleError(
                f"official RingID pattern generation is only defined for ring_width = 1, got "
                f"{self.ring_width}"
            )

        self.torch_dtype_name = str(rid_torch_dtype)
        self.inversion_prompt = str(rid_inversion_prompt)
        self.inversion_guidance = float(rid_inversion_guidance)
        self.inversion_steps = int(rid_inversion_steps)
        self.vae_sample = bool(rid_vae_sample)
        self.vae_scaling_factor = float(rid_vae_scaling_factor)
        self.target_fpr = float(rid_target_fpr)
        self.user_threshold = None if rid_threshold is None else float(rid_threshold)
        self.top_k = int(rid_top_k)

        self.model_id = modelid_target
        self.model_revision = model_revision
        self.scheduler = scheduler_target
        self.resolution = int(resolution)

        self._pipe = None
        self._keybook_masked: typing.Optional[torch.Tensor] = None
        self._keybook_sha256: typing.Optional[str] = None
        self._keybook_indices: typing.List[int] = []
        self._keybook_indices_key: typing.Optional[typing.Tuple[int, ...]] = None

        # Masks are pure functions of the frozen constants.
        self.heter_watermark_region_mask = self.__heter_region_mask()
        self.watermarking_mask = self.__get_watermarking_mask()

        # Key/keybook state: from a bundle when one is given, otherwise built.
        self.bundle: typing.Optional[RidBundle] = None
        self.state_source = "in_memory"
        self.gt_patch = self.__init_key(rid_bundle_dir, rid_create_bundle)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_wm_type(self) -> str:
        return "RID"

    @staticmethod
    def apply_arg_defaults(args, argv):
        return apply_arg_defaults(args, argv)

    @property
    def candidate_count(self) -> int:
        """Keybook capacity: ``quantization_levels ** (RADIUS - RADIUS_CUTOFF)``."""
        if self.assigned_keys > 0:
            return int(self.assigned_keys)
        return int(self.quantization_levels ** ((RADIUS - RADIUS_CUTOFF) * len(RING_WATERMARK_CHANNEL)))

    @property
    def quantization_values(self) -> typing.List[float]:
        return np.linspace(-self.ring_value_range, self.ring_value_range,
                           self.quantization_levels).tolist()

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def __heter_region_mask(self) -> typing.Optional[torch.Tensor]:
        if len(HETER_WATERMARK_CHANNEL) == 0:
            return None
        single = torch.tensor(
            ring_mask(size=self.latent_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)
        )
        return single.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(self.device)

    def __get_watermarking_mask(self) -> torch.Tensor:
        """Official per-channel watermark region mask, shape ``[len(WATERMARK_CHANNEL), S, S]``."""
        ring = torch.tensor(ring_mask(size=self.latent_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF))
        heter = torch.tensor(ring_mask(size=self.latent_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF))
        region = []
        for channel_idx in WATERMARK_CHANNEL:
            region.append(ring if channel_idx in RING_WATERMARK_CHANNEL else heter)
        return torch.stack(region).to(self.device)

    # ------------------------------------------------------------------
    # Keybook
    # ------------------------------------------------------------------

    def key_value_combinations(self) -> typing.List[typing.Tuple]:
        """Official candidate ordering (``itertools.product`` over quantized slots)."""
        single_channel_num_slots = RADIUS - RADIUS_CUTOFF
        key_value_list = [
            [list(combo) for combo in itertools.product(
                self.quantization_values, repeat=len(RING_WATERMARK_CHANNEL))]
            for _ in range(single_channel_num_slots)
        ]
        combinations = list(itertools.product(*key_value_list))
        if self.assigned_keys > 0:
            if self.assigned_keys > len(combinations):
                raise RidBundleError(
                    f"--assigned_keys {self.assigned_keys} exceeds the capacity {len(combinations)}"
                )
            # Official: random.sample from the global RNG seeded by set_random_seed.
            combinations = random.sample(combinations, k=self.assigned_keys)
        return combinations

    def _base_latents(self) -> torch.Tensor:
        """The key-construction base latent draw.

        Official ``verify.py``/``identify.py`` take it from
        ``pipe.get_random_latents()`` (a device-side ``torch.randn`` in the text
        encoder dtype) and immediately cast it to float64. Its *values* never
        reach the pattern (which starts from ``zeros_like``), but the draw
        advances the same global RNG stream the heterogeneous noise comes from,
        so it must be reproduced exactly.

        RAVEN's canonical key RNG is a **CPU float32** draw so a key id means the
        same tensor on every machine. ``--rid_key_rng_device cuda
        --rid_key_rng_dtype float16`` reproduces the official runtime draw
        instead; either way the choice is recorded in the bundle manifest.
        """
        latents = torch.randn(
            tuple(self.latent_shape), device=self.key_rng_device, dtype=self.key_rng_dtype
        )
        return latents.to(torch.float64)

    def _shift_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Official spatial (time) shift of the ring channel.

        ``official_code_exact`` reproduces the released line, in which the
        ``* args.time_shift_factor`` multiplication is commented out.
        ``paper_described_shift`` applies the paper's eta explicitly.
        """
        shifted = torch.fft.fftshift(ifft(pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2))
        if self.shift_semantics == "paper_described_shift":
            shifted = shifted * self.time_shift_factor
        pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(shifted)
        return pattern

    def iter_candidate_patterns(
        self,
        indices: typing.Optional[typing.Collection[int]] = None,
    ) -> typing.Iterator[typing.Tuple[int, torch.Tensor]]:
        """Walk the candidate keybook in official order under the frozen key RNG.

        Yields ``(index, pattern)`` for the requested indices. Candidates that
        are not requested are still *drawn* so the RNG stream — and therefore
        the identity of every later key — is bit-identical to building the whole
        keybook, exactly as the official scripts do.

        ``fix_gt`` and the spatial shift are applied per pattern; both are
        per-pattern maps that consume no randomness, so fusing them into this
        loop is identical to the official "build all, then map all" order.
        """
        # The *whole* key construction runs on the declared key RNG device so
        # the same seed yields the same keybook regardless of the device the
        # provider itself runs on. Official code draws both the base latent and
        # every heterogeneous-noise tensor on one device; mixing devices here
        # would silently change every key.
        key_device = torch.device(self.key_rng_device)
        heter_mask = (
            None if self.heter_watermark_region_mask is None
            else self.heter_watermark_region_mask.to(key_device)
        )
        set_random_seed(self.key_seed)
        base_latents = self._base_latents()
        combinations = self.key_value_combinations()
        wanted = None if indices is None else set(int(i) for i in indices)
        if wanted is not None:
            unknown = [i for i in wanted if i < 0 or i >= len(combinations)]
            if unknown:
                raise RidBundleError(
                    f"candidate index/indices {sorted(unknown)} outside keybook of size "
                    f"{len(combinations)}"
                )
        remaining = None if wanted is None else set(wanted)

        for index, combo in enumerate(combinations):
            if remaining is not None and index not in remaining:
                # Consume exactly the randomness this candidate would have used.
                torch.randn(*base_latents.shape, device=key_device)
                continue
            pattern = make_Fourier_ringid_pattern(
                key_device, list(combo), base_latents,
                radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                ring_watermark_channel=RING_WATERMARK_CHANNEL,
                heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                heter_watermark_region_mask=heter_mask,
                ring_width=self.ring_width,
            )
            if self.fix_gt:
                pattern = fft(ifft(pattern).real)
            if self.time_shift:
                pattern = self._shift_pattern(pattern)
            yield index, pattern.to(self.device)
            if remaining is not None:
                remaining.discard(index)
                if not remaining:
                    return

    def build_key_pattern(self, index: int) -> torch.Tensor:
        """The official candidate pattern for one key index."""
        for _, pattern in self.iter_candidate_patterns([index]):
            return pattern
        raise RidBundleError(f"key index {index} not produced by the keybook")

    def masked_pattern_vector(self, pattern: torch.Tensor) -> torch.Tensor:
        """Masked complex coefficients of the watermark channels, shape ``[C, M]``."""
        mask = self.watermarking_mask
        selected = pattern[0][WATERMARK_CHANNEL]
        return torch.stack([selected[c][mask[c]] for c in range(len(mask))])

    def build_keybook(
        self,
        candidate_indices: typing.Optional[typing.Sequence[int]] = None,
        force: bool = False,
    ) -> typing.Dict[str, typing.Any]:
        """Materialize the candidate keybook for identification.

        Only the masked coefficients of the watermark channels are kept in
        memory (2048 x 2 x 556 complex128 = ~36 MB instead of ~537 MB of full
        patterns), while ``keybook_sha256`` is streamed over the **full**
        patterns so the recorded hash is a hash of the official tensors.

        ``candidate_indices`` declares a smaller keybook (official
        ``--assigned_keys``-style capacity study). The candidates are still
        drawn in official order under the same RNG stream, so a key id keeps its
        meaning; the resulting hash is explicitly a hash of *that* declared
        keybook, never of the full one.
        """
        if candidate_indices is not None:
            indices = [int(i) for i in candidate_indices]
            if len(set(indices)) != len(indices):
                raise RidBundleError("candidate_indices contains duplicates")
            indices = sorted(indices)
        else:
            indices = None

        cache_key = None if indices is None else tuple(indices)
        if self._keybook_masked is not None and not force and cache_key == self._keybook_indices_key:
            return {
                "masked": self._keybook_masked,
                "keybook_sha256": self._keybook_sha256,
                "candidate_count": int(self._keybook_masked.shape[0]),
                "candidate_indices": list(self._keybook_indices),
                "is_full_keybook": indices is None,
            }

        import hashlib

        digest = hashlib.sha256()
        vectors, produced = [], []
        for index, pattern in self.iter_candidate_patterns(indices):
            digest.update(f"{index}:{rid_bundle.sha256_tensor(pattern)}".encode("utf-8"))
            vectors.append(self.masked_pattern_vector(pattern))
            produced.append(int(index))
        if not vectors:
            raise RidBundleError("keybook construction produced no candidates")

        self._keybook_masked = torch.stack(vectors)
        self._keybook_sha256 = digest.hexdigest()
        self._keybook_indices = produced
        self._keybook_indices_key = cache_key

        if indices is None and self.bundle is not None:
            recorded = self.bundle.manifest.get("keybook_sha256")
            if recorded is not None and recorded != self._keybook_sha256:
                raise RidBundleError(
                    "regenerated keybook does not match the bundle: keybook_sha256 "
                    f"{self._keybook_sha256} != {recorded}. The same key id would denote a "
                    "different tensor; nothing was scored."
                )
        return {
            "masked": self._keybook_masked,
            "keybook_sha256": self._keybook_sha256,
            "candidate_count": int(self._keybook_masked.shape[0]),
            "candidate_indices": list(produced),
            "is_full_keybook": indices is None,
        }

    def candidate_order_sha256(self) -> str:
        """Hash of the official candidate ordering (independent of the tensors)."""
        combos = [[list(slot) for slot in combo] for combo in self.key_value_combinations()]
        return rid_bundle.canonical_sha256({"candidate_order": combos})

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------

    def __init_key(self, bundle_dir, create_bundle: bool) -> torch.Tensor:
        if bundle_dir is None:
            self.state_source = "in_memory"
            return self.build_key_pattern(self.key_index)

        bundle = RidBundle(bundle_dir)
        if bundle.complete():
            bundle = RidBundle.load(bundle_dir)
            bundle.assert_compatible(
                self.bundle_compat_config(),
                required_fields=rid_bundle.REQUIRED_BUNDLE_COMPAT_FIELDS,
            )
            bundle_key_index = bundle.manifest.get("selected_key_index")
            if self.requested_key_index is not None and self.requested_key_index != bundle_key_index:
                raise RidBundleError(
                    f"--rid_key_index {self.requested_key_index} disagrees with the bundle key "
                    f"{bundle_key_index} in {bundle.dir}. A key id must denote the same tensor "
                    "everywhere, so this is rejected instead of silently substituted."
                )
            self.key_index = int(bundle_key_index)
            self.bundle = bundle
            self.state_source = "bundle"
            mask_sha = rid_bundle.sha256_tensor(self.watermarking_mask)
            if bundle.manifest.get("mask_sha256") != mask_sha:
                raise RidBundleError(
                    "the mask rebuilt by this code does not match the bundle mask "
                    f"({mask_sha} != {bundle.manifest.get('mask_sha256')})"
                )
            return bundle.pattern.to(self.device)

        if not create_bundle:
            raise RidBundleError(
                f"{bundle_dir} is not a complete RingID bundle and this run may not create one "
                "(verification never creates watermark state)"
            )

        pattern = self.build_key_pattern(self.key_index)
        keybook_info = None
        # The keybook hash is only recorded when it is affordable to compute it
        # here; identification recomputes and re-verifies it on demand.
        self.bundle = RidBundle.create(
            bundle_dir,
            pattern=pattern,
            mask=self.watermarking_mask,
            config=self.bundle_manifest_config(keybook_sha256=keybook_info),
        )
        self.state_source = "created"
        return self.bundle.pattern.to(self.device)

    def bundle_compat_config(self) -> typing.Dict[str, typing.Any]:
        """Fields that must agree between this run and an existing bundle."""
        return {
            "latent_shape": [int(d) for d in self.latent_shape],
            "radius": RADIUS,
            "radius_cutoff": RADIUS_CUTOFF,
            "ring_width": self.ring_width,
            "rounder_ring": USE_ROUNDER_RING,
            "anchor_x_offset": ANCHOR_X_OFFSET,
            "anchor_y_offset": ANCHOR_Y_OFFSET,
            "heterogeneous_channels": list(HETER_WATERMARK_CHANNEL),
            "ring_channels": list(RING_WATERMARK_CHANNEL),
            "quantization_levels": self.quantization_levels,
            "ring_value_range": self.ring_value_range,
            "quantization_values": self.quantization_values,
            "assigned_keys": self.assigned_keys,
            "candidate_count": self.candidate_count,
            "candidate_order_sha256": self.candidate_order_sha256(),
            "fix_gt": self.fix_gt,
            "spatial_shift": self.time_shift,
            "spatial_shift_factor": self.time_shift_factor,
            "spatial_shift_factor_semantics": self.shift_semantics,
            "rng_algorithm": RNG_ALGORITHM,
            "rng_seed": self.key_seed,
            "rng_device": self.key_rng_device,
            "rng_dtype": self.key_rng_dtype_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "torch_dtype": self.torch_dtype_name,
            "scheduler": self.scheduler,
            "resolution": self.resolution,
            "inversion_prompt_sha256": rid_bundle.sha256_text(self.inversion_prompt),
            "inversion_guidance_scale": self.inversion_guidance,
            "inversion_steps": self.inversion_steps,
            "vae_sample": self.vae_sample,
            "vae_scaling_factor": self.vae_scaling_factor,
            "channel_min": self.channel_min,
            "score_definition": RID_SCORE_DEFINITION,
        }

    def bundle_manifest_config(self, keybook_sha256: typing.Optional[str] = None
                               ) -> typing.Dict[str, typing.Any]:
        config = self.bundle_compat_config()
        config.update({
            "profile_name": self.profile,
            "profile_is_official": self.profile_is_official,
            "profile_overrides": dict(self.profile_overrides),
            "selected_key_index": self.key_index,
            "selected_key_id": f"rid-key-{self.key_index:06d}",
            "keybook_sha256": keybook_sha256,
            "upstream_unused_args": list(OFFICIAL_UNUSED_ARGS),
            "legacy_rid_seed": self.legacy_rid_seed,
            "score_direction": "higher_is_watermarked",
        })
        return config

    def binding_config(self) -> typing.Dict[str, typing.Any]:
        binding = self.bundle_compat_config()
        binding.update({
            "profile_name": self.profile,
            "selected_key_index": self.key_index,
            "selected_pattern_sha256": rid_bundle.sha256_tensor(self.gt_patch),
            "mask_sha256": rid_bundle.sha256_tensor(self.watermarking_mask),
        })
        if self.bundle is not None:
            binding["bundle_config_sha256"] = self.bundle.manifest.get("bundle_config_sha256")
        return binding

    @property
    def selected_key_id(self) -> str:
        return f"rid-key-{self.key_index:06d}"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def sample_clean_latent(self, sample_seed: int) -> torch.Tensor:
        """One *complete* initial latent for one sample, from an explicit RNG.

        Official ``verify.py``/``identify.py`` call ``set_random_seed(this_seed)``
        and draw a fresh ``pipe.get_random_latents()`` per prompt. RAVEN uses an
        explicit ``torch.Generator`` instead of the process-global RNG so a
        sample never depends on iteration order, batch history or resume
        position, and so the same ``sample_seed`` reproduces the same latent
        after a restart.
        """
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
        latent = torch.randn(tuple(self.latent_shape), generator=generator, dtype=torch.float32)
        return latent.to(self.device)

    def inject(self, latents_clean: torch.Tensor,
               pattern: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        """Official Fourier-domain replacement followed by the real-valued IFFT."""
        pattern = self.gt_patch if pattern is None else pattern
        return generate_Fourier_watermark_latents(
            device=self.device,
            radius=RADIUS,
            radius_cutoff=RADIUS_CUTOFF,
            original_latents=latents_clean,
            watermark_pattern=pattern.to(latents_clean.device),
            watermark_channel=WATERMARK_CHANNEL,
            watermark_region_mask=self.watermarking_mask,
        )

    def build_sample_latents(self, sample_seed: int) -> typing.Dict[str, typing.Any]:
        """Matched clean/watermarked pair for one sample, with pairing hashes."""
        clean = self.sample_clean_latent(sample_seed)
        pre_injection_sha = rid_bundle.sha256_tensor(clean)
        watermarked = self.inject(clean.clone().to(torch.float64))
        return {
            "sample_seed": int(sample_seed),
            "clean_latent": clean,
            "watermarked_latent": watermarked,
            "clean_latent_sha256": pre_injection_sha,
            "pre_injection_latent_sha256": pre_injection_sha,
            "post_injection_latent_sha256": rid_bundle.sha256_tensor(watermarked),
            "selected_pattern_sha256": rid_bundle.sha256_tensor(self.gt_patch),
            "mask_sha256": rid_bundle.sha256_tensor(self.watermarking_mask),
        }

    def generate(self, pipe_provider_target, prompts, latents, num_inference_steps: int,
                 guidance_scale: float):
        """Delegate image synthesis to the shared pipeline provider."""
        return pipe_provider_target.generate(
            prompts=prompts,
            latents=latents.to(pipe_provider_target.pipe.dtype)
            if hasattr(pipe_provider_target, "pipe") and pipe_provider_target.pipe is not None
            else latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def get_wm_latents(self,
                       latents_clean: torch.Tensor = None,
                       seed: int = None,
                       sample_seed: int = None) -> typing.Dict[str, typing.Any]:
        """Legacy single-latent entry point kept for the generic runner/validate().

        Prefer :meth:`build_sample_latents`, which binds the latent to an
        explicit per-sample seed instead of the process-global RNG.
        """
        if sample_seed is not None:
            sample = self.build_sample_latents(sample_seed)
            latents_clean = sample["clean_latent"]
            latents_w = sample["watermarked_latent"]
        else:
            if latents_clean is None:
                if seed is not None:
                    set_random_seed(seed)
                latents_clean = torch.randn(tuple(self.latent_shape))
            latents_clean = latents_clean.clone().to(self.device, torch.float64)
            latents_w = self.inject(latents_clean)

        latents_clean_torch = latents_clean.to(self.device)
        latents_w_torch = latents_w.to(self.device)
        ch = RING_WATERMARK_CHANNEL[0]

        clean_fft = torch.fft.fftshift(
            torch.fft.fft2(latents_clean_torch.to(torch.float32)), dim=(-1, -2)
        ).real
        wm_fft = torch.fft.fftshift(
            torch.fft.fft2(latents_w_torch.to(torch.float32)), dim=(-1, -2)
        ).real

        return {
            "zT_clean_torch": latents_clean_torch,
            "zT_clean_PIL": torch_to_PIL(latents_clean_torch),
            "zT_clean": torch_to_PIL(latents_clean_torch),
            "zT_clean_fft_torch": clean_fft,
            "zT_clean_fft_PIL": torch_to_PIL(clean_fft),
            "zT_clean_fft": torch_to_PIL(clean_fft),
            "zT_clean_fft_wchannel_torch": clean_fft[:, ch:ch + 1],
            "zT_clean_fft_wchannel_PIL": torch_to_PIL(clean_fft[:, ch:ch + 1]),
            "zT_clean_fft_wchannel": torch_to_PIL(clean_fft[:, ch:ch + 1]),
            "zT_torch": latents_w_torch,
            "zT_PIL": torch_to_PIL(latents_w_torch),
            "zT": torch_to_PIL(latents_w_torch),
            "zT_fft_torch": wm_fft,
            "zT_fft_PIL": torch_to_PIL(wm_fft),
            "zT_fft": torch_to_PIL(wm_fft),
            "zT_fft_wchannel_torch": wm_fft[:, ch:ch + 1],
            "zT_fft_wchannel_PIL": torch_to_PIL(wm_fft[:, ch:ch + 1]),
            "zT_fft_wchannel": torch_to_PIL(wm_fft[:, ch:ch + 1]),
        }

    # ------------------------------------------------------------------
    # Inversion (official parity adapter)
    # ------------------------------------------------------------------

    @staticmethod
    def transform_img(image, target_size: int = 512) -> torch.Tensor:
        """Official ``utils.transform_img``: resize, center crop, to tensor, to [-1, 1]."""
        from torchvision import transforms

        tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        return 2.0 * tform(image) - 1.0

    def image_seed(self, image_sha256: str) -> int:
        digest = rid_bundle.sha256_text(f"rid:{image_sha256}")
        return int(digest[:8], 16)

    @staticmethod
    def get_text_embedding(pipe, prompt: str, device) -> torch.Tensor:
        """Official ``get_text_embedding`` (empty prompt at detection time)."""
        text_input_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return pipe.text_encoder(text_input_ids.to(device))[0]

    @torch.no_grad()
    def get_image_latents(self, image_tensor: torch.Tensor, generator=None) -> torch.Tensor:
        """Official ``InversableStableDiffusionPipeline.get_image_latents``."""
        encoding_dist = self._pipe.vae.encode(image_tensor).latent_dist
        encoding = encoding_dist.sample(generator=generator) if self.vae_sample else encoding_dist.mode()
        return encoding * self.vae_scaling_factor

    @torch.no_grad()
    def invert_pil_image(self, image, pipe_provider_target,
                         image_sha256: typing.Optional[str] = None,
                         num_inference_steps: typing.Optional[int] = None
                         ) -> typing.Dict[str, typing.Any]:
        """Official RingID detection front-end for one PIL image.

        Reuses the shared ``official_forward_diffusion`` transcription of the
        official ``InversableStableDiffusionPipeline.forward_diffusion``: the DPM
        scheduler supplies only the timestep grid, ``init_noise_sigma``,
        ``scale_model_input`` and ``alphas_cumprod`` while the state update is
        the manual DDIM equation. ``DPMSolverMultistepInverseScheduler`` is a
        *different* update rule and is deliberately not used here.
        """
        pipe = pipe_provider_target.pipe
        if pipe is None:
            raise RuntimeError("pipe provider has no loaded pipeline")
        self._pipe = pipe
        device = pipe_provider_target.device
        steps = self.inversion_steps if num_inference_steps is None else num_inference_steps

        text_embeddings = self.get_text_embedding(pipe, self.inversion_prompt, device)

        image_tensor = self.transform_img(image, target_size=self.resolution)
        image_tensor = image_tensor.unsqueeze(0).to(text_embeddings.dtype).to(device)

        seed = self.image_seed(image_sha256) if image_sha256 is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed) if self.vae_sample else None
        z0 = self.get_image_latents(image_tensor, generator=generator)

        scheduler = pipe_provider_target.scheduler
        pipe.scheduler = scheduler
        zT = official_forward_diffusion(
            unet=pipe.unet,
            scheduler=scheduler,
            latents=z0,
            text_embeddings=text_embeddings,
            guidance_scale=self.inversion_guidance,
            num_inference_steps=steps,
            device=device,
        )
        return {
            "z0_torch": z0,
            "zT_torch": zT,
            "inversion_seed": None if not self.vae_sample else seed,
            "inversion_steps": steps,
            "recovered_latent_sha256": rid_bundle.sha256_tensor(zT),
        }

    @torch.no_grad()
    def invert_images(self, images, pipe_provider_target, num_inference_steps: int = 50,
                      callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
        """Hook used by ``utils.imprint_utils.validate``."""
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError("RingID inversion supports a single image per call")
            images = images[0]
        result = self.invert_pil_image(
            images, pipe_provider_target=pipe_provider_target,
            num_inference_steps=num_inference_steps,
        )
        z0, zT = result["z0_torch"], result["zT_torch"]
        return {
            "z0_torch": z0, "z0_PIL": torch_to_PIL(z0), "z0": torch_to_PIL(z0),
            "zT_torch": zT, "zT_PIL": torch_to_PIL(zT), "zT": torch_to_PIL(zT),
        }

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def recovered_fft(self, reversed_latents: typing.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Official ``fft(reconstructed_latents)`` of the *continuous* latent."""
        if isinstance(reversed_latents, np.ndarray):
            reversed_latents = torch.from_numpy(reversed_latents)
        if reversed_latents.dim() != 4:
            raise ValueError(f"expected a 4D latent batch, got {tuple(reversed_latents.shape)}")
        return fft(reversed_latents.to(self.device).to(torch.complex128)
                   if reversed_latents.is_complex()
                   else reversed_latents.to(self.device).to(torch.float64))

    def channel_distances(self, reversed_latents, pattern: typing.Optional[torch.Tensor] = None
                          ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Official per-channel complex L1 and the channel-wise minimum.

        Returns **one record per image in the batch**. The official
        ``get_distance`` indexes ``tensor[0]``, i.e. it scores exactly one image;
        collapsing a batch (including its batch dimension) into a single scalar —
        as the previous RAVEN ``__get_l1_distance`` did — merges several suspect
        images into one detector score.
        """
        pattern = self.gt_patch if pattern is None else pattern
        pattern = pattern.to(self.device)
        recovered = self.recovered_fft(reversed_latents)
        mask = self.watermarking_mask

        records = []
        for i in range(recovered.shape[0]):
            single = recovered[i][None, ...]
            distances = official_channel_distances(
                pattern, single, mask, channel=WATERMARK_CHANNEL, mode="complex"
            )
            record = {
                f"rid_channel_{ch}_l1": float(distances[idx])
                for idx, ch in enumerate(WATERMARK_CHANNEL)
            }
            # channel_min=1 (official default): the minimum over the watermarked
            # channels. channel_min=0 keeps the official single masked mean over
            # all watermarked channels together.
            if self.channel_min:
                combined = float(min(distances))
            else:
                combined = float(
                    torch.mean(
                        torch.abs(pattern[0][WATERMARK_CHANNEL] - single[0][WATERMARK_CHANNEL])[mask]
                    ).item()
                )
            record["rid_channel_min_l1"] = combined
            record["rid_channel_distances"] = [float(d) for d in distances]
            record["rid_score"] = -combined
            record["score_definition"] = RID_SCORE_DEFINITION
            record["score_direction"] = "higher_is_watermarked"
            records.append(record)
        return records

    def identify_key(self, reversed_latents,
                     keybook: typing.Optional[typing.Mapping[str, typing.Any]] = None,
                     top_k: typing.Optional[int] = None,
                     true_key_index: typing.Optional[int] = None,
                     candidate_indices: typing.Optional[typing.Sequence[int]] = None
                     ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Official multi-key identification: argmin over the candidate keybook.

        Mirrors ``identify.py``: for every candidate key the official channel-min
        complex L1 is computed against the recovered latent and the argmin is the
        predicted key. Ties resolve to the lowest candidate index (stable sort,
        same as upstream ``np.argmin``).
        """
        top_k = self.top_k if top_k is None else int(top_k)
        info = self.build_keybook(candidate_indices) if keybook is None else dict(keybook)
        masked_keybook = info["masked"]
        key_ids = list(info["candidate_indices"])
        candidate_count = int(masked_keybook.shape[0])

        recovered = self.recovered_fft(reversed_latents)
        mask = self.watermarking_mask
        # Masked coefficients of the recovered latent, shape [B, C, M].
        recovered_masked = torch.stack([
            torch.stack([recovered[i][WATERMARK_CHANNEL][c][mask[c]] for c in range(len(mask))])
            for i in range(recovered.shape[0])
        ])

        results = []
        keys = masked_keybook.to(recovered_masked.device)
        for i in range(recovered_masked.shape[0]):
            # [K, C, M] -> per-channel masked mean |difference| -> [K, C]
            diff = torch.abs(keys - recovered_masked[i][None, ...]).mean(dim=-1)
            per_key = diff.min(dim=-1).values if self.channel_min else diff.mean(dim=-1)
            per_key_np = per_key.real.to(torch.float64).cpu().numpy()
            if not np.isfinite(per_key_np).all():
                raise RidBundleError("identification produced a non-finite candidate distance")
            order = np.argsort(per_key_np, kind="stable")
            best_pos = int(order[0])
            best = int(key_ids[best_pos])
            second = float(per_key_np[order[1]]) if candidate_count > 1 else None
            record = {
                "predicted_key_index": best,
                "predicted_key_id": f"rid-key-{best:06d}",
                "best_distance": float(per_key_np[best_pos]),
                "second_best_distance": second,
                "identification_margin": (None if second is None
                                          else float(second - per_key_np[best_pos])),
                "candidate_count": candidate_count,
                "keybook_sha256": info["keybook_sha256"],
                "top_k_key_indices": [int(key_ids[j]) for j in order[:max(1, top_k)]],
                "top_k_distances": [float(per_key_np[j]) for j in order[:max(1, top_k)]],
                "score_definition": RID_SCORE_DEFINITION,
                "score_direction": "higher_is_watermarked",
            }
            if true_key_index is not None:
                record["true_key_index"] = int(true_key_index)
                record["identification_correct"] = bool(best == int(true_key_index))
            else:
                record["true_key_index"] = None
                record["identification_correct"] = None
            results.append(record)
        return results

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.ndarray]
                       ) -> typing.Dict[str, typing.Any]:
        """Detector entry point used by ``utils.imprint_utils.validate``.

        Emits the raw per-channel distances, the official channel-min distance
        and the canonical score. ``l1_dist`` stays the positive channel-min
        distance so existing callers keep their meaning.
        """
        records = self.channel_distances(latents)
        return {
            "l1_dist": [record["rid_channel_min_l1"] for record in records],
            "rid_records": records,
            "rid_channel_min_l1": [record["rid_channel_min_l1"] for record in records],
            "rid_score": [record["rid_score"] for record in records],
            "score_definition": RID_SCORE_DEFINITION,
            "score_direction": "higher_is_watermarked",
        }

    # ------------------------------------------------------------------
    # Threshold / decision
    # ------------------------------------------------------------------

    @staticmethod
    def decide(score: typing.Optional[float], threshold: typing.Optional[float]
               ) -> typing.Optional[bool]:
        """Canonical decision rule: ``score >= threshold`` on ``-channel_min_l1``."""
        if score is None or threshold is None:
            return None
        return bool(float(score) >= float(threshold))

    def resolve_threshold(self) -> typing.Dict[str, typing.Any]:
        """Pick the deployment threshold, or declare that there is none."""
        info = {
            "threshold": None,
            "threshold_source": "none",
            "threshold_available": False,
            "score_definition": RID_SCORE_DEFINITION,
            "score_direction": "higher_is_watermarked",
            "comparison_operator": ">=",
            "report_label": "official_profile_raw_scores",
            "threshold_target_fpr": None,
            "threshold_empirical_fpr": None,
        }
        if self.user_threshold is not None:
            info.update({
                "threshold": float(self.user_threshold),
                "threshold_source": "user_supplied",
                "threshold_available": True,
                "report_label": "user_supplied_threshold",
            })
            return info
        if self.bundle is not None and self.bundle.has_threshold():
            artifact = self.bundle.load_threshold()
            rid_bundle.assert_threshold_compatible(artifact, self.binding_config())
            info.update({
                "threshold": float(artifact["threshold"]),
                "threshold_source": artifact["threshold_source"],
                "threshold_available": True,
                "report_label": artifact["report_label"],
                "threshold_target_fpr": artifact.get("target_fpr"),
                "threshold_empirical_fpr": artifact.get("empirical_fpr"),
            })
        return info

    def is_detection_successful(self, value: typing.Optional[float]) -> typing.Optional[bool]:
        """Legacy hook: ``value`` is the positive channel-min L1 distance."""
        info = self.resolve_threshold()
        if value is None or not info["threshold_available"]:
            return None
        return self.decide(-float(value), info["threshold"])
