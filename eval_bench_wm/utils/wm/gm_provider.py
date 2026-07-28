"""GaussMarker provider — the single authoritative GaussMarker implementation.

Official reference (frozen):
    https://github.com/SunnierLee/GaussMarker
    commit 4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155

Everything algorithmic lives here:

* official ChaCha20 state creation and loading (``watermark.Gaussian_Shading_chacha``)
* per-sample truncated-Gaussian latent sampling with official legacy-RNG semantics
* Tree-Ring style ring pattern / mask / complex injection (``tr_utils``)
* GM image inversion (official ``InversableStableDiffusionPipeline.forward_diffusion``)
* raw bit recovery, GNR restoration, copy-dimension voting
* ring L1 from the *continuous* recovered latent and the ensemble classifier
* threshold-compatibility checks and the final ``score >= threshold`` decision

Runners (``run_watermark.py``, ``run_verify_watermark.py``) only do CLI parsing,
IO and serialization. They must never reimplement any of the above.

Two operating modes are distinguished and never conflated:

``paper_eval``
    the official paired positive/negative cohort protocol with a cohort-derived
    ROC threshold (``report_label="official_paper_evaluation"``).
``verify``
    a RAVEN deployment extension that scores individual suspect images against a
    pre-calibrated compatible threshold
    (``report_label="calibrated_deployment_verification"`` /
    ``"deployment_verification_extension"`` / ``"user_supplied_threshold"``).
"""

from __future__ import annotations

import argparse
import copy
import typing
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, truncnorm

from . import gm_bundle
from .ddim_inversion import official_forward_diffusion
from .gm_bundle import (
    GmBundle,
    GmBundleError,
    OFFICIAL_GAUSSMARKER_COMMIT,
    OFFICIAL_GAUSSMARKER_REPO,
)
from .gm_unet import GmUNet
from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

#: Immutable official Stable Diffusion 2.1 profile (Issue #1 clarification §5).
#: Generic RAVEN parser defaults must not silently override any of these values;
#: an explicit CLI override is recorded and downgrades the run to an ablation.
GM_OFFICIAL_SD21_PROFILE: typing.Dict[str, typing.Any] = {
    # generation
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "model_revision": "fp16",
    "gm_torch_dtype": "float16",
    "scheduler_target": "DPM",
    "resolution": 512,
    "num_inference_steps_target": 50,
    "guidance_scale_target": 7.5,
    "gm_channel_copy": 1,
    "gm_w_copy": 8,
    "gm_h_copy": 8,
    "gm_fpr": 1e-6,
    "gm_user_number": 1000000,
    "w_seed": 999999,
    "w_channel": 3,
    "w_pattern": "ring",
    "w_mask_shape": "circle",
    "w_radius": 4,
    "w_measurement": "l1_complex",
    "w_injection": "complex",
    "w_pattern_const": 0.0,
    # detection
    "gm_inversion_prompt": "",
    "gm_inversion_guidance": 1.0,
    "gm_inversion_steps": 50,
    "gm_target_fpr": 0.01,
    "gm_classifier_type": 0,
    "gm_model_nf": 128,
    "gm_vae_sample": True,
    "gm_vae_scaling_factor": 0.18215,
}

GM_PROFILES = {
    "official_sd21": GM_OFFICIAL_SD21_PROFILE,
    # "legacy" applies nothing and never claims official parity.
    "legacy": {},
}

#: Kept for backward compatibility with callers that only need the model/scheduler.
GM_ARG_DEFAULTS = {
    "modelid_target": GM_OFFICIAL_SD21_PROFILE["modelid_target"],
    "scheduler_target": GM_OFFICIAL_SD21_PROFILE["scheduler_target"],
}

GM_SCORE_DEFINITION = "gm_ensemble_classifier_probability"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gm_profile", default="official_sd21", type=str, choices=sorted(GM_PROFILES))
parser.add_argument("--gm_bundle_dir", default=None, type=str,
                    help="Reusable GM bundle (manifest.json + w1.pth + w2.pth [+ threshold.json]).")
parser.add_argument("--gm_channel_copy", default=1, type=int)
parser.add_argument("--gm_w_copy", default=8, type=int)
parser.add_argument("--gm_h_copy", default=8, type=int)
parser.add_argument("--gm_fpr", default=1e-6, type=float)
parser.add_argument("--gm_user_number", default=1000000, type=int)
parser.add_argument("--gm_torch_dtype", default="float16", type=str, choices=sorted(TORCH_DTYPES))
parser.add_argument("--gm_utils_dir", default="GM_utils", type=str)
parser.add_argument("--gm_w1_path", default=None, type=str,
                    help="Official w1.pth to import (used only when creating a bundle or in legacy mode).")
parser.add_argument("--gm_w2_path", default=None, type=str,
                    help="Official w2.pth to import (used only when creating a bundle or in legacy mode).")
parser.add_argument("--gm_watermark_bits_seed", default=None, type=int,
                    help="Deterministic seed for the 256 identity bits. Omit for a random identity.")
parser.add_argument("--gm_use_gnr", action="store_true", default=True)
parser.add_argument("--gm_no_gnr", dest="gm_use_gnr", action="store_false")
parser.add_argument("--gm_gnr_path", default="GM_utils/GNR_bits256/model_final.pth", type=str)
parser.add_argument("--gm_model_nf", default=128, type=int)
parser.add_argument("--gm_classifier_type", default=0, type=int)
parser.add_argument("--gm_use_classifier", action="store_true", default=True)
parser.add_argument("--gm_no_classifier", dest="gm_use_classifier", action="store_false")
parser.add_argument("--gm_classifier_path", default="GM_utils/sd21_cls2.pkl", type=str)
# inversion
parser.add_argument("--gm_inversion_prompt", default="", type=str)
parser.add_argument("--gm_inversion_guidance", default=1.0, type=float)
parser.add_argument("--gm_inversion_steps", default=50, type=int)
parser.add_argument("--gm_inversion_seed", default=0, type=int)
parser.add_argument("--gm_vae_sample", action="store_true", default=True,
                    help="Official detection samples the VAE posterior (not the mean).")
parser.add_argument("--gm_vae_posterior_mean", dest="gm_vae_sample", action="store_false")
parser.add_argument("--gm_vae_scaling_factor", default=0.18215, type=float)
# thresholds
parser.add_argument("--gm_target_fpr", default=0.01, type=float)
parser.add_argument("--gm_threshold", default=None, type=float,
                    help="Explicit user-supplied decision threshold; labelled user_supplied_threshold.")


#: Profile entries that are booleans have an explicit "off" switch; setting that
#: switch counts as an explicit override of the profile value.
GM_NEGATION_FLAGS = {
    "gm_use_gnr": "--gm_no_gnr",
    "gm_use_classifier": "--gm_no_classifier",
    "gm_vae_sample": "--gm_vae_posterior_mean",
}


def apply_arg_defaults(args, argv) -> typing.Dict[str, typing.Any]:
    """Apply the selected GM profile without letting generic defaults win.

    Any value explicitly present on the command line takes precedence and is
    recorded in ``gm_profile_overrides``; a run with overrides is *not* an
    official run and is labelled as an ablation.
    """
    profile_name = getattr(args, "gm_profile", "official_sd21")
    profile = GM_PROFILES.get(profile_name, {})
    argv = list(argv or [])

    def explicitly_set(name: str) -> bool:
        flag = f"--{name}"
        return any(token == flag or token.startswith(flag + "=") for token in argv)

    applied, overrides = {}, {}
    for name, value in profile.items():
        if explicitly_set(name):
            overrides[name] = getattr(args, name, None)
            continue
        negated = GM_NEGATION_FLAGS.get(name)
        if negated is not None and negated in argv:
            overrides[name] = getattr(args, name, None)
            continue
        setattr(args, name, value)
        applied[name] = value

    args.gm_profile_overrides = overrides
    args.gm_profile_is_official = (profile_name == "official_sd21" and not overrides)
    return {"profile": profile_name, "applied": applied, "overrides": overrides,
            "is_official": args.gm_profile_is_official}


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    """Official ``tr_utils.circle_mask``."""
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    if r >= 0:
        return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2
    return ((x - x0) ** 2 + (y - y0) ** 2) <= -1


class GmProvider(WmProvider):
    """Official-compatible GaussMarker generation and detection."""

    def __init__(
        self,
        gm_profile: str = "official_sd21",
        gm_bundle_dir: typing.Optional[str] = None,
        gm_channel_copy: int = 1,
        gm_w_copy: int = 8,
        gm_h_copy: int = 8,
        gm_fpr: float = 1e-6,
        gm_user_number: int = 1000000,
        gm_torch_dtype: str = "float16",
        gm_utils_dir: str = "GM_utils",
        gm_w1_path: typing.Optional[str] = None,
        gm_w2_path: typing.Optional[str] = None,
        gm_watermark_bits_seed: typing.Optional[int] = None,
        gm_use_gnr: bool = True,
        gm_gnr_path: str = "GM_utils/GNR_bits256/model_final.pth",
        gm_model_nf: int = 128,
        gm_classifier_type: int = 0,
        gm_use_classifier: bool = True,
        gm_classifier_path: str = "GM_utils/sd21_cls2.pkl",
        gm_inversion_prompt: str = "",
        gm_inversion_guidance: float = 1.0,
        gm_inversion_steps: int = 50,
        gm_inversion_seed: int = 0,
        gm_vae_sample: bool = True,
        gm_vae_scaling_factor: float = 0.18215,
        gm_target_fpr: float = 0.01,
        gm_threshold: typing.Optional[float] = None,
        gm_profile_overrides: typing.Optional[typing.Mapping[str, typing.Any]] = None,
        gm_profile_is_official: typing.Optional[bool] = None,
        gm_create_bundle: bool = False,
        gm_allow_in_memory_state: bool = True,
        modelid_target: typing.Optional[str] = None,
        model_revision: typing.Optional[str] = None,
        scheduler_target: typing.Optional[str] = None,
        resolution: int = 512,
        w_seed: int = 999999,
        w_channel: int = 3,
        w_pattern: str = "ring",
        w_mask_shape: str = "circle",
        w_radius: int = 4,
        w_measurement: str = "l1_complex",
        w_injection: str = "complex",
        w_pattern_const: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tuple(self.latent_shape)[1:] != (4, 64, 64):
            raise ValueError(
                "GM supports SD-style 512px latents with shape (B, 4, 64, 64), "
                f"got {tuple(self.latent_shape)}."
            )

        self.profile = gm_profile
        self.profile_overrides = dict(gm_profile_overrides or {})
        self.profile_is_official = (
            gm_profile_is_official
            if gm_profile_is_official is not None
            else (gm_profile == "official_sd21" and not self.profile_overrides)
        )

        self.ch = gm_channel_copy
        self.w = gm_w_copy
        self.h = gm_h_copy
        self.fpr = gm_fpr
        self.user_number = gm_user_number
        self.gm_torch_dtype = gm_torch_dtype
        self.dtype = TORCH_DTYPES[gm_torch_dtype]
        self.watermark_bits_seed = gm_watermark_bits_seed
        self._pipe = None

        if 4 % self.ch != 0 or 64 % self.w != 0 or 64 % self.h != 0:
            raise ValueError("GM copy factors must divide latent dimensions 4x64x64.")
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.w * self.h)
        self.vote_threshold = 1 if (self.ch == 1 and self.w == 1 and self.h == 1) else self.ch * self.w * self.h // 2

        self.utils_dir = self._resolve_path(gm_utils_dir)
        self.import_w1_path = self._resolve_path(gm_w1_path) if gm_w1_path else None
        self.import_w2_path = self._resolve_path(gm_w2_path) if gm_w2_path else None

        self.use_gnr = gm_use_gnr
        self.gnr_path = self._resolve_path(gm_gnr_path) if gm_gnr_path else None
        self.model_nf = gm_model_nf
        self.classifier_type = gm_classifier_type
        self.use_classifier = gm_use_classifier
        self.classifier_path = self._resolve_path(gm_classifier_path) if gm_classifier_path else None

        self.inversion_prompt = gm_inversion_prompt
        self.inversion_guidance = gm_inversion_guidance
        self.inversion_steps = gm_inversion_steps
        self.inversion_seed = gm_inversion_seed
        self.vae_sample = gm_vae_sample
        self.vae_scaling_factor = gm_vae_scaling_factor

        self.target_fpr = gm_target_fpr
        self.user_threshold = gm_threshold

        self.model_id = modelid_target
        self.model_revision = model_revision
        self.scheduler_name = scheduler_target
        self.resolution = resolution

        self.w_seed = w_seed
        self.w_channel = w_channel
        self.w_pattern = w_pattern
        self.w_mask_shape = w_mask_shape
        self.w_radius = w_radius
        self.w_measurement = w_measurement
        self.w_injection = w_injection
        self.w_pattern_const = w_pattern_const

        # ---- state ----------------------------------------------------
        self.bundle: typing.Optional[GmBundle] = None
        self.state_source = None      # "bundle" | "bundle_created" | "w1_file" | "in_memory"
        self.watermark = None         # torch int64 (1, 4//ch, 64//w, 64//h)
        self.m_flat = None            # numpy uint8 (16384,) — official encrypted message
        self.m = None                 # torch int64 (1, 4, 64, 64)
        self.key = None               # bytes (secret, never logged)
        self.nonce = None             # bytes (secret, never logged)
        self.gt_patch = None          # complex (1, 4, 64, 64)

        self._init_state(gm_bundle_dir, gm_create_bundle, gm_allow_in_memory_state)

        self.watermarking_mask = self._get_watermarking_mask()
        self._gnr = None
        self._classifier = None

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def get_wm_type(self) -> str:
        return "GM"

    @staticmethod
    def _provider_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_path(cls, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return cls._provider_root() / path

    @staticmethod
    def apply_arg_defaults(args, argv):
        """Hook used by ``run_watermark.py``; delegates to :func:`apply_arg_defaults`."""
        return apply_arg_defaults(args, argv)

    # ------------------------------------------------------------------
    # State: official ChaCha20 semantics
    # ------------------------------------------------------------------

    def _init_state(self, bundle_dir, create_bundle: bool, allow_in_memory: bool) -> None:
        if bundle_dir is not None:
            bundle_path = Path(bundle_dir)
            existing = GmBundle(bundle_path)
            if existing.complete():
                self.bundle = GmBundle.load(bundle_path)
                self.bundle.assert_compatible(self.bundle_identity_config())
                self._adopt_w1(self.bundle.w1)
                self.gt_patch = self.bundle.w2.to(self.device)
                self.state_source = "bundle"
                return
            if existing.exists():
                raise GmBundleError(
                    f"GM bundle directory {bundle_path} contains partial artifacts; refusing to "
                    "regenerate or overwrite them"
                )
            if not create_bundle:
                raise GmBundleError(
                    f"no GM bundle at {bundle_path}. Generation creates one automatically; "
                    "verification requires an existing bundle."
                )
            state = self._create_or_import_w1()
            patch = self._load_or_create_w2()
            self.bundle = GmBundle.create(bundle_path, state, patch, self.bundle_manifest_config())
            self._adopt_w1(self.bundle.w1)
            self.gt_patch = self.bundle.w2.to(self.device)
            self.state_source = "bundle_created"
            return

        # No bundle: legacy / single-process paths (run_removal, run_reprompting…).
        if self.import_w1_path is not None and self.import_w1_path.exists():
            self._adopt_w1(gm_bundle.load_official_w1(self.import_w1_path))
            self.gt_patch = self._load_or_create_w2().to(self.device)
            self.state_source = "w1_file"
            return

        if not allow_in_memory:
            raise GmBundleError("GM state requires --gm_bundle_dir or an existing --gm_w1_path")

        self._adopt_w1(self._create_or_import_w1())
        self.gt_patch = self._load_or_create_w2().to(self.device)
        self.state_source = "in_memory"

    def _adopt_w1(self, state: typing.Mapping[str, typing.Any]) -> None:
        watermark = torch.as_tensor(state["w"]).to(torch.int64)
        if watermark.dim() == 3:
            watermark = watermark.unsqueeze(0)
        expected = (1, 4 // self.ch, 64 // self.w, 64 // self.h)
        if tuple(watermark.shape) != expected:
            raise GmBundleError(
                f"w1['w'] shape {tuple(watermark.shape)} does not match the configured copy "
                f"factors (expected {expected})"
            )
        self.watermark = watermark.to(self.device)
        self.m_flat = np.ascontiguousarray(state["m"]).astype(np.uint8).reshape(-1)
        if self.m_flat.size != self.latentlength:
            raise GmBundleError(f"w1['m'] must hold {self.latentlength} bits, got {self.m_flat.size}")
        self.m = torch.from_numpy(self.m_flat.astype(np.int64)).reshape(1, 4, 64, 64).to(self.device)
        self.key = bytes(state["key"])
        self.nonce = bytes(state["nonce"])

    def _create_or_import_w1(self) -> typing.Dict[str, typing.Any]:
        if self.import_w1_path is not None:
            if not self.import_w1_path.exists():
                raise GmBundleError(f"--gm_w1_path {self.import_w1_path} does not exist")
            return gm_bundle.load_official_w1(self.import_w1_path)
        return self.create_official_state(
            channel_copy=self.ch,
            w_copy=self.w,
            h_copy=self.h,
            bits_seed=self.watermark_bits_seed,
        )

    @staticmethod
    def create_official_state(
        channel_copy: int = 1,
        w_copy: int = 8,
        h_copy: int = 8,
        bits_seed: typing.Optional[int] = None,
        key: typing.Optional[bytes] = None,
        nonce: typing.Optional[bytes] = None,
    ) -> typing.Dict[str, typing.Any]:
        """Official ``Gaussian_Shading_chacha.create_watermark_and_return_w_m`` state.

        ``watermark bits -> repeat/copy to latent shape -> pack bits ->
        ChaCha20 encryption -> unpack encrypted bits``.
        """
        try:
            from Crypto.Cipher import ChaCha20
            from Crypto.Random import get_random_bytes
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise ImportError(
                "official GaussMarker state requires pycryptodome (Crypto.Cipher.ChaCha20)"
            ) from exc

        if bits_seed is None:
            watermark = torch.randint(0, 2, [1, 4 // channel_copy, 64 // w_copy, 64 // h_copy])
        else:
            generator = torch.Generator().manual_seed(int(bits_seed))
            watermark = torch.randint(
                0, 2, [1, 4 // channel_copy, 64 // w_copy, 64 // h_copy], generator=generator
            )
        sd = watermark.repeat(1, channel_copy, w_copy, h_copy)

        key = get_random_bytes(32) if key is None else bytes(key)
        nonce = get_random_bytes(12) if nonce is None else bytes(nonce)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        m_byte = cipher.encrypt(np.packbits(sd.flatten().numpy()).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return {"w": watermark, "m": m_bit, "key": key, "nonce": nonce}

    def stream_key_encrypt(self, sd_bits: np.ndarray) -> np.ndarray:
        """Official ``stream_key_encrypt``."""
        from Crypto.Cipher import ChaCha20

        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(np.asarray(sd_bits).reshape(-1)).tobytes())
        return np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))

    def stream_key_decrypt(self, reversed_m: np.ndarray) -> torch.Tensor:
        """Official ``stream_key_decrypt``."""
        from Crypto.Cipher import ChaCha20

        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(np.asarray(reversed_m).reshape(-1)).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        return torch.from_numpy(sd_bit.copy()).reshape(1, 4, 64, 64).to(torch.uint8)

    # ------------------------------------------------------------------
    # Ring pattern (w2) and mask
    # ------------------------------------------------------------------

    def _load_or_create_w2(self) -> torch.Tensor:
        if self.import_w2_path is not None:
            if not self.import_w2_path.exists():
                raise GmBundleError(f"--gm_w2_path {self.import_w2_path} does not exist")
            return gm_bundle.load_official_w2(self.import_w2_path).to(self.device)
        return self.build_watermarking_pattern()

    def build_watermarking_pattern(self) -> torch.Tensor:
        """Official ``tr_utils.get_watermarking_pattern`` with ``shape=(1, 4, 64, 64)``."""
        from utils import utils as raven_utils

        raven_utils.set_random_seed(self.w_seed)
        gt_init = torch.randn(1, 4, 64, 64, device=self.device)

        if "zeros" in self.w_pattern:
            return torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        if "const" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            return gt_patch + self.w_pattern_const
        if "rand" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
            return gt_patch
        if "ring" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i), device=self.device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
            return gt_patch
        raise NotImplementedError(f"w_pattern: {self.w_pattern}")

    def _get_watermarking_mask(self) -> torch.Tensor:
        """Official ``tr_utils.get_watermarking_mask`` on the (1, 4, 64, 64) pattern."""
        shape = (1, 4, 64, 64)
        watermarking_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        if self.w_mask_shape == "circle":
            np_mask = circle_mask(shape[-1], r=self.w_radius)
            watermarking_mask[:, self.w_channel if self.w_channel != -1 else slice(None)] = torch.tensor(
                np_mask, device=self.device
            )
        elif self.w_mask_shape == "square":
            anchor_p = shape[-1] // 2
            lo, hi = anchor_p - self.w_radius, anchor_p + self.w_radius
            if self.w_channel == -1:
                watermarking_mask[:, :, lo:hi, lo:hi] = True
            else:
                watermarking_mask[:, self.w_channel, lo:hi, lo:hi] = True
        else:
            raise NotImplementedError(f"w_mask_shape: {self.w_mask_shape}")
        return watermarking_mask

    # ------------------------------------------------------------------
    # Per-sample latent sampling (official legacy-RNG semantics)
    # ------------------------------------------------------------------

    def trunc_sampling(self, message_bits: np.ndarray, rng: np.random.RandomState) -> torch.Tensor:
        """Official ``Gaussian_Shading_chacha.truncSampling``.

        The official code calls ``set_random_seed(seed)`` (which sets
        ``np.random.seed(seed + 3)``) and then draws every element with
        ``scipy.stats.truncnorm.rvs`` from the legacy global NumPy RNG. We use a
        local ``np.random.RandomState`` seeded identically, which produces the
        identical sequence without mutating process-wide RNG state.
        """
        message = np.asarray(message_bits).reshape(-1)
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i: i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1], random_state=rng)
        return torch.from_numpy(z).reshape(1, 4, 64, 64).half()

    def sample_pre_frequency_latent(self, sample_seed: int) -> torch.Tensor:
        """Independently sample the *complete* initial latent for one sample.

        The bundle identity (bits, encrypted message, key/nonce, ring target) is
        fixed; only the truncated-Gaussian draw changes per sample, exactly as in
        official ``gaussmarker_gen.py``.
        """
        if self.m_flat is None:
            raise GmBundleError("GM watermark state is not initialised")
        rng = np.random.RandomState(int(sample_seed) + 3)
        return self.trunc_sampling(self.m_flat, rng)

    def inject_ring(self, latents: torch.Tensor) -> torch.Tensor:
        """Official ``tr_utils.inject_watermark``."""
        latents = latents.to(self.device).float()
        gt_patch = self.gt_patch.to(latents.device)
        if self.w_injection == "seed":
            latents_w = latents.clone()
            latents_w[self.watermarking_mask] = gt_patch.real.to(latents_w.dtype)[self.watermarking_mask].clone()
            return latents_w
        if self.w_injection != "complex":
            raise NotImplementedError(f"w_injection: {self.w_injection}")
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        latents_fft[self.watermarking_mask] = gt_patch[self.watermarking_mask].clone()
        return torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real

    def build_sample_latents(self, sample_seed: int) -> typing.Dict[str, typing.Any]:
        """Full official generation step for one sample."""
        pre = self.sample_pre_frequency_latent(sample_seed)
        injected = self.inject_ring(pre)
        # Official casts the injected latent to fp16 before handing it to the pipeline.
        injected_official = injected.half()
        return {
            "sample_seed": int(sample_seed),
            "pre_frequency_latent": pre,
            "pre_injection_latent_sha256": gm_bundle.sha256_tensor(pre),
            "post_injection_latent_sha256": gm_bundle.sha256_tensor(injected_official),
            "latent": injected_official.to(device=self.device, dtype=self.dtype),
        }

    # ------------------------------------------------------------------
    # Provider API (generation)
    # ------------------------------------------------------------------

    def get_wm_latents(self, sample_seed: int = 0, **kwargs) -> typing.Dict[str, typing.Any]:
        sample = self.build_sample_latents(sample_seed)
        clean = sample["pre_frequency_latent"].to(device=self.device, dtype=self.dtype)
        watermarked = sample["latent"]
        return {
            "zT_clean_torch": clean,
            "zT_clean_PIL": torch_to_PIL(clean),
            "zT_clean": torch_to_PIL(clean),
            "zT_torch": watermarked,
            "zT_PIL": torch_to_PIL(watermarked),
            "zT": torch_to_PIL(watermarked),
            "gm_sample_seed": sample["sample_seed"],
            "gm_pre_injection_latent_sha256": sample["pre_injection_latent_sha256"],
            "gm_post_injection_latent_sha256": sample["post_injection_latent_sha256"],
            "gm_watermark_torch": self.watermark,
            "gm_m_torch": self.m,
            "gm_watermark_sha256": gm_bundle.sha256_tensor(self.watermark),
            "gm_m_sha256": gm_bundle.sha256_array(self.m_flat),
            "gm_target_sha256": gm_bundle.sha256_tensor(self.gt_patch),
            "gm_mask_sha256": gm_bundle.sha256_tensor(self.watermarking_mask),
        }

    def generate(
        self,
        pipe_provider_target,
        prompts,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: typing.Optional[torch.Tensor] = None,
        sample_seed: int = 0,
        **kwargs,
    ):
        if latents is None:
            latents = self.get_wm_latents(sample_seed=sample_seed)["zT_torch"]
        return pipe_provider_target.generate(
            prompts=prompts,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    # ------------------------------------------------------------------
    # Detection artifacts
    # ------------------------------------------------------------------

    @property
    def gnr(self):
        if self._gnr is None and self.use_gnr:
            self._gnr = self._load_gnr()
        return self._gnr

    @property
    def classifier(self):
        if self._classifier is None and self.use_classifier:
            self._classifier = self._load_classifier()
        return self._classifier

    def gnr_available(self) -> bool:
        return bool(self.use_gnr and self.gnr_path is not None and self.gnr_path.exists())

    def classifier_available(self) -> bool:
        return bool(self.use_classifier and self.classifier_path is not None and self.classifier_path.exists())

    def _load_gnr(self):
        if self.gnr_path is None or not self.gnr_path.exists():
            raise FileNotFoundError(
                f"GM GNR is enabled but the checkpoint was not found: {self.gnr_path}. "
                "Train it with the official train_GNR.py, place it under "
                "GM_utils/GNR_bits256/model_final.pth, or pass --gm_no_gnr to emit raw scores only."
            )
        in_channels = 8 if self.classifier_type == 1 else 4
        model = GmUNet(in_channels, 4, nf=self.model_nf).to(self.device)
        state_dict = torch.load(self.gnr_path, map_location=self.device, weights_only=False)
        # Strict load: an architecture/nf/classifier_type mismatch must fail closed.
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _load_classifier(self):
        if self.classifier_path is None or not self.classifier_path.exists():
            raise FileNotFoundError(
                f"GM classifier is enabled but the file was not found: {self.classifier_path}."
            )
        try:
            import joblib
            from sklearn.exceptions import InconsistentVersionWarning
        except ImportError as exc:
            raise ImportError(
                "GM ensemble detection requires scikit-learn and joblib for the official "
                "sd21_cls2.pkl classifier (built with scikit-learn 1.5.2)."
            ) from exc
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(self.classifier_path)

    # ------------------------------------------------------------------
    # Inversion (official parity adapter)
    # ------------------------------------------------------------------

    @staticmethod
    def transform_img(image, target_size: int = 512) -> torch.Tensor:
        """Official ``utils.transform_img``: resize, center crop, to tensor, to [-1, 1]."""
        from torchvision import transforms

        tform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ]
        )
        return 2.0 * tform(image) - 1.0

    def image_seed(self, image_sha256: str) -> int:
        """Deterministic per-image VAE/inversion seed so verification is reproducible."""
        digest = gm_bundle.sha256_text(f"{int(self.inversion_seed)}:{image_sha256}")
        return int(digest[:8], 16)

    @torch.no_grad()
    def get_image_latents(self, image_tensor: torch.Tensor, generator=None) -> torch.Tensor:
        """Official ``InversableStableDiffusionPipeline.get_image_latents``."""
        encoding_dist = self._pipe.vae.encode(image_tensor).latent_dist
        encoding = encoding_dist.sample(generator=generator) if self.vae_sample else encoding_dist.mode()
        return encoding * self.vae_scaling_factor

    @torch.no_grad()
    def invert_pil_image(
        self,
        image,
        pipe_provider_target,
        image_sha256: typing.Optional[str] = None,
        num_inference_steps: typing.Optional[int] = None,
    ) -> typing.Dict[str, typing.Any]:
        """Official detection front-end for one PIL image."""
        pipe = pipe_provider_target.pipe
        if pipe is None:
            raise RuntimeError("pipe provider has no loaded pipeline")
        self._pipe = pipe
        device = pipe_provider_target.device
        steps = self.inversion_steps if num_inference_steps is None else num_inference_steps

        text_embeddings = self.get_text_embedding(pipe, self.inversion_prompt, device)

        image_tensor = self.transform_img(image, target_size=self.resolution)
        image_tensor = image_tensor.unsqueeze(0).to(text_embeddings.dtype).to(device)

        seed = self.image_seed(image_sha256) if image_sha256 is not None else int(self.inversion_seed)
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
            "inversion_seed": seed,
            "inversion_steps": steps,
            "recovered_latent_sha256": gm_bundle.sha256_tensor(zT),
        }

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
    def invert_images(
        self,
        images,
        pipe_provider_target,
        num_inference_steps: int = 50,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Hook used by ``utils.imprint_utils.validate``."""
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError("GM inversion currently supports a single image per call")
            images = images[0]
        result = self.invert_pil_image(
            images,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps=num_inference_steps,
        )
        z0, zT = result["z0_torch"], result["zT_torch"]
        return {
            "z0_torch": z0,
            "z0_PIL": torch_to_PIL(z0),
            "z0": torch_to_PIL(z0),
            "zT_torch": zT,
            "zT_PIL": torch_to_PIL(zT),
            "zT": torch_to_PIL(zT),
        }

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def diffusion_inverse(self, watermark_r: torch.Tensor) -> torch.Tensor:
        """Official ``diffusion_inverse`` copy-dimension voting."""
        ch_stride = 4 // self.ch
        w_stride = 64 // self.w
        h_stride = 64 // self.h
        split_dim1 = torch.cat(torch.split(watermark_r, (ch_stride,) * self.ch, dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, (w_stride,) * self.w, dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, (h_stride,) * self.h, dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.vote_threshold] = 0
        vote[vote > self.vote_threshold] = 1
        return vote

    def pred_w_from_m(self, reversed_m: torch.Tensor) -> torch.Tensor:
        """Official ``pred_w_from_m``: ChaCha20 decrypt then copy-dimension voting."""
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        return self.diffusion_inverse(reversed_sd)

    def bit_accuracy(self, predicted_watermark: torch.Tensor) -> float:
        target = self.watermark.to(predicted_watermark.device)
        return float((predicted_watermark == target).float().mean().item())

    def ring_l1(self, latents: torch.Tensor) -> float:
        """Official ``tr_utils.eval_watermark`` distance *before* the 0.01 scale.

        Computed from the original continuous recovered latent — never from the
        GNR output, the thresholded sign map or the restored binary map.
        """
        latents = latents.to(self.device).float()
        if "complex" in self.w_measurement:
            evaluated = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
            target = self.gt_patch.to(evaluated.device)
        elif "seed" in self.w_measurement:
            evaluated = latents
            target = self.gt_patch.real.to(latents.device)
        else:
            raise NotImplementedError(f"w_measurement: {self.w_measurement}")
        if "l1" not in self.w_measurement:
            raise NotImplementedError(f"w_measurement: {self.w_measurement}")
        mask = self.watermarking_mask.to(evaluated.device)
        return float(torch.abs(evaluated[mask] - target[mask]).mean().item())

    @staticmethod
    def ring_classifier_feature(ring_l1: float) -> float:
        """Official ensemble feature: ``eval_watermark`` scales by 0.01, the detector negates."""
        return -0.01 * float(ring_l1)

    def classifier_probability(self, restored_bit_accuracy: float, ring_l1: float) -> float:
        """Official ``clf.predict_proba(x)[:, 1]`` on ``[bit_acc, -0.01 * ring_l1]``."""
        features = np.array([[float(restored_bit_accuracy), self.ring_classifier_feature(ring_l1)]])
        return float(self.classifier.predict_proba(features)[:, 1][0])

    def restore_with_gnr(self, raw_m: torch.Tensor) -> torch.Tensor:
        """Official GNR restoration of the raw sign map."""
        gnr_input = raw_m.float().to(self.device)
        if self.classifier_type == 1:
            target_m = self.m.to(self.device).float().expand_as(gnr_input)
            gnr_input = torch.cat([target_m, gnr_input], dim=1)
        with torch.no_grad():
            return (torch.sigmoid(self.gnr(gnr_input)) > 0.5).to(torch.int64)

    def detect_from_latent(self, zT_hat: torch.Tensor) -> typing.Dict[str, typing.Any]:
        """The official detector data flow. Raw scores are always emitted."""
        if self.watermark is None:
            raise GmBundleError("GM watermark state is missing")
        latents = zT_hat.to(self.device).float()
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
        if latents.shape[0] != 1:
            raise ValueError("GM detection processes one image at a time")
        if not torch.isfinite(latents).all():
            raise ValueError("recovered latent contains NaN or Inf")

        raw_m = (latents > 0).to(torch.int64)
        raw_watermark = self.pred_w_from_m(raw_m)
        raw_bit_accuracy = self.bit_accuracy(raw_watermark)

        raw_ring_l1 = self.ring_l1(latents)
        ring_feature = self.ring_classifier_feature(raw_ring_l1)

        restored_bit_accuracy = None
        if self.gnr_available():
            restored_m = self.restore_with_gnr(raw_m)
            restored_bit_accuracy = self.bit_accuracy(self.pred_w_from_m(restored_m))

        classifier_probability = None
        if restored_bit_accuracy is not None and self.classifier_available():
            classifier_probability = self.classifier_probability(restored_bit_accuracy, raw_ring_l1)

        return {
            "raw_bit_accuracy": raw_bit_accuracy,
            "restored_bit_accuracy": restored_bit_accuracy,
            "raw_ring_l1": raw_ring_l1,
            "ring_classifier_feature": ring_feature,
            "classifier_probability": classifier_probability,
            "score_definition": GM_SCORE_DEFINITION if classifier_probability is not None else None,
            "recovered_latent_sha256": gm_bundle.sha256_tensor(latents),
            "gnr_used": self.gnr_available(),
            "classifier_used": classifier_probability is not None,
        }

    def ensemble_score(self, detection: typing.Mapping[str, typing.Any]) -> typing.Optional[float]:
        return detection.get("classifier_probability")

    # ------------------------------------------------------------------
    # Thresholds and decisions
    # ------------------------------------------------------------------

    def _detector_config(self) -> typing.Dict[str, typing.Any]:
        """Model / inversion / detector-artifact configuration (no state hashes).

        Safe to call before the watermark state exists, so bundle creation and
        threshold binding share one definition.
        """
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "torch_dtype": self.gm_torch_dtype,
            "scheduler": self.scheduler_name,
            "resolution": int(self.resolution),
            "inversion_prompt_sha256": gm_bundle.sha256_text(self.inversion_prompt),
            "inversion_guidance_scale": float(self.inversion_guidance),
            "inversion_steps": int(self.inversion_steps),
            "vae_sample": bool(self.vae_sample),
            "vae_scaling_factor": float(self.vae_scaling_factor),
            "gnr_sha256": gm_bundle.optional_file_sha256(self.gnr_path if self.use_gnr else None),
            "classifier_sha256": gm_bundle.optional_file_sha256(
                self.classifier_path if self.use_classifier else None
            ),
            "classifier_type": int(self.classifier_type),
            "model_nf": int(self.model_nf),
            "channel_copy": int(self.ch),
            "w_copy": int(self.w),
            "h_copy": int(self.h),
            "w_seed": int(self.w_seed),
            "w_channel": int(self.w_channel),
            "w_pattern": self.w_pattern,
            "w_mask_shape": self.w_mask_shape,
            "w_radius": int(self.w_radius),
            "w_measurement": self.w_measurement,
            "w_injection": self.w_injection,
        }

    def binding_config(self) -> typing.Dict[str, typing.Any]:
        """Everything a threshold artifact is bound to (detector config + state)."""
        binding = self._detector_config()
        binding.update(
            {
                "watermark_sha256": gm_bundle.sha256_tensor(self.watermark),
                "m_sha256": gm_bundle.sha256_array(self.m_flat),
                "w2_tensor_sha256": gm_bundle.sha256_tensor(self.gt_patch),
            }
        )
        if self.bundle is not None:
            binding["bundle_config_sha256"] = self.bundle.manifest.get("bundle_config_sha256")
            binding["w1_file_sha256"] = self.bundle.manifest.get("w1_file_sha256")
            binding["w2_file_sha256"] = self.bundle.manifest.get("w2_file_sha256")
        return binding

    def bundle_identity_config(self) -> typing.Dict[str, typing.Any]:
        """Fields an existing bundle must agree on before it may be reused."""
        return {
            "channel_copy": int(self.ch),
            "w_copy": int(self.w),
            "h_copy": int(self.h),
            "w_seed": int(self.w_seed),
            "w_channel": int(self.w_channel),
            "w_pattern": self.w_pattern,
            "w_mask_shape": self.w_mask_shape,
            "w_radius": int(self.w_radius),
            "w_measurement": self.w_measurement,
            "w_injection": self.w_injection,
            "latent_shape": [1, 4, 64, 64],
        }

    def bundle_manifest_config(self) -> typing.Dict[str, typing.Any]:
        config = self.bundle_identity_config()
        config.update(self._detector_config())
        config.update(
            {
                "profile": self.profile,
                "profile_is_official": bool(self.profile_is_official),
                "profile_overrides": dict(self.profile_overrides),
                "fpr": float(self.fpr),
                "user_number": int(self.user_number),
                "marklength": int(self.marklength),
                "w_pattern_const": float(self.w_pattern_const),
                "w2_generation_device": str(self.device),
                "watermark_bits_seed": self.watermark_bits_seed,
            }
        )
        return config

    def resolve_threshold(self) -> typing.Dict[str, typing.Any]:
        """Resolve the decision threshold and its provenance, or fail closed."""
        if self.user_threshold is not None:
            return {
                "threshold": float(self.user_threshold),
                "threshold_source": "user_supplied",
                "report_label": "user_supplied_threshold",
                "comparison_operator": ">=",
                "score_direction": "higher_is_watermarked",
                "score_definition": GM_SCORE_DEFINITION,
                "threshold_available": True,
            }
        if self.bundle is not None and self.bundle.has_threshold():
            artifact = self.bundle.load_threshold()
            gm_bundle.assert_threshold_compatible(artifact, self.binding_config())
            return {
                "threshold": float(artifact["threshold"]),
                "threshold_source": artifact["threshold_source"],
                "report_label": "calibrated_deployment_verification",
                "comparison_operator": artifact["comparison_operator"],
                "score_direction": artifact["score_direction"],
                "score_definition": artifact["score_definition"],
                "threshold_target_fpr": artifact.get("target_fpr"),
                "threshold_empirical_fpr": artifact.get("empirical_fpr"),
                "threshold_available": True,
            }
        return {
            "threshold": None,
            "threshold_source": "none",
            "report_label": "official_profile_raw_scores" if self.profile_is_official else "legacy_or_ablation_mode",
            "comparison_operator": ">=",
            "score_direction": "higher_is_watermarked",
            "score_definition": GM_SCORE_DEFINITION,
            "threshold_available": False,
        }

    @staticmethod
    def decide(score: typing.Optional[float], threshold: typing.Optional[float]) -> typing.Optional[bool]:
        """Official-compatible ROC decision: ``score >= threshold``."""
        if score is None or threshold is None:
            return None
        return bool(score >= threshold)

    # ------------------------------------------------------------------
    # Legacy provider API used by run_watermark / run_removal / validate
    # ------------------------------------------------------------------

    def get_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        detection = self.detect_from_latent(latents)
        threshold_info = self.resolve_threshold()
        score = self.ensemble_score(detection)
        if score is None:
            # No GNR/classifier: emit raw scores and never fabricate a decision.
            score = detection["restored_bit_accuracy"]
            if score is None:
                score = detection["raw_bit_accuracy"]
            score_definition = "gm_raw_bit_accuracy"
            detection_success = None
        else:
            score_definition = GM_SCORE_DEFINITION
            detection_success = self.decide(score, threshold_info["threshold"])

        bit_acc = detection["restored_bit_accuracy"]
        if bit_acc is None:
            bit_acc = detection["raw_bit_accuracy"]

        report_label = threshold_info["report_label"]
        if detection["classifier_probability"] is None and report_label == "calibrated_deployment_verification":
            report_label = "official_profile_raw_scores"

        threshold_value = threshold_info["threshold"]
        threshold_text = "none" if threshold_value is None else f"{threshold_value:.6f}"
        return {
            "accuracies": [bit_acc],
            "bit_accuracies": [bit_acc],
            "p_values": [0.0],
            "l1_dist": [-detection["raw_ring_l1"]],
            "gm_raw_bit_accuracy": detection["raw_bit_accuracy"],
            "gm_restored_bit_accuracy": detection["restored_bit_accuracy"],
            "gm_raw_ring_l1": detection["raw_ring_l1"],
            "gm_ring_classifier_feature": detection["ring_classifier_feature"],
            "gm_classifier_probability": detection["classifier_probability"],
            "gm_score_definition": score_definition,
            "gm_used_gnr": detection["gnr_used"],
            "gm_used_classifier": detection["classifier_used"],
            "gm_report_label": report_label,
            "gm_threshold_source": threshold_info["threshold_source"],
            "gm_comparison_operator": threshold_info["comparison_operator"],
            "gm_official_reference_commit": OFFICIAL_GAUSSMARKER_COMMIT,
            "gm_official_reference_repo": OFFICIAL_GAUSSMARKER_REPO,
            "gm_state_source": self.state_source,
            "value": float(score),
            "detection_success": detection_success,
            "threshold": threshold_value,
            "log_message": (
                f"(WM type GM) label: {report_label}; score[{score_definition}]: {score:.6f}; "
                f"threshold: {threshold_text} ({threshold_info['threshold_source']}, "
                f"{threshold_info['comparison_operator']}); detection: {detection_success}"
            ),
        }

    def is_detection_successful(self, score: typing.Optional[float]) -> typing.Optional[bool]:
        return self.decide(score, self.resolve_threshold()["threshold"])
