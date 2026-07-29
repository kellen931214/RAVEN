"""HSQR (SFWMark) watermark provider — the single authoritative HSQR implementation.

Every HSQR algorithm lives here: the QR keybook, the center-region RFFT sign
injection, the inverse RFFT, the center-region FFT and the official complex L1
detector. Runners (``run_watermark.py``, ``run_verify_watermark.py``) and the
artifact helpers (``utils/wm/sfw_bundle.py``) may enumerate files, save/load
artifacts and call the methods below, but must never reimplement any of it.

Official reference:
    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb
    (``src/generate.py`` / ``src/detect.py`` / ``src/utils.py``)

Profiles
--------
``official_sfwmark_sd21``
    Immutable official profile: SD2.1-base, DDIM, float32, 512px, 50 steps,
    guidance 7.5, latent ``(B, 4, 64, 64)``, center slice ``10:54``, QR version 1
    / box size 2 / border 0 / EC ``H``, ``delta=0``, capacity 2048 and official
    base key seed **7433** (key ``i`` uses seed ``7433 + i``, payload
    ``HSQR{seed % 10000}``). Key identity is chosen by an explicit
    ``--hsqr_key_index``; no process-global RNG is involved.

``legacy_raven``
    The pre-Issue-#5 RAVEN behaviour, preserved byte-for-byte for the existing
    formal cohorts: base key seed 999999, ``fix_gt`` overloaded as an index into
    a process-global-RNG mapping of length 8192. This is **not**
    official-equivalent and is labelled as such everywhere.

Score semantics
---------------
``hsqr_l1_distance`` is the official raw mean complex L1 distance (lower means
more likely watermarked). ``hsqr_score = -hsqr_l1_distance`` is the official
canonical ROC score (higher means more likely watermarked) and is the only value
a threshold may be compared against, with ``score >= threshold``.
"""

import argparse
import typing
import warnings

import numpy as np
import torch

from .wm_provider import WmProvider
from . import sfw_bundle, sfw_inversion
from .sfw_bundle import SfwBundleError
from utils.image_utils import torch_to_PIL
from utils import utils


# [HSQR] hyperparameters
RADIUS = 14
RADIUS_CUTOFF = 3
w_channel = 3
TREE_WATERMARK_CHANNEL = [w_channel]
HSQR_WATERMARK_CHANNEL = [w_channel]
assert TREE_WATERMARK_CHANNEL == HSQR_WATERMARK_CHANNEL, "HSQR and Tree-Ring have the same channel in the paper."

#: Nominal official key capacity: 2^(14-3).
HSQR_WM_CAPACITY = 2 ** (RADIUS - RADIUS_CUTOFF)

#: Official SFWMark base watermark/key seed. Key ``i`` uses seed ``7433 + i``.
OFFICIAL_BASE_KEY_SEED = 7433

#: Legacy RAVEN base seed, kept only for the pre-Issue-#5 cohorts.
LEGACY_BASE_KEY_SEED = 999999

HSQR_SCORE_DEFINITION = "hsqr_negative_mean_complex_l1_distance"
HSQR_SCORE_DIRECTION = "higher_is_watermarked"
HSQR_COMPARISON_OPERATOR = ">="

#: The QR boolean pattern is mapped to these complex magnitudes by the official
#: detector target.
QR_TARGET_MAGNITUDE = 45.0

OFFICIAL_PROFILE_NAME = "official_sfwmark_sd21"
LEGACY_PROFILE_NAME = "legacy_raven"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

ERROR_CORRECTION_LEVELS = ("L", "M", "Q", "H")

#: Immutable official profile. Values are applied to ``args`` by
#: :func:`apply_arg_defaults` unless the user set them explicitly on the command
#: line; any explicit value is recorded as an override and the run stops being
#: an official run.
HSQR_OFFICIAL_SD21_PROFILE = {
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "scheduler_target": "DDIM",
    "resolution": 512,
    "num_inference_steps_target": 50,
    "guidance_scale_target": 7.5,
    "hsqr_torch_dtype": "float32",
    "hsqr_base_key_seed": OFFICIAL_BASE_KEY_SEED,
    "hsqr_key_selection": "explicit_index",
    "qr_version": 1,
    "box_size": 2,
    "hsqr_border": 0,
    "hsqr_error_correction": "H",
    "delta": 0,
    "hsqr_center_start": 10,
    "hsqr_center_end": 54,
    "hsqr_inversion_prompt": "",
    "hsqr_inversion_guidance": 0.0,
    "hsqr_inversion_steps": 50,
}

HSQR_PROFILES = {
    OFFICIAL_PROFILE_NAME: HSQR_OFFICIAL_SD21_PROFILE,
    # "legacy_raven" applies nothing and never claims official parity.
    LEGACY_PROFILE_NAME: {},
}


parser = argparse.ArgumentParser(add_help=False)
# ``legacy_raven`` remains the parser default so the pre-existing formal
# generators (experiments/generate_watermarked_images.py, run_removal.py) keep
# producing exactly the cohorts they produced before Issue #5. The standalone
# runners opt into the official profile explicitly.
parser.add_argument('--hsqr_profile', default=LEGACY_PROFILE_NAME, type=str,
                    choices=sorted(HSQR_PROFILES))
parser.add_argument('--hsqr_seed', default=LEGACY_BASE_KEY_SEED, type=int,
                    help="Legacy base key seed (and legacy global RNG seed).")
parser.add_argument('--hsqr_base_key_seed', default=None, type=int,
                    help="Base key seed; official SFWMark uses 7433. Defaults to --hsqr_seed.")
parser.add_argument('--hsqr_key_index', default=0, type=int,
                    help="Explicit key identity in [0, wm_capacity). Replaces the fix_gt overload.")
parser.add_argument('--hsqr_key_selection', default=None, type=str,
                    choices=["explicit_index", "legacy_fix_gt"],
                    help="Official runs use explicit_index. legacy_fix_gt reproduces the "
                         "pre-Issue-#5 global-RNG mapping and is never official.")
parser.add_argument('--hsqr_key_policy', default="fixed", type=str,
                    choices=["fixed", "per_sample"],
                    help="fixed: one key for the whole run. per_sample: a deterministic "
                         "per-sample key derived from the base key seed and persisted.")
parser.add_argument('--box_size', default=2, type=int)
parser.add_argument('--qr_version', default=1, type=int)
parser.add_argument('--hsqr_border', default=0, type=int)
parser.add_argument('--hsqr_error_correction', default="H", type=str, choices=list(ERROR_CORRECTION_LEVELS))
parser.add_argument('--delta', default=0, type=int)
parser.add_argument('--hsqr_wm_capacity', default=HSQR_WM_CAPACITY, type=int)
parser.add_argument('--hsqr_center_start', default=None, type=int)
parser.add_argument('--hsqr_center_end', default=None, type=int)
parser.add_argument('--hsqr_torch_dtype', default="float32", type=str, choices=sorted(TORCH_DTYPES))
# artifacts
parser.add_argument('--hsqr_bundle_dir', default=None, type=str,
                    help="Reusable HSQR bundle (manifest.json + selected_pattern.pt "
                         "[+ keybook.pt + key_mapping.json + threshold.json]).")
parser.add_argument('--hsqr_save_keybook', action='store_true', default=False,
                    help="Also persist the full 2048-pattern keybook for identification experiments.")
parser.add_argument('--hsqr_paired', action='store_true', default=False,
                    help="Paper mode: also generate the matched clean image from the same "
                         "pre-injection base latent.")
# inversion (official detect.py front-end)
parser.add_argument('--hsqr_inversion_prompt', default="", type=str)
parser.add_argument('--hsqr_inversion_guidance', default=0.0, type=float)
parser.add_argument('--hsqr_inversion_steps', default=50, type=int)
parser.add_argument('--hsqr_vae_scaling_factor', default=None, type=float,
                    help="Defaults to the loaded VAE's own scaling factor.")
# thresholds
parser.add_argument('--hsqr_target_fpr', default=0.01, type=float)
parser.add_argument('--hsqr_threshold', default=None, type=float,
                    help="Explicit user-supplied score threshold (score = -L1 distance).")
parser.add_argument('--hsqr_allow_legacy_threshold', action='store_true', default=False,
                    help="Permit the legacy nominal-FPR=1e-3 score threshold "
                         f"({-65.86233520507812}). Never a TPR@1%%FPR result.")


def apply_arg_defaults(args, argv) -> typing.Dict[str, typing.Any]:
    """Apply the selected HSQR profile without letting generic defaults win.

    Any value explicitly present on the command line takes precedence and is
    recorded in ``hsqr_profile_overrides``; a run with overrides is *not* an
    official run and is labelled as an ablation.
    """
    profile_name = getattr(args, "hsqr_profile", LEGACY_PROFILE_NAME)
    profile = HSQR_PROFILES.get(profile_name, {})
    argv = list(argv or [])

    def explicitly_set(name: str) -> bool:
        flag = f"--{name}"
        return any(token == flag or token.startswith(flag + "=") for token in argv)

    applied, overrides = {}, {}
    for name, value in profile.items():
        if explicitly_set(name):
            overrides[name] = getattr(args, name, None)
            continue
        setattr(args, name, value)
        applied[name] = value

    args.hsqr_profile_overrides = overrides
    args.hsqr_profile_is_official = (profile_name == OFFICIAL_PROFILE_NAME and not overrides)
    return {"profile": profile_name, "applied": applied, "overrides": overrides,
            "is_official": args.hsqr_profile_is_official}


class HSQRProvider(WmProvider):
    def __init__(self,
                 hsqr_seed: int = None,
                 box_size: int = 2,
                 qr_version: int = 1,
                 delta: int = 0,
                 latent_channel: int = 4,
                 start: int = 10,
                 end: int = 54, # 64-10 = hw_latent-start
                 hw_latent : int = 64,
                 fix_gt: int = 1,
                 wm_capacity: int = HSQR_WM_CAPACITY,
                 hsqr_profile: str = LEGACY_PROFILE_NAME,
                 hsqr_base_key_seed: typing.Optional[int] = None,
                 hsqr_key_index: int = 0,
                 hsqr_key_selection: typing.Optional[str] = None,
                 hsqr_key_policy: str = "fixed",
                 hsqr_border: int = 0,
                 hsqr_error_correction: str = "H",
                 hsqr_wm_capacity: typing.Optional[int] = None,
                 hsqr_center_start: typing.Optional[int] = None,
                 hsqr_center_end: typing.Optional[int] = None,
                 hsqr_torch_dtype: str = "float32",
                 hsqr_bundle_dir: typing.Optional[str] = None,
                 hsqr_inversion_prompt: str = "",
                 hsqr_inversion_guidance: float = 0.0,
                 hsqr_inversion_steps: int = 50,
                 hsqr_vae_scaling_factor: typing.Optional[float] = None,
                 hsqr_target_fpr: float = 0.01,
                 hsqr_threshold: typing.Optional[float] = None,
                 hsqr_allow_legacy_threshold: bool = False,
                 hsqr_profile_overrides: typing.Optional[typing.Mapping[str, typing.Any]] = None,
                 hsqr_profile_is_official: typing.Optional[bool] = None,
                 hsqr_pattern: typing.Optional[torch.Tensor] = None,
                 modelid_target: typing.Optional[str] = None,
                 model_revision: typing.Optional[str] = None,
                 scheduler_target: typing.Optional[str] = None,
                 resolution: int = 512,
                 **kwargs):

        super().__init__(**kwargs)

        self.profile = hsqr_profile
        self.profile_overrides = dict(hsqr_profile_overrides or {})
        self.profile_is_official = (
            hsqr_profile_is_official
            if hsqr_profile_is_official is not None
            else (hsqr_profile == OFFICIAL_PROFILE_NAME and not self.profile_overrides)
        )
        official = self.profile == OFFICIAL_PROFILE_NAME

        # ---- key space -----------------------------------------------------
        if hsqr_wm_capacity is not None:
            wm_capacity = hsqr_wm_capacity
        if wm_capacity != HSQR_WM_CAPACITY:
            raise ValueError(
                f"HSQR nominal key capacity is 2^(14-3)={HSQR_WM_CAPACITY}, got {wm_capacity}"
            )
        self.wm_capacity = wm_capacity

        self.hsqr_seed = hsqr_seed
        base_key_seed = hsqr_base_key_seed
        if base_key_seed is None:
            base_key_seed = hsqr_seed if hsqr_seed is not None else (
                OFFICIAL_BASE_KEY_SEED if official else LEGACY_BASE_KEY_SEED
            )
        self.base_key_seed = int(base_key_seed)
        if official and self.base_key_seed != OFFICIAL_BASE_KEY_SEED:
            raise ValueError(
                f"profile {OFFICIAL_PROFILE_NAME} requires the official base key seed "
                f"{OFFICIAL_BASE_KEY_SEED}, got {self.base_key_seed}"
            )

        self.key_selection = hsqr_key_selection or (
            "explicit_index" if official else "legacy_fix_gt"
        )
        if official and self.key_selection != "explicit_index":
            raise ValueError(
                f"profile {OFFICIAL_PROFILE_NAME} forbids key selection {self.key_selection!r}; "
                "the fix_gt/global-RNG overload is legacy only"
            )
        self.key_policy = hsqr_key_policy
        if self.key_policy not in ("fixed", "per_sample"):
            raise ValueError(f"unknown --hsqr_key_policy {self.key_policy!r}")

        # ---- QR profile ----------------------------------------------------
        if hsqr_error_correction not in ERROR_CORRECTION_LEVELS:
            raise ValueError(f"unknown QR error correction level {hsqr_error_correction!r}")
        self.qr_version = qr_version
        self.box_size = box_size
        self.border = hsqr_border
        self.error_correction = hsqr_error_correction
        self.delta = delta

        # ---- geometry ------------------------------------------------------
        start = hsqr_center_start if hsqr_center_start is not None else start
        end = hsqr_center_end if hsqr_center_end is not None else end
        self.start = start
        self.end = end
        self.center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
        self.shape = (1, latent_channel, hw_latent, hw_latent)
        self.hw_latent = hw_latent
        self.watermark_channels = list(HSQR_WATERMARK_CHANNEL)
        self._assert_geometry(latent_channel, hw_latent, official)

        self.hsqr_torch_dtype = hsqr_torch_dtype
        self.dtype = TORCH_DTYPES[hsqr_torch_dtype]

        # ---- detection / inversion configuration ---------------------------
        self.model_id = modelid_target
        self.model_revision = model_revision
        self.scheduler_type = scheduler_target
        self.resolution = int(resolution)
        self.inversion_prompt = hsqr_inversion_prompt
        self.inversion_guidance = float(hsqr_inversion_guidance)
        self.inversion_steps = int(hsqr_inversion_steps)
        self.vae_scaling_factor = hsqr_vae_scaling_factor
        # Official detect.py takes the deterministic VAE posterior mode.
        self.vae_sample = False
        self.target_fpr = float(hsqr_target_fpr)
        self.user_threshold = hsqr_threshold
        self.allow_legacy_threshold = bool(hsqr_allow_legacy_threshold)

        self.qr_generator = QRCodeGenerator(box_size=box_size, border=self.border,
                                            qr_version=qr_version,
                                            error_correction=self.error_correction)

        # ---- key identity --------------------------------------------------
        self._keybook: typing.Optional[torch.Tensor] = None
        self.legacy_identify_gt_indices: typing.Optional[typing.List[int]] = None
        self.sample_index = fix_gt  # kept for legacy call sites

        if hsqr_pattern is not None:
            # Loaded from a bundle: use the persisted pattern verbatim, never
            # silently regenerate and replace it.
            self.selected_key_index = int(hsqr_key_index)
            self.gt_patch = hsqr_pattern.to(torch.bool)
            sfw_bundle.validate_pattern(self.gt_patch, len(self.watermark_channels))
            self.pattern_source = "bundle"
        elif self.key_selection == "legacy_fix_gt":
            self.selected_key_index = self._legacy_key_index(fix_gt)
            self.gt_patch = self.make_pattern(self.selected_key_index)
            self.pattern_source = "legacy_fix_gt_global_rng"
        else:
            if not 0 <= int(hsqr_key_index) < self.wm_capacity:
                raise ValueError(
                    f"--hsqr_key_index must be in [0, {self.wm_capacity}), got {hsqr_key_index}"
                )
            self.selected_key_index = int(hsqr_key_index)
            self.gt_patch = self.make_pattern(self.selected_key_index)
            self.pattern_source = "explicit_index"

        self.bundle: typing.Optional[sfw_bundle.SfwBundle] = None
        self.bundle_dir = hsqr_bundle_dir

        # The official detection front-end differs from the generic RAVEN
        # inversion (fixed 512x512 resize, VAE posterior mode, inverse-scheduler
        # step). Exposing it as ``invert_images`` makes ``imprint_utils.validate``
        # use it — but only for official-profile runs, so the pre-existing legacy
        # cohorts keep the exact inversion path they were produced and scored with.
        if official:
            self.invert_images = self._sfw_invert_images

    # ------------------------------------------------------------------
    # Configuration / validation
    # ------------------------------------------------------------------

    @staticmethod
    def apply_arg_defaults(args, argv):
        """Hook used by ``run_watermark.py``; delegates to :func:`apply_arg_defaults`."""
        return apply_arg_defaults(args, argv)

    def _assert_geometry(self, latent_channel: int, hw_latent: int, official: bool) -> None:
        """Never apply the fixed HSQR geometry to an incompatible latent shape."""
        actual = tuple(int(d) for d in self.latent_shape)
        expected_tail = (latent_channel, hw_latent, hw_latent)
        center = self.end - self.start
        problems = []
        if actual[1:] != expected_tail:
            problems.append(
                f"pipeline latent shape {actual} does not match the configured HSQR geometry "
                f"(*, {latent_channel}, {hw_latent}, {hw_latent})"
            )
        if not 0 <= self.start < self.end <= hw_latent:
            problems.append(f"center slice {self.start}:{self.end} is outside 0:{hw_latent}")
        if max(self.watermark_channels) >= actual[1]:
            problems.append(
                f"watermark channel {self.watermark_channels} is outside the {actual[1]} latent channels"
            )
        if official:
            if actual[1:] != (4, 64, 64):
                problems.append(
                    f"profile {OFFICIAL_PROFILE_NAME} requires latent shape (B, 4, 64, 64), got {actual}"
                )
            if (self.start, self.end) != (10, 54):
                problems.append(
                    f"profile {OFFICIAL_PROFILE_NAME} requires center slice 10:54, "
                    f"got {self.start}:{self.end}"
                )
            if center != 44:
                problems.append(f"official center region must be 44x44, got {center}x{center}")
        if problems:
            message = "HSQR geometry is not usable: " + "; ".join(problems)
            if official:
                raise ValueError(message)
            warnings.warn(f"{message} (legacy profile: continuing unchanged)", RuntimeWarning)

    # ------------------------------------------------------------------
    # Key book
    # ------------------------------------------------------------------

    def key_seed(self, key_index: int) -> int:
        """Official key seed for an index: ``base_key_seed + i``."""
        if not 0 <= int(key_index) < self.wm_capacity:
            raise ValueError(f"key index {key_index} outside [0, {self.wm_capacity})")
        return self.base_key_seed + int(key_index)

    def payload_text(self, key_index: int) -> str:
        """Official QR payload format: ``HSQR{key_seed % 10000}``."""
        return f"HSQR{self.key_seed(key_index) % 10000}"

    def make_pattern(self, key_index: int) -> torch.Tensor:
        """Official HSQR pattern for one key index: boolean ``(c_wm, qr, qr)``."""
        return self._make_hsqr_pattern(self.key_seed(key_index))

    def keybook(self) -> torch.Tensor:
        """The full ``(wm_capacity, c_wm, qr, qr)`` boolean keybook (cached).

        Mirrors official ``pattern_list-2048.pt``.
        """
        if self._keybook is None:
            patterns = [self.make_pattern(index) for index in range(self.wm_capacity)]
            assert len(patterns) == self.wm_capacity
            self._keybook = torch.stack(patterns, dim=0)
        return self._keybook

    def _legacy_key_index(self, fix_gt: int) -> int:
        """Reproduce the pre-Issue-#5 ``fix_gt`` overload exactly.

        NOT official: the mapping comes from the process-global NumPy RNG seeded
        by ``--hsqr_seed`` and was never persisted, which is precisely why the
        official profile forbids it.
        """
        if self.hsqr_seed is not None:
            utils.set_random_seed(self.hsqr_seed)
        # The legacy implementation materialized the whole keybook before drawing
        # the mapping; keep that ordering so the drawn indices are unchanged.
        self.keybook()
        self.legacy_identify_gt_indices = np.random.choice(self.wm_capacity, size=8192).tolist()
        return int(self.legacy_identify_gt_indices[fix_gt])

    def pattern_sha256(self, pattern: typing.Optional[torch.Tensor] = None) -> str:
        return sfw_bundle.sha256_tensor(self.gt_patch if pattern is None else pattern)

    def key_identity(self) -> typing.Dict[str, typing.Any]:
        """Everything needed to reproduce and audit the selected watermark key."""
        return {
            "base_key_seed": self.base_key_seed,
            "selected_key_index": self.selected_key_index,
            "selected_key_seed": self.key_seed(self.selected_key_index),
            "payload_text": self.payload_text(self.selected_key_index),
            "selected_pattern_sha256": self.pattern_sha256(),
            "key_selection": self.key_selection,
            "pattern_source": self.pattern_source,
        }

    def sample_key_index(self, sample_id: int) -> int:
        """Key identity for one sample under the configured key policy.

        ``fixed`` keeps one key for the whole run. ``per_sample`` derives the key
        deterministically from the base key seed and the sample id — never from
        process-global RNG — and the resulting mapping is persisted in the bundle
        so verification can reload it.
        """
        if self.key_policy == "fixed":
            return self.selected_key_index
        digest = sfw_bundle.sha256_text(f"hsqr|{self.base_key_seed}|{int(sample_id)}")
        return int(digest[:16], 16) % self.wm_capacity

    def get_wm_type(self) -> str:
        return "HSQR"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def sample_base_latent(self, sample_seed: int) -> torch.Tensor:
        """One *independent complete* base latent per sample from an explicit seed.

        The generator is local, so the latent depends only on ``sample_seed`` —
        never on loop order, batch history or resume position.
        """
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
        latent = torch.randn(tuple(self.latent_shape), generator=generator, dtype=torch.float32)
        return latent.to(self.device, self.dtype)

    def inject(self, base_latent: torch.Tensor,
               pattern: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inject an HSQR pattern into a *clone* of ``base_latent``."""
        pattern = self.gt_patch if pattern is None else pattern
        return self.__inject_watermark(
            inverted_latent=base_latent.clone().to(self.device, self.dtype),
            qr_tensor=pattern,
            center=True,
        )

    def build_sample_latents(self, sample_seed: int,
                             pattern: typing.Optional[torch.Tensor] = None
                             ) -> typing.Dict[str, typing.Any]:
        """Paired clean / watermarked latents for one sample, with their hashes.

        ``clean_base_latent_sha256`` and ``watermark_pre_injection_base_latent_sha256``
        are computed from the two tensors that are actually used, so the pairing
        invariant is proven rather than assumed.
        """
        base_latent = self.sample_base_latent(sample_seed)
        clean_latent = base_latent.clone()
        pre_injection = base_latent.clone()
        watermarked = self.inject(pre_injection, pattern=pattern)
        return {
            "sample_seed": int(sample_seed),
            "clean_latent": clean_latent,
            "watermarked_latent": watermarked,
            "base_latent_sha256": sfw_bundle.sha256_tensor(base_latent),
            "clean_base_latent_sha256": sfw_bundle.sha256_tensor(clean_latent),
            "watermark_pre_injection_base_latent_sha256": sfw_bundle.sha256_tensor(pre_injection),
            "watermarked_latent_sha256": sfw_bundle.sha256_tensor(watermarked),
        }

    def get_wm_latents(self, latents_clean: torch.Tensor = None, seed: int = None) -> typing.Dict[str, typing.Any]:
        """
        Create (or inject into) latents and return a dict matching tr_provider.get_wm_latents() keys.
        """
        if seed is not None:
            utils.set_random_seed(seed)

        # Prepare clean latents
        if latents_clean is None:
            latents_clean = torch.randn(self.latent_shape)
        latents_clean = latents_clean.clone().to(self.device, self.dtype)

        latents_w = self.__inject_watermark(inverted_latent = latents_clean, qr_tensor=self.gt_patch, center=True)

        # clean
        latents_clean_torch = latents_clean.to(self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean_torch)
        # clean fft
        latents_clean_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_clean.to(torch.float32)), dim=(-1, -2)).real.to(self.device)
        latents_clean_fft_PIL = torch_to_PIL(latents_clean_fft_torch)
        # clean fft wchannel
        ch = TREE_WATERMARK_CHANNEL[0]
        latents_clean_fft_wchannel_torch = latents_clean_fft_torch[:, ch: ch + 1]
        latents_clean_fft_wchannel_PIL = torch_to_PIL(latents_clean_fft_wchannel_torch)

        # watermarked
        latents_w_torch = latents_w.to(self.device)
        latents_w_PIL = torch_to_PIL(latents_w_torch)
        # watermarked fft
        latents_w_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_w_torch), dim=(-1, -2)).real.to(self.device)
        latents_w_fft_PIL = torch_to_PIL(latents_w_fft_torch)
        # watermarked fft wchannel
        latents_w_fft_wchannel_torch = latents_w_fft_torch[:, ch: ch + 1].to(self.device)
        latents_w_fft_wchannel_PIL = torch_to_PIL(latents_w_fft_wchannel_torch)

        return {
            # clean
            "zT_clean_torch": latents_clean_torch,
            "zT_clean_PIL": latents_clean_PIL,
            "zT_clean": latents_clean_PIL,
            # clean fft
            "zT_clean_fft_torch": latents_clean_fft_torch,
            "zT_clean_fft_PIL": latents_clean_fft_PIL,
            "zT_clean_fft": latents_clean_fft_PIL,
            # clean fft wchannel
            "zT_clean_fft_wchannel_torch": latents_clean_fft_wchannel_torch,
            "zT_clean_fft_wchannel_PIL": latents_clean_fft_wchannel_PIL,
            "zT_clean_fft_wchannel": latents_clean_fft_wchannel_PIL,

            # watermarked
            "zT_torch": latents_w_torch,
            "zT_PIL": latents_w_PIL,
            "zT": latents_w_PIL,
            # watermarked fft
            "zT_fft_torch": latents_w_fft_torch,
            "zT_fft_PIL": latents_w_fft_PIL,
            "zT_fft": latents_w_fft_PIL,
            # watermarked fft wchannel
            "zT_fft_wchannel_torch": latents_w_fft_wchannel_torch,
            "zT_fft_wchannel_PIL": latents_w_fft_wchannel_PIL,
            "zT_fft_wchannel": latents_w_fft_wchannel_PIL,
        }

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def detector_target(self, qr_gt_bool: torch.Tensor) -> torch.Tensor:
        """Official detector target: QR booleans -> ±45, two halves -> complex (42,21)."""
        qr_pix_len = qr_gt_bool.shape[-1]
        qr_pix_half = (qr_pix_len + 1) // 2
        qr_gt_f32 = torch.where(
            qr_gt_bool,
            torch.tensor(QR_TARGET_MAGNITUDE),
            torch.tensor(-QR_TARGET_MAGNITUDE),
        ).to(torch.float32)
        qr_left = qr_gt_f32[0, :, :qr_pix_half]
        qr_right = qr_gt_f32[0, :, qr_pix_half:]
        return torch.complex(qr_left, qr_right)

    def __get_l1_distance(self, reversed_latents_w: typing.Union[torch.Tensor, np.ndarray],
                          qr_gt_bool, channel=HSQR_WATERMARK_CHANNEL,
                          p=1, center=False):
        """
        qr_gt_bool : (c_wm,42,42) boolean
        target_fft : (N,4,64,64) complex64

        Returns one distance per batch item, in input order. Batch element ``i``
        is scored against its own recovered latent — the pre-Issue-#5 code hard
        coded index 0 and silently ignored every later image.
        """
        Fourier_wm_zT_fft = torch.zeros_like(reversed_latents_w, dtype=torch.complex64)
        Fourier_wm_zT_fft[self.center_slice] = HSQRProvider.fft(reversed_latents_w[self.center_slice])

        target_fft = Fourier_wm_zT_fft

        center_row = target_fft.shape[-2] // 2 # 32
        qr_pix_len = qr_gt_bool.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_complex = self.detector_target(qr_gt_bool).to(target_fft.device) # (42,21) complex64
        if center:
            row_start = self.start + 1 # 11
            row_end = row_start + qr_pix_len # 53 = 11+42
            col_start = center_row + 1 # 33 = 32+1
            col_end = col_start + qr_pix_half # 54 = 33+21
        else:
            row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
            row_end = center_row + qr_pix_half
            col_start = center_row + 1 # 33
            col_end = col_start + qr_pix_half # 33+21
            # [TBD] the odd case will be updated
        qr_slice = (slice(None), channel, slice(row_start, row_end), slice(col_start, col_end))

        diff = torch.abs(qr_complex - target_fft[qr_slice]) # (N,c_wm,42,21)
        reduce_dims = tuple(range(1, diff.ndim))
        if p != 1:
            per_item = torch.norm(diff.flatten(1), p=p, dim=1) / diff[0].numel()
        else:
            per_item = diff.mean(dim=reduce_dims)

        return {
            "l1_dist": [float(value) for value in per_item.tolist()],
        }

    def l1_distances(self, latents: torch.Tensor,
                     pattern: typing.Optional[torch.Tensor] = None) -> typing.List[float]:
        """Official raw mean complex L1 distance, one value per batch item."""
        pattern = self.gt_patch if pattern is None else pattern
        results = self.__get_l1_distance(
            reversed_latents_w=latents, qr_gt_bool=pattern,
            channel=self.watermark_channels, p=1, center=True,
        )
        return results["l1_dist"]

    @staticmethod
    def score_from_distance(distance: float) -> float:
        """Official canonical ROC score: ``-L1 distance`` (higher is watermarked)."""
        return -float(distance)

    def detect_from_latent(self, latents: torch.Tensor,
                           pattern: typing.Optional[torch.Tensor] = None
                           ) -> typing.List[typing.Dict[str, typing.Any]]:
        """One detector record per batch item, in deterministic input order."""
        # A NaN/Inf anywhere in the recovered latent means the inversion failed;
        # the region-restricted L1 could otherwise still return a plausible-looking
        # number, which would read as a confident "not watermarked".
        if not torch.isfinite(latents).all():
            raise SfwBundleError(
                "HSQR detector received a non-finite recovered latent; the inversion failed"
            )
        if tuple(latents.shape)[1:] != tuple(self.latent_shape)[1:]:
            raise SfwBundleError(
                f"HSQR detector received latent shape {tuple(latents.shape)}, "
                f"expected (*, {', '.join(str(int(d)) for d in tuple(self.latent_shape)[1:])})"
            )
        distances = self.l1_distances(latents, pattern=pattern)
        identity = self.key_identity()
        records = []
        for distance in distances:
            if not np.isfinite(distance):
                raise SfwBundleError(f"HSQR detector produced a non-finite distance: {distance!r}")
            record = {
                "hsqr_l1_distance": float(distance),
                "hsqr_score": self.score_from_distance(distance),
                "score_definition": HSQR_SCORE_DEFINITION,
                "score_direction": HSQR_SCORE_DIRECTION,
            }
            record.update(identity)
            records.append(record)
        return records

    def identify(self, latents: torch.Tensor,
                 candidate_indices: typing.Optional[typing.Sequence[int]] = None
                 ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Compare each item against several candidate keys (paper identification).

        This is a *separate* API: it never changes the single-key verification
        semantics of :meth:`detect_from_latent`.
        """
        indices = (
            list(range(self.wm_capacity)) if candidate_indices is None
            else [int(index) for index in candidate_indices]
        )
        keybook = self.keybook()
        per_candidate = {index: self.l1_distances(latents, pattern=keybook[index]) for index in indices}
        batch = len(next(iter(per_candidate.values())))
        results = []
        for item in range(batch):
            scored = sorted(
                ((index, per_candidate[index][item]) for index in indices),
                key=lambda pair: (pair[1], pair[0]),
            )
            best_index, best_distance = scored[0]
            results.append(
                {
                    "identified_key_index": best_index,
                    "identified_key_seed": self.key_seed(best_index),
                    "identified_payload_text": self.payload_text(best_index),
                    "identified_l1_distance": float(best_distance),
                    "identified_score": self.score_from_distance(best_distance),
                    "candidate_count": len(indices),
                    "candidate_distances": {int(index): float(per_candidate[index][item])
                                            for index in indices},
                }
            )
        return results

    def get_accuracies(self,
                       latents: typing.Union[torch.Tensor, np.ndarray]) -> typing.Dict[str, typing.Any]:
        """
        Get the accuracy of the watermarking scheme

        @param latents: torch.Tensor or np.array, shape: self.latent_shape,

        @return: dict with one raw L1 distance per batch item (input order)
        """
        return {"l1_dist": self.l1_distances(latents)}

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def resolve_threshold(self) -> typing.Dict[str, typing.Any]:
        """Resolve the decision threshold, fail-closed and fully labelled.

        Precedence: explicit ``--hsqr_threshold`` > a bundled, hash-compatible
        ``threshold.json`` > the explicitly requested legacy threshold. When none
        applies, no binary decision is produced and raw scores are still emitted.
        """
        info = {
            "threshold": None,
            "distance_threshold": None,
            "threshold_source": "none",
            "threshold_available": False,
            "report_label": "official_profile_raw_scores" if self.profile_is_official
                            else "legacy_or_ablation_mode",
            "score_definition": HSQR_SCORE_DEFINITION,
            "score_direction": HSQR_SCORE_DIRECTION,
            "comparison_operator": HSQR_COMPARISON_OPERATOR,
            "threshold_target_fpr": None,
            "threshold_empirical_fpr": None,
            "threshold_nominal_fpr": None,
        }

        if self.user_threshold is not None:
            info.update(
                {
                    "threshold": float(self.user_threshold),
                    "distance_threshold": -float(self.user_threshold),
                    "threshold_source": "user_supplied",
                    "threshold_available": True,
                    "report_label": "user_supplied_threshold",
                }
            )
            return info

        if self.bundle is not None and self.bundle.has_threshold():
            artifact = self.bundle.load_threshold()
            sfw_bundle.assert_threshold_compatible(artifact, self.binding_config())
            info.update(
                {
                    "threshold": float(artifact["threshold"]),
                    "distance_threshold": float(artifact["distance_threshold"]),
                    "threshold_source": artifact["threshold_source"],
                    "threshold_available": True,
                    "report_label": artifact["report_label"],
                    "threshold_target_fpr": artifact.get("target_fpr"),
                    "threshold_empirical_fpr": artifact.get("empirical_fpr"),
                }
            )
            return info

        if self.allow_legacy_threshold:
            from utils.utils import describe_legacy_detection_threshold

            legacy = describe_legacy_detection_threshold("HSQR", self.model_id)
            info.update(
                {
                    "threshold": float(legacy["threshold"]),
                    "distance_threshold": -float(legacy["threshold"]),
                    "threshold_source": "legacy_default_threshold",
                    "threshold_available": True,
                    "report_label": "legacy_threshold",
                    "threshold_nominal_fpr": legacy["nominal_fpr"],
                }
            )
            return info

        return info

    @staticmethod
    def decide(score: typing.Optional[float],
               threshold: typing.Optional[float]) -> typing.Optional[bool]:
        """``score >= threshold`` with ``score = -L1 distance``.

        Returns ``None`` (undecided) rather than ``False`` when no threshold is
        available: a missing threshold must never look like a negative detection.
        """
        if score is None or threshold is None:
            return None
        return bool(float(score) >= float(threshold))

    def is_detection_successful(self, distance: typing.Optional[float]) -> typing.Optional[bool]:
        """Decide from a *raw distance* (the value `get_accuracies` returns)."""
        if distance is None:
            return None
        return self.decide(self.score_from_distance(distance), self.resolve_threshold()["threshold"])

    # ------------------------------------------------------------------
    # Bundle binding
    # ------------------------------------------------------------------

    def detector_config(self) -> typing.Dict[str, typing.Any]:
        """The configuration a score is only comparable within."""
        return {
            "method": "HSQR",
            "profile_name": self.profile,
            "profile_is_official": bool(self.profile_is_official),
            "profile_overrides": dict(self.profile_overrides),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "scheduler_type": self.scheduler_type,
            "torch_dtype": self.hsqr_torch_dtype,
            "resolution": int(self.resolution),
            "latent_shape": [int(d) for d in self.latent_shape],
            "center_slice": [int(self.start), int(self.end)],
            "watermark_channels": list(self.watermark_channels),
            "qr_version": int(self.qr_version),
            "box_size": int(self.box_size),
            "border": int(self.border),
            "error_correction": self.error_correction,
            "delta": int(self.delta),
            "wm_capacity": int(self.wm_capacity),
            "inversion_prompt_sha256": sfw_bundle.sha256_text(self.inversion_prompt),
            "inversion_guidance_scale": float(self.inversion_guidance),
            "inversion_steps": int(self.inversion_steps),
            "inversion_impl_version": sfw_inversion.SFW_INVERSION_IMPL_VERSION,
            "inversion_parity_status": sfw_inversion.SFW_INVERSION_PARITY_STATUS,
            "inversion_weights_parity": sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
            "vae_sample": bool(self.vae_sample),
            "vae_scaling_factor": float(self.vae_scaling_factor)
                                   if self.vae_scaling_factor is not None else None,
            "score_definition": HSQR_SCORE_DEFINITION,
            "score_direction": HSQR_SCORE_DIRECTION,
        }

    def bundle_manifest_config(self) -> typing.Dict[str, typing.Any]:
        config = self.detector_config()
        config.update(self.key_identity())
        config["key_policy"] = self.key_policy
        config.pop("selected_pattern_sha256", None)  # written by the bundle itself
        return config

    def bundle_compat_config(self) -> typing.Dict[str, typing.Any]:
        """Fields compared against an existing bundle before it may be reused."""
        config = self.bundle_manifest_config()
        # ``pattern_source`` legitimately differs (a reloaded provider says
        # "bundle"); every value that defines the watermark identity is compared.
        config.pop("pattern_source", None)
        config.pop("profile_is_official", None)
        return config

    def binding_config(self) -> typing.Dict[str, typing.Any]:
        config = self.detector_config()
        config.update(self.key_identity())
        config.pop("pattern_source", None)
        config.pop("profile_is_official", None)
        if self.bundle is not None:
            config["bundle_config_sha256"] = self.bundle.manifest.get("bundle_config_sha256")
        return config

    def attach_bundle(self, bundle: "sfw_bundle.SfwBundle") -> None:
        """Bind a loaded bundle after validating it against this configuration."""
        bundle.assert_compatible(self.bundle_compat_config())
        if bundle.manifest["selected_pattern_sha256"] != self.pattern_sha256():
            raise SfwBundleError(
                "HSQR bundle selected pattern does not match the provider's pattern "
                f"({bundle.manifest['selected_pattern_sha256']} != {self.pattern_sha256()})"
            )
        self.bundle = bundle
        self.bundle_dir = bundle.dir.as_posix()

    def create_bundle(self, directory, save_keybook: bool = False,
                      key_mapping: typing.Optional[typing.Sequence[int]] = None
                      ) -> "sfw_bundle.SfwBundle":
        bundle = sfw_bundle.SfwBundle.create(
            directory,
            pattern=self.gt_patch,
            config=self.bundle_manifest_config(),
            keybook=self.keybook() if save_keybook else None,
            key_mapping=key_mapping,
        )
        self.attach_bundle(bundle)
        return bundle

    @classmethod
    def from_bundle(cls, bundle: "sfw_bundle.SfwBundle", latent_shape, device,
                    **overrides) -> "HSQRProvider":
        """Rebuild a provider from a persisted bundle, using its stored pattern.

        The persisted pattern is used verbatim — never regenerated — so a bundle
        remains verifiable even if the QR library or the key derivation changes.
        """
        manifest = bundle.manifest
        kwargs = {
            "hsqr_profile": manifest["profile_name"],
            "hsqr_base_key_seed": manifest["base_key_seed"],
            "hsqr_key_index": manifest["selected_key_index"],
            # Officialness is a property of the run that *created* the bundle: a
            # generation run with profile overrides must not become "official"
            # again just because the bundle records the profile name.
            "hsqr_profile_is_official": manifest.get("profile_is_official"),
            "hsqr_profile_overrides": manifest.get("profile_overrides") or {},
            "hsqr_key_selection": manifest.get("key_selection", "explicit_index"),
            "hsqr_key_policy": manifest.get("key_policy", "fixed"),
            "hsqr_pattern": bundle.pattern,
            "qr_version": manifest["qr_version"],
            "box_size": manifest["box_size"],
            "hsqr_border": manifest["border"],
            "hsqr_error_correction": manifest["error_correction"],
            "delta": manifest["delta"],
            "hsqr_wm_capacity": manifest["wm_capacity"],
            "hsqr_center_start": manifest["center_slice"][0],
            "hsqr_center_end": manifest["center_slice"][1],
            "hsqr_torch_dtype": manifest["torch_dtype"],
            "modelid_target": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "scheduler_target": manifest["scheduler_type"],
            "resolution": manifest["resolution"],
            "hsqr_inversion_guidance": manifest["inversion_guidance_scale"],
            "hsqr_inversion_steps": manifest["inversion_steps"],
            "hsqr_vae_scaling_factor": manifest["vae_scaling_factor"],
            "latent_channel": manifest["latent_shape"][1],
            "hw_latent": manifest["latent_shape"][2],
        }
        kwargs.update(overrides)
        provider = cls(latent_shape=latent_shape, device=device, **kwargs)
        # The manifest stores the *hash* of the inversion prompt, so the prompt
        # itself is supplied by the caller and checked here.
        provider.attach_bundle(bundle)
        return provider

    # ------------------------------------------------------------------
    # Inversion (official SFWMark detect.py front-end)
    # ------------------------------------------------------------------

    def invert_pil_image(self, image, pipe_provider_target,
                         num_inference_steps: typing.Optional[int] = None
                         ) -> typing.Dict[str, typing.Any]:
        result = sfw_inversion.invert_pil_image(
            image,
            pipe_provider_target=pipe_provider_target,
            resolution=self.resolution,
            inversion_prompt=self.inversion_prompt,
            guidance_scale=self.inversion_guidance,
            num_inference_steps=self.inversion_steps if num_inference_steps is None
                                else num_inference_steps,
            vae_scaling_factor=self.vae_scaling_factor,
        )
        result["recovered_latent_sha256"] = sfw_bundle.sha256_tensor(result["zT_torch"])
        return result

    def _sfw_invert_images(self, images, pipe_provider_target,
                           num_inference_steps: int = 50,
                           callback_on_step_end=None,
                           callback_on_step_end_tensor_inputs=None):
        """Hook used by ``utils.imprint_utils.validate`` (official profile only)."""
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError("SFW inversion supports a single image per call")
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
    # Official HSQR math (unchanged)
    # ------------------------------------------------------------------

    def _make_hsqr_pattern(self, idx: int) -> torch.Tensor:

        data = f"HSQR{idx % 10000}"
        qr_tensor = self.qr_generator.make_qr_tensor(data=data) # (42,42) boolean tensor
        qr_tensor = qr_tensor.repeat(len(HSQR_WATERMARK_CHANNEL), 1, 1) # (c_wm,42,42) boolean tensor
        return qr_tensor # (c_wm,42,42) boolean tensor

    def __inject_watermark(self, inverted_latent, qr_tensor, center=False):

        qr_tensor = qr_tensor.unsqueeze(0) # (1,c_wm,42,42)
        assert len(qr_tensor.shape) == 4 # (N,c_wm,42,42)
        inverted_latent = inverted_latent.to(self.device)
        qr_tensor = qr_tensor.to(self.device)
        qr_pix_len = qr_tensor.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_left = qr_tensor[:, :, :, :qr_pix_half]    # (N,c_wm,42,21) boolean
        qr_right = qr_tensor[:, :, :, qr_pix_half:]   # (N,c_wm,42,21) boolean
        if center:
            # rfft
            center_latent_rfft = HSQRProvider.rfft(inverted_latent[self.center_slice]) # (N,4,44,44) -> # (N,4,44,23) complex64
            center_real_batch = center_latent_rfft.real # (N,4,44,23) f32
            center_imag_batch = center_latent_rfft.imag # (N,4,44,23) f32
            real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
            imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
            #center=True  [:,[3], 1:43,1:22] (N,1,42,21)
            center_real_batch[real_slice] = HSQRProvider.qr_abs(qr_left, center_real_batch[real_slice], delta=self.delta) # (N,c_wm,42,21)
            center_imag_batch[imag_slice] = HSQRProvider.qr_abs(qr_right, center_imag_batch[imag_slice], delta=self.delta) # (N,c_wm,42,21)
            center_latent_ifft = HSQRProvider.irfft(torch.complex(center_real_batch, center_imag_batch)) # (N,4,44,44) f32
            inverted_latent = inverted_latent.clone()
            inverted_latent[self.center_slice] = center_latent_ifft
            return inverted_latent # (N,4,64,64)
        else:
            # Coordinates for HSQR injection
            center_row = inverted_latent.shape[-2] // 2 # 32
            row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
            row_end = center_row + qr_pix_half
            col_end_left = 1 + qr_pix_half
            col_end_right = qr_pix_half if qr_pix_len % 2 else col_end_left # if odd length QR, shortend by 1 pix
            # rfft
            latent_rfft = HSQRProvider.rfft(inverted_latent) # (N,4,64,64) -> (N,4,64,33) complex64
            real_batch = latent_rfft.real # (N,4,64,33) f32
            imag_batch = latent_rfft.imag # (N,4,64,33) f32
            # Inject HSQR
            # [row_start 11 = 32-21 : row_end 53 = 32+21] / [col_start 1 : col_end_left 22 = 1+21 = col_end_right 22]
            real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_left))
            imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_right))
            #center=False [:,[3],11:53,1:22] (N,1,42,21)
            real_batch[real_slice] = HSQRProvider.qr_abs(qr_left, real_batch[real_slice], delta=self.delta) # (N,c_wm,42,21)
            imag_batch[imag_slice] = HSQRProvider.qr_abs(qr_right, imag_batch[imag_slice], delta=self.delta) # (N,c_wm,42,21)
            return HSQRProvider.irfft(torch.complex(real_batch, imag_batch)) # (N,4,64,64)

    @staticmethod
    def fft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))

    @staticmethod
    def ifft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))

    @staticmethod
    def rfft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.rfft2(input_tensor, dim=(-2,-1)), dim=-2)

    @staticmethod
    def irfft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.irfft2(torch.fft.ifftshift(input_tensor, dim=-2), dim=(-2,-1), s=(input_tensor.shape[-2],input_tensor.shape[-2]))

    @staticmethod
    def qr_abs(boolean_tensor, input_tensor, delta=0): # boolean → qr_abs tensor
        return torch.where(boolean_tensor, input_tensor.abs() + delta, -input_tensor.abs() - delta)


# qrcoder mask
import qrcode
class QRCodeGenerator:
    def __init__(self, box_size=2, border=1, qr_version=1, error_correction="H"):
        self.qr_version = qr_version
        self.box_size = box_size
        self.border = border
        self.error_correction = error_correction
        self.qr = qrcode.QRCode(
            version=qr_version, box_size=box_size, border=border,
            error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{error_correction}"),
        )

    def make_qr_tensor(self, data, filename='qrcode.png', save_img=False):
        self.qr.add_data(data)
        self.qr.make(fit=True)
        if self.qr.version != self.qr_version:
            # ``fit=True`` silently upgrades the version when the payload does not
            # fit, which would change the pattern size and break the fixed latent
            # geometry. Fail instead of producing an off-spec pattern.
            version = self.qr.version
            self.clear()
            raise ValueError(
                f"QR payload {data!r} does not fit version {self.qr_version} at error "
                f"correction {self.error_correction}; qrcode selected version {version}"
            )
        img = self.qr.make_image(fill_color="black", back_color="white")
        if save_img:
            img.save(filename)
        self.clear()
        img_array = np.array(img)
        tensor = torch.from_numpy(img_array)
        return tensor.clone().detach() # boolean (h,w)

    def clear(self):
        self.qr.clear()
        self.qr.version = self.qr_version
