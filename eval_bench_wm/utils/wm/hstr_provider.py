from __future__ import annotations

import argparse
import typing

import numpy as np
import torch
import torchvision

from . import hstr_bundle
from .wm_provider import WmProvider
from utils.canonical import canonical_json_sha256, tensor_sha256
from utils.image_utils import torch_to_PIL
from utils import utils


OFFICIAL_HSTR_PROFILE = "official_sfwmark_sd21"
LEGACY_HSTR_PROFILE = "legacy_raven"
OFFICIAL_BASE_KEY_SEED = 7433
OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
OFFICIAL_SCHEDULER = "DDIM"
OFFICIAL_RESOLUTION = 512
OFFICIAL_STEPS = 50
OFFICIAL_GUIDANCE_SCALE = 7.5
HSTR_SCORE_DEFINITION = "hstr_score=-min(channel_0_l1,channel_3_l1)"
HSTR_SCORE_DIRECTION = "higher_is_watermarked"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--hstr_seed", default=None, type=int)
parser.add_argument("--hstr_profile", default=LEGACY_HSTR_PROFILE, choices=[LEGACY_HSTR_PROFILE, OFFICIAL_HSTR_PROFILE])
parser.add_argument("--hstr_key_index", default=1, type=int)
parser.add_argument("--hstr_bundle_dir", default=None, type=str)
parser.add_argument("--hstr_create_bundle", action="store_true", default=False)
parser.add_argument("--hstr_save_full_keybook", action="store_true", default=False)
parser.add_argument("--hstr_overwrite_bundle", action="store_true", default=False)
parser.add_argument("--hstr_rng_device", default=None, choices=["cpu", "cuda"])
parser.add_argument("--hstr_threshold", default=None, type=float)
parser.add_argument("--latent_channel", default=4, type=int)

RADIUS = 14
RADIUS_CUTOFF = 3
USE_ROUNDER_RING = True

w_channel = 3
TREE_WATERMARK_CHANNEL = [w_channel]
HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [w_channel]
RINGID_WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)


class HSTRProvider(WmProvider):
    def __init__(
        self,
        hstr_seed: int | None = None,
        latent_channel: int = 4,
        start: int = 10,
        end: int = 54,
        hw_latent: int = 64,
        fix_gt: int = 1,
        wm_capacity: int = 2 ** (RADIUS - RADIUS_CUTOFF),
        hstr_profile: str = LEGACY_HSTR_PROFILE,
        hstr_key_index: int = 1,
        hstr_bundle_dir: str | None = None,
        hstr_create_bundle: bool = False,
        hstr_save_full_keybook: bool = False,
        hstr_overwrite_bundle: bool = False,
        hstr_rng_device: str | None = None,
        hstr_threshold: float | None = None,
        modelid_target: str | None = None,
        model_revision: str | None = None,
        scheduler_target: str | None = None,
        resolution: int = OFFICIAL_RESOLUTION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hstr_profile not in (LEGACY_HSTR_PROFILE, OFFICIAL_HSTR_PROFILE):
            raise ValueError(f"unknown HSTR profile {hstr_profile!r}")
        if wm_capacity != 2048:
            raise ValueError("official HSTR wm_capacity must be 2048")

        self.profile = hstr_profile
        self.shape = (1, latent_channel, hw_latent, hw_latent)
        self.hstr_seed = hstr_seed
        self.wm_capacity = int(wm_capacity)
        self.base_key_seed = OFFICIAL_BASE_KEY_SEED if self.profile == OFFICIAL_HSTR_PROFILE else (999999 if hstr_seed is None else int(hstr_seed))
        self.hstr_seed_list = list(range(self.base_key_seed, self.base_key_seed + self.wm_capacity))
        self.fix_gt = int(fix_gt)
        self.key_index = int(hstr_key_index if self.profile == OFFICIAL_HSTR_PROFILE else self._legacy_key_index(self.fix_gt))
        if not 0 <= self.key_index < self.wm_capacity:
            raise ValueError(f"--hstr_key_index must be in [0, {self.wm_capacity - 1}], got {self.key_index}")
        self.selected_key_seed = int(self.hstr_seed_list[self.key_index])
        self.model_id = modelid_target
        self.model_revision = model_revision
        self.scheduler_type = scheduler_target
        self.resolution = int(resolution)
        self.hstr_threshold = hstr_threshold
        self.inversion_steps = OFFICIAL_STEPS
        self.inversion_prompt = ""
        self.inversion_guidance = 0.0
        self.rng_device = hstr_rng_device or ("cuda" if self.device.type == "cuda" else "cpu")
        self.start = int(start)
        self.end = int(end)
        self.center_slice = (slice(None), slice(None), slice(self.start, self.end), slice(self.start, self.end))
        if self.profile == OFFICIAL_HSTR_PROFILE:
            self._validate_official_profile()

        self.masks, self.watermark_region_mask_hstr = self.__get_watermarking_mask()
        self.bundle: hstr_bundle.HstrBundle | None = None
        self.state_source = "in_memory"
        full_keybook = None
        if hstr_bundle_dir and not hstr_create_bundle:
            self.bundle = hstr_bundle.HstrBundle.load(hstr_bundle_dir)
            self.gt_patch = self.bundle.selected_pattern.to(self.device)
            self.bundle.assert_compatible(self.provider_config())
            self.state_source = "bundle"
        else:
            self.gt_patch = self.__get_watermarking_pattern()
            if hstr_save_full_keybook:
                full_keybook = torch.stack(self.__get_watermarking_pattern_list(), dim=0)
            if hstr_bundle_dir:
                self.bundle = hstr_bundle.HstrBundle.create(
                    hstr_bundle_dir,
                    provider_config=self.provider_config(),
                    selected_pattern=self.gt_patch,
                    full_keybook=full_keybook,
                    overwrite=hstr_overwrite_bundle,
                )
                self.gt_patch = self.bundle.selected_pattern.to(self.device)
                self.state_source = "bundle_created"

        self.selected_pattern_sha256 = tensor_sha256(self.gt_patch)
        self.watermark_mask_sha256 = tensor_sha256(self.watermark_region_mask_hstr)

    @staticmethod
    def _legacy_key_index(fix_gt: int) -> int:
        rng = np.random.default_rng(999999 + 3)
        return int(rng.choice(2048, size=8192).tolist()[int(fix_gt)])

    @classmethod
    def apply_arg_defaults(cls, args, argv=None):
        argv = list(argv or [])
        if getattr(args, "hstr_profile", LEGACY_HSTR_PROFILE) != OFFICIAL_HSTR_PROFILE:
            return {"profile": getattr(args, "hstr_profile", LEGACY_HSTR_PROFILE), "overrides": {}}
        explicit = {item for item in argv if item.startswith("--")}
        defaults = {
            "modelid_target": OFFICIAL_MODEL_ID,
            "scheduler_target": OFFICIAL_SCHEDULER,
            "resolution": OFFICIAL_RESOLUTION,
            "num_inference_steps_target": OFFICIAL_STEPS,
            "guidance_scale_target": OFFICIAL_GUIDANCE_SCALE,
        }
        applied = {"profile": OFFICIAL_HSTR_PROFILE, "overrides": {}}
        for field, value in defaults.items():
            if f"--{field}" not in explicit:
                setattr(args, field, value)
            elif getattr(args, field, None) != value:
                applied["overrides"][field] = getattr(args, field, None)
        return applied

    def _validate_official_profile(self) -> None:
        if tuple(self.latent_shape) != (1, 4, 64, 64):
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires latent_shape=(1,4,64,64), got {tuple(self.latent_shape)}")
        if self.shape != (1, 4, 64, 64):
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires shape=(1,4,64,64), got {self.shape}")
        if self.start != 10 or self.end != 54:
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires center slice 10:54")
        if self.model_id not in (None, OFFICIAL_MODEL_ID):
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires model {OFFICIAL_MODEL_ID}")
        if self.scheduler_type not in (None, OFFICIAL_SCHEDULER):
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires scheduler {OFFICIAL_SCHEDULER}")
        if self.resolution != OFFICIAL_RESOLUTION:
            raise ValueError(f"{OFFICIAL_HSTR_PROFILE} requires resolution {OFFICIAL_RESOLUTION}")

    def get_wm_type(self) -> str:
        return "HSTR"

    def provider_config(self) -> dict[str, typing.Any]:
        return hstr_bundle.build_provider_config(
            profile_name=self.profile,
            model_id=self.model_id,
            model_revision=self.model_revision,
            scheduler_type=self.scheduler_type,
            resolution=self.resolution,
            latent_shape=list(self.latent_shape),
            center_slice=[self.start, self.end],
            radius=RADIUS,
            radius_cutoff=RADIUS_CUTOFF,
            watermark_channels=TREE_WATERMARK_CHANNEL,
            heterogeneous_channels=HETER_WATERMARK_CHANNEL,
            wm_capacity=self.wm_capacity,
            base_key_seed=self.base_key_seed,
            selected_key_index=self.key_index,
            selected_key_seed=self.selected_key_seed,
            rng_algorithm=(
                "torch.Generator(device).manual_seed(key_seed)+torch.randn_like_diffusers_prepare_latents"
                if self.profile == OFFICIAL_HSTR_PROFILE else "legacy_explicit_reconstruction_of_previous_key_index"
            ),
            rng_device=str(self.rng_device),
            runtime_dtype=str(self.dtype),
        )

    def provider_config_sha256(self) -> str:
        return canonical_json_sha256(self.provider_config())

    def binding_config(self) -> dict[str, typing.Any]:
        payload = self.provider_config()
        payload.update({
            "provider_config_sha256": self.provider_config_sha256(),
            "selected_pattern_sha256": self.selected_pattern_sha256,
            "watermark_mask_sha256": self.watermark_mask_sha256,
        })
        if self.bundle is not None and self.bundle.manifest is not None:
            payload["bundle_sha256"] = hstr_bundle.sha256_file(self.bundle.manifest_path)
        return payload

    def sample_base_latent(self, sample_seed: int | None = None) -> torch.Tensor:
        if sample_seed is None:
            sample_seed = self.base_key_seed
        if self.profile == OFFICIAL_HSTR_PROFILE:
            device = torch.device(self.rng_device)
            generator = torch.Generator(device=device).manual_seed(int(sample_seed))
            return torch.randn(tuple(self.latent_shape), generator=generator, device=device, dtype=self.dtype).to(self.device)
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
        return torch.randn(tuple(self.latent_shape), generator=generator, dtype=torch.float32).to(self.device, self.dtype)

    def get_wm_latents(self, latents_clean: torch.Tensor = None, seed: int = None) -> dict[str, typing.Any]:
        if seed is not None and self.profile != OFFICIAL_HSTR_PROFILE:
            utils.set_random_seed(seed)
        if latents_clean is None:
            latents_clean = self.sample_base_latent(seed)
        latents_clean = latents_clean.clone().to(self.device, self.dtype)
        latents_w, _ = self.__inject_watermark(latents_clean, self.gt_patch, self.masks, center=True, cut_real=False)

        latents_clean_torch = latents_clean.to(self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean_torch)
        latents_clean_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_clean.to(torch.float32)), dim=(-1, -2)).real.to(self.device)
        latents_clean_fft_PIL = torch_to_PIL(latents_clean_fft_torch)
        ch = TREE_WATERMARK_CHANNEL[0]
        latents_clean_fft_wchannel_torch = latents_clean_fft_torch[:, ch: ch + 1]
        latents_clean_fft_wchannel_PIL = torch_to_PIL(latents_clean_fft_wchannel_torch)

        latents_w_torch = latents_w.to(self.device)
        latents_w_PIL = torch_to_PIL(latents_w_torch)
        latents_w_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_w_torch), dim=(-1, -2)).real.to(self.device)
        latents_w_fft_PIL = torch_to_PIL(latents_w_fft_torch)
        latents_w_fft_wchannel_torch = latents_w_fft_torch[:, ch: ch + 1].to(self.device)
        latents_w_fft_wchannel_PIL = torch_to_PIL(latents_w_fft_wchannel_torch)

        return {
            "zT_clean_torch": latents_clean_torch,
            "zT_clean_PIL": latents_clean_PIL,
            "zT_clean": latents_clean_PIL,
            "zT_clean_fft_torch": latents_clean_fft_torch,
            "zT_clean_fft_PIL": latents_clean_fft_PIL,
            "zT_clean_fft": latents_clean_fft_PIL,
            "zT_clean_fft_wchannel_torch": latents_clean_fft_wchannel_torch,
            "zT_clean_fft_wchannel_PIL": latents_clean_fft_wchannel_PIL,
            "zT_clean_fft_wchannel": latents_clean_fft_wchannel_PIL,
            "zT_torch": latents_w_torch,
            "zT_PIL": latents_w_PIL,
            "zT": latents_w_PIL,
            "zT_fft_torch": latents_w_fft_torch,
            "zT_fft_PIL": latents_w_fft_PIL,
            "zT_fft": latents_w_fft_PIL,
            "zT_fft_wchannel_torch": latents_w_fft_wchannel_torch,
            "zT_fft_wchannel_PIL": latents_w_fft_wchannel_PIL,
            "zT_fft_wchannel": latents_w_fft_wchannel_PIL,
            "selected_key_index": self.key_index,
            "selected_key_seed": self.selected_key_seed,
            "selected_pattern_sha256": self.selected_pattern_sha256,
        }

    def _pattern_for_batch(self, batch_size: int) -> torch.Tensor:
        pattern = self.gt_patch.to(self.device)
        if pattern.shape[0] == batch_size:
            return pattern
        if pattern.shape[0] == 1:
            return pattern.repeat(batch_size, 1, 1, 1)
        raise ValueError(f"HSTR pattern batch {pattern.shape[0]} does not match latent batch {batch_size}")

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.ndarray]) -> dict[str, typing.Any]:
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        latents = latents.to(self.device)
        results = self.__get_l1_distance(
            reversed_latents_w=latents,
            mask=self.watermark_region_mask_hstr,
            channel=RINGID_WATERMARK_CHANNEL,
            p=1,
            mode="complex",
            channel_min=True,
            center=True,
        )
        items = results["items"]
        l1_dist = [item["l1_dist"] for item in items]
        scores = [-float(value) for value in l1_dist]
        return {
            "l1_dist": l1_dist,
            "hstr_channel_0_l1": [item["hstr_channel_0_l1"] for item in items],
            "hstr_channel_3_l1": [item["hstr_channel_3_l1"] for item in items],
            "hstr_channel_min_l1": l1_dist,
            "hstr_score": scores,
            "score_direction": HSTR_SCORE_DIRECTION,
            "score_definition": HSTR_SCORE_DEFINITION,
            "selected_key_index": self.key_index,
            "selected_key_seed": self.selected_key_seed,
            "selected_pattern_sha256": self.selected_pattern_sha256,
            "provider_config_sha256": self.provider_config_sha256(),
        }

    def __get_l1_distance(self, reversed_latents_w, mask, channel=RINGID_WATERMARK_CHANNEL, p=1, mode="complex", channel_min=False, center=False):
        Fourier_wm_zT_fft = torch.zeros_like(reversed_latents_w, dtype=torch.complex64)
        Fourier_wm_zT_fft[self.center_slice] = HSTRProvider.fft(reversed_latents_w[self.center_slice])
        target = self._pattern_for_batch(Fourier_wm_zT_fft.shape[0])
        if Fourier_wm_zT_fft.shape != target.shape:
            raise ValueError(f"Shape mismatch during eval: {Fourier_wm_zT_fft.shape} vs {target.shape}")
        if mode not in ["complex", "real", "imag"]:
            raise NotImplementedError(f"Eval mode not implemented: {mode}")

        def calc_diff(t1, t2, m):
            if mode == "complex":
                diff = torch.abs(t1 - t2)
            elif mode == "real":
                diff = torch.abs(t1.real - t2.real)
            else:
                diff = torch.abs(t1.imag - t2.imag)
            return diff if m is None else diff[m]

        items = []
        if center:
            temp_tensor1 = Fourier_wm_zT_fft[self.center_slice].clone()
            temp_tensor2 = target[self.center_slice].clone()
            temp_mask = mask[None, ...][self.center_slice][0].clone()
            for batch_index in range(Fourier_wm_zT_fft.shape[0]):
                if not channel_min:
                    diff = calc_diff(temp_tensor1[batch_index][channel], temp_tensor2[batch_index][channel], temp_mask)
                    value = torch.norm(diff, p=p).item() / torch.sum(temp_mask) if p != 1 else torch.mean(diff).item()
                    items.append({"l1_dist": float(value)})
                else:
                    if p != 1 or channel != RINGID_WATERMARK_CHANNEL:
                        raise NotImplementedError
                    diff = calc_diff(temp_tensor1[batch_index][channel], temp_tensor2[batch_index][channel], None)
                    l1_list = [torch.mean(diff[i][temp_mask[i]]).item() for i in range(len(channel))]
                    items.append({
                        "hstr_channel_0_l1": float(l1_list[0]),
                        "hstr_channel_3_l1": float(l1_list[1]),
                        "l1_dist": float(min(l1_list)),
                    })
        else:
            for batch_index in range(Fourier_wm_zT_fft.shape[0]):
                if not channel_min:
                    diff = calc_diff(Fourier_wm_zT_fft[batch_index][channel], target[batch_index][channel], mask)
                    value = torch.norm(diff, p=p).item() / torch.sum(mask) if p != 1 else torch.mean(diff).item()
                    items.append({"l1_dist": float(value)})
                else:
                    if p != 1:
                        raise NotImplementedError
                    diff = calc_diff(Fourier_wm_zT_fft[batch_index][channel], target[batch_index][channel], None)
                    l1_list = [torch.mean(diff[i][mask[i]]).item() for i in range(len(channel))]
                    items.append({
                        "hstr_channel_0_l1": float(l1_list[0]),
                        "hstr_channel_3_l1": float(l1_list[1]),
                        "l1_dist": float(min(l1_list)),
                    })
        return {"items": items}

    def __get_watermarking_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        single_channel_tree_watermark_mask = torch.tensor(circle_mask(size=self.latent_shape[-1], r=RADIUS))
        single_channel_heter_watermark_mask = torch.tensor(ring_mask(size=self.latent_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF))
        masks = torch.zeros(self.latent_shape, dtype=torch.bool)
        masks[:, TREE_WATERMARK_CHANNEL] = single_channel_tree_watermark_mask
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask
        watermark_region_mask_hstr = torch.stack([
            single_channel_heter_watermark_mask,
            single_channel_tree_watermark_mask,
        ]).to(self.device)
        return masks, watermark_region_mask_hstr

    def __get_watermarking_pattern(self) -> torch.Tensor:
        return self.__make_Fourier_treering_pattern(
            self.shape,
            self.selected_key_seed,
            hs=True,
            center=True,
            heter=True,
        )

    def __get_watermarking_pattern_list(self) -> list[torch.Tensor]:
        return [
            self.__make_Fourier_treering_pattern(self.shape, seed, hs=True, center=True, heter=True)
            for seed in self.hstr_seed_list
        ]

    def __key_seed_latent(self, hstr_seed: int) -> torch.Tensor:
        if self.profile == OFFICIAL_HSTR_PROFILE:
            device = torch.device(self.rng_device)
            generator = torch.Generator(device=device).manual_seed(int(hstr_seed))
            return torch.randn(self.shape, generator=generator, device=device, dtype=self.dtype).to(self.device)
        generator = torch.Generator(device="cpu").manual_seed(int(hstr_seed))
        return torch.randn(self.shape, generator=generator, dtype=torch.float32).to(self.device)

    def __make_Fourier_treering_pattern(self, shape, hstr_seed, hs=False, center=False, heter=False):
        assert shape[-1] == shape[-2]
        gt_init = self.__key_seed_latent(hstr_seed)
        if center:
            watermarked_latents_fft = HSTRProvider.fft(torch.zeros(shape, device=self.device))
            gt_patch_tmp = HSTRProvider.fft(gt_init[self.center_slice]).clone().detach().to(torch.complex64)
            center_len = gt_patch_tmp.shape[-1] // 2
            for radius in range(center_len - 1, 0, -1):
                tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius), device=self.device)
                for j in range(watermarked_latents_fft.shape[1]):
                    watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item()
            if heter:
                watermarked_latents_fft[:, HETER_WATERMARK_CHANNEL, self.start:self.end, self.start:self.end] = gt_patch_tmp[:, HETER_WATERMARK_CHANNEL]
        else:
            watermarked_latents_fft = HSTRProvider.fft(gt_init)
            gt_patch_tmp = watermarked_latents_fft.clone().detach()
            center_len = shape[-1] // 2
            for radius in range(center_len - 1, 0, -1):
                tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius), device=self.device)
                for j in range(watermarked_latents_fft.shape[1]):
                    watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item()
        if hs:
            return HSTRProvider.enforce_hermitian_symmetry(watermarked_latents_fft)
        return watermarked_latents_fft

    def __inject_watermark(self, inverted_latent, w_pattern, w_mask, cut_real=True, center=False):
        assert len(w_pattern.shape) == 4
        assert len(w_mask.shape) == 4
        batch_size = inverted_latent.shape[0]
        w_mask = w_mask.repeat(batch_size, 1, 1, 1).to(self.device)
        inverted_latent = inverted_latent.to(self.device)
        w_pattern = self._pattern_for_batch(batch_size).to(self.device)
        if center:
            center_latent_fft = HSTRProvider.fft(inverted_latent[self.center_slice])
            temp_mask = w_mask[self.center_slice]
            temp_pattern = w_pattern[self.center_slice]
            center_latent_fft[temp_mask] = temp_pattern[temp_mask].clone()
            center_latent_ifft = HSTRProvider.ifft(center_latent_fft)
            center_latent_ifft = center_latent_ifft.real if cut_real or center_latent_ifft.imag.abs().max() < 1e-3 else center_latent_ifft
            inverted_latent = inverted_latent.clone()
            inverted_latent[self.center_slice] = center_latent_ifft
            inverted_latent_fft = None
        else:
            inverted_latent_fft = HSTRProvider.fft(inverted_latent)
            inverted_latent_fft[w_mask] = w_pattern[w_mask].clone()
            inverted_latent = HSTRProvider.ifft(inverted_latent_fft)
            inverted_latent = inverted_latent.real if cut_real or inverted_latent.imag.abs().max() < 1e-3 else inverted_latent
        inverted_latent[inverted_latent == float("Inf")] = 4
        inverted_latent[inverted_latent == float("-Inf")] = -4
        return inverted_latent, inverted_latent_fft


    def generate(
        self,
        pipe_provider_target,
        prompts,
        latents: torch.Tensor,
        num_inference_steps: int = OFFICIAL_STEPS,
        guidance_scale: float = OFFICIAL_GUIDANCE_SCALE,
    ) -> dict[str, typing.Any]:
        return pipe_provider_target.generate(
            prompts=prompts,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def _official_image_tensor(self, image_pil):
        from PIL import Image

        images = [image_pil] if isinstance(image_pil, Image.Image) else list(image_pil)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.resolution, self.resolution)),
            torchvision.transforms.ToTensor(),
        ])
        tensor = torch.stack([2.0 * transform(image.convert("RGB")) - 1.0 for image in images])
        return tensor

    @torch.no_grad()
    def invert_pil_image(self, image_pil, pipe_provider_target, image_sha256: str | None = None) -> dict[str, typing.Any]:
        """Frozen SFWMark detection inversion: VAE mode + DDIM inverse + guidance 0."""
        from diffusers import DDIMInverseScheduler
        from utils.canonical import tensor_sha256 as _tensor_sha256

        # Load through the existing provider so model provenance and scheduler setup stay centralized.
        pipe_provider_target._PipeProvider__load_pipe()
        pipe = pipe_provider_target.pipe
        current_scheduler = pipe.scheduler
        try:
            pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
            image_tensor = self._official_image_tensor(image_pil).to(pipe.unet.dtype).to(pipe.device)
            z0_torch = pipe.vae.encode(image_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor
            prompt = [""] * z0_torch.shape[0]
            zT_torch = pipe(
                prompt=prompt,
                latents=z0_torch,
                guidance_scale=self.inversion_guidance,
                num_inference_steps=self.inversion_steps,
                output_type="latent",
            ).images
        finally:
            pipe.scheduler = current_scheduler
        return {
            "z0_torch": z0_torch.detach(),
            "zT_torch": zT_torch.detach(),
            "inversion_seed": None,
            "inversion_steps": int(self.inversion_steps),
            "inversion_prompt_sha256": hstr_bundle.sha256_text(self.inversion_prompt),
            "inversion_guidance_scale": float(self.inversion_guidance),
            "recovered_latent_sha256": _tensor_sha256(zT_torch.detach()),
            "image_sha256": image_sha256,
        }

    def detect_from_latent(self, latents: torch.Tensor) -> dict[str, typing.Any]:
        if not torch.isfinite(latents.detach().to(torch.float32)).all():
            raise ValueError("HSTR detection received a non-finite latent")
        scores = self.get_accuracies(latents)
        result = {key: (value[0] if isinstance(value, list) else value) for key, value in scores.items()}
        result["score"] = float(result["hstr_score"])
        result["comparison_operator"] = ">="
        return result

    def decide(self, score: float | None, threshold: float | None) -> bool | None:
        if score is None or threshold is None:
            return None
        return bool(float(score) >= float(threshold))

    def is_detection_successful(self, score: float) -> bool | None:
        return self.decide(score, self.hstr_threshold)

    def resolve_threshold(self) -> dict[str, typing.Any]:
        if self.bundle is None:
            if self.hstr_threshold is None:
                return {
                    "threshold_available": False,
                    "threshold": None,
                    "threshold_source": None,
                    "score_direction": HSTR_SCORE_DIRECTION,
                    "comparison_operator": ">=",
                }
            return {
                "threshold_available": True,
                "threshold": float(self.hstr_threshold),
                "threshold_source": "user_supplied",
                "score_direction": HSTR_SCORE_DIRECTION,
                "comparison_operator": ">=",
            }
        return self.bundle.load_threshold(self.bundle.binding_config(), explicit_threshold=self.hstr_threshold)

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
        return torch.fft.fftshift(torch.fft.rfft2(input_tensor, dim=(-2, -1)), dim=-2)

    @staticmethod
    def irfft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.irfft2(torch.fft.ifftshift(input_tensor, dim=-2), dim=(-2, -1), s=(input_tensor.shape[-2], input_tensor.shape[-2]))

    @staticmethod
    def enforce_hermitian_symmetry(freq_tensor):
        B, C, H, W = freq_tensor.shape
        assert H == W, "H != W"
        freq_tensor = freq_tensor.clone()
        freq_tensor_tmp = freq_tensor.clone()
        freq_tensor[:, :, H // 2, W // 2] = torch.real(freq_tensor_tmp[:, :, H // 2, W // 2])
        if H % 2 == 0:
            freq_tensor[:, :, 0, 0] = torch.real(freq_tensor_tmp[:, :, 0, 0])
            freq_tensor[:, :, H // 2, 0] = torch.real(freq_tensor_tmp[:, :, H // 2, 0])
            freq_tensor[:, :, 0, W // 2] = torch.real(freq_tensor_tmp[:, :, 0, W // 2])
            freq_tensor[:, :, 0, 1:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0, W // 2 + 1:], dims=[2]))
            freq_tensor[:, :, H // 2, 1:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2, W // 2 + 1:], dims=[2]))
            freq_tensor[:, :, 1:H // 2, 0] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2 + 1:, 0], dims=[2]))
            freq_tensor[:, :, 1:H // 2, W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2 + 1:, W // 2], dims=[2]))
            freq_tensor[:, :, 1:H // 2, 1:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2 + 1:, W // 2 + 1:], dims=[2, 3]))
            freq_tensor[:, :, H // 2 + 1:, 1:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 1:H // 2, W // 2 + 1:], dims=[2, 3]))
        else:
            freq_tensor[:, :, H // 2, 0:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2, W // 2 + 1:], dims=[2]))
            freq_tensor[:, :, 0:H // 2, W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2 + 1:, W // 2], dims=[2]))
            freq_tensor[:, :, 0:H // 2, 0:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H // 2 + 1:, W // 2 + 1:], dims=[2, 3]))
            freq_tensor[:, :, H // 2 + 1:, 0:W // 2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0:H // 2, W // 2 + 1:], dims=[2, 3]))
        return freq_tensor


class RounderRingMask:
    def __init__(self, size=65, r_out=RADIUS):
        assert size >= 3
        self.size = size
        self.r_out = r_out
        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        ring_vector = torch.tensor([(200 - i * 4) * (-1) ** i for i in range(num_rings)])
        zero_bg_freq[center, center:center + num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()
        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            res[angle] = torchvision.transforms.functional.rotate(zero_bg_freq, angle=angle)
        res = res.numpy()
        self.res = res
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
            mask = mask[:self.size - 1, :self.size - 1]
        return mask


if USE_ROUNDER_RING:
    mask_obj = RounderRingMask(size=65, r_out=RADIUS)

    def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF):
        assert size == 64
        return mask_obj.get_ring_mask(r_out=r_out, r_in=r_in)
else:
    def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF):
        outer_mask = circle_mask(size=size, r=r_out)
        inner_mask = circle_mask(size=size, r=r_in)
        return outer_mask & (~inner_mask)


def circle_mask(size: int, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def apply_arg_defaults(args, argv=None):
    return HSTRProvider.apply_arg_defaults(args, argv)
