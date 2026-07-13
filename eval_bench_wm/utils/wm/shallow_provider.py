import argparse
import copy
import random
import types
import typing

import numpy as np
import scipy
import torch
from torchvision.transforms.functional import to_pil_image

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL
from utils import utils


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--shallow_edit_time", default=0.3, type=float)
parser.add_argument("--shallow_detection_threshold", default=-58.96875, type=float)
parser.add_argument("--shallow_dpm_disable_maxsive_before_edit", action="store_true", default=True)
parser.add_argument("--shallow_dpm_enable_maxsive_before_edit", dest="shallow_dpm_disable_maxsive_before_edit", action="store_false")
parser.add_argument("--shallow_maxsive_template_scale", type=float, default=5.0)
parser.add_argument("--shallow_maxsive_template_c", type=int, default=3)
parser.add_argument("--shallow_maxsive_disable_peak", action="store_true", default=False)
parser.add_argument("--shallow_maxsive_debug", action="store_true", default=False)


SHALLOW_ARG_DEFAULTS = {
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "scheduler_target": "DDIM",
    "w_seed": 42,
    "w_channel": 3,
    "w_pattern": "complex2_ring",
    "w_measurement": "l1_complex2",
    "w_injection": "complex2",
}


def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    if r >= 0:
        return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2
    return ((x - x0) ** 2 + (y - y0) ** 2) <= -1


class ShallowProvider(WmProvider):
    """
    Shallow Diffuse T2I provider.

    Unlike Tree-Ring, this watermark is injected at an intermediate DDIM latent
    rather than directly in zT. The run script calls this provider for custom
    generation and inversion while keeping the shared validation/reporting path.
    """

    def __init__(
        self,
        w_seed: int = 42,
        w_channel: int = 3,
        w_pattern: str = "complex2_ring",
        w_mask_shape: str = "circle",
        w_radius: int = 10,
        w_measurement: str = "l1_complex2",
        w_injection: str = "complex2",
        shallow_edit_time: float = 0.3,
        shallow_detection_threshold: float = -58.96875,
        shallow_dpm_disable_maxsive_before_edit: bool = True,
        shallow_maxsive_template_scale: float = 5.0,
        shallow_maxsive_template_c: int = 3,
        shallow_maxsive_disable_peak: bool = False,
        shallow_maxsive_debug: bool = False,
        scheduler_target: str = "DDIM",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tuple(self.latent_shape)[1:] != (4, 64, 64):
            raise ValueError(
                "SHALLOW currently supports SD-style 512px latents with shape "
                f"(B, 4, 64, 64), got {tuple(self.latent_shape)}."
            )

        self.w_seed = w_seed
        self.w_channel = w_channel
        self.w_pattern = self._normalize_pattern(w_pattern, w_injection)
        self.w_mask_shape = w_mask_shape
        self.w_radius = w_radius
        self.w_measurement = w_measurement
        self.w_injection = w_injection
        self.edit_time = shallow_edit_time
        self.threshold = shallow_detection_threshold
        self.disable_maxsive_before_edit = shallow_dpm_disable_maxsive_before_edit
        self.shallow_maxsive_template_scale = shallow_maxsive_template_scale
        self.shallow_maxsive_template_c = shallow_maxsive_template_c
        self.shallow_maxsive_disable_peak = shallow_maxsive_disable_peak
        self.shallow_maxsive_debug = shallow_maxsive_debug
        self.scheduler_target = scheduler_target

        utils.set_random_seed(self.w_seed)
        self.gt_patch = self._get_watermarking_pattern()
        self.watermarking_mask = self._get_watermarking_mask(self.latent_shape)
        self.last_clean_zT = None

    def get_wm_type(self) -> str:
        return "SHALLOW"

    @staticmethod
    def apply_arg_defaults(args, argv):
        for arg_name, default_value in SHALLOW_ARG_DEFAULTS.items():
            cli_name = f"--{arg_name}"
            if cli_name not in argv:
                setattr(args, arg_name, default_value)

    @staticmethod
    def _normalize_pattern(w_pattern, w_injection):
        if w_pattern in {"ring", "rand", "zero"}:
            return f"{w_injection}_{w_pattern}"
        return w_pattern

    def _get_watermarking_mask(self, latent_shape):
        watermarking_mask = torch.zeros(latent_shape, dtype=torch.bool, device=self.device)
        if self.w_mask_shape == "circle":
            np_mask = circle_mask(latent_shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask, device=self.device)
            if self.w_channel == -1:
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == "square":
            anchor_p = latent_shape[-1] // 2
            if self.w_channel == -1:
                watermarking_mask[:, :, anchor_p - self.w_radius : anchor_p + self.w_radius, anchor_p - self.w_radius : anchor_p + self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p - self.w_radius : anchor_p + self.w_radius, anchor_p - self.w_radius : anchor_p + self.w_radius] = True
        elif self.w_mask_shape == "whole":
            if self.w_channel == -1:
                watermarking_mask[:, :] = True
            else:
                watermarking_mask[:, self.w_channel] = True
        elif self.w_mask_shape == "outercircle":
            np_mask = circle_mask(latent_shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(~np_mask, device=self.device)
            if self.w_channel == -1:
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        else:
            raise NotImplementedError(f"w_mask_shape: {self.w_mask_shape}")
        return watermarking_mask

    def _get_watermarking_pattern(self):
        gt_init = torch.randn(*self.latent_shape, device=self.device)

        if "seed_ring" in self.w_pattern:
            gt_patch = gt_init
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i), device=self.device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif "seed_zero" in self.w_pattern:
            gt_patch = gt_init * 0
        elif "seed_rand" in self.w_pattern:
            gt_patch = gt_init
        elif "complex2_rand" in self.w_pattern:
            gt_patch = torch.fft.fft2(gt_init)
            gt_patch[:] = gt_patch[0]
        elif "complex2_zero" in self.w_pattern:
            gt_patch = torch.fft.fft2(gt_init) * 0
        elif "complex2_ring" in self.w_pattern:
            gt_patch = torch.fft.fft2(gt_init)
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i), device=self.device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif "complex_zero" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif "complex_ring" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i), device=self.device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif "complex_rand" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        else:
            raise NotImplementedError(f"w_pattern: {self.w_pattern}")
        return gt_patch

    def _inject_watermark(self, latents):
        latents_w = latents.float()
        if self.w_injection == "complex":
            latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents_w), dim=(-1, -2))
            latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
            return torch.fft.ifft2(torch.fft.ifftshift(latents_w_fft, dim=(-1, -2))).real.to(latents.dtype)
        if self.w_injection == "complex2":
            latents_w_fft = torch.fft.fft2(latents_w)
            latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
            return torch.fft.ifft2(latents_w_fft).real.to(latents.dtype)
        if "seed" in self.w_injection:
            gt_patch = self.gt_patch.to(latents_w.dtype)
            latents_w[self.watermarking_mask] = gt_patch[self.watermarking_mask].clone()
            return latents_w.to(latents.dtype)
        raise NotImplementedError(f"w_injection: {self.w_injection}")

    def get_wm_latents(self, latents_clean: torch.Tensor = None, seed: int = None) -> typing.Dict[str, typing.Any]:
        if seed is not None:
            utils.set_random_seed(seed)
        if latents_clean is None:
            latents_clean = torch.randn(self.latent_shape)
        latents_clean = latents_clean.clone().to(device=self.device, dtype=self.dtype)
        self.last_clean_zT = latents_clean.detach().clone()
        return {
            "zT_clean_torch": latents_clean,
            "zT_clean_PIL": torch_to_PIL(latents_clean),
            "zT_clean": torch_to_PIL(latents_clean),
            "zT_torch": latents_clean,
            "zT_PIL": torch_to_PIL(latents_clean),
            "zT": torch_to_PIL(latents_clean),
        }

    @staticmethod
    def _ensure_pipe(pipe_provider):
        pipe_provider._PipeProvider__load_pipe()
        pipe_provider.set_scheduler()
        return pipe_provider.pipe

    @staticmethod
    def _text_embedding(pipe, prompt):
        text_input_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return pipe.text_encoder(text_input_ids.to(pipe.device))[0]

    @torch.no_grad()
    def _ddim_walk(
        self,
        pipe,
        scheduler,
        latents,
        text_embeddings,
        text_embeddings_null,
        guidance_scale,
        num_inference_steps,
        reverse_process,
        start_timestep,
        end_timestep,
    ):
        do_cfg = guidance_scale > 1.0
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.to(pipe.device)
        latents = latents * scheduler.init_noise_sigma

        iterator = reversed(timesteps) if reverse_process else timesteps
        for i, t in enumerate(iterator):
            if i < start_timestep:
                continue
            if i == end_timestep:
                return latents

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = text_embeddings
            if do_cfg:
                encoder_hidden_states = torch.cat([text_embeddings_null, text_embeddings])

            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    @staticmethod
    def _maxsive_create_template_points(theta, lengths):
        axis = []
        for length in lengths:
            radius = int(32 * length)
            for theta_ in theta:
                axis.append(
                    [
                        32 + int(radius * np.cos(np.radians(theta_))),
                        32 + int(radius * np.sin(np.radians(theta_))),
                    ]
                )
        return axis

    @staticmethod
    def _maxsive_add_peak(scheduler, inputs, x, y):
        inputs_modify = inputs.clone()
        mean = inputs.mean()
        std = inputs.std()
        inputs_modify[0, scheduler.template_c, x - 1 : x + 1, y - 1 : y + 1] = 0
        inputs_modify[0, scheduler.template_c, x, y] = mean + scheduler.template_scale * std
        return inputs_modify

    @staticmethod
    def _maxsive_peak_injection(scheduler, inputs, theta, lengths):
        points = ShallowProvider._maxsive_create_template_points(theta, lengths)
        z_fft = torch.fft.fftshift(torch.fft.fft2(inputs.float()), dim=(-1, -2))
        for x, y in points:
            z_fft = ShallowProvider._maxsive_add_peak(scheduler, z_fft, x, y)
        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(z_fft, dim=(-1, -2))).real

        scheduler.shallow_maxsive_peak_calls += 1
        if scheduler.shallow_maxsive_debug:
            print(
                "SHALLOW MAXSIVE_DPM template injection "
                f"call={scheduler.shallow_maxsive_peak_calls}, "
                f"template_c={scheduler.template_c}, "
                f"template_scale={scheduler.template_scale}"
            )
        return init_latents_w.to(dtype=inputs.dtype)

    def _set_maxsive_template_enabled(self, scheduler, enabled: bool):
        if not hasattr(scheduler, "peak_injection"):
            raise ValueError("SHALLOW dpm_maxsive sampler requires a scheduler with peak_injection().")

        scheduler.template_c = self.shallow_maxsive_template_c
        scheduler.template_scale = self.shallow_maxsive_template_scale
        scheduler.shallow_maxsive_debug = self.shallow_maxsive_debug
        if not hasattr(scheduler, "shallow_maxsive_peak_calls"):
            scheduler.shallow_maxsive_peak_calls = 0

        if enabled and not self.shallow_maxsive_disable_peak:
            scheduler.peak_injection = types.MethodType(
                ShallowProvider._maxsive_peak_injection,
                scheduler,
            )
            return

        def _identity_peak_injection(self, inputs, theta, lengths):
            return inputs

        scheduler.peak_injection = types.MethodType(_identity_peak_injection, scheduler)

    @staticmethod
    def _reset_dpm_scheduler_state(scheduler):
        if hasattr(scheduler.config, "solver_order"):
            scheduler.model_outputs = [
                None,
            ] * scheduler.config.solver_order
        if hasattr(scheduler, "lower_order_nums"):
            scheduler.lower_order_nums = 0
        if hasattr(scheduler, "_step_index"):
            scheduler._step_index = None
        if hasattr(scheduler, "begin_index"):
            scheduler.set_begin_index(None)

    @torch.no_grad()
    def _predict_noise(
        self,
        pipe,
        latents,
        timestep,
        text_embeddings,
        text_embeddings_null,
        guidance_scale,
        scheduler,
    ):
        do_cfg = guidance_scale > 1.0
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        encoder_hidden_states = text_embeddings
        if do_cfg:
            encoder_hidden_states = torch.cat([text_embeddings_null, text_embeddings])

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def _dpm_step_walk(
        self,
        pipe,
        scheduler,
        latents,
        timesteps,
        text_embeddings,
        text_embeddings_null,
        guidance_scale,
        start_index,
        end_index,
    ):
        for i, timestep in enumerate(timesteps):
            if i < start_index:
                continue
            if end_index is not None and i >= end_index:
                break

            noise_pred = self._predict_noise(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                text_embeddings=text_embeddings,
                text_embeddings_null=text_embeddings_null,
                guidance_scale=guidance_scale,
                scheduler=scheduler,
            )
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents

    @torch.no_grad()
    def _dpm_inverse_walk(
        self,
        pipe,
        scheduler,
        latents,
        text_embeddings,
        num_inference_steps,
        end_index,
    ):
        scheduler.set_timesteps(num_inference_steps, device=pipe.device)
        self._reset_dpm_scheduler_state(scheduler)
        timesteps = scheduler.timesteps.to(pipe.device)

        for i, timestep in enumerate(timesteps):
            if i >= end_index:
                break

            latent_model_input = scheduler.scale_model_input(latents, timestep)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents

    @staticmethod
    def _decode_latents(pipe, latents):
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return [to_pil_image(img.detach().cpu()) for img in image]

    @torch.no_grad()
    def _generate_dpm(
        self,
        pipe,
        scheduler,
        pipe_provider_target,
        prompts,
        num_inference_steps,
        guidance_scale,
        latents,
    ):
        scheduler.set_timesteps(num_inference_steps, device=pipe.device)
        self._reset_dpm_scheduler_state(scheduler)
        timesteps = scheduler.timesteps.to(pipe.device)
        latents = latents * scheduler.init_noise_sigma

        edit_timestep = int(self.edit_time * num_inference_steps)
        split_index = num_inference_steps - edit_timestep
        prompt_embeddings = self._text_embedding(pipe, prompts[0])
        null_embeddings = self._text_embedding(pipe, "")

        use_maxsive_template = self.scheduler_target == "MAXSIVE_DPM"
        if use_maxsive_template:
            self._set_maxsive_template_enabled(
                scheduler,
                enabled=not self.disable_maxsive_before_edit,
            )
        xt_no_w = self._dpm_step_walk(
            pipe=pipe,
            scheduler=scheduler,
            latents=latents,
            timesteps=timesteps,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=guidance_scale,
            start_index=0,
            end_index=split_index,
        )

        xt_w = self._inject_watermark(xt_no_w)

        scheduler_no_w = copy.deepcopy(scheduler)
        scheduler_w = copy.deepcopy(scheduler)
        if use_maxsive_template:
            self._set_maxsive_template_enabled(scheduler_no_w, enabled=True)
            self._set_maxsive_template_enabled(scheduler_w, enabled=True)

        x0_no_w = self._dpm_step_walk(
            pipe=pipe,
            scheduler=scheduler_no_w,
            latents=xt_no_w,
            timesteps=timesteps,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=1.0,
            start_index=split_index,
            end_index=None,
        )
        x0_w = self._dpm_step_walk(
            pipe=pipe,
            scheduler=scheduler_w,
            latents=xt_w,
            timesteps=timesteps,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=1.0,
            start_index=split_index,
            end_index=None,
        )

        averaged_latent = x0_w.clone()
        if self.w_channel != -1:
            for channel_idx in range(averaged_latent.shape[1]):
                if channel_idx != self.w_channel:
                    averaged_latent[:, channel_idx] = x0_no_w[:, channel_idx]

        images_PIL = self._decode_latents(pipe, averaged_latent)
        images_torch = pipe_provider_target.PIL_to_torch(images_PIL)

        return {
            "prompts": prompts,
            "images_torch": images_torch,
            "images_PIL": images_PIL,
            "images": images_PIL,
            "z0_torch": averaged_latent,
            "z0_PIL": torch_to_PIL(averaged_latent),
            "z0": torch_to_PIL(averaged_latent),
            "zT_torch": self.last_clean_zT,
            "zT_PIL": torch_to_PIL(self.last_clean_zT),
            "zT": torch_to_PIL(self.last_clean_zT),
        }

    @torch.no_grad()
    def generate(
        self,
        pipe_provider_target,
        prompts,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        pipe = self._ensure_pipe(pipe_provider_target)
        scheduler = pipe_provider_target.scheduler

        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) != 1:
            raise ValueError("SHALLOW currently supports batch size 1.")

        if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder") or not hasattr(pipe, "unet"):
            raise ValueError("SHALLOW currently supports StableDiffusionPipeline-style models only.")

        if latents is None:
            latents = torch.randn(self.latent_shape, device=self.device, dtype=pipe_provider_target.get_dtype())
        latents = latents.to(device=self.device, dtype=pipe_provider_target.get_dtype())
        self.last_clean_zT = latents.detach().clone()

        if self.scheduler_target in {"DPM", "LOCAL_DPM", "MAXSIVE_DPM"}:
            return self._generate_dpm(
                pipe=pipe,
                scheduler=scheduler,
                pipe_provider_target=pipe_provider_target,
                prompts=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latents,
            )

        edit_timestep = int(self.edit_time * num_inference_steps)
        prompt_embeddings = self._text_embedding(pipe, prompts[0])
        null_embeddings = self._text_embedding(pipe, "")

        xt_no_w = self._ddim_walk(
            pipe=pipe,
            scheduler=scheduler,
            latents=latents,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            reverse_process=False,
            start_timestep=0,
            end_timestep=num_inference_steps - edit_timestep,
        )
        xt_w = self._inject_watermark(xt_no_w)

        x0_no_w = self._ddim_walk(
            pipe=pipe,
            scheduler=scheduler,
            latents=xt_no_w,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            reverse_process=False,
            start_timestep=num_inference_steps - edit_timestep,
            end_timestep=-1,
        )
        x0_w = self._ddim_walk(
            pipe=pipe,
            scheduler=scheduler,
            latents=xt_w,
            text_embeddings=prompt_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            reverse_process=False,
            start_timestep=num_inference_steps - edit_timestep,
            end_timestep=-1,
        )

        averaged_latent = x0_w.clone()
        if self.w_channel != -1:
            for channel_idx in range(averaged_latent.shape[1]):
                if channel_idx != self.w_channel:
                    averaged_latent[:, channel_idx] = x0_no_w[:, channel_idx]

        images_PIL = self._decode_latents(pipe, averaged_latent)
        images_torch = pipe_provider_target.PIL_to_torch(images_PIL)

        return {
            "prompts": prompts,
            "images_torch": images_torch,
            "images_PIL": images_PIL,
            "images": images_PIL,
            "z0_torch": averaged_latent,
            "z0_PIL": torch_to_PIL(averaged_latent),
            "z0": torch_to_PIL(averaged_latent),
            "zT_torch": latents,
            "zT_PIL": torch_to_PIL(latents),
            "zT": torch_to_PIL(latents),
        }

    @torch.no_grad()
    def invert_images(
        self,
        images,
        pipe_provider_target,
        num_inference_steps: int = 50,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        pipe = self._ensure_pipe(pipe_provider_target)
        z0_torch = pipe_provider_target.imgs_to_latents(images)
        null_embeddings = self._text_embedding(pipe, "")
        edit_timestep = int(self.edit_time * num_inference_steps)

        if self.scheduler_target in {"DPM", "LOCAL_DPM", "MAXSIVE_DPM"}:
            xt_torch = self._dpm_inverse_walk(
                pipe=pipe,
                scheduler=pipe_provider_target.scheduler_inverse,
                latents=z0_torch,
                text_embeddings=null_embeddings,
                num_inference_steps=num_inference_steps,
                end_index=edit_timestep,
            )
            return {
                "z0_torch": z0_torch,
                "z0_PIL": torch_to_PIL(z0_torch),
                "z0": torch_to_PIL(z0_torch),
                "zT_torch": xt_torch,
                "zT_PIL": torch_to_PIL(xt_torch),
                "zT": torch_to_PIL(xt_torch),
            }

        xt_torch = self._ddim_walk(
            pipe=pipe,
            scheduler=pipe_provider_target.scheduler,
            latents=z0_torch,
            text_embeddings=null_embeddings,
            text_embeddings_null=null_embeddings,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            reverse_process=True,
            start_timestep=0,
            end_timestep=edit_timestep,
        )
        return {
            "z0_torch": z0_torch,
            "z0_PIL": torch_to_PIL(z0_torch),
            "z0": torch_to_PIL(z0_torch),
            "zT_torch": xt_torch,
            "zT_PIL": torch_to_PIL(xt_torch),
            "zT": torch_to_PIL(xt_torch),
        }

    def _eval_watermark(self, latents):
        latents = latents.to(self.device).float()
        if "complex" in self.w_measurement and "complex2" not in self.w_measurement:
            latents_eval = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        elif "complex2" in self.w_measurement:
            latents_eval = torch.fft.fft2(latents)
        elif "seed" in self.w_measurement:
            latents_eval = latents
        else:
            raise NotImplementedError(f"w_measurement: {self.w_measurement}")

        target_patch = self.gt_patch
        masked = latents_eval[self.watermarking_mask]
        target = target_patch[self.watermarking_mask]
        l1 = torch.abs(masked - target).mean().item() if masked.numel() > 0 else float("inf")

        p_value = None
        if "p_value" in self.w_measurement and masked.numel() > 0:
            latent_vec = torch.concatenate([masked.flatten().real, masked.flatten().imag])
            target_vec = torch.concatenate([target.flatten().real, target.flatten().imag])
            sigma = latent_vec.std()
            lambd = (target_vec**2 / sigma**2).sum().item()
            x = (((latent_vec - target_vec) / sigma) ** 2).sum().item()
            p_value = scipy.stats.ncx2.cdf(x=x, df=len(target_vec), nc=lambd)

        return l1, p_value

    def get_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        l1, p_value = self._eval_watermark(latents)
        score = -l1
        if p_value is not None:
            detection_success = p_value < self.threshold
            value = 1.0 - float(p_value)
        else:
            value = score
            detection_success = value > self.threshold
            p_value = 0.0

        return {
            "p_values": [float(p_value)],
            "l1_dist": [-score],
            # "value": float(value),
            "bit_accuracies": [0.0],
            "detection_success": detection_success,
            "threshold": self.threshold,
            "log_message": (
                f"(WM type SHALLOW) measurement: {self.w_measurement}; "
                f"p_value: {float(p_value):.6g}; l1: {-l1:.6f}; "
                f"threshold: {self.threshold:.6g}; detection: {detection_success}"
            ),
        }
