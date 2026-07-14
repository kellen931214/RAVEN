"""End-to-end RAVEN reproduction pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from .attention import install_view_guided_attention, restore_default_attention
from .color_transfer import color_contrast_transfer_pil, color_transfer_diagnostics
from .inversion import partial_diffusion_inversion
from .resource_guard import limit_cpu_threads
from .utils import image_size_divisible_by_8, save_image, save_json, seed_everything, tensor_to_image
from .warp import (
    create_nfpa_translation_flow,
    latent_grid_warp,
    latent_grid_warp_nearest_reflection,
    nfpa_warp_single_latent,
    sample_translation,
    translate_latent,
)


class RavenPipeline:
    def __init__(
        self,
        model_id: str = "RedbeardNZ/stable-diffusion-2-1-base",
        device: str = "cuda",
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        try:
            import torch
            from diffusers import DDIMScheduler, StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "RavenPipeline requires torch and diffusers. Install raven_repro/requirements.txt first."
            ) from exc

        limit_cpu_threads(1)

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        self.torch = torch
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.model_id = model_id
        self.model_revision = revision
        self.model_variant = "default"

        # This checkpoint has standard safetensors but no filename-level fp16 variant.
        model_variant = None
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant=model_variant,
            low_cpu_mem_usage=True,
            revision=revision,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self._default_attn_processors = dict(self.pipe.unet.attn_processors)
        self.vae_scale_factor = getattr(self.pipe, "vae_scale_factor", 8)

    def _resolve_dtype(self, dtype: Optional[str]):
        torch = self.torch
        if dtype in {None, "auto"}:
            return torch.float16 if self.device == "cuda" else torch.float32
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _make_generator(self, seed: int):
        return self.torch.Generator(device=self.device).manual_seed(seed)

    def _encode_prompt(self, prompt: str, negative_prompt: str, guidance_scale: float, num_images_per_prompt: int):
        do_cfg = guidance_scale > 1.0
        if hasattr(self.pipe, "encode_prompt"):
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )
            if do_cfg:
                return self.torch.cat([negative_prompt_embeds, prompt_embeds])
            return prompt_embeds

        return self.pipe._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
        )

    def _prepare_extra_step_kwargs(self, generator):
        kwargs: Dict[str, Any] = {}
        step_params = inspect.signature(self.pipe.scheduler.step).parameters
        if "eta" in step_params:
            kwargs["eta"] = 0.0
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    def _conditioning_diagnostics(self, prompt_embeds, num_images_per_prompt: int, guidance_scale: float):
        diagnostics: Dict[str, Any] = {
            "shape": list(prompt_embeds.shape),
            "cfg_enabled": guidance_scale > 1.0,
            "cfg_layout": "[unconditional, conditional]" if guidance_scale > 1.0 else "conditional_only",
        }
        if guidance_scale > 1.0:
            unconditional = prompt_embeds[:num_images_per_prompt]
            conditional = prompt_embeds[num_images_per_prompt:]
            max_abs_difference = float(
                (unconditional.float() - conditional.float()).abs().max().detach().cpu().item()
            )
            diagnostics.update({
                "unconditional_shape": list(unconditional.shape),
                "conditional_shape": list(conditional.shape),
                "unconditional_conditional_max_abs_difference": max_abs_difference,
                "unconditional_conditional_identical": max_abs_difference == 0.0,
            })
        return diagnostics

    def _decode_latents(self, latents):
        image, _ = self._decode_latents_with_diagnostics(latents)
        return image

    def _decode_latents_with_diagnostics(self, latents):
        scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        decoded = self.pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
        normalized = decoded / 2 + 0.5
        diagnostics = {
            "latent_min": float(latents.detach().float().min().cpu().item()),
            "latent_max": float(latents.detach().float().max().cpu().item()),
            "decoded_min_before_clamp": float(normalized.detach().float().min().cpu().item()),
            "decoded_max_before_clamp": float(normalized.detach().float().max().cpu().item()),
            "fraction_below_zero": float((normalized < 0).float().mean().cpu().item()),
            "fraction_above_one": float((normalized > 1).float().mean().cpu().item()),
        }
        return tensor_to_image(normalized.clamp(0, 1)), diagnostics

    def run(
        self,
        input_image: Image.Image,
        output_dir: str | Path,
        steps: int = 50,
        strength: float = 0.15,
        guidance_scale: float = 2.5,
        shift_min: int = 24,
        shift_max: int = 32,
        shift_sign: str = "random",
        shift_sampling: str = "independent_axes",
        shift_space: str = "image_pixels",
        warp_mode: str = "integer",
        padding_mode: str = "zeros",
        latent_sampling_mode: Optional[str] = None,
        shift_x: Optional[float] = None,
        shift_y: Optional[float] = None,
        view_guided_attention: bool = True,
        color_transfer: bool = True,
        seed: int = 42,
        prompt: str = "",
        negative_prompt: str = "",
        debug: bool = False,
        inversion_mode: str = "ddim",
    ) -> Image.Image:
        torch = self.torch
        restore_default_attention(self.pipe.unet, self._default_attn_processors)
        seed_everything(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_size_divisible_by_8(input_image)
        save_image(input_image, output_dir / "input.png")

        generator = self._make_generator(seed)
        inversion_prompt_embeds = self._encode_prompt(
            prompt="",
            negative_prompt="",
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )
        inversion = partial_diffusion_inversion(
            vae=self.pipe.vae,
            scheduler=self.pipe.scheduler,
            image=input_image,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            mode=inversion_mode,
            unet=self.pipe.unet,
            prompt_embeds=inversion_prompt_embeds,
            guidance_scale=guidance_scale,
        )
        if (shift_x is None) != (shift_y is None):
            raise ValueError("shift_x and shift_y must be provided together")
        if shift_x is None:
            dx, dy = sample_translation(
                shift_min, shift_max, shift_sign, seed=seed, sampling=shift_sampling
            )
            shift_source = "sample_translation"
        else:
            dx, dy = float(shift_x), float(shift_y)
            shift_source = "explicit_plan"
        inverse_sampling_modes = {
            "nfpa_exact", "nfpa_pixel_center", "latent_grid_nearest_reflection", "latent_grid"
        }
        inverse_sampling_warp = warp_mode in inverse_sampling_modes
        nfpa_warp_metadata = None
        if warp_mode in {"nfpa_exact", "nfpa_pixel_center"}:
            if shift_space != "image_pixels":
                raise ValueError(f"{warp_mode} requires image-space flow")
            nfpa_flow = create_nfpa_translation_flow(
                dx_image_px=float(dx),
                dy_image_px=float(dy),
                batch=inversion.noisy_latents.shape[0],
                height=inversion.noisy_latents.shape[-2] * self.vae_scale_factor,
                width=inversion.noisy_latents.shape[-1] * self.vae_scale_factor,
                device=inversion.noisy_latents.device,
                dtype=inversion.noisy_latents.dtype,
            )
            shifted_latents, nfpa_warp_metadata = nfpa_warp_single_latent(
                inversion.noisy_latents, nfpa_flow, return_metadata=True,
                pixel_center_offset=0.5 if warp_mode == "nfpa_pixel_center" else 0.0,
            )
        elif warp_mode == "latent_grid_nearest_reflection":
            if shift_space != "image_pixels":
                raise ValueError("latent_grid_nearest_reflection requires image-space shifts")
            shifted_latents, nfpa_warp_metadata = latent_grid_warp_nearest_reflection(
                inversion.noisy_latents, float(dx), float(dy),
                vae_scale_factor=self.vae_scale_factor, return_metadata=True,
            )
        elif warp_mode == "latent_grid":
            if shift_space != "image_pixels":
                raise ValueError("latent_grid requires image-space shifts")
            shifted_latents, nfpa_warp_metadata = latent_grid_warp(
                inversion.noisy_latents, float(dx), float(dy),
                vae_scale_factor=self.vae_scale_factor,
                sampling_mode=latent_sampling_mode or "nearest",
                padding_mode=padding_mode,
                return_metadata=True,
            )
        else:
            shifted_latents = translate_latent(
                inversion.noisy_latents,
                dx=dx,
                dy=dy,
                shift_space=shift_space,
                vae_scale_factor=self.vae_scale_factor,
                padding_mode=padding_mode,
                warp_mode=warp_mode,
            )
        save_image(self._decode_latents(shifted_latents), output_dir / "latent_shift_only.png")

        latents = torch.cat([inversion.noisy_latents, shifted_latents], dim=0)
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=2,
        )
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator)

        if view_guided_attention:
            processors = install_view_guided_attention(self.pipe.unet, debug=debug)
            attention_mode = "view_guided_self_attention"
        else:
            restore_default_attention(self.pipe.unet, self._default_attn_processors)
            processors = {}
            attention_mode = "default_self_attention"

        debug_info: Dict[str, Any] = {
            "model_id": self.model_id,
            "model_revision": self.model_revision or "unspecified",
            "model_variant": self.model_variant,
            "vae_scale_factor": self.vae_scale_factor,
            "vae_scaling_factor": float(getattr(self.pipe.vae.config, "scaling_factor", 0.18215)),
            "denoise_scheduler": inversion.denoise_scheduler,
            "inverse_scheduler": inversion.inverse_scheduler,
            "prediction_type": inversion.prediction_type,
            "eta": inversion.eta,
            "scheduler_config": {
                "num_train_timesteps": int(getattr(self.pipe.scheduler.config, "num_train_timesteps", 1000)),
                "beta_schedule": str(getattr(self.pipe.scheduler.config, "beta_schedule", "unspecified")),
                "steps_offset": int(getattr(self.pipe.scheduler.config, "steps_offset", 0)),
                "set_alpha_to_one": bool(getattr(self.pipe.scheduler.config, "set_alpha_to_one", True)),
            },
            "device": self.device,
            "dtype": str(self.dtype),
            "inversion_prompt": "",
            "reconstruction_prompt": prompt,
            "negative_prompt": negative_prompt,
            "inversion_prompt_is_empty": True,
            "reconstruction_prompt_is_empty": prompt == "",
            "inversion_conditioning": self._conditioning_diagnostics(
                inversion_prompt_embeds, 1, guidance_scale
            ),
            "reconstruction_conditioning": self._conditioning_diagnostics(
                prompt_embeds, 2, guidance_scale
            ),
            "input_image_size": list(input_image.size),
            "latent_shape": list(inversion.clean_latents.shape),
            "reference_latent_shape": list(inversion.noisy_latents.shape),
            "shifted_latent_shape": list(shifted_latents.shape),
            "timesteps": [int(t) for t in inversion.timesteps.detach().cpu().tolist()],
            "strength": strength,
            "selected_tau": int(inversion.start_timestep[0].detach().cpu().item()),
            "inversion_mode": inversion.mode,
            "exact_timestep": inversion.target_timestep,
            "inversion_timesteps": (
                [int(t) for t in inversion.inversion_timesteps.detach().cpu().tolist()]
                if inversion.inversion_timesteps is not None
                else []
            ),
            "dx": dx,
            "dy": dy,
            "flow_dx_image_px": dx if shift_space == "image_pixels" else float(dx) * self.vae_scale_factor,
            "flow_dy_image_px": dy if shift_space == "image_pixels" else float(dy) * self.vae_scale_factor,
            "visual_shift_dx_image_px": (
                -(dx if shift_space == "image_pixels" else float(dx) * self.vae_scale_factor)
                if inverse_sampling_warp else (dx if shift_space == "image_pixels" else float(dx) * self.vae_scale_factor)
            ),
            "visual_shift_dy_image_px": (
                -(dy if shift_space == "image_pixels" else float(dy) * self.vae_scale_factor)
                if inverse_sampling_warp else (dy if shift_space == "image_pixels" else float(dy) * self.vae_scale_factor)
            ),
            "image_dx": dx if shift_space == "image_pixels" else float(dx) * self.vae_scale_factor,
            "image_dy": dy if shift_space == "image_pixels" else float(dy) * self.vae_scale_factor,
            "shift_space": shift_space,
            "shift_sampling": shift_sampling,
            "shift_source": shift_source,
            "latent_dx": float(dx) / self.vae_scale_factor if shift_space == "image_pixels" else float(dx),
            "latent_dy": float(dy) / self.vae_scale_factor if shift_space == "image_pixels" else float(dy),
            "padding_mode": (
                nfpa_warp_metadata.get("padding_mode")
                if inverse_sampling_warp and nfpa_warp_metadata
                else padding_mode
            ),
            "guidance_scale": guidance_scale,
            "attention_processor_mode": attention_mode,
            "warp_mode": warp_mode,
            "interpolation_mode": (
                nfpa_warp_metadata.get("latent_sampling_mode")
                if inverse_sampling_warp and nfpa_warp_metadata
                else ("none" if warp_mode == "integer" else "bilinear")
            ),
            "align_corners": (
                False if inverse_sampling_warp else (None if warp_mode == "integer" else True)
            ),
            "normalized_coordinate_formula": (
                nfpa_warp_metadata.get("normalization_formula")
                if inverse_sampling_warp and nfpa_warp_metadata
                else (
                    None if warp_mode == "integer"
                    else "source_grid = output_grid - (2*latent_shift/(size-1)); align_corners=True"
                )
            ),
            "half_pixel_offset": (
                0.5 if warp_mode == "nfpa_pixel_center" else (0.0 if inverse_sampling_warp else None)
            ),
            "nfpa_warp_metadata": nfpa_warp_metadata,
            "circular": False,
            "output_alignment": "shifted_novel_view",
            "overlap_handling": "valid overlap crop for paired quality metrics",
            "view_guided_attention": view_guided_attention,
            "color_transfer": color_transfer,
        }

        if debug:
            for key, value in debug_info.items():
                print(f"{key}: {value}")

        try:
            for i, timestep in enumerate(inversion.timesteps):
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep)
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.pipe.scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                if debug:
                    print(f"denoise_step: {i + 1}/{len(inversion.timesteps)} timestep={int(timestep)}")
        finally:
            restore_default_attention(self.pipe.unet, self._default_attn_processors)

        view_latent = latents[1:2]
        view_image, clipping_diagnostics = self._decode_latents_with_diagnostics(view_latent)
        debug_info["clipping_diagnostics"] = clipping_diagnostics
        save_image(view_image, output_dir / "view_guided_output.png")

        final_image = color_contrast_transfer_pil(view_image, input_image) if color_transfer else view_image
        final_name = "final_color_corrected.png" if color_transfer else "final.png"
        save_image(final_image, output_dir / final_name)

        if color_transfer:
            debug_info["color_transfer_diagnostics"] = color_transfer_diagnostics(
                view_image, input_image, final_image
            )

        if processors:
            debug_info["attention_processor_count"] = len(processors)
            self_processors = [processor for processor in processors.values() if hasattr(processor, "state")]
            debug_info["attention_debug"] = {
                "self_processor_count": len(self_processors),
                "total_calls": sum(processor.state.calls for processor in self_processors),
                "processors_with_calls": sum(processor.state.calls > 0 for processor in self_processors),
                "registration_failures": [],
            }
            if debug and self_processors:
                debug_info["attention_debug"]["layers"] = [
                    {
                        "name": processor.state.layer_name,
                        "calls": processor.state.calls,
                        "self_calls": processor.state.self_calls,
                        "last_shape": processor.state.last_shape,
                        "last_batch_size": processor.state.last_batch_size,
                        "cfg_layout": processor.state.cfg_layout,
                        "reference_query_fingerprint": (processor.state.last_query_fingerprints or [None, None])[-2],
                        "target_query_fingerprint": (processor.state.last_query_fingerprints or [None, None])[-1],
                        "transformed_query_fingerprint": (processor.state.last_query_fingerprints or [None, None])[-1],
                        "reference_key_fingerprint": (processor.state.last_injected_key_fingerprints or [None, None])[-2],
                        "target_key_before_injection": (processor.state.last_original_key_fingerprints or [None, None])[-1],
                        "target_key_after_injection": (processor.state.last_injected_key_fingerprints or [None, None])[-1],
                        "reference_value_fingerprint": (processor.state.last_injected_value_fingerprints or [None, None])[-2],
                        "target_value_before_injection": (processor.state.last_original_value_fingerprints or [None, None])[-1],
                        "target_value_after_injection": (processor.state.last_injected_value_fingerprints or [None, None])[-1],
                    } for processor in self_processors
                ]
                state = self_processors[0].state
                debug_info["attention_debug"]["sample_processor"] = {
                    "last_shape": state.last_shape,
                    "last_batch_size": state.last_batch_size,
                    "last_query_checksums": state.last_query_checksums,
                    "last_key_source_checksums": state.last_key_source_checksums,
                    "last_value_source_checksums": state.last_value_source_checksums,
                }
        save_json(debug_info, output_dir / "debug_info.json")
        return final_image
