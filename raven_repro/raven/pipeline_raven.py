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
from .warp import sample_translation, translate_latent


class RavenPipeline:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
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

        model_variant = "fp16" if self.dtype == torch.float16 else None
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

    def _decode_latents(self, latents):
        scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        image = self.pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
        return tensor_to_image((image / 2 + 0.5).clamp(0, 1))

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
        padding_mode: str = "reflection",
        view_guided_attention: bool = True,
        color_transfer: bool = True,
        seed: int = 42,
        prompt: str = "",
        negative_prompt: str = "",
        debug: bool = False,
        inversion_mode: str = "ddim",
    ) -> Image.Image:
        torch = self.torch
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
        dx, dy = sample_translation(
            shift_min, shift_max, shift_sign, seed=seed, sampling=shift_sampling
        )
        shifted_latents = translate_latent(
            inversion.noisy_latents,
            dx=dx,
            dy=dy,
            shift_space=shift_space,
            vae_scale_factor=self.vae_scale_factor,
            padding_mode=padding_mode,
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
            restore_default_attention(self.pipe.unet)
            processors = {}
            attention_mode = "default_self_attention"

        debug_info: Dict[str, Any] = {
            "model_id": self.model_id,
            "model_revision": self.model_revision or "unspecified",
            "device": self.device,
            "dtype": str(self.dtype),
            "input_image_size": list(input_image.size),
            "latent_shape": list(inversion.clean_latents.shape),
            "reference_latent_shape": list(inversion.noisy_latents.shape),
            "shifted_latent_shape": list(shifted_latents.shape),
            "timesteps": [int(t) for t in inversion.timesteps.detach().cpu().tolist()],
            "strength": strength,
            "selected_tau": int(inversion.start_timestep[0].detach().cpu().item()),
            "inversion_mode": inversion.mode,
            "inversion_timesteps": (
                [int(t) for t in inversion.inversion_timesteps.detach().cpu().tolist()]
                if inversion.inversion_timesteps is not None
                else []
            ),
            "dx": dx,
            "dy": dy,
            "shift_space": shift_space,
            "shift_sampling": shift_sampling,
            "latent_dx": float(dx) / self.vae_scale_factor if shift_space == "image_pixels" else float(dx),
            "latent_dy": float(dy) / self.vae_scale_factor if shift_space == "image_pixels" else float(dy),
            "padding_mode": padding_mode,
            "guidance_scale": guidance_scale,
            "attention_processor_mode": attention_mode,
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
            restore_default_attention(self.pipe.unet)

        view_latent = latents[1:2]
        view_image = self._decode_latents(view_latent)
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
            }
            if debug and self_processors:
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
