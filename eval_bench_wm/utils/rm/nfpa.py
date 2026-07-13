import random
import typing

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from .base import BaseRemover


def rearrange_3(tensor, f):
    frames, dim, channels = tensor.size()
    return torch.reshape(tensor, (frames // f, f, dim, channels))


def rearrange_4(tensor):
    batch, frames, dim, channels = tensor.size()
    return torch.reshape(tensor, (batch * frames, dim, channels))


class CrossFrameAttnProcessor:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if not is_cross_attention:
            video_length = key.size()[0] // self.batch_size
            first_frame_index = [0] * video_length
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]
            key = rearrange_4(key)
            value = rearrange_4(value)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def coords_grid(batch, height, width, device):
    coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def warp_single_latent(latent, reference_flow):
    _, _, height, width = reference_flow.size()
    _, _, latent_height, latent_width = latent.size()
    coords0 = coords_grid(1, height, width, device=latent.device).to(latent.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= width
    coords_t0[:, 1] /= height
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(latent_height, latent_width), mode="bilinear")
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))

    return F.grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection")


def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, device, dtype):
    seq_length = len(frame_ids)
    reference_flow = torch.zeros((seq_length, 2, 512, 512), device=device, dtype=dtype)
    for fr_idx in range(seq_length):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * frame_ids[fr_idx]
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * frame_ids[fr_idx]
    return reference_flow


def create_motion_field_z_translation(motion_strength, frame_ids, device, dtype, height=512, width=512):
    seq_length = len(frame_ids)
    coords = coords_grid(1, height, width, device=device)[0]
    center_x = width / 2
    center_y = height / 2
    x_offset = coords[0] - center_x
    y_offset = coords[1] - center_y
    norm = torch.sqrt(x_offset**2 + y_offset**2 + 1e-8)
    unit_x = x_offset / norm
    unit_y = y_offset / norm

    reference_flow = torch.zeros((seq_length, 2, height, width), device=device, dtype=dtype)
    for fr_idx in range(seq_length):
        scale = motion_strength * frame_ids[fr_idx]
        reference_flow[fr_idx, 0] = unit_x * scale
        reference_flow[fr_idx, 1] = unit_y * scale
    return reference_flow


def create_motion_field_and_warp_latents_xy(
    latents,
    motion_field_strength_x=12,
    motion_field_strength_y=12,
    video_length=2,
):
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        frame_ids=list(range(video_length))[1:],
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


def create_motion_field_and_warp_latents_z(latents, motion_field_strength_z=12, video_length=2):
    motion_field = create_motion_field_z_translation(
        motion_strength=motion_field_strength_z,
        frame_ids=list(range(video_length))[1:],
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None], motion_field[i][None])
    return warped_latents


def max_warp_latents(latents, x=0, y=0, z=0):
    max_loss, max_x, max_y, max_z = 0, 0, 0, 0

    for k in np.arange(-z, z + 1):
        if k < z // 2 and k > -z // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_z(latents, motion_field_strength_z=k)
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_z = k
            warped_latents_z = warped_latents

    if max_z != 0:
        latents = warped_latents_z

    max_loss = 0
    for i in np.arange(-x, x + 1):
        if i < x // 2 and i > -x // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_xy(
            latents,
            motion_field_strength_x=i,
            motion_field_strength_y=0,
        )
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_x = i

    max_loss = 0
    for j in np.arange(-y, y + 1):
        if j < y // 2 and j > -y // 2:
            continue
        warped_latents = create_motion_field_and_warp_latents_xy(
            latents,
            motion_field_strength_x=0,
            motion_field_strength_y=j,
        )
        loss = torch.abs(warped_latents - latents).mean().item()
        if loss > max_loss:
            max_loss = loss
            max_y = j

    return int(max_x), int(max_y), int(max_z)


class NFPAWatermarkRemover(BaseRemover):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: typing.Union[str, torch.device] = None,
        dtype: torch.dtype = None,
        resolution: int = 512,
        disable_tqdm: bool = True,
        seed: int = None,
    ):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.resolution = resolution

        if seed is not None:
            self.set_seed(seed)

        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        if disable_tqdm:
            self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def set_seed(seed=1234):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def image_to_latents(self, image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.resolution, self.resolution))
            image = transforms.ToTensor()(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        else:
            raise ValueError("NFPA remover expects a PIL image or torch tensor.")

        image = image.to(device=self.device, dtype=self.dtype)
        image = image * 2 - 1
        latents = self.pipe.vae.encode(image).latent_dist.mean
        return latents * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def invert_image(self, image, num_inference_steps=10, guidance_scale=7.5):
        latents = self.image_to_latents(image)
        original_scheduler = self.pipe.scheduler
        self.pipe.scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
        inverted_latents = self.pipe(
            prompt="",
            negative_prompt="",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=latents,
            output_type="latent",
            width=self.resolution,
            height=self.resolution,
        ).images
        self.pipe.scheduler = DDIMScheduler.from_config(original_scheduler.config)
        return inverted_latents, latents

    def _make_next_frame_latents(self, latents, xy=40, z=0):
        max_x, max_y, max_z = max_warp_latents(latents.detach(), x=xy, y=xy, z=z)
        warped_latents = create_motion_field_and_warp_latents_z(
            latents.detach(),
            motion_field_strength_z=max_z,
        )
        warped_latents = create_motion_field_and_warp_latents_xy(
            warped_latents,
            motion_field_strength_x=max_x,
            motion_field_strength_y=max_y,
        )
        warped_latents_timestep = torch.tensor([0], dtype=torch.long, device=self.device)
        warped_latents_noise = torch.randn(
            warped_latents.shape,
            device=warped_latents.device,
            dtype=warped_latents.dtype,
        )
        warped_latents = self.pipe.scheduler.add_noise(
            warped_latents,
            warped_latents_noise,
            warped_latents_timestep,
        )
        return torch.cat([latents, warped_latents], dim=0)

    @torch.no_grad()
    def remove(self, image, num_inference_steps=10, xy=40, z=0, guidance_scale=7.5, **kwargs):
        inverted_latents, _ = self.invert_image(image, 
                                                num_inference_steps=num_inference_steps,
                                                guidance_scale=guidance_scale,)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        latents = self._make_next_frame_latents(inverted_latents, xy=xy, z=z)
        images = self.pipe(
            prompt="",
            negative_prompt="",
            guidance_scale=guidance_scale,
            num_images_per_prompt=2,
            latents=latents,
            num_inference_steps=num_inference_steps,
            width=self.resolution,
            height=self.resolution,
        ).images
        return images[1]
