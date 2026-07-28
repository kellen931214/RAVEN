"""Test-only transcription of the official GaussMarker reference logic.

Source: https://github.com/SunnierLee/GaussMarker
Commit: 4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155
Files:  watermark.py, tr_utils.py, utils.py, inverse_stable_diffusion.py

This module exists **only** so the parity tests can compare the RAVEN provider
against the official call graph element-by-element. It is never imported by any
runner or provider, and it must never be used to produce results. The
authoritative implementation is ``utils/wm/gm_provider.py``.

The transcription is deliberately literal (including the official global-RNG
usage) so a divergence in the provider shows up as a test failure rather than
being hidden by a "cleaner" rewrite.
"""

from __future__ import annotations

import copy
import random
from functools import reduce

import numpy as np
import torch
from scipy.stats import norm, truncnorm

OFFICIAL_COMMIT = "4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155"


# --- utils.py --------------------------------------------------------------

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    from torchvision import transforms

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


# --- watermark.py ----------------------------------------------------------

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, w_factor, h_factor, fpr, user_number,
                 watermark=None, key=None, nonce=None, m=None):
        self.ch = ch_factor
        self.w = w_factor
        self.h = h_factor
        self.nonce = nonce
        self.key = key
        self.watermark = watermark
        self.m = m
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.w * self.h)
        self.threshold = 1 if self.h == 1 and self.w == 1 and self.ch == 1 else self.ch * self.w * self.h // 2

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i: i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z

    def create_watermark_and_return_w_m(self):
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.w, 64 // self.h])
            sd = self.watermark.repeat(1, self.ch, self.w, self.h)
            self.m = self.stream_key_encrypt(sd.flatten().numpy())
        w = self.truncSampling(self.m)
        return w, torch.from_numpy(self.m).reshape(1, 4, 64, 64)

    def stream_key_encrypt(self, sd):
        from Crypto.Cipher import ChaCha20
        from Crypto.Random import get_random_bytes

        if self.key is None or self.nonce is None:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_decrypt(self, reversed_m):
        from Crypto.Cipher import ChaCha20

        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor

    def diffusion_inverse(self, watermark_r):
        ch_stride = 4 // self.ch
        w_stride = 64 // self.w
        h_stride = 64 // self.h
        ch_list = [ch_stride] * self.ch
        w_list = [w_stride] * self.w
        h_list = [h_stride] * self.h
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(w_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def pred_w_from_latent(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        return self.diffusion_inverse(reversed_sd)

    def pred_w_from_m(self, reversed_m):
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        return self.diffusion_inverse(reversed_sd)


# --- tr_utils.py -----------------------------------------------------------

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)
    if args.w_mask_shape == "circle":
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)
        if args.w_channel == -1:
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == "square":
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            watermarking_mask[:, :, anchor_p - args.w_radius:anchor_p + args.w_radius,
                              anchor_p - args.w_radius:anchor_p + args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p - args.w_radius:anchor_p + args.w_radius,
                              anchor_p - args.w_radius:anchor_p + args.w_radius] = True
    else:
        raise NotImplementedError(f"w_mask_shape: {args.w_mask_shape}")
    return watermarking_mask


def get_watermarking_pattern(args, device, shape=(1, 4, 64, 64)):
    set_random_seed(args.w_seed)
    gt_init = torch.randn(*shape, device=device)

    if "zeros" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif "rand" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif "ring" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    else:
        raise NotImplementedError(f"w_pattern: {args.w_pattern}")
    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == "complex":
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == "seed":
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        raise NotImplementedError(f"w_injection: {args.w_injection}")
    return torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real


def eval_watermark(reversed_latents_w, watermarking_mask, gt_patch, args):
    """Official positive-branch of ``eval_watermark`` (returns ``l1 * 0.01``)."""
    if "complex" in args.w_measurement:
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif "seed" in args.w_measurement:
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f"w_measurement: {args.w_measurement}")
    if "l1" in args.w_measurement:
        return torch.abs(
            reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]
        ).mean().item() * 0.01
    raise NotImplementedError(f"w_measurement: {args.w_measurement}")


# --- inverse_stable_diffusion.py -------------------------------------------

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def forward_diffusion(unet, scheduler, latents, text_embeddings, guidance_scale,
                      num_inference_steps, device):
    """``backward_diffusion(..., reverse_process=True)`` from the official pipeline."""
    do_classifier_free_guidance = guidance_scale > 1.0
    scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = scheduler.timesteps.to(device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in enumerate(reversed(timesteps_tensor)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        latents = backward_ddim(
            x_t=latents, alpha_t=alpha_prod_t, alpha_tm1=alpha_prod_t_prev, eps_xt=noise_pred
        )
    return latents


class OfficialArgs:
    """Minimal stand-in for the official argparse namespace."""

    def __init__(self, **kwargs):
        defaults = {
            "w_seed": 999999,
            "w_channel": 3,
            "w_pattern": "ring",
            "w_mask_shape": "circle",
            "w_radius": 4,
            "w_measurement": "l1_complex",
            "w_injection": "complex",
            "w_pattern_const": 0,
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)
