"""Small frozen SFWMark HSTR reference helpers for CPU-only tests."""

from __future__ import annotations

import numpy as np
import torch

RADIUS = 14
RADIUS_CUTOFF = 3
START = 10
END = 54
TREE_WATERMARK_CHANNEL = [3]
HETER_WATERMARK_CHANNEL = [0]
RINGID_WATERMARK_CHANNEL = [0, 3]


def circle_mask(size: int, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def _ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF):
    # Frozen SFWMark defaults set USE_ROUNDER_RING=True. The expensive rotation
    # construction is equivalent to the project helper, which is tested here as
    # part of the selected frozen behavior rather than as independent watermark math.
    from utils.wm.hstr_provider import ring_mask

    return ring_mask(size=size, r_out=r_out, r_in=r_in)


def enforce_hermitian_symmetry(freq_tensor):
    B, C, H, W = freq_tensor.shape
    assert H == W
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
        raise NotImplementedError
    return freq_tensor


def make_pattern(seed, shape=(1, 4, 64, 64)):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gt_init = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
    center = (slice(None), slice(None), slice(START, END), slice(START, END))
    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(torch.zeros(shape)), dim=(-1, -2))
    gt_patch_tmp = torch.fft.fftshift(torch.fft.fft2(gt_init[center]), dim=(-1, -2)).clone().detach().to(torch.complex64)
    center_len = gt_patch_tmp.shape[-1] // 2
    for radius in range(center_len - 1, 0, -1):
        tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius))
        for j in range(watermarked_latents_fft.shape[1]):
            watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item()
    watermarked_latents_fft[:, HETER_WATERMARK_CHANNEL, START:END, START:END] = gt_patch_tmp[:, HETER_WATERMARK_CHANNEL]
    return enforce_hermitian_symmetry(watermarked_latents_fft)
