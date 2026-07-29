"""Test-only transcription of the official RingID reference logic.

Source: https://github.com/showlab/RingID
Commit: 45631a59aecd7d63ccdb640aaaf3e616fdb89fb9
Files:  utils.py, verify.py, identify.py

This module exists **only** so the parity tests can compare the RAVEN provider
against the official call graph element-by-element. It is never imported by any
runner or provider and must never be used to produce results. The authoritative
implementation is ``utils/wm/ringid_provider.py``.

The transcription is deliberately literal — including the official global-RNG
usage, the ``USE_ROUNDER_RING`` rebinding of ``ring_mask`` and the commented-out
``* args.time_shift_factor`` in the spatial shift — so a divergence in the
provider shows up as a test failure instead of being hidden by a "cleaner"
rewrite.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch
from torchvision import transforms

OFFICIAL_COMMIT = "45631a59aecd7d63ccdb640aaaf3e616fdb89fb9"

# --- utils.py constants ----------------------------------------------------

RADIUS = 14
RADIUS_CUTOFF = 3
ANCHOR_X_OFFSET = 0
ANCHOR_Y_OFFSET = 0
USE_ROUNDER_RING = True

HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [3]
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def circle_mask(size=64, r=RADIUS, x_offset=ANCHOR_X_OFFSET, y_offset=ANCHOR_Y_OFFSET, mode='full'):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset - 1
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    if mode == 'left':
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & ((x > x0) + ((x == x0) & (y > y0)))
    if mode == 'right':
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & ((x < x0) + ((x == x0) & (y < y0)))
    if mode == 'full':
        return (((x - x0)**2 + (y - y0)**2) <= r**2) & (
            ((x > x0) + ((x == x0) & (y > y0))) + ((x < x0) + ((x == x0) & (y < y0)))
        )
    raise NotImplementedError(f'Circle mask "{mode}" not implemented.')


def _plain_ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF, x_offset=ANCHOR_X_OFFSET,
                     y_offset=ANCHOR_Y_OFFSET, mode='full'):
    outer_mask = circle_mask(size=size, r=r_out, x_offset=x_offset, y_offset=y_offset, mode=mode)
    inner_mask = circle_mask(size=size, r=r_in, x_offset=x_offset, y_offset=y_offset, mode=mode)
    return outer_mask & (~(inner_mask))


class RounderRingMask:
    def __init__(self, size=64, r_out=RADIUS, x_offset=ANCHOR_X_OFFSET,
                 y_offset=ANCHOR_Y_OFFSET, mode='full'):
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
            zero_bg_freq_rot = transforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

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
        if r_in - 1 < 0:
            right_end = 0
        else:
            right_end = r_in - 1
        cand_list = self.ring_vector_np[r_out - 1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)

        if self.size % 2:
            mask = mask[:self.size - 1, :self.size - 1]  # [64, 64]

        return mask


_mask_obj = None


def _rounder_obj():
    """The official module-level ``mask_obj`` (built lazily here: it costs ~8s)."""
    global _mask_obj
    if _mask_obj is None:
        _mask_obj = RounderRingMask(size=65, r_out=RADIUS, x_offset=ANCHOR_X_OFFSET,
                                    y_offset=ANCHOR_Y_OFFSET)
    return _mask_obj


def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF, x_offset=ANCHOR_X_OFFSET,
              y_offset=ANCHOR_Y_OFFSET, mode='full'):
    """``USE_ROUNDER_RING = True`` rebinds ``ring_mask`` to the rounder ring."""
    if not USE_ROUNDER_RING:
        return _plain_ring_mask(size, r_out, r_in, x_offset, y_offset, mode)
    assert size == 64
    assert mode == 'full', f'not implemented mode {mode}'
    return _rounder_obj().get_ring_mask(r_out=r_out, r_in=r_in)


def fft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))


def ifft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))


def make_Fourier_ringid_pattern(device, key_value_combination, no_watermark_latents,
                                radius, radius_cutoff, ring_watermark_channel,
                                heter_watermark_channel, heter_watermark_region_mask=None,
                                ring_width=1):
    if ring_width != 1:
        raise NotImplementedError('Proposed watermark generation only implemented for ring width = 1.')

    if len(key_value_combination) != (RADIUS - RADIUS_CUTOFF):
        raise ValueError('Mismatch between #key values and #slots')

    shape = no_watermark_latents.shape
    if len(shape) != 4:
        raise ValueError(f'Invalid shape for initial latent: {shape}')

    latents_fft = fft(no_watermark_latents)
    watermarked_latents_fft = torch.zeros_like(latents_fft)

    radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        mask = torch.tensor(ring_mask(size=shape[-1], r_out=this_r_out, r_in=this_r_in)).to(device).to(torch.float64)
        for batch_index in range(shape[0]):
            for channel_index in range(len(ring_watermark_channel)):
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real + mask * key_value_combination[radius_index][channel_index]
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag + mask * key_value_combination[radius_index][channel_index]

    if len(heter_watermark_channel) > 0:
        assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
        heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
        w_type = 'noise'

        if w_type == 'noise':
            w_content = fft(torch.randn(*shape, device=device))  # [N, c, h, w]
        elif w_type == 'zeros':
            w_content = fft(torch.zeros(*shape, device=device))
        else:
            raise NotImplementedError

        for batch_index in range(shape[0]):
            for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                watermarked_latents_fft[batch_index, channel_id].real = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * w_content[batch_index][channel_id].real
                watermarked_latents_fft[batch_index, channel_id].imag = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * w_content[batch_index][channel_id].imag

    return watermarked_latents_fft


def generate_Fourier_watermark_latents(device, radius, radius_cutoff, watermark_region_mask,
                                       watermark_channel, original_latents=None,
                                       watermark_pattern=None):
    if original_latents is None:
        raise NotImplementedError('Original latents should be provided.')
    if watermark_pattern is None:
        raise NotImplementedError('Fourier watermark pattern should be provided.')

    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim=(-1, -2))

    assert len(watermark_channel) == len(watermark_region_mask)
    for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
        watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~channel_mask + watermark_pattern[:, channel] * channel_mask

    return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim=(-1, -2))).real


def get_distance(tensor1, tensor2, mask, p, mode, channel_min=False, channel=WATERMARK_CHANNEL):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
    if mode not in ['complex', 'real', 'imag']:
        raise NotImplementedError(f'Eval mode not implemented: {mode}')

    if not channel_min:
        if p == 1:
            if mode == 'complex':
                return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
            if mode == 'real':
                return torch.mean(torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)[mask]).item()
            if mode == 'imag':
                return torch.mean(torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)[mask]).item()
        raise NotImplementedError('only p = 1 is transcribed')
    else:
        if len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
            if mode == 'complex':
                diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])
            elif mode == 'real':
                diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)
            elif mode == 'imag':
                diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)
            l1_list = []
            for c_idx in range(len(mask)):
                mask_c = torch.zeros_like(mask)
                mask_c[c_idx] = mask[c_idx]
                l1_list.append(torch.mean(diff[mask_c]).item())
            return min(l1_list)
        raise NotImplementedError('only the released 1 ring + 1 heterogeneous channel case')


# --- verify.py / identify.py setup -----------------------------------------

def official_watermark_region_mask(size=64, device="cpu"):
    """The mask block shared verbatim by verify.py and identify.py."""
    sing_channel_ring_watermark_mask = torch.tensor(
        ring_mask(size=size, r_out=RADIUS, r_in=RADIUS_CUTOFF)
    )
    single_channel_heter_watermark_mask = torch.tensor(
        ring_mask(size=size, r_out=RADIUS, r_in=RADIUS_CUTOFF)
    )
    heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(
        len(HETER_WATERMARK_CHANNEL), 1, 1
    ).to(device)

    watermark_region_mask = []
    for channel_idx in WATERMARK_CHANNEL:
        if channel_idx in RING_WATERMARK_CHANNEL:
            watermark_region_mask.append(sing_channel_ring_watermark_mask)
        else:
            watermark_region_mask.append(single_channel_heter_watermark_mask)
    watermark_region_mask = torch.stack(watermark_region_mask).to(device)
    return watermark_region_mask, heter_watermark_region_mask


def official_key_value_combinations(ring_value_range=64, quantization_levels=2, assigned_keys=-1):
    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [
        [list(combo) for combo in itertools.product(
            np.linspace(-ring_value_range, ring_value_range, quantization_levels).tolist(),
            repeat=len(RING_WATERMARK_CHANNEL))]
        for _ in range(single_channel_num_slots)
    ]
    key_value_combinations = list(itertools.product(*key_value_list))
    if assigned_keys > 0:
        assert assigned_keys <= len(key_value_combinations)
        key_value_combinations = random.sample(key_value_combinations, k=assigned_keys)
    return key_value_combinations


def official_keybook(general_seed=42, latent_shape=(1, 4, 64, 64), device="cpu",
                     ring_value_range=64, quantization_levels=2, assigned_keys=-1,
                     fix_gt=1, time_shift=1, max_index=None):
    """The verify.py / identify.py key construction block.

    ``base_latents`` stands in for ``pipe.get_random_latents()``; on CPU that is
    the same ``torch.randn`` draw the official code performs on the device. Only
    the *draw* matters — the pattern itself starts from ``zeros_like``.

    ``max_index`` stops after that many candidates (the RNG stream up to that
    point is identical, so the produced keys are the official ones).
    """
    set_random_seed(general_seed)
    base_latents = torch.randn(*latent_shape, device=device)
    base_latents = base_latents.to(torch.float64)

    _, heter_watermark_region_mask = official_watermark_region_mask(
        size=latent_shape[-1], device=device
    )
    key_value_combinations = official_key_value_combinations(
        ring_value_range, quantization_levels, assigned_keys
    )
    if max_index is not None:
        key_value_combinations = key_value_combinations[:max_index]

    Fourier_watermark_pattern_list = [
        make_Fourier_ringid_pattern(
            device, list(combo), base_latents,
            radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
            ring_watermark_channel=RING_WATERMARK_CHANNEL,
            heter_watermark_channel=HETER_WATERMARK_CHANNEL,
            heter_watermark_region_mask=heter_watermark_region_mask,
        )
        for _, combo in enumerate(key_value_combinations)
    ]

    if fix_gt:
        Fourier_watermark_pattern_list = [
            fft(ifft(pattern).real) for pattern in Fourier_watermark_pattern_list
        ]

    if time_shift:
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            # official: the "* args.time_shift_factor" multiplication is commented out
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(
                torch.fft.fftshift(
                    ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]),
                    dim=(-1, -2),
                )
            )

    return Fourier_watermark_pattern_list
