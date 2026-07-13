import typing

import random
import numpy as np
import torch
import torchvision

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL
from utils import utils 

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--hstr_seed', default=999999, type=int)
parser.add_argument('--latent_channel', default=4, type=int)

RADIUS = 14
RADIUS_CUTOFF = 3
USE_ROUNDER_RING = True

w_channel = 3
TREE_WATERMARK_CHANNEL = [w_channel]
HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [w_channel]
RINGID_WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL) # [0,3]

class HSTRProvider(WmProvider):
    def __init__(self,
                 hstr_seed: int = None,
                 latent_channel: int = 4,
                 start: int = 10,
                 end: int = 54, # 64-10 = hw_latent-start
                 hw_latent : int = 64,
                 fix_gt: int = 1,
                 wm_capacity: int = 2**(RADIUS-RADIUS_CUTOFF),
                 **kwargs):
        
        super().__init__(**kwargs)

        if hstr_seed is not None:
            utils.set_random_seed(hstr_seed)

        self.shape = (1, latent_channel, hw_latent, hw_latent)
        self.hstr_seed = hstr_seed
        
        assert wm_capacity == 2048
        self.wm_capacity = wm_capacity
        self.hstr_seed_list = [*range(hstr_seed, hstr_seed + wm_capacity)] # 2048 seed numbers
        self.sample_index = fix_gt
        
        self.start = start
        self.end = end
        
        self.center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))

        self.masks, self.watermark_region_mask_hstr = self.__get_watermarking_mask()
        self.gt_patch = self.__get_watermarking_pattern()

    def get_wm_type(self) -> str:
        return "HSTR"
    
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

        latents_w, pristine_latents_w_fft = self.__inject_watermark(latents_clean, self.gt_patch, self.masks, center=True, cut_real=False)

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

        # # watermarked fft pristine
        # pristine_latents_w_fft_torch = pristine_latents_w_fft.to(self.device)
        # pristine_latents_w_fft_PIL = torch_to_PIL(pristine_latents_w_fft_torch)
        # # watermarked fft wchannel pristine
        # pristine_latents_w_fft_wchannel_torch = pristine_latents_w_fft[:, self.w_channel: self.w_channel + 1].to(self.device)
        # pristine_latents_w_fft_wchannel_PIL = torch_to_PIL(pristine_latents_w_fft_wchannel_torch)

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

            # # pristine watermarked fft (before IFFT drop)
            # "pristine_zT_fft_torch": pristine_latents_w_fft_torch,
            # "pristine_zT_fft_PIL": pristine_latents_w_fft_PIL,
            # "pristine_zT_fft": pristine_latents_w_fft_PIL,
            # # pristine wchannel
            # "pristine_zT_fft_wchannel_torch": pristine_latents_w_fft_wchannel_torch,
            # "pristine_zT_fft_wchannel_PIL": pristine_latents_w_fft_wchannel_PIL,
            # "pristine_zT_fft_wchannel": pristine_latents_w_fft_wchannel_PIL,
        }
    
    def __get_l1_distance(self, reversed_latents_w: typing.Union[torch.Tensor, np.ndarray], 
                          mask, channel=RINGID_WATERMARK_CHANNEL, p=1,
                          mode='complex', channel_min=False, center=False):
        
        Fourier_wm_zT_fft = torch.zeros_like(reversed_latents_w, dtype=torch.complex64)
        Fourier_wm_zT_fft[self.center_slice] = HSTRProvider.fft(reversed_latents_w[self.center_slice])

        if Fourier_wm_zT_fft.shape != self.gt_patch.shape:
            raise ValueError(f'Shape mismatch during eval: {Fourier_wm_zT_fft.shape} vs {self.gt_patch.shape}')
        if mode not in ['complex', 'real', 'imag']:
            raise NotImplementedError(f'Eval mode not implemented: {mode}')

        w_metric = None

        def calc_diff(t1, t2, m):
            if mode == 'complex':
                diff = torch.abs(t1 - t2)
            elif mode == 'real':
                diff = torch.abs(t1.real - t2.real)
            else:  # 'imag'
                diff = torch.abs(t1.imag - t2.imag)
            return diff if m is None else diff[m]
        
        if center:
            temp_tensor1 = Fourier_wm_zT_fft[self.center_slice].clone() # 1,4,64,64 -> 1,4,44,44
            temp_tensor2 = self.gt_patch[self.center_slice].clone() # 1,4,44,44
            temp_mask = mask[None, ...][self.center_slice][0].clone() # (C_R+C_H,64,64) -> (C_R+C_H,44,44)
            if not channel_min: # C_H=0. Only non-hetero watermarked channels C_R.
                diff = calc_diff(temp_tensor1[0][channel], temp_tensor2[0][channel], temp_mask) # (C_R,44,44) masked.
                w_metric = torch.norm(diff, p=p).item() / torch.sum(temp_mask) if p != 1 else torch.mean(diff).item()    
            else:
                assert p == 1
                diff = calc_diff(temp_tensor1[0][channel], temp_tensor2[0][channel], None)  # (C_R+C_H,44,44) unmasked.
                l1_list = [torch.mean(diff[i][temp_mask[i]]).item() for i in range(len(channel))]
                if channel == RINGID_WATERMARK_CHANNEL: # [0,3]
                    w_metric = min(l1_list)
                else:
                    raise NotImplementedError
        else:
            if not channel_min:
                diff = calc_diff(Fourier_wm_zT_fft[0][channel], self.gt_patch[0][channel], mask) # (C_R,64,64) masked.
                w_metric = torch.norm(diff, p=p).item() / torch.sum(mask) if p != 1 else torch.mean(diff).item()
            else:
                diff = calc_diff(Fourier_wm_zT_fft[0][channel], self.gt_patch[0][channel], None)  # (C_R+C_H,64,64) unmasked.
                l1_list = [torch.mean(diff[i][mask[i]]).item() for i in range(len(channel))]
                
                if len(RING_WATERMARK_CHANNEL) > 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                    ring_indices = [i for i, c in enumerate(RINGID_WATERMARK_CHANNEL) if c in RING_WATERMARK_CHANNEL]
                    heter_indices = [i for i, c in enumerate(RINGID_WATERMARK_CHANNEL) if c in HETER_WATERMARK_CHANNEL]
                    ring_mean = sum(l1_list[i] * torch.sum(mask[i]).item() for i in ring_indices) / sum(torch.sum(mask[i]).item() for i in ring_indices)
                    w_metric = min(ring_mean, min(l1_list[i] for i in heter_indices))
                elif len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                    w_metric = min(l1_list)
                else:
                    raise NotImplementedError
                
        # reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        # target_patch = self.gt_patch

        # reversed_latents_w_fft = reversed_latents_w_fft[:, WATERMARK_CHANNEL, :, :]   # (1, 2, 64, 64)
        # target_patch  = target_patch[:, WATERMARK_CHANNEL, :, :]
        # mask = self.watermarking_mask.unsqueeze(0)
        # w_metric = torch.abs(reversed_latents_w_fft[mask] - target_patch[mask]).mean().item()

        return {
            "l1_dist": w_metric,
        }
    
    def get_accuracies(self,
                       latents: typing.Union[torch.Tensor, np.ndarray]) -> typing.Dict[str, typing.Any]:
        """
        Get the accuracy of the watermarking scheme

        @param latents: torch.Tensor or np.array, shape: self.latent_shape,

        @return: dict
        """
        
        # results = self.__get_p_value(latents)
        # p_values = results["p_values"]
        # accuracies = [1 - p for p in p_values]

        is_center = True
        channel_min = True
        
        results = self.__get_l1_distance(reversed_latents_w=latents, mask=self.watermark_region_mask_hstr, channel=RINGID_WATERMARK_CHANNEL, 
                                         p=1, mode='complex', channel_min=channel_min, center=is_center)
        l1_dist = [results["l1_dist"]]

        return {
            # "accuracies": accuracies,
            # "p_values": p_values,
            "l1_dist": l1_dist
            # "zT_fft_torch": results["zT_fft_wchannel_PIL"],
            # "zT_fft_PIL": results["zT_fft_PIL"],
            # "zT_fft": results["zT_fft"],
            # "zT_fft_wchannel_torch": results["zT_fft_wchannel_PIL"],
            # "zT_fft_wchannel_PIL": results["zT_fft_wchannel_PIL"],
            # "zT_fft_wchannel": results["zT_fft_wchannel"],
        }
    
    def __get_watermarking_mask(self) -> torch.tensor:
        single_channel_tree_watermark_mask = torch.tensor(circle_mask(size=self.latent_shape[-1], r=RADIUS))

        tree_masks = torch.zeros(self.latent_shape, dtype=torch.bool)
        tree_masks[:, TREE_WATERMARK_CHANNEL] = single_channel_tree_watermark_mask

        if len(HETER_WATERMARK_CHANNEL) > 0:
            single_channel_heter_watermark_mask = torch.tensor(ring_mask(size=self.latent_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)) # (64,64)

        masks = tree_masks
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask # (64,64) RounderRingMask for Hetero Watermark (noise)

        # [get_distance - input] watermark_region_mask for detection process
        watermark_region_mask_hstr = torch.stack([
            single_channel_heter_watermark_mask, 
            single_channel_tree_watermark_mask]).to(self.device) # # (C_R+C_H,64,64) - cuda
            
        return masks, watermark_region_mask_hstr
    
    def __get_watermarking_pattern(self) -> torch.tensor:
        
        Fourier_watermark_pattern_list = [self.__make_Fourier_treering_pattern(self.shape, this_hstr_seed,
                                                                               hs=True, center=True, heter=True) for this_hstr_seed in self.hstr_seed_list]
        assert len(Fourier_watermark_pattern_list) == self.wm_capacity

        pattern_gt = Fourier_watermark_pattern_list
        if len(pattern_gt[0].shape) in (4, 16):
            pattern_gt = torch.cat(pattern_gt, dim=0) # (N,4,64,64) for Tree-Ring, RingID, HSTR
        
        identify_gt_indices = np.random.choice(self.wm_capacity, size=8192).tolist()
        key_index = identify_gt_indices[self.sample_index]
        pattern_gt = Fourier_watermark_pattern_list[key_index]
        
        return pattern_gt
    

    def __make_Fourier_treering_pattern(self, shape, hstr_seed, hs=False, center=False, heter=False):
        
        assert shape[-1] == shape[-2] # 64==64
        gt_init = torch.randn(self.latent_shape).to(self.device, torch.float64) # (1,4,64,64)

        # [HSTR] center-aware design
        if center:
            watermarked_latents_fft = HSTRProvider.fft(torch.zeros(shape, device=self.device)) # (1,4,64,64) complex64
            gt_patch_tmp = HSTRProvider.fft(gt_init[self.center_slice]).clone().detach().to(torch.complex64) # (1,4,44,44) complex64
            center_len = gt_patch_tmp.shape[-1] // 2 # 22
            for radius in range(center_len-1, 0, -1): # [21,20,...,1]
                tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius)) # (64,64)
                for j in range(watermarked_latents_fft.shape[1]): # GT : all channel Tree-Ring
                    watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item() # Use (22,22+radius) element.
            if heter: # Gaussian noise key (Heterogenous watermark in RingID)
                watermarked_latents_fft[:, HETER_WATERMARK_CHANNEL, self.start:self.end, self.start:self.end] = gt_patch_tmp[:, HETER_WATERMARK_CHANNEL] # (1,1,44,44) complex64
        # # [Original Tree-Ring]
        # else:
        #     watermarked_latents_fft = HSTRProvider.fft(gt_init) # (1,4,64,64)
        #     # constant ring values chosen from a Gaussian distribution.
        #     gt_patch_tmp = watermarked_latents_fft.clone().detach()
        #     center_len = shape[-1] // 2 # 32
        #     for radius in range(center_len-1, 0, -1): # [31,30,...,1]
        #         tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius))
        #         for j in range(watermarked_latents_fft.shape[1]): # GT : all channel Tree-Ring
        #             watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item() # Use (32,32+radius) element.
        
        # [Hermitian Symmetric Fourier] HSTR or TR
        if hs: 
            return HSTRProvider.enforce_hermitian_symmetry(watermarked_latents_fft)
        
        return watermarked_latents_fft # (1,4,64,64) complex64
    
    def __inject_watermark(self, inverted_latent, w_pattern, w_mask, cut_real=True, center=False):

        assert len(w_pattern.shape) == 4
        assert len(w_mask.shape) == 4
        batch_size = inverted_latent.shape[0]
        w_mask = w_mask.repeat(batch_size, 1, 1, 1)

        inverted_latent = inverted_latent.to(self.device)
        w_pattern = w_pattern.to(self.device)
        w_mask = w_mask.to(self.device)
        # inject watermarks in fourier space
        # masking processing according to center option
        if center:
            center_latent_fft = HSTRProvider.fft(inverted_latent[self.center_slice]) # (N,4,44,44) complex64
            # insert watermark
            temp_mask = w_mask[self.center_slice] # (N,4,44,44) boolean
            temp_pattern = w_pattern[self.center_slice] # (N,4,44,44) complex64
            center_latent_fft[temp_mask] = temp_pattern[temp_mask].clone() # (N,4,44,44) complex64
            # IFFT and restore to original location
            center_latent_ifft = HSTRProvider.ifft(center_latent_fft) # (N,4,44,44)
            center_latent_ifft = center_latent_ifft.real if cut_real or center_latent_ifft.imag.abs().max() < 1e-3 else center_latent_ifft
            # restore to original tensor
            inverted_latent = inverted_latent.clone()
            inverted_latent[self.center_slice] = center_latent_ifft
            inverted_latent_fft = None
        else:
            # maintain existing logic
            inverted_latent_fft = HSTRProvider.fft(inverted_latent) # complex64
            inverted_latent_fft[w_mask] = w_pattern[w_mask].clone()
            inverted_latent = HSTRProvider.ifft(inverted_latent_fft) # complex64
            inverted_latent = inverted_latent.real if cut_real or inverted_latent.imag.abs().max() < 1e-3 else inverted_latent
            # if cut_real: # enforcing to discard imaginary part regardless of its values.
            #     inverted_latent = inverted_latent.real # float32
            # else:
            #     if inverted_latent.imag.abs().max() < 1e-3: # discard numerical error in imaginary part.
            #         inverted_latent = inverted_latent.real # float32
            #     else:
            #         raise
        # hot fix to prevent out of bounds values. will "properly" fix this later
        inverted_latent[inverted_latent == float("Inf")] = 4
        inverted_latent[inverted_latent == float("-Inf")] = -4

        return inverted_latent, inverted_latent_fft # float32, complex64

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
    def enforce_hermitian_symmetry(freq_tensor):
        B, C, H, W = freq_tensor.shape # fftshifted frequency (complex tensor) - center (32,32)
        assert H == W, "H != W"
        freq_tensor = freq_tensor.clone()
        freq_tensor_tmp = freq_tensor.clone()
        # DC point (no imaginary)
        freq_tensor[:, :, H//2, W//2] = torch.real(freq_tensor_tmp[:, :, H//2, W//2])
        if H % 2 == 0: # Even
            # Nyquist Points (no imaginary)
            freq_tensor[:, :, 0, 0] = torch.real(freq_tensor_tmp[:, :, 0, 0])
            freq_tensor[:, :, H//2, 0] = torch.real(freq_tensor_tmp[:, :, H//2, 0])  # (32, 0)
            freq_tensor[:, :, 0, W//2] = torch.real(freq_tensor_tmp[:, :, 0, W//2])  # (0, 32)
        
            # Nyquist axis - conjugate
            freq_tensor[:, :, 0, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0, W//2+1:], dims=[2]))
            freq_tensor[:, :, H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
            freq_tensor[:, :, 1:H//2, 0] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, 0], dims=[2]))
            freq_tensor[:, :, 1:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
            # Square quadrants - conjugate
            freq_tensor[:, :, 1:H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
            freq_tensor[:, :, H//2+1:, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 1:H//2, W//2+1:], dims=[2, 3]))
        else: # Odd
            # Nyquist axis - conjugate
            freq_tensor[:, :, H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
            freq_tensor[:, :, 0:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
            # Square quadrants - conjugate
            freq_tensor[:, :, 0:H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
            freq_tensor[:, :, H//2+1:, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0:H//2, W//2+1:], dims=[2, 3]))
        return freq_tensor

# ring mask
class RounderRingMask:
    def __init__(self, size=65, r_out=RADIUS):
        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center, center
        # center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            zero_bg_freq_rot = torchvision.transforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

        res = res.numpy()
        self.res = res
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y],  return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values!=0][0]
        
    def get_ring_mask(self, r_out, r_in):
        # get mask from pure_bg
        assert r_out <= self.r_out
        if r_in - 1 < 0:
            right_end = 0  # None, to take the center
        else:
            right_end = r_in - 1
        cand_list = self.ring_vector_np[r_out-1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)
        if self.size % 2:
            mask = mask[:self.size-1, :self.size-1]  # [64, 64]
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
        return outer_mask & (~(inner_mask))
    

def circle_mask(size: int, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    # y = y[::-1]
    # This original tree-ring code is wrong since (0,0) is on the top-left corner. (not bottom-left)
    # Plus, RingID's Y-center adjustment with -1 offset is done with this.
    return ((x - x0)**2 + (y-y0)**2)<= r**2