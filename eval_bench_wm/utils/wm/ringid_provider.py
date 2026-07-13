
"""
RingID provider (FFT-based) implemented to mirror the public API and style of tr_provider.py (Tree-Ring).
It relies on the existing project utilities in utils.py/optim_utils.py without changing their math.
"""
import copy
import typing

import random
import numpy as np
import scipy
import torch
import itertools

import torch.nn.functional as F

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL
from utils import utils 

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--rid_seed', default=999999, type=int)
parser.add_argument('--ring_width', default=1, type=int)
parser.add_argument('--quantization_levels', default=2, type=int)
parser.add_argument('--ring_value_range', default=64, type=int)

parser.add_argument('--fix_gt', type=int, default=1, help='use watermark after discarding the imag part on space domain as gt.')
parser.add_argument('--time_shift', type=int, default=1, help='use time-shift')
parser.add_argument('--time_shift_factor', type=float, default=1.0, help='factor to scale the value after time-shift')
parser.add_argument('--assigned_keys', type=int, default=-1, help='number of assigned keys, -1 for all possible kyes')
parser.add_argument('--channel_min', type=int, default=1, help='only for heterogeous watermark, when match gt, take min among channels as the result')

RADIUS = 14
RADIUS_CUTOFF = 3
ANCHOR_X_OFFSET = 0     
ANCHOR_Y_OFFSET = 0     # 1 = not correct, 0 = correct
USE_ROUNDER_RING = True

HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [3]
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)

class RingIDProvider(WmProvider):
    def __init__(self,
                 rid_seed: int = None,
                 channel_min: int = 1,
                 ring_value_range: int = 64,
                 quantization_levels: int = 2,
                 assigned_keys: int = -1,
                 fix_gt: int = 1,
                 time_shift: int = 1,
                 **kwargs):
        """
        Args mirror tr_provider; extra RingID params are added with safe defaults.
        """
        super().__init__(**kwargs)

        if rid_seed is not None:
            utils.set_random_seed(rid_seed)

        self.channel_min = channel_min
        if self.channel_min: assert len(HETER_WATERMARK_CHANNEL) > 0

        self.rid_seed = rid_seed
        self.ring_value_range = ring_value_range
        self.quantization_levels = quantization_levels
        self.assigned_keys = assigned_keys
        self.fix_gt = fix_gt
        self.time_shift = time_shift

        self.heter_watermark_region_mask = None
        
        self.watermarking_mask = self.__get_watermarking_mask() # bool mask, shape=self.latent_shape
        self.gt_patch = self.__get_watermarking_pattern()       # complex FFT pattern, shape=self.latent_shape


    def get_wm_type(self) -> str:
        return "RID"


    def get_wm_latents(self,
                       latents_clean: torch.Tensor = None,
                       seed: int = None) -> typing.Dict[str, typing.Any]:
        """
        Create (or inject into) latents and return a dict matching tr_provider.get_wm_latents() keys.
        """
        if seed is not None:
            utils.set_random_seed(seed)

        # Prepare clean latents
        if latents_clean is None:
            latents_clean = torch.randn(self.latent_shape)
        latents_clean = latents_clean.clone().to(self.device, self.dtype)
        
        # latents_w = latents_clean
        # Inject watermark
        latents_w, pristine_latents_w_fft = self.__generate_Fourier_watermark_latents(device=self.device, 
                                                                           radius=RADIUS,
                                                                           radius_cutoff = RADIUS_CUTOFF,
                                                                           original_latents = latents_clean,
                                                                           watermark_pattern = self.gt_patch,
                                                                           watermark_channel = WATERMARK_CHANNEL,
                                                                           watermark_region_mask = self.watermarking_mask)
        pristine_latents_w_fft = pristine_latents_w_fft.real

        # clean
        latents_clean_torch = latents_clean.to(self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean_torch)
        # clean fft
        latents_clean_fft_torch = torch.fft.fftshift(torch.fft.fft2(latents_clean.to(torch.float32)), dim=(-1, -2)).real.to(self.device)
        latents_clean_fft_PIL = torch_to_PIL(latents_clean_fft_torch)
        # clean fft wchannel
        ch = RING_WATERMARK_CHANNEL[0]
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
        
        # watermarked fft pristine
        pristine_latents_w_fft_torch = pristine_latents_w_fft.to(self.device)
        pristine_latents_w_fft_PIL = torch_to_PIL(pristine_latents_w_fft_torch)
        # watermarked fft wchannel pristine
        pristine_latents_w_fft_wchannel_torch = pristine_latents_w_fft[:, ch: ch + 1].to(self.device)
        pristine_latents_w_fft_wchannel_PIL = torch_to_PIL(pristine_latents_w_fft_wchannel_torch)

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

            # pristine watermarked fft (before IFFT drop)
            "pristine_zT_fft_torch": pristine_latents_w_fft_torch,
            "pristine_zT_fft_PIL": pristine_latents_w_fft_PIL,
            "pristine_zT_fft": pristine_latents_w_fft_PIL,
            # pristine wchannel
            "pristine_zT_fft_wchannel_torch": pristine_latents_w_fft_wchannel_torch,
            "pristine_zT_fft_wchannel_PIL": pristine_latents_w_fft_wchannel_PIL,
            "pristine_zT_fft_wchannel": pristine_latents_w_fft_wchannel_PIL,
        }

    def __get_watermarking_mask(self) -> torch.tensor:

        sing_channel_ring_watermark_mask = torch.tensor(
            RingIDProvider.__ring_mask(
                size = self.latent_shape[-1], 
                r_out = RADIUS, 
                r_in = RADIUS_CUTOFF)
            )

        # get heterogeneous watermark mask
        if len(HETER_WATERMARK_CHANNEL) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                    RingIDProvider.__ring_mask(
                        size = self.latent_shape[-1], 
                        r_out = RADIUS, 
                        r_in = RADIUS_CUTOFF)  # TODO: change to whole mask
                    )       
            self.heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(self.device)

        watermark_region_mask = []
        for channel_idx in WATERMARK_CHANNEL:
            if channel_idx in RING_WATERMARK_CHANNEL:
                watermark_region_mask.append(sing_channel_ring_watermark_mask)
            else:
                watermark_region_mask.append(single_channel_heter_watermark_mask)
        watermark_region_mask = torch.stack(watermark_region_mask).to(self.device)  # [C, 64, 64]
        return watermark_region_mask      
       

    def __get_watermarking_pattern(self) -> torch.tensor:

        base_latents = torch.randn(self.latent_shape)
        base_latents = base_latents.to(torch.float64)

        single_channel_num_slots = RADIUS - RADIUS_CUTOFF
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-self.ring_value_range, self.ring_value_range, self.quantization_levels).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))

        # random select from all possible value combinations, then generate patterns for selected ones.
        if self.assigned_keys > 0:
            assert self.assigned_keys <= len(key_value_combinations)
            key_value_combinations = random.sample(key_value_combinations, k=self.assigned_keys)

        Fourier_watermark_pattern_list = [self.__make_Fourier_ringid_pattern(self.device, list(combo), base_latents, 
                                                                                    radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                                                                                    ring_watermark_channel=RING_WATERMARK_CHANNEL, 
                                                                                    heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                                                                                    heter_watermark_region_mask=self.heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
                                                                                    for _, combo in enumerate(key_value_combinations)]            

        ring_capacity = len(Fourier_watermark_pattern_list)

        if self.fix_gt:
            Fourier_watermark_pattern_list = [RingIDProvider.fft(RingIDProvider.ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]

        if self.time_shift:
            for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
                # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
                Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = RingIDProvider.fft(torch.fft.fftshift(RingIDProvider.ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))

        # Use a single ring pattern for verification
        Fourier_watermark_pattern = Fourier_watermark_pattern_list[628] #(1, 4, 64, 64)

        # print(f'[Info] Ring capacity = {ring_capacity}') # 2048
        return Fourier_watermark_pattern


    def __generate_Fourier_watermark_latents(
        self,
        device, 
        radius, 
        radius_cutoff, 
        watermark_region_mask, 
        watermark_channel, 
        original_latents = None, 
        watermark_pattern = None):
    
        #set_random_seed(seed)

        if original_latents is None:
            #original_latents = torch.randn(*shape, device = device)
            raise NotImplementedError('Original latents should be provided.')

        if watermark_pattern is None:
            raise NotImplementedError('Fourier watermark pattern should be provided.')

        # circular_mask = torch.tensor(ring_mask(size = original_latents.shape[-1], r_out = radius, r_in = radius_cutoff)).to(device)
        watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim = (-1, -2))

        # for channel in watermark_channel:
        #     watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~circular_mask + watermark_pattern[:, channel] * circular_mask
        
        assert len(watermark_channel) == len(watermark_region_mask)
        for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
            watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~channel_mask + watermark_pattern[:, channel] * channel_mask
        
        latents_w = torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim = (-1, -2))).real
        
        pristine_latents_w_fft = watermarked_latents_fft.clone()

        ## Undo: have bugs
        # latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
        # latents_w = torch.fft.ifft2(torch.fft.ifftshift(latents_w_fft, dim=(-1, -2)))
        # latents_w = latents_w.real

        return latents_w, pristine_latents_w_fft
    
    def __get_l1_distance(self, reversed_latents_w: typing.Union[torch.Tensor, np.ndarray]):
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = self.gt_patch

        reversed_latents_w_fft = reversed_latents_w_fft[:, WATERMARK_CHANNEL, :, :]   # (1, 2, 64, 64)
        target_patch  = target_patch[:, WATERMARK_CHANNEL, :, :]
        mask = self.watermarking_mask.unsqueeze(0)
        w_metric = torch.abs(reversed_latents_w_fft[mask] - target_patch[mask]).mean().item()

        return {
            "l1_dist": w_metric,
        }


    def __get_p_value(self, reversed_latents_w: typing.Union[torch.Tensor, np.ndarray]) -> typing.Dict[str, typing.Any]:
        """
        Low means -> Watermark is present

        Will calulate the fft of latetns and search for WM pattern

        @param latent: torch.tensor, shape: self.latent_shape, on device

        @return: dict
        """
        
        zT_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))

        zT_fft_real = zT_fft.real
        zero_mask = torch.zeros(zT_fft.shape, dtype=torch.bool).to(self.device)
        for i, ch in enumerate(WATERMARK_CHANNEL):
            if i == 1:
                zero_mask[:, ch] = self.watermarking_mask[i].to(self.device)
        self.watermarking_mask = zero_mask

        watermarking_mask = self.watermarking_mask[0]
        gt_patch = self.gt_patch[0]

        p_values = []
        for latent_fft in zT_fft:
            # get the watermarking mask
            latent_vec = latent_fft[watermarking_mask].flatten()
            target_vec = gt_patch[watermarking_mask].flatten()
            target_vec = torch.concatenate([target_vec.real, target_vec.imag])
            latent_vec = torch.concatenate([latent_vec.real, latent_vec.imag])

            # p-test
            sigma = latent_vec.std()
            lambd = (target_vec ** 2 / sigma ** 2).sum().item()
            x = (((latent_vec - target_vec) / sigma) ** 2).sum().item()
            p = scipy.stats.ncx2.cdf(x=x, df=len(target_vec), nc=lambd)
            p_values.append(p)

        return {
            "p_values": p_values,
            "zT_fft_PIL": torch_to_PIL(zT_fft_real),
            "zT_fft": torch_to_PIL(zT_fft_real),
            # "zT_fft_wchannel_torch": zT_fft_wch,
            # "zT_fft_wchannel_PIL": torch_to_PIL(zT_fft_wch),
            # "zT_fft_wchannel": torch_to_PIL(zT_fft_wch),
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

        results = self.__get_l1_distance(latents)
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
    
    def __make_Fourier_ringid_pattern(self,
                                    device,
                                    key_value_combination, 
                                    no_watermark_latents, 
                                    radius,
                                    radius_cutoff, 
                                    ring_watermark_channel, 
                                    heter_watermark_channel, 
                                    heter_watermark_region_mask=None,
                                    ring_width = 1, 
                                    ):
        if ring_width != 1:
            raise NotImplementedError(f'Proposed watermark generation only implemented for ring width = 1.')

        if len(key_value_combination) != (RADIUS - RADIUS_CUTOFF):
            raise ValueError('Mismatch between #key values and #slots')
        
        shape = no_watermark_latents.shape
        if len(shape) != 4:
            raise ValueError(f'Invalid shape for initial latent: {shape}')
        
        latents_fft = RingIDProvider.fft(no_watermark_latents)
        # watermarked_latents_fft = copy.deepcopy(latents_fft)
        watermarked_latents_fft = torch.zeros_like(latents_fft).to(device)

        radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

        # put ring
        for radius_index in range(len(radius_list)):
            this_r_out = radius_list[radius_index]
            this_r_in = this_r_out - ring_width
            mask = torch.tensor(RingIDProvider.__ring_mask(size = shape[-1], r_out = this_r_out, r_in = this_r_in)).to(device).to(torch.float64)  # sector_idx default to -1
            
            for batch_index in range(shape[0]):
                for channel_index in range(len(ring_watermark_channel)):
                    watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real + mask * key_value_combination[radius_index][channel_index]
                    watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag + mask * key_value_combination[radius_index][channel_index]

        # put noise or zeros
        if len(heter_watermark_channel) > 0:
            assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
            heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
            w_type = 'noise'
            
            if w_type == 'noise':
                w_content = RingIDProvider.fft(torch.randn(*shape, device = device))  # [N, c, h, w]
            elif w_type == 'zeros':
                w_content = RingIDProvider.fft(torch.zeros(*shape, device = device))  # [N, c, h, w]
            else:
                raise NotImplementedError
            
            for batch_index in range(shape[0]):
                for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                    watermarked_latents_fft[batch_index, channel_id].real = \
                        (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * w_content[batch_index][channel_id].real
                    watermarked_latents_fft[batch_index, channel_id].imag = \
                        (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * w_content[batch_index][channel_id].imag

        return watermarked_latents_fft

    def get_distance(tensor1, tensor2, mask, p, mode, channel_min=False, channel = WATERMARK_CHANNEL):
        if tensor1.shape != tensor2.shape:
            raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
        if mode not in ['complex', 'real', 'imag']:
            raise NotImplemented(f'Eval mode not implemented: {mode}')
        
        if not channel_min:
            if p == 1:
                # a faster implementation for l1 distance
                if mode == 'complex':
                    return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
                if mode == 'real':
                    return torch.mean(torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)[mask]).item()
                if mode == 'imag':
                    return torch.mean(torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)[mask]).item()
            else:
                if mode == 'complex':
                    return torch.norm(torch.abs(tensor1[0][channel][mask] - tensor2[0][channel][mask]), p = p).item() / torch.sum(mask)
                if mode == 'real':
                    return torch.norm(torch.abs(tensor1[0][channel][mask].real - tensor2[0][channel][mask].real), p = p).item() / torch.sum(mask)
                if mode == 'imag':
                    return torch.norm(torch.abs(tensor1[0][channel][mask].imag - tensor2[0][channel][mask].imag), p = p).item() / torch.sum(mask)
        else:
            # argmin TODO: normalize
            if  len(RING_WATERMARK_CHANNEL) > 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                ring_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in RING_WATERMARK_CHANNEL]
                heter_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in HETER_WATERMARK_CHANNEL]
                if mode == 'complex':
                    diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
                elif mode == 'real':
                    diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
                elif mode == 'imag':
                    diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
                l1_list = []
                num_items = []
                for c_idx in range(len(mask)):
                    mask_c = torch.zeros_like(mask)
                    mask_c[c_idx] = mask[c_idx]
                    l1_list.append(torch.mean(diff[mask_c]).item())
                    num_items.append(torch.sum(mask_c).item())
                total = 0
                num = 0
                for ring_channel_idx in ring_channel_idx_list:
                    total += l1_list[ring_channel_idx] * num_items[ring_channel_idx]
                    num += num_items[ring_channel_idx]
                ring_channels_mean = total / num
                return min(ring_channels_mean, min([l1_list[idx] for idx in heter_channel_idx_list]))
            elif len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                if mode == 'complex':
                    diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
                elif mode == 'real':
                    diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
                elif mode == 'imag':
                    diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
                l1_list = []
                for c_idx in range(len(mask)):
                    mask_c = torch.zeros_like(mask)
                    mask_c[c_idx] = mask[c_idx]
                    l1_list.append(torch.mean(diff[mask_c]).item())
                return min(l1_list)
            else:
                raise NotImplementedError
            

    # ---------- Internal functions ----------    
    @staticmethod
    def __ring_mask(size=64, r_out = RADIUS, r_in = RADIUS_CUTOFF, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full') -> np.ndarray:
        """Generate a boolean ring (annulus) mask."""
        outer_mask = RingIDProvider.__circle_mask(size = size, r = r_out, x_offset = x_offset, y_offset = y_offset, mode = mode)
        inner_mask = RingIDProvider.__circle_mask(size = size, r = r_in, x_offset = x_offset, y_offset = y_offset, mode = mode)
        return outer_mask & (~(inner_mask))
    
    @staticmethod
    def __circle_mask(size = 64, r = RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full') -> np.ndarray:
        '''
        Returns a (size, size) bool type numpy array.
        '''
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset - 1
        y, x = np.ogrid[:size, :size]
        y = y[::-1]

        if mode == 'left':
            return (((x - x0)**2 + (y - y0)**2)<= r**2) & ((x > x0) + ((x == x0) & (y > y0)))
        if mode == 'right':
            return (((x - x0)**2 + (y - y0)**2)<= r**2) & ((x < x0) + ((x == x0) & (y < y0)))
        if mode == 'full':
            return (((x - x0)**2 + (y - y0)**2)<= r**2) & (((x > x0) + ((x == x0) & (y > y0))) + ((x < x0) + ((x == x0) & (y < y0))))
        raise NotImplementedError(f'Circle mask "{mode}" not implemented.')

    @staticmethod
    def fft(input_tensor: torch.Tensor) -> torch.Tensor:
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim = (-1, -2))

    @staticmethod
    def ifft(input_tensor: torch.Tensor) -> torch.Tensor:
        assert len(input_tensor.shape) == 4
        return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim = (-1, -2)))

    @staticmethod
    def _fftshift(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftshift(x, dim=(-1, -2))

    @staticmethod
    def _ifftshift(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftshift(x, dim=(-1, -2))
    

class RounderRingMask:
    def __init__(self, size = 64, r_out = RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        import torchvision
        for angle in range(1, 360):
            zero_bg_freq_rot = torchvision.transforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

        res = res.numpy()
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y],  return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values!=0][0]
        
    def get_ring_mask(self, r_out, r_in):
        """
            get mask from pure_bg
            sector_idx == -1, no sectors, get the whole ring
            sector_idx in [0, 1] for 2 effective sectors
        """
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
    mask_obj = RounderRingMask(size=65, r_out=RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET)
    def ring_mask(size = 64, r_out = RADIUS, r_in = RADIUS_CUTOFF, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
        assert size == 64
        assert mode == 'full', f'not implemented mode {mode}'
        return mask_obj.get_ring_mask(r_out=r_out, r_in=r_in)