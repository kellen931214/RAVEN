import typing
import numpy as np
import torch
import torchvision

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL
from utils import utils 

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--hsqr_seed', default=999999, type=int)
parser.add_argument('--box_size', default=2, type=int)
parser.add_argument('--qr_version', default=1, type=int)
parser.add_argument('--delta', default=0, type=int)

# [HSQR] hyperparameters
RADIUS = 14
RADIUS_CUTOFF = 3
w_channel = 3
TREE_WATERMARK_CHANNEL = [w_channel]
HSQR_WATERMARK_CHANNEL = [w_channel]
assert TREE_WATERMARK_CHANNEL == HSQR_WATERMARK_CHANNEL, "HSQR and Tree-Ring have the same channel in the paper."

class HSQRProvider(WmProvider):
    def __init__(self,
                 hsqr_seed: int = None,
                 box_size: int = 2,
                 qr_version: int = 1,
                 delta: int = 0,
                 latent_channel: int = 4,
                 start: int = 10,
                 end: int = 54, # 64-10 = hw_latent-start
                 hw_latent : int = 64,
                 fix_gt: int = 1,
                 wm_capacity: int = 2**(RADIUS-RADIUS_CUTOFF),
                 **kwargs):
        
        super().__init__(**kwargs)

        if hsqr_seed is not None:
            utils.set_random_seed(hsqr_seed)

        self.shape = (1, latent_channel, hw_latent, hw_latent)
        self.hsqr_seed = hsqr_seed
        
        assert wm_capacity == 2048
        self.wm_capacity = wm_capacity
        self.hsqr_seed_list = [*range(hsqr_seed, hsqr_seed + wm_capacity)] # 2048 seed numbers
        self.sample_index = fix_gt
        
        self.start = start
        self.end = end
        self.qr_generator = QRCodeGenerator(box_size=box_size, border=0, qr_version=qr_version)
        self.delta = delta
        self.center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))

        self.gt_patch = self.__get_watermarking_pattern()

    def get_wm_type(self) -> str:
        return "HSQR"
    
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

        latents_w = self.__inject_watermark(inverted_latent = latents_clean, qr_tensor=self.gt_patch, center=True)

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
                          qr_gt_bool, channel=HSQR_WATERMARK_CHANNEL,
                          p=1, center=False):
        """
        qr_gt_bool : (c_wm,42,42) boolean
        target_fft : (1,4,64,64) complex64
        """
        Fourier_wm_zT_fft = torch.zeros_like(reversed_latents_w, dtype=torch.complex64)
        Fourier_wm_zT_fft[self.center_slice] = HSQRProvider.fft(reversed_latents_w[self.center_slice])

        # if Fourier_wm_zT_fft.shape != self.gt_patch.shape:
            # raise ValueError(f'Shape mismatch during eval: {Fourier_wm_zT_fft.shape} vs {self.gt_patch.shape}')

        target_fft = Fourier_wm_zT_fft

        w_metric = None
        center_row = target_fft.shape[-2] // 2 # 32
        qr_pix_len = qr_gt_bool.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_gt_f32 = torch.where(qr_gt_bool, torch.tensor(45.0), torch.tensor(-45.0)).to(torch.float32) # (c_wm,42,42) boolean -> float32
        qr_left = qr_gt_f32[0, :, :qr_pix_half]   # (42,21) float32
        qr_right = qr_gt_f32[0, :, qr_pix_half:]  # (42,21) float32
        qr_complex = torch.complex(qr_left, qr_right).to(target_fft.device) # (42,21) complex64
        if center:
            row_start = 10 + 1 # 11
            row_end = row_start + qr_pix_len # 53 = 11+42
            col_start = center_row + 1 # 33 = 32+1
            col_end = col_start + qr_pix_half # 54 = 33+21
        else:
            row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
            row_end = center_row + qr_pix_half
            col_start = center_row + 1 # 33
            col_end = col_start + qr_pix_half # 33+21
            # [TBD] the odd case will be updated
        qr_slice = (0, channel, slice(row_start, row_end), slice(col_start, col_end)) # (42,21)

        diff = torch.abs(qr_complex - target_fft[qr_slice]) # (42,21)
        w_metric = torch.norm(diff, p=p).item() / diff.numel() if p != 1 else torch.mean(diff).item()

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
        
        results = self.__get_l1_distance(reversed_latents_w=latents, qr_gt_bool=self.gt_patch, 
                                         channel=TREE_WATERMARK_CHANNEL,p=1, center=is_center)
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
    
    def __get_watermarking_pattern(self) -> torch.tensor:
        
        Fourier_watermark_pattern_list = [self.__make_hsqr_pattern(idx=this_hsqr_seed) for this_hsqr_seed in self.hsqr_seed_list]
        assert len(Fourier_watermark_pattern_list) == self.wm_capacity

        pattern_gt = Fourier_watermark_pattern_list
        if len(pattern_gt[0].shape) == 3:
            pattern_gt = torch.stack(pattern_gt, dim=0) # (N,c_wm,42,42) for HSQR
        
        identify_gt_indices = np.random.choice(self.wm_capacity, size=8192).tolist()
        key_index = identify_gt_indices[self.sample_index]
        pattern_gt = Fourier_watermark_pattern_list[key_index]

        return pattern_gt

    def __make_hsqr_pattern(self, idx: int):
        
        # assert shape[-1] == shape[-2] # 64==64
        # gt_init = torch.randn(self.latent_shape).to(self.device, torch.float64) # (1,4,64,64)

        data = f"HSQR{idx % 10000}"
        qr_tensor = self.qr_generator.make_qr_tensor(data=data) # (42,42) boolean tensor
        qr_tensor = qr_tensor.repeat(len(HSQR_WATERMARK_CHANNEL), 1, 1) # (c_wm,42,42) boolean tensor
        return qr_tensor # (c_wm,42,42) boolean tensor
    
    def __inject_watermark(self, inverted_latent, qr_tensor, center=False):

        qr_tensor = qr_tensor.unsqueeze(0) # (1,c_wm,42,42)
        assert len(qr_tensor.shape) == 4 # (N,c_wm,42,42)
        inverted_latent = inverted_latent.to(self.device)
        qr_tensor = qr_tensor.to(self.device)
        qr_pix_len = qr_tensor.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_left = qr_tensor[:, :, :, :qr_pix_half]    # (N,c_wm,42,21) boolean
        qr_right = qr_tensor[:, :, :, qr_pix_half:]   # (N,c_wm,42,21) boolean
        if center:
            # rfft
            center_latent_rfft = HSQRProvider.rfft(inverted_latent[self.center_slice]) # (N,4,44,44) -> # (N,4,44,23) complex64
            center_real_batch = center_latent_rfft.real # (N,4,44,23) f32
            center_imag_batch = center_latent_rfft.imag # (N,4,44,23) f32
            real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
            imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
            #center=True  [:,[3], 1:43,1:22] (N,1,42,21)
            center_real_batch[real_slice] = HSQRProvider.qr_abs(qr_left, center_real_batch[real_slice], delta=self.delta) # (N,c_wm,42,21)
            center_imag_batch[imag_slice] = HSQRProvider.qr_abs(qr_right, center_imag_batch[imag_slice], delta=self.delta) # (N,c_wm,42,21)
            center_latent_ifft = HSQRProvider.irfft(torch.complex(center_real_batch, center_imag_batch)) # (N,4,44,44) f32
            inverted_latent = inverted_latent.clone()
            inverted_latent[self.center_slice] = center_latent_ifft
            return inverted_latent # (N,4,64,64)
        else:
            # Coordinates for HSQR injection
            center_row = inverted_latent.shape[-2] // 2 # 32
            row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
            row_end = center_row + qr_pix_half
            col_end_left = 1 + qr_pix_half
            col_end_right = qr_pix_half if qr_pix_len % 2 else col_end_left # if odd length QR, shortend by 1 pix
            # rfft
            latent_rfft = HSQRProvider.rfft(inverted_latent) # (N,4,64,64) -> (N,4,64,33) complex64
            real_batch = latent_rfft.real # (N,4,64,33) f32
            imag_batch = latent_rfft.imag # (N,4,64,33) f32
            # Inject HSQR
            # [row_start 11 = 32-21 : row_end 53 = 32+21] / [col_start 1 : col_end_left 22 = 1+21 = col_end_right 22]
            real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_left))
            imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_right))
            #center=False [:,[3],11:53,1:22] (N,1,42,21)
            real_batch[real_slice] = HSQRProvider.qr_abs(qr_left, real_batch[real_slice], delta=self.delta) # (N,c_wm,42,21)
            imag_batch[imag_slice] = HSQRProvider.qr_abs(qr_right, imag_batch[imag_slice], delta=self.delta) # (N,c_wm,42,21)
            return HSQRProvider.irfft(torch.complex(real_batch, imag_batch)) # (N,4,64,64)

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
    def qr_abs(boolean_tensor, input_tensor, delta=0): # boolean → qr_abs tensor
        return torch.where(boolean_tensor, input_tensor.abs() + delta, -input_tensor.abs() - delta)


# qrcoder mask
import qrcode
class QRCodeGenerator:
    def __init__(self, box_size=2, border=1, qr_version=1):
        self.qr = qrcode.QRCode(version=qr_version, box_size=box_size, border=border,
            error_correction = qrcode.constants.ERROR_CORRECT_H)
    
    def make_qr_tensor(self, data, filename='qrcode.png', save_img=False):
        self.qr.add_data(data)
        self.qr.make(fit=True)
        img = self.qr.make_image(fill_color="black", back_color="white")
        if save_img:
            img.save(filename)
        self.clear()
        img_array = np.array(img)
        tensor = torch.from_numpy(img_array)
        return tensor.clone().detach() # boolean (h,w)
    
    def clear(self):
        self.qr.clear()