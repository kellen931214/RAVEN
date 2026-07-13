import typing
import torch
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from scipy.stats import norm, truncnorm
import math
from Crypto.Cipher import ChaCha20
from scipy.special import betainc
from .wm_provider import WmProvider
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--wm_len', default=256, type=int)
parser.add_argument('--wm_times', default=9, type=int)
parser.add_argument('--adaptable_wm_times', default=True, type=bool)
parser.add_argument('--wm_rough_ratio', default=0.3, type=float)

class TagProvider(WmProvider):
    def __init__(self,
                 wm_len: int = 256,
                 center_interval_ratio: float = 0.5,
                 wm_times: int = 9,
                 adaptable_wm_times: bool = True,
                 tlt_intervals_num: int = 3,
                #  fpr=1e-6, 
                #  user_number=1000000,
                 shuffle_random_seed: int = 133563,
                 encrypt_random_seed: int = 133563,
                 offset: int = 0,
                 message: typing.Optional[str] = None,
                 key: bytes = None,
                 nonce: bytes = None,
                 **kwargs):
  
        super().__init__(**kwargs)
        self.wm_len = wm_len
        self.wm_times = wm_times
        self.adaptable_wm_times = adaptable_wm_times
        self.offset = offset
        self.latent_len = self.latent_shape[1]* self.latent_shape[2]* self.latent_shape[3]

        self.center_interval_ratio = center_interval_ratio
        self.tlt_intervals_num = tlt_intervals_num  # 3 or 4
        # abs_truncdot used by sampling and reverse-sampling
        self.abs_truncdot = float(np.abs(norm.ppf(0.5 - self.center_interval_ratio / 2)))

        # --- message/key/nonce init (align with gs_provider.py) ---
        # if message is None:
        #     from .messages import MESSAGES
        #     from .keys import KEYS
        #     from .nonces import NONCES
        #     self.messages = MESSAGES
        #     self.keys = KEYS
        #     self.nonces = NONCES
        # else:
        #     assert key is not None and nonce is not None
        #     self.messages = [message for _ in range(self.batch_size)]
        #     self.keys = [key for _ in range(self.batch_size)]
        #     self.nonces = [nonce for _ in range(self.batch_size)]

        # --- shuffle / crypto settings ---
        self.shuffle_random_seed = shuffle_random_seed
        self.encrypt_random_seed = encrypt_random_seed
        
        self.shuffle_seed_generator = torch.Generator(device=self.device)
        self.encrypt_seed_generator = torch.Generator(device=self.device)

        self.key = self._get_random_bytes(32)
        self.nonce = self._get_random_bytes(24)

        # (TPR thresholds optional; enable if needed)
        # self.tp_onebit_count = 0
        # self.tp_bits_count = 0
        # self.tau_onebit = 0.6328125
        # self.tau_bits = 0.70703125
        # self.tau_onebit, self.tau_bits = self.__init_tau(fpr, user_number)

    def get_wm_type(self) -> str:
        return "TAG"

    # ================= Crypto utils =================
    def _get_random_bytes(self, length: int):
        if self.encrypt_random_seed is None:
            return None
        self.encrypt_seed_generator.manual_seed(self.encrypt_random_seed)
        random_ints = torch.randint(0, 256, (length,), dtype=torch.uint8,
                                    generator=self.encrypt_seed_generator,
                                    device=self.encrypt_seed_generator.device)
        return random_ints.cpu().numpy().tobytes()

    def stream_key_encrypt(self, sd):
        if self.encrypt_random_seed is None:
            return sd
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_decrypt(self, reversed_m):
        if self.encrypt_random_seed is None:
            return reversed_m
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        return sd_bit

    # ================= Shuffle utils =================
    def shuffle(self, data: torch.Tensor):  # [data_length]
        if self.shuffle_random_seed is None:
            return data
        data_length = data.size(0)
        self.shuffle_seed_generator.manual_seed(self.shuffle_random_seed)
        shuffle_indices = torch.randperm(data_length, generator=self.shuffle_seed_generator, device=data.device)
        return data[shuffle_indices]

    def inverse_shuffle(self, shuffled_data: torch.Tensor):  # [data_length]
        if self.shuffle_random_seed is None:
            return shuffled_data
        data_length = shuffled_data.size(0)
        self.shuffle_seed_generator.manual_seed(self.shuffle_random_seed)
        shuffle_indices = torch.randperm(data_length, generator=self.shuffle_seed_generator, device=shuffled_data.device)
        inverse_indices = torch.empty_like(shuffle_indices)
        inverse_indices[shuffle_indices] = torch.arange(data_length, device=shuffled_data.device)
        return shuffled_data[inverse_indices]

    # ================= Embedding utilities =================
    def __message_bytes_to_tensor(self, message_bytes: bytes) -> torch.Tensor:
        """Convert bytes to a bit tensor (uint8 {0,1}) on self.device."""
        bits = ''.join(format(b, '08b') for b in message_bytes)
        arr = np.fromiter((1 if c == '1' else 0 for c in bits), dtype=np.uint8, count=len(bits))
        return torch.tensor(arr, dtype=torch.uint8, device=self.device)

    # ================= Core interface =================
    def get_wm_latents(self, **kwargs) -> typing.Dict[str, any]:
        """
        Embed watermark into latent space using trunc-sampling + shuffle.
        tlt: template (flat) tensor of length latent_len with {0,1} values.
        latent_size: (C, H, W) to reshape back.
        Returns dict with list of latents and wm_repeats per-sample.
        """
    
        # from .tag_message import MESSAGES
        # self.wm = torch.tensor(MESSAGES, device=self.device, requires_grad=False)

        self.wm = torch.randint(0, 2, [self.wm_len], requires_grad=False, device=self.device)
        tlt = np.arange(self.latent_len) % 2
        latent_size = self.latent_shape[1:]
        self.wm_latents, self.wm_repeat = self.__embedding_wm_tlt(self.wm, tlt, latent_size=latent_size)
        self.wm_latents = self.wm_latents.to(torch.float32)
       
        return {
            "zT_torch": self.wm_latents,
            # "message_bits_str_list": message_bits_str_list
            # TODO: check after
        }
        
    
    # def __recover_messages_from_latents(self,
    #                                 reversed_latent: torch.Tensor,
    #                                 wm_len: int,
    #                                 pred_tamper_loc_latent: typing.Optional[torch.Tensor] = None,
    #                                 with_tamper_loc: bool = True,
    #                                 key_bytes: typing.Optional[bytes] = None,
    #                                 nonce_bytes: typing.Optional[bytes] = None,
    #                                 sample_index: int = 0) -> typing.Dict[str, any]:
    #     """
    #     Recover watermark using: deshuffle -> reverse sampling -> decrypt -> voting.
    #     Return dict aligned with gs_provider.py format.
    #     """

    def get_accuracies(self, reversed_latents_w: torch.Tensor) -> typing.Dict[str, any]:

        wm_accs = []
        wm_global_accs = []

        reversed_wm_repeat, reversed_tlt = self.__deembedding_wm_tlt(reversed_latents_w)
        reversed_wm = self.__calc_watermark(wm_len=self.wm_len, 
                              wm_repeat=reversed_wm_repeat,
                              pred_tamper_loc_latent=None,
                              with_tamper_loc=False)
        
        wm_acc = self.__eval_watermark(wm=self.wm, reversed_wm=reversed_wm)
        wm_accs.append(wm_acc)

        wm_global_acc = (reversed_wm_repeat == self.wm_repeat).float().mean().item()
        wm_global_accs.append(wm_global_acc)

        return {
            "bit_accuracies": wm_accs,
        }
                                                 

    def denseWMandDenseFixedTLTtruncSampling(self, wm, tlt):  
        """  More effective 
            When self.tlt_intervals_num == 3:
                (wm, tlt) intervals: (0, 1), (0, 0), (1, 0), (1, 1) 
            When self.tlt_intervals_num == 4:
                (wm, tlt) intervals: (0, 0), (0, 1), (1, 0), (1, 1) 
        """
        if self.tlt_intervals_num == 3:
            z = np.zeros(wm.shape[0])
            ppf = [-math.inf, -self.abs_truncdot, 0, self.abs_truncdot, math.inf]  # split 4 sampling range

            # Sample wm and tlt noise using vectorized operations
            indices_wm_0_tlt_1 = (wm == 0) & (tlt == 1)
            indices_wm_0_tlt_0 = (wm == 0) & (tlt == 0)
            indices_wm_1_tlt_0 = (wm == 1) & (tlt == 0)
            indices_wm_1_tlt_1 = (wm == 1) & (tlt == 1)

            z[indices_wm_0_tlt_1] = truncnorm.rvs(ppf[0], ppf[1], size=np.sum(indices_wm_0_tlt_1))
            z[indices_wm_0_tlt_0] = truncnorm.rvs(ppf[1], ppf[2], size=np.sum(indices_wm_0_tlt_0))
            z[indices_wm_1_tlt_0] = truncnorm.rvs(ppf[2], ppf[3], size=np.sum(indices_wm_1_tlt_0))
            z[indices_wm_1_tlt_1] = truncnorm.rvs(ppf[3], ppf[4], size=np.sum(indices_wm_1_tlt_1))

        elif self.tlt_intervals_num == 4:
            z = np.zeros(wm.shape[0])
            self.abs_truncdot = np.abs(norm.ppf(0.25))
            ppf = [-math.inf, -self.abs_truncdot, 0, self.abs_truncdot, math.inf]  # split 4 sampling range
            indices_wm_0_tlt_0 = (wm == 0) & (tlt == 0)
            indices_wm_0_tlt_1 = (wm == 0) & (tlt == 1)
            indices_wm_1_tlt_0 = (wm == 1) & (tlt == 0)
            indices_wm_1_tlt_1 = (wm == 1) & (tlt == 1)

            z[indices_wm_0_tlt_0] = truncnorm.rvs(ppf[0], ppf[1], size=np.sum(indices_wm_0_tlt_0))
            z[indices_wm_0_tlt_1] = truncnorm.rvs(ppf[1], ppf[2], size=np.sum(indices_wm_0_tlt_1))
            z[indices_wm_1_tlt_0] = truncnorm.rvs(ppf[2], ppf[3], size=np.sum(indices_wm_1_tlt_0))
            z[indices_wm_1_tlt_1] = truncnorm.rvs(ppf[3], ppf[4], size=np.sum(indices_wm_1_tlt_1))

        return torch.from_numpy(z).half().to(self.device)

    def reverseTruncSampling(self, sampled_messege):
        # Tamper Localization Template: [0, 0, 0, 0, 1, 1, 1,....]
        if isinstance(sampled_messege, torch.Tensor):
            sampled_messege = sampled_messege.detach().cpu().numpy() 
            
        # reverse wm 
        reversed_wm = (sampled_messege > 0).astype(int)

        # reverse tlt
        if self.tlt_intervals_num == 3:
            reversed_tlt = (np.abs(sampled_messege) > self.abs_truncdot).astype(int)
            
        elif self.tlt_intervals_num == 4:
            # Define the conditions for each interval
            conditions = [
                (sampled_messege < -self.abs_truncdot),  # Condition for the first interval
                (sampled_messege >= -self.abs_truncdot) & (sampled_messege < 0),  # Condition for the second interval
                (sampled_messege >= 0) & (sampled_messege < self.abs_truncdot),  # Condition for the third interval
                (sampled_messege >= self.abs_truncdot)  # Condition for the fourth interval
            ]
            # Define the corresponding values for each condition
            choices = [0, 1, 0, 1]
            # Use np.select() to apply the conditions in batch
            reversed_tlt = np.select(conditions, choices, default=None)  # 'default' is used for cases that don't satisfy any condition
            # Check if any condition was met, if not, raise an error
            if reversed_tlt is None or np.any(reversed_tlt is None):
                raise ValueError("An unexpected value encountered. No condition met.")
        return reversed_wm, reversed_tlt    
    
    def __embedding_wm_tlt(self, wm, tlt, latent_size):
        # calculate latent length
        wm_len = wm.shape[0]
        latent_len = tlt.shape[0]
        # repeat watermark
        wm_times = int(latent_len // wm_len)
        remain_wm_len = latent_len % wm_len
        wm_repeat = torch.concat([wm.repeat(wm_times), wm[:remain_wm_len]], dim=0)
        # encrypt
        wm_repeat_encrypt = self.stream_key_encrypt(wm_repeat.cpu().numpy())
        # sampling
        flat_latent = self.denseWMandDenseFixedTLTtruncSampling(wm_repeat_encrypt, tlt)
        # shuffle
        shuffled_flat_latent = self.shuffle(flat_latent)
        # reshape
        latent_noise = shuffled_flat_latent.reshape(1, *latent_size)
        return latent_noise, wm_repeat
    
    def __deembedding_wm_tlt(self, reversed_latent: torch.Tensor):
        # flaten
        flat_reversed_latent = reversed_latent.view(-1)
        # de-shuffle
        deshuffled_flat_reversed_latent = self.inverse_shuffle(flat_reversed_latent)
        # de-sampling
        wm_repeat_encrypt, reversed_tlt = self.reverseTruncSampling(deshuffled_flat_reversed_latent)
        # decrypt
        wm_repeat = self.stream_key_decrypt(wm_repeat_encrypt)    
        wm_repeat = torch.from_numpy(wm_repeat).to(reversed_latent.device).float()
        return wm_repeat, reversed_tlt
        

    # def __calculate_bit_accuracy(self,
    #                          original_message_hex: any,  # bytes
    #                          extracted_message_bits_str: str) -> float:
    #     """
    #     Get bit accuracy between extracted bits and the original message hex

    #     @param original_message_hex: original message in hex (bytes-like object)
    #     @param extracted_message_bits_str: extracted message in bits (string of '0'/'1')

    #     @return: bit accuracy
    #     """
    #     # Convert the original hex message to binary
    #     original_message_bin = ''.join(format(byte, '08b') for byte in original_message_hex)

    #     # Ensure both binary strings are of the same length
    #     min_length = min(len(original_message_bin), len(extracted_message_bits_str))
    #     original_message_bin = original_message_bin[:min_length]
    #     extracted_message_bits_str = extracted_message_bits_str[:min_length]

    #     # Calculate bit accuracy
    #     matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bits_str) if x == y)
    #     bit_accuracy = matching_bits / min_length if min_length > 0 else 0.0

    #     return bit_accuracy


    def __calc_watermark(self, wm_len, wm_repeat, pred_tamper_loc_latent=None, with_tamper_loc=True):
        # if 'int' not in str(pred_tamper_loc_latent.dtype):
        #     pred_tamper_loc_latent = (pred_tamper_loc_latent - torch.min(pred_tamper_loc_latent)) / (torch.max(pred_tamper_loc_latent) - torch.min(pred_tamper_loc_latent))
        latent_len = wm_repeat.size(0)   
        wm_repeat_times = latent_len // wm_len
        complete_wm_len = wm_len * wm_repeat_times
        remain_wm_len = latent_len - complete_wm_len
        
        if with_tamper_loc == False or pred_tamper_loc_latent is None:
            pred_tamper_loc_latent = torch.zeros_like(wm_repeat)
            
        pred_notamper_loc_latent = 1 - pred_tamper_loc_latent.view(-1)
        pred_notamper_loc_latent = self.inverse_shuffle(pred_notamper_loc_latent)
        wm_repeat_notamper = wm_repeat * pred_notamper_loc_latent

        # de-repeat
        split_notamper_wm = torch.cat(torch.split(wm_repeat_notamper[:complete_wm_len].unsqueeze(0), wm_len, dim=1), dim=0)
        remaining_notamper_wm = torch.cat([wm_repeat_notamper[complete_wm_len:], torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)]).unsqueeze(0)
        split_notamper_wm = torch.cat([split_notamper_wm, remaining_notamper_wm], dim=0)

        split_notamper = torch.cat(torch.split(pred_notamper_loc_latent[:complete_wm_len].unsqueeze(0), wm_len, dim=1), dim=0)
        remaining_notamper = torch.cat([pred_notamper_loc_latent[complete_wm_len:], torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)]).unsqueeze(0)
        split_notamper = torch.cat([split_notamper, remaining_notamper], dim=0)
        
        # watermark vote
        vote = torch.sum(split_notamper_wm, dim=0)
        # print('vote:', vote)
        vote_num = torch.sum(split_notamper, dim=0)
        # print('vote_num:', vote_num)
        reversed_watermark = (vote / vote_num >= 0.5).int()
        # print('reversed_watermark:', reversed_watermark)
            
        return reversed_watermark
    
    # def get_accuracies(self, latents: typing.Union[torch.Tensor, np.ndarray]) -> typing.Dict[str, any]:
    #     """
    #     Get bit accuracy between original and extracted messages

    #     @param latents: latent either tensor with batch dim or numpy with batch dim
    #     @return: dict
    #     """
    #     # get the extracted message
    #     recovered = self.__recover_messages_from_latents(latents)
    #     recovered_messages_bits_str = recovered["messages_bits_str"]
    #     recovered_barcodes_torch = recovered["barcodes_torch"]
    #     recovered_barcodes_PIL = recovered["barcodes_PIL"]

    #     # iterate and calculate bit accuracies
    #     bit_accuracies = []
    #     recovered_message_bits_str_list = []
    #     for i in range(self.offset, self.batch_size + self.offset):
    #         original_message_hex = self.messages[i]
    #         recovered_message_bits_str = recovered_messages_bits_str[i - self.offset]

    #         # get the bit accuracy
    #         bit_accuracy = self.__calculate_bit_accuracy(original_message_hex, recovered_message_bits_str)
    #         bit_accuracies.append(bit_accuracy)
    #         recovered_message_bits_str_list.append(recovered_message_bits_str)

    #     return {
    #         "accuracies": bit_accuracies,
    #         "bit_accuracies": bit_accuracies,
    #         "barcodes_torch": recovered_barcodes_torch,
    #         "barcodes_PIL": recovered_barcodes_PIL,
    #         "barcodes": recovered_barcodes_PIL,
    #         "message_bits_str_list": recovered_message_bits_str_list,
    #     }

    # ================= Optional helper =================
    def __init_tau(self, fpr: float, user_number: int):
        tau_onebit = None
        tau_bits = None
        for i in range(self.wm_len):
            fpr_onebit = betainc(i+1, self.wm_len - i, 0.5)
            fpr_bits = betainc(i+1, self.wm_len - i, 0.5) * user_number
            if fpr_onebit <= fpr and tau_onebit is None:
                tau_onebit = i / self.wm_len
            if fpr_bits <= fpr and tau_bits is None:
                tau_bits = i / self.wm_len
        return tau_onebit, tau_bits
    
    
    def __eval_watermark(self, wm, reversed_wm):
        acc = (reversed_wm == wm).float().mean().item()
        # if acc >= self.tau_onebit:
        #     self.tp_onebit_count = self.tp_onebit_count+1
        # if acc >= self.tau_bits:
        #     self.tp_bits_count = self.tp_bits_count + 1
        # print(f"Copyright Watermark Accuracy: {acc: .4f}")
        return acc