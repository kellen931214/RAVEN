import argparse
import os
import pickle
import random
import string
import typing

import numpy as np
import torch
from torch.distributions import Chi2

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--sph_from_file",
    type=str,
    default="qLWEAFAHapBLw4B6Tl5RUb1SlxXKrpd2jbiim3d1W75GlmNe1R9ZnsrZq7YKMb2K_1_31_16384_512",
)
parser.add_argument("--sph_keys_path", type=str, default="keys/sph")
parser.add_argument("--sph_N", type=int, default=31)
parser.add_argument("--sph_lm", type=int, default=512)
parser.add_argument("--sph_lr", type=int, default=512)
parser.add_argument("--sph_t", type=int, default=1)
parser.add_argument("--sph_K_len", type=int, default=128)
parser.add_argument("--sph_partial_channels", type=int, default=4)
parser.add_argument("--sph_channel_start", type=int, default=0)

class SphericalCodes:
    """
    Spherical watermark mapping from submission14944_codes, adapted as an
    in-process mapping module for geodistort-wm providers.
    """

    def __init__(self, from_file, keys_path, N, lm, lr, t, K_len, batch_size, latent_shape, device, P=None):
        self.device = device
        self.ablate_signet = False
        self.ablate_rotation = False

        if from_file:
            self.from_file = from_file
            self.keys_path = keys_path
            file_path = os.path.join(keys_path, f"{from_file}.pkl")
            with open(file_path, "rb") as f:
                self.P = P.to(device) if P is not None else None
                self.N, self.lm, self.lr, self.t, self.latent_shape, R_positions, K_seed, self.K_len = pickle.load(f)
                self.total_len = self.N * self.lm + self.lr
                self.T, self.T_inv, self.K, self.K_inv = self.construct_signet_from_file(R_positions, K_seed)
                self.T = self.T.to(device)
                self.T_inv = self.T_inv.to(device)
                self.K = self.K.to(device)
                self.K_inv = self.K_inv.to(device)
        else:
            self.N = N
            self.lm = lm
            self.lr = lr
            self.t = t
            self.K_len = K_len
            self.P = P.to(device) if P is not None else None
            self.total_len = N * lm + lr
            self.latent_shape = latent_shape

            K_seed = random.getrandbits(64)
            T, T_inv, K, K_inv, R_positions = self.construct_signet(K_seed)
            self.T = T
            self.T_inv = T_inv
            self.K = K
            self.K_inv = K_inv

            os.makedirs(keys_path, exist_ok=True)
            from_file = f"{self._random_file_name()}_{t}_{N}_{self.total_len}_{self.lr}"
            with open(os.path.join(keys_path, f"{from_file}.pkl"), "wb") as f:
                pickle.dump((N, lm, lr, t, self.latent_shape, R_positions, K_seed, self.K_len), f)
            self.from_file = from_file
            self.keys_path = keys_path
            print(f"Constructed watermark signet located at {keys_path}/{self.from_file}.pkl")

        self.batch_size = batch_size
        self.chisquare = Chi2(df=self.K_len)

    def _random_file_name(self):
        letters = string.ascii_letters + string.digits
        return "".join(random.choice(letters) for _ in range(64))

    def _construct_R(self, N, lm, lr, t):
        if lr < N * t:
            raise ValueError("lr must be greater than N * t")

        R = np.zeros((N, lm, lr), dtype=int)
        positions = []
        for round_index in range(lm):
            perm = np.random.permutation(lr)
            selected = perm[: N * t]
            for i in range(N):
                group = np.sort(selected[i * t : (i + 1) * t])
                R[i, round_index, list(group)] = 1
                positions.append((i, round_index, list(group)))
        return torch.from_numpy(R).reshape((-1, lr)), positions

    def _construct_R_from_positions(self, N, lm, lr, R_positions):
        R = np.zeros((N, lm, lr), dtype=int)
        for item in R_positions:
            i, round_index, group = item
            R[i, round_index, group] = 1
        return torch.from_numpy(R).reshape((-1, lr))

    def _construct_T(self, N, lm, lr, t, P, device):
        I_s = torch.eye(N * lm, dtype=torch.float32, device=device)
        R, R_positions = self._construct_R(N, lm, lr, t)
        R = R.to(I_s)
        null_matrix = torch.zeros((lr, N * lm), dtype=torch.float32, device=device)
        P = torch.eye(lr, dtype=torch.float32, device=device) if P is None else P.to(device)
        P_inv = P.transpose(0, 1)
        T_top = torch.cat((I_s, R), dim=1)
        T_bottom = torch.cat((null_matrix, P), dim=1)
        T = torch.cat((T_top, T_bottom), dim=0)

        T_inv_top = torch.cat((I_s, (R @ P_inv) % 2), dim=1)
        T_inv_bottom = torch.cat((null_matrix, P_inv), dim=1)
        T_inv = torch.cat((T_inv_top, T_inv_bottom), dim=0)
        return T, T_inv, R_positions

    def _construct_T_from_positions(self, N, lm, lr, R_positions, P, device):
        I_s = torch.eye(N * lm, dtype=torch.float32, device=device)
        R = self._construct_R_from_positions(N, lm, lr, R_positions).to(I_s)
        null_matrix = torch.zeros((lr, N * lm), dtype=torch.float32, device=device)
        P = torch.eye(lr, dtype=torch.float32, device=device) if P is None else P.to(device)
        P_inv = P.transpose(0, 1)
        T_top = torch.cat((I_s, R), dim=1)
        T_bottom = torch.cat((null_matrix, P), dim=1)
        T = torch.cat((T_top, T_bottom), dim=0)

        T_inv_top = torch.cat((I_s, (R @ P_inv) % 2), dim=1)
        T_inv_bottom = torch.cat((null_matrix, P_inv), dim=1)
        T_inv = torch.cat((T_inv_top, T_inv_bottom), dim=0)
        return T, T_inv

    def _construct_K(self, K_len, device, K_seed):
        generator = torch.Generator()
        generator.manual_seed(K_seed)
        K_init = torch.randn((K_len, K_len), generator=generator).to(device)
        K, _ = torch.linalg.qr(K_init)
        return K, K.transpose(0, 1)

    def construct_signet(self, K_seed):
        T, T_inv, R_positions = self._construct_T(self.N, self.lm, self.lr, self.t, self.P, self.device)
        K, K_inv = self._construct_K(self.K_len, self.device, K_seed)
        return T, T_inv, K, K_inv, R_positions

    def construct_signet_from_file(self, R_positions, K_seed):
        T, T_inv = self._construct_T_from_positions(self.N, self.lm, self.lr, R_positions, self.P, self.device)
        K, K_inv = self._construct_K(self.K_len, self.device, K_seed)
        return T, T_inv, K, K_inv

    def embed_watermark(self, message):
        if len(message.shape) == 2:
            message = message.unsqueeze(-1)
        repeated_message = message.repeat(1, self.N, 1).to(self.device)
        batch_size = message.shape[0]
        random_stream = torch.randint(0, 2, (batch_size, self.lr, 1)).to(self.T)
        input_message = torch.cat((repeated_message, random_stream), dim=1)

        out_sign = (self.T @ input_message) % 2 if not self.ablate_signet else input_message
        out_sign = out_sign * 2 - 1

        if not self.ablate_rotation:
            out_sign = out_sign.reshape((batch_size, self.K_len, -1))
            out_noise = self.K @ out_sign
            chi_rand = self.chisquare.sample((batch_size * out_noise.shape[-1],)).to(out_noise)
            chi_rand = chi_rand.reshape((batch_size, 1, -1))
            out_noise = out_noise * torch.sqrt(chi_rand) / (self.K_len**0.5)
        else:
            out_noise = out_sign * torch.abs(torch.randn_like(out_sign).to(out_sign))
        return out_noise.reshape((batch_size, *self.latent_shape))

    def extract_watermark(self, pred_noise):
        batch_size = pred_noise.shape[0]
        if not self.ablate_rotation:
            pred_noise = pred_noise.reshape((batch_size, self.K_len, -1))
            pred_sign = self.K_inv @ pred_noise
        else:
            pred_sign = torch.sign(pred_noise)
        pred_sign = pred_sign.reshape((batch_size, -1, 1))
        pred_sign = (torch.sign(pred_sign) + 1) / 2
        pred_sign = torch.round(pred_sign)

        pred_input_message = (self.T_inv @ pred_sign) % 2 if not self.ablate_signet else pred_sign
        pred_repeated_message = pred_input_message[:, : (self.lm * self.N), :].reshape((-1, self.N, self.lm))
        return torch.round(torch.mean(pred_repeated_message, dim=1))


class SPHProvider(WmProvider):
    def __init__(
        self,
        sph_from_file: str = "qLWEAFAHapBLw4B6Tl5RUb1SlxXKrpd2jbiim3d1W75GlmNe1R9ZnsrZq7YKMb2K_1_31_16384_512",
        sph_keys_path: str = "keys/sph",
        sph_N: int = 31,
        sph_lm: int = 512,
        sph_lr: int = 512,
        sph_t: int = 1,
        sph_K_len: int = 128,
        sph_partial_channels: int = 4,
        sph_channel_start: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if sph_partial_channels != 4:
            raise ValueError(
                f"SPH partial-channel mode currently expects --sph_partial_channels=4, got {sph_partial_channels}."
            )
        if self.latent_resolution != 64:
            raise ValueError(
                f"SPH currently expects latent resolution 64, got {self.latent_resolution}. "
                "Use --resolution 512 for the current SPH key."
            )
        if self.num_channels not in (4, 16):
            raise ValueError(
                f"SPH supports 4-channel latents or 16-channel partial mode, got {tuple(self.latent_shape)}."
            )
        if sph_channel_start < 0 or sph_channel_start + sph_partial_channels > self.num_channels:
            raise ValueError(
                f"Invalid SPH channel slice [{sph_channel_start}:{sph_channel_start + sph_partial_channels}] "
                f"for latent shape {tuple(self.latent_shape)}."
            )

        self.sph_partial_channels = sph_partial_channels
        self.sph_channel_start = sph_channel_start
        self.sph_channel_end = sph_channel_start + sph_partial_channels
        self.uses_partial_channels = self.num_channels != sph_partial_channels
        spherical_latent_shape = (sph_partial_channels, self.latent_resolution, self.latent_resolution)

        self.mapping = SphericalCodes(
            from_file=sph_from_file,
            keys_path=sph_keys_path,
            N=sph_N,
            lm=sph_lm,
            lr=sph_lr,
            t=sph_t,
            K_len=sph_K_len,
            batch_size=self.batch_size,
            latent_shape=spherical_latent_shape,
            device=self.device,
            P=None,
        )
        self.messages_torch = None
        self.message_bits_str_list = None

    def get_wm_type(self) -> str:
        return "SPH"

    @staticmethod
    def _bits_to_str(bits: torch.Tensor) -> str:
        return "".join(str(int(bit)) for bit in bits.detach().flatten().cpu().tolist())

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        messages = torch.randint(0, 2, (self.batch_size, self.mapping.lm), device=self.device)
        sph_latents = self.mapping.embed_watermark(messages).to(device=self.device, dtype=self.dtype)
        if self.uses_partial_channels:
            latents = torch.randn(self.latent_shape, device=self.device, dtype=self.dtype)
            latents[:, self.sph_channel_start : self.sph_channel_end, :, :] = sph_latents
        else:
            latents = sph_latents
        self.messages_torch = messages.detach().clone()
        self.message_bits_str_list = [self._bits_to_str(message) for message in self.messages_torch]

        return {
            "zT_torch": latents,
            "zT_PIL": torch_to_PIL(latents),
            "zT": torch_to_PIL(latents),
            "message_bits_str_list": self.message_bits_str_list,
        }

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.ndarray]) -> typing.Dict[str, typing.Any]:
        if self.messages_torch is None:
            raise RuntimeError("SPH messages are not initialized. Call get_wm_latents() before get_accuracies().")

        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        latents = latents.to(device=self.device, dtype=torch.float32)
        sph_latents = latents[:, self.sph_channel_start : self.sph_channel_end, :, :]
        recovered = self.mapping.extract_watermark(sph_latents)
        expected = self.messages_torch.to(recovered)
        bit_accuracies = (recovered == expected).float().mean(dim=1).detach().cpu().tolist()
        recovered_message_bits_str_list = [self._bits_to_str(message) for message in recovered]

        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "message_bits_str_list": recovered_message_bits_str_list,
        }
