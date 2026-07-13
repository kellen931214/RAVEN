import argparse
import typing
from functools import reduce

import torch
from scipy.stats import norm

from .wm_provider import WmProvider


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--t2s_channels", type=int, default=0, choices=[0, 4, 16], help="0 auto-detects from latent shape")
parser.add_argument("--t2s_key_channel_idx", type=int, default=0)
parser.add_argument("--t2s_key_length", type=int, default=16)
parser.add_argument("--t2s_msg_length", type=int, default=256)
parser.add_argument("--t2s_tau", type=float, default=0.674)
parser.add_argument("--t2s_fix_key", action="store_true", default=False)


class T2SMark:
    def __init__(self, m: int, tau: float, latent_shape: typing.Tuple[int, int, int], device: torch.device):
        self.latent_shape = tuple(latent_shape)
        self.n = reduce(lambda x, y: x * y, self.latent_shape, 1)
        self.m = m
        self.r = int(2 * norm.cdf(-tau) * self.n / m)
        self.k = self.m * self.r
        self.noise_size = self.n - self.k
        self.device = device
        self.prng = torch.Generator(device=device)

        if self.r <= 0 or self.k > self.n:
            raise ValueError(
                f"Invalid T2SMark parameters: m={m}, tau={tau}, latent_shape={latent_shape}, r={self.r}, k={self.k}"
            )

    def binlist2int(self, binlist):
        res = reduce(lambda x, y: x * 2 + y, binlist)
        if isinstance(binlist, torch.Tensor):
            return int(res.item())
        return int(res)

    def encode(self, bits: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        z = torch.randn(self.latent_shape, device=self.device).flatten()
        self.prng.manual_seed(self.binlist2int(key))

        v_value = torch.randint(0, 2, (self.k,), generator=self.prng, device=self.device).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng, device=self.device)[: self.k]

        repeated_bits = (1 - 2 * bits).repeat(self.r).to(self.device).float()
        codeword = repeated_bits * v_value

        watermark_mask = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        watermark_mask[v_support] = True

        tail = torch.topk(z.abs(), k=self.k, dim=0, largest=True, sorted=False)
        central = torch.topk(z.abs(), k=self.noise_size, dim=0, largest=False, sorted=False)

        z_w = torch.zeros(self.n, device=self.device)
        z_w[watermark_mask] = tail.values * codeword
        z_w[~watermark_mask] = central.values * (
            torch.randint(0, 2, (self.noise_size,), device=self.device).float() * 2 - 1
        )
        return z_w.reshape(self.latent_shape)

    def decode(self, reversed_noise: torch.Tensor, key: torch.Tensor, detection: bool = False):
        self.prng.manual_seed(self.binlist2int(key))
        v_value = torch.randint(0, 2, (self.k,), generator=self.prng, device=self.device).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=self.prng, device=self.device)[: self.k]

        watermark_mask = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        watermark_mask[v_support] = True

        watermarked_vec = reversed_noise.flatten().to(self.device)[watermark_mask] * v_value
        votes = watermarked_vec.reshape(self.r, self.m).sum(dim=0)
        bits = (votes < 0).int()
        if detection:
            return bits, torch.norm(votes.flatten(), p=1).item()
        return bits


class T2SProvider(WmProvider):
    def __init__(
        self,
        t2s_channels: int = 0,
        t2s_key_channel_idx: int = 0,
        t2s_key_length: int = 16,
        t2s_msg_length: int = 256,
        t2s_tau: float = 0.674,
        t2s_fix_key: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.t2s_channels = t2s_channels or self.num_channels
        if self.t2s_channels not in (4, 16):
            raise ValueError(f"T2S supports only 4-channel or 16-channel latents, got {self.t2s_channels}")
        if self.t2s_channels != self.num_channels:
            raise ValueError(
                f"--t2s_channels={self.t2s_channels} does not match latent channels={self.num_channels}"
            )
        if self.batch_size != 1:
            raise ValueError("T2SProvider currently supports batch_size=1")

        self.key_channel_idx = t2s_key_channel_idx
        self.key_length = t2s_key_length
        self.msg_length = t2s_msg_length
        self.tau = t2s_tau
        self.fix_key = t2s_fix_key
        self.seed = seed

        _, _, height, width = self.latent_shape
        if self.t2s_channels == 4:
            if not 0 <= self.key_channel_idx < 4:
                raise ValueError("--t2s_key_channel_idx must be in [0, 3] for 4-channel T2S")
            self.key_channels = [self.key_channel_idx]
            self.msg_channels = [i for i in range(4) if i != self.key_channel_idx]
        else:
            self.key_channels = [0, 1, 2, 3]
            self.msg_channels = [i for i in range(16) if i not in self.key_channels]

        self.t2s_key = T2SMark(
            m=self.key_length,
            tau=self.tau,
            latent_shape=(len(self.key_channels), height, width),
            device=self.device,
        )
        self.t2s_msg = T2SMark(
            m=self.msg_length,
            tau=self.tau,
            latent_shape=(len(self.msg_channels), height, width),
            device=self.device,
        )

        self.fixed_master_key = None
        self.fixed_msg = None
        if self.fix_key:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.seed)
            self.fixed_master_key = torch.randint(0, 2, (self.key_length,), generator=gen, device=self.device)
            self.fixed_msg = torch.randint(0, 2, (self.msg_length,), generator=gen, device=self.device)

    def get_wm_type(self) -> str:
        return "T2S"

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        if self.fix_key:
            self.master_key = self.fixed_master_key.clone()
            self.msg = self.fixed_msg.clone()
        else:
            self.master_key = torch.randint(0, 2, (self.key_length,), device=self.device)
            self.msg = torch.randint(0, 2, (self.msg_length,), device=self.device)

        self.session_key = torch.randint(0, 2, (self.key_length,), device=self.device)
        self.fake_key = 1 - self.master_key

        z_key = self.t2s_key.encode(self.session_key, self.master_key)
        z_msg = self.t2s_msg.encode(self.msg, self.session_key)

        latents = torch.zeros(self.latent_shape, device=self.device, dtype=self.dtype)
        latents[:, self.key_channels, :, :] = z_key.to(dtype=self.dtype).unsqueeze(0)
        latents[:, self.msg_channels, :, :] = z_msg.to(dtype=self.dtype).unsqueeze(0)
        self.wm_latents = latents

        return {
            "zT_torch": self.wm_latents,
            "message_bits_str_list": ["".join(str(int(bit.item())) for bit in self.msg)],
        }

    def get_accuracies(self, reversed_latents_w: torch.Tensor) -> typing.Dict[str, typing.Any]:
        if not hasattr(self, "master_key"):
            raise RuntimeError("get_wm_latents() must be called before get_accuracies() for T2SProvider")

        reversed_latents_w = reversed_latents_w.to(self.device)
        reversed_key_channel = reversed_latents_w[0, self.key_channels, :, :]
        reversed_msg_channel = reversed_latents_w[0, self.msg_channels, :, :]

        _, norm1_no_w = self.t2s_key.decode(reversed_key_channel, self.fake_key, detection=True)
        reversed_key, norm1_w = self.t2s_key.decode(reversed_key_channel, self.master_key, detection=True)
        reversed_msg = self.t2s_msg.decode(reversed_msg_channel, reversed_key)

        acc_key = (reversed_key == self.session_key).float().mean().item()
        acc_msg = (reversed_msg == self.msg).float().mean().item()
        detection_success = norm1_w > norm1_no_w
        recovered_msg_bits = "".join(str(int(bit.item())) for bit in reversed_msg)

        return {
            "accuracies": [acc_msg],
            "bit_accuracies": [acc_msg],
            "message_bits_str_list": [recovered_msg_bits],
            "value": norm1_w,
            "detection_success": detection_success,
            "log_message": (
                f"T2S key_acc: {acc_key:.5f}; msg_acc: {acc_msg:.5f}; "
                f"norm1_w: {norm1_w:.5f}; norm1_no_w: {norm1_no_w:.5f}"
            ),
            "t2s_key_accuracy": acc_key,
            "t2s_norm1_w": norm1_w,
            "t2s_norm1_no_w": norm1_no_w,
        }
