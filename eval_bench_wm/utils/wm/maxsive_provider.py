import argparse
import math
import typing

import torch
import torch.nn.functional as F
import torchvision

from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--maxsive_channel_copy", type=int, default=1)
parser.add_argument("--maxsive_hw_copy", type=int, default=2)
parser.add_argument("--maxsive_distant_func", type=str, default="cos", choices=["corr", "cos", "mse", "l1"])
parser.add_argument("--maxsive_threshold_path", type=str, default="keys/maxsive/MaXsive-cos.pt")
parser.add_argument("--maxsive_fpr", type=float, default=0.001)

# Shared single implementation (previously an identical local copy).
from .ddim_inversion import backward_ddim  # noqa: E402  (kept at original position)


class MaxsiveShuffler:
    def __init__(self, channel_copy: int, hw_copy: int, device: torch.device):
        self.channel_copy = channel_copy
        self.hw_copy = hw_copy
        self.device = device

    def shuffle_key_gen(self):
        key_len = int(64 / self.hw_copy) * int(64 / self.hw_copy) * int(4 / self.channel_copy)
        return [
            torch.randperm(key_len, device=self.device)
            for _ in range(self.channel_copy * self.hw_copy * self.hw_copy)
        ]

    def shuffle(self, z: torch.Tensor, shuffle_keys: typing.List[torch.Tensor]) -> torch.Tensor:
        all_z = []
        z_shape = z.shape
        z_flat = z.reshape(-1)

        for i in range(self.channel_copy):
            channel_blocks = []
            for j in range(self.hw_copy):
                row_blocks = []
                for k in range(self.hw_copy):
                    key_index = i * self.hw_copy * self.hw_copy + j * self.hw_copy + k
                    row_blocks.append(z_flat[shuffle_keys[key_index]].reshape(z_shape))
                channel_blocks.append(torch.concat(row_blocks, axis=3))
            all_z.append(torch.concat(channel_blocks, axis=2))

        return torch.concat(all_z, axis=1)

    def unshuffle(self, shuffled_z: torch.Tensor, shuffle_keys: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        ch_stride = 4 // self.channel_copy
        hw_stride = 64 // self.hw_copy
        ch_list = [ch_stride] * self.channel_copy
        hw_list = [hw_stride] * self.hw_copy

        split_dim1 = torch.cat(torch.split(shuffled_z, tuple(hw_list), dim=3), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(ch_list), dim=1), dim=0)
        unshuffled_z = []

        key_len = int(64 / self.hw_copy) * int(64 / self.hw_copy) * int(4 / self.channel_copy)
        for i in range(self.channel_copy):
            for j in range(self.hw_copy):
                for k in range(self.hw_copy):
                    key_index = i * self.hw_copy * self.hw_copy + j * self.hw_copy + k
                    unshuffle_order = torch.zeros(key_len, dtype=torch.long, device=self.device)
                    unshuffle_order[shuffle_keys[key_index].to(self.device)] = torch.arange(
                        key_len, device=self.device
                    )
                    temp_shuffled_z = split_dim3[key_index].clone().reshape(-1)
                    unshuffled_z.append(
                        temp_shuffled_z[unshuffle_order].reshape(
                            1,
                            int(4 / self.channel_copy),
                            int(64 / self.hw_copy),
                            int(64 / self.hw_copy),
                        )
                    )
        return unshuffled_z


class HoughTransform:
    def __init__(self, x: int, y: int, dim: int, interval: int):
        self.dim = dim
        self.x = x
        self.y = y
        self.interval = interval
        self.init_map(dim, interval)

    def init_map(self, dim: int, interval: int):
        self.degree_list = [interval * x for x in range(int(180 / interval))]
        self.mask = torch.zeros(len(self.degree_list), dim, dim)
        for i, degree in enumerate(self.degree_list):
            for r in range(int(dim / 2 * (2) ** 0.5)):
                x_r = int(r * math.cos(math.radians(degree)))
                y_r = int(r * math.sin(math.radians(degree)))
                if abs(x_r) > int(dim / 2) - 1 or abs(y_r) > int(dim / 2) - 1:
                    break
                self.mask[i, self.x + x_r, self.y + y_r] = 1
                self.mask[i, self.x - x_r, self.y - y_r] = 1
        self.w = self.mask.sum(dim=(1, 2))

    def detect(self, x: torch.Tensor, include_angle: int, pre_angle: int):
        mask = self.mask.to(device=x.device, dtype=x.dtype)
        w = self.w.to(device=x.device, dtype=x.dtype)
        score = (x[None, :] * mask).sum(dim=(1, 2)) / w
        x_score = []
        x_degree = []
        gap = int(include_angle / self.interval)
        for i in range(score.shape[0]):
            x_degree.append((self.degree_list[i], self.degree_list[(i + gap) % score.shape[0]]))
            x_score.append(score[i] + score[(i + gap) % score.shape[0]])

        x_score = torch.stack(x_score)
        best_idx = torch.topk(x_score, 1)[1].item()
        rotated_degree = self.degree_list[best_idx] - pre_angle
        if_scale = self.detect_scale(x, include_angle, rotated_degree)
        return {
            "rotated_degree": rotated_degree,
            "if_scale": if_scale,
            "x_degree": x_degree[best_idx],
            "value": torch.topk(x_score, 1)[0].item(),
        }

    def detect_scale(self, ifft_zt: torch.Tensor, include_angle: int, predict_angle: int):
        all_hist = []
        for boundwide in [-2, -1, 0, 1, 2]:
            degree_list = [predict_angle - include_angle + boundwide, predict_angle + boundwide]
            mask = torch.zeros(int(64 / 2 * (2) ** 0.5), 64, 64, device=ifft_zt.device, dtype=ifft_zt.dtype)
            for r in range(int(64 / 2 * (2) ** 0.5)):
                for degree in degree_list:
                    x_r = int(r * math.cos(math.radians(degree)))
                    y_r = int(r * math.sin(math.radians(degree)))
                    if abs(x_r) > int(64 / 2) - 1 or abs(y_r) > int(64 / 2) - 1:
                        break
                    mask[r, 32 + x_r, 32 + y_r] = 1
                    mask[r, 32 - x_r, 32 - y_r] = 1
            test = abs(ifft_zt) * mask
            r_strength = test.reshape(45, -1).sum(axis=1) / mask.reshape(45, -1).sum(axis=1)
            if r_strength[16] > 50 and r_strength[16] > r_strength[17] and r_strength[16] > r_strength[18]:
                all_hist.append(False)
            all_hist.append(True)
        return False not in all_hist


class MaxsiveProvider(WmProvider):
    def __init__(
        self,
        maxsive_channel_copy: int = 1,
        maxsive_hw_copy: int = 2,
        maxsive_template_c: int = 3,
        maxsive_distant_func: str = "cos",
        maxsive_threshold_path: str = "keys/maxsive/MaXsive-cos.pt",
        maxsive_fpr: float = 0.001,
        maxsive_rotation_restore: bool = True,
        maxsive_debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tuple(self.latent_shape) != (1, 4, 64, 64):
            raise ValueError(
                f"MAXSIVE latent-only baseline expects latent shape (1, 4, 64, 64), got {tuple(self.latent_shape)}."
            )

        self.channel_copy = maxsive_channel_copy
        self.hw_copy = maxsive_hw_copy
        self.template_c = maxsive_template_c
        self.distant_func = maxsive_distant_func
        self.rotation_restore = maxsive_rotation_restore
        self.debug = maxsive_debug

        self.shuffler = MaxsiveShuffler(self.channel_copy, self.hw_copy, self.device)
        self.barlett_window2d = self.create_2d_window(64).to(self.device)
        self.line_detection = HoughTransform(32, 32, 64, 5)

        threshold_data = torch.load(maxsive_threshold_path, weights_only=False)
        threshold_data = threshold_data.sort().values
        self.threshold = threshold_data[-int(len(threshold_data) * maxsive_fpr)].item()

        self.wm_data = None

    def get_wm_type(self) -> str:
        return "MAXSIVE"

    @staticmethod
    def create_2d_window(dim: int):
        window1d = torch.bartlett_window(dim)
        return torch.sqrt(torch.outer(window1d, window1d))

    def watermark_injection(self):
        w = torch.randn(
            1,
            4 // self.channel_copy,
            64 // self.hw_copy,
            64 // self.hw_copy,
            device=self.device,
            dtype=self.dtype,
        )
        w = w / w.std() - w.mean()
        shuffle_key = self.shuffler.shuffle_key_gen()
        z = self.shuffler.shuffle(w, shuffle_key).to(device=self.device, dtype=self.dtype)
        data = {"z": z.detach().clone(), "keys": [w.detach().clone(), shuffle_key]}
        return z, data

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        z, data = self.watermark_injection()
        self.wm_data = data
        return {
            "zT_torch": z,
            "zT_PIL": torch_to_PIL(z),
            "zT": torch_to_PIL(z),
        }

    @torch.no_grad()
    def invert_images(
        self,
        images,
        pipe_provider_target,
        num_inference_steps: int = 50,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        z0_torch = pipe_provider_target.imgs_to_latents(images)
        pipe = pipe_provider_target.pipe
        scheduler = pipe_provider_target.scheduler
        pipe.scheduler = scheduler

        if self.debug:
            print(
                "MAXSIVE debug: using provider custom inversion with "
                f"{type(scheduler).__name__}"
            )

        text_input_ids = pipe.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = pipe.text_encoder(text_input_ids.to(pipe_provider_target.device))[0]

        scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = scheduler.timesteps.to(pipe_provider_target.device)
        latents = z0_torch * scheduler.init_noise_sigma

        for i, t in enumerate(reversed(timesteps_tensor)):
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            if callback_on_step_end is not None:
                callback_on_step_end(pipe, i, t, {"latents": latents})

            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return {
            "z0_torch": z0_torch,
            "z0_PIL": torch_to_PIL(z0_torch),
            "z0": torch_to_PIL(z0_torch),
            "zT_torch": latents,
            "zT_PIL": torch_to_PIL(latents),
            "zT": torch_to_PIL(latents),
        }

    def template_restore(self, z: torch.Tensor, include_angle: int = 45, pre_angle: int = 135):
        z_fft = torch.fft.fftshift(torch.fft.fft2(z.float()), dim=(-1, -2))
        ifft_zt = z_fft[0, self.template_c] * self.barlett_window2d
        detection_result = self.line_detection.detect(abs(ifft_zt), include_angle, pre_angle)
        degree = detection_result["rotated_degree"]
        if_scale = detection_result["if_scale"]

        if degree < 0:
            degree += 180
        if if_scale and degree != 0:
            scale = math.sin(abs(degree % 90) / 180 * math.pi) + math.cos(abs(degree % 90) / 180 * math.pi)
            z = torchvision.transforms.functional.resize(z, int(64 / scale))
            z = torchvision.transforms.functional.resize(
                torchvision.transforms.functional.pad(z, int((64 - z.shape[2]) / 2)),
                64,
            )
        return torchvision.transforms.functional.rotate(z, -degree)

    def detection(self, z: torch.Tensor, data: typing.Dict[str, typing.Any]):
        keys = data["keys"]
        if self.rotation_restore:
            z = self.template_restore(z)

        rotate_zs = self.shuffler.unshuffle(z.to(self.device), keys[1])
        vote_rotate_z = torch.mean(torch.concat(rotate_zs), dim=0).clone()

        w = keys[0].reshape(1, -1).to(self.device, dtype=torch.float32)
        vote_rotate_z = vote_rotate_z.reshape(1, -1).to(self.device, dtype=torch.float32)

        if self.distant_func == "corr":
            corr = torch.corrcoef(torch.concat([w, vote_rotate_z]))
            distance = abs(corr[0, 1].item())
        elif self.distant_func == "cos":
            distance = F.cosine_similarity(vote_rotate_z, w).item()
        elif self.distant_func == "mse":
            distance = -F.mse_loss(vote_rotate_z, w).item()
        elif self.distant_func == "l1":
            distance = -abs(vote_rotate_z - w).mean().item()
        else:
            raise ValueError(f"Unknown MAXSIVE distance function: {self.distant_func}")

        return distance, distance > self.threshold

    def get_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        if self.wm_data is None:
            raise RuntimeError("MAXSIVE wm data is not initialized. Call get_wm_latents() before get_accuracies().")

        value, detection_success = self.detection(latents.to(self.device), self.wm_data)
        return {
            "value": value,
            "bit_accuracies": [0.0],
            "message_bits_str_list": [None],
            "detection_success": detection_success,
            "log_message": (
                f"MAXSIVE; {self.distant_func}: {value:.6f}; "
                f"threshold: {self.threshold:.6f}; detection: {detection_success}"
            ),
            "threshold": self.threshold,
        }
