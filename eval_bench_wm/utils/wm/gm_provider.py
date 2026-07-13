import argparse
import copy
from functools import reduce
from pathlib import Path
import typing

import numpy as np
import torch
from scipy.stats import norm, truncnorm

from .gm_unet import GmUNet
from .wm_provider import WmProvider
from utils.image_utils import torch_to_PIL
from utils import utils


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gm_channel_copy", default=1, type=int)
parser.add_argument("--gm_w_copy", default=8, type=int)
parser.add_argument("--gm_h_copy", default=8, type=int)
parser.add_argument("--gm_detection_threshold", default=0.9, type=float)
parser.add_argument("--gm_ring_weight", default=0.1, type=float)
parser.add_argument("--gm_utils_dir", default="GM_utils", type=str)
parser.add_argument("--gm_w1_path", default="GM_utils/w1_256.pth", type=str)
parser.add_argument("--gm_w2_path", default="GM_utils/w2.pth", type=str)
parser.add_argument("--gm_use_gnr", action="store_true", default=False)
parser.add_argument("--gm_gnr_path", default="GM_utils/GNR_bits256/model_final.pth", type=str)
parser.add_argument("--gm_model_nf", default=128, type=int)
parser.add_argument("--gm_classifier_type", default=0, type=int)
parser.add_argument("--gm_use_classifier", action="store_true", default=False)
parser.add_argument("--gm_classifier_path", default="GM_utils/sd21_cls2.pkl", type=str)
parser.add_argument("--gm_classifier_threshold", default=0.7618901319786762, type=float)
# FPR 10^-2: 0.9371684180856926 (pure)
# FPR 10^-3: 0.9323549943910325 (pure)
# FPR 10^-3: 0.7618901319786762 with template injection

GM_ARG_DEFAULTS = {
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "scheduler_target": "DPM",
}


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    if r >= 0:
        return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2
    return ((x - x0) ** 2 + (y - y0) ** 2) <= -1


class GmProvider(WmProvider):
    """
    Phase-4 GaussMarker provider.

    This integrates the basic GaussMarker idea into the benchmark provider API:
    Gaussian-Shading-style multi-bit initial latents plus a Tree-Ring-like
    frequency patch. Optional GNR restores extracted Gaussian bits after
    image manipulations. Optional classifier mode follows GaussMarker's
    ensemble detector with [bit_accuracy, ring_score] features.
    """

    def __init__(
        self,
        gm_channel_copy: int = 1,
        gm_w_copy: int = 8,
        gm_h_copy: int = 8,
        gm_detection_threshold: float = 0.9,
        gm_ring_weight: float = 0.1,
        gm_utils_dir: str = "GM_utils",
        gm_w1_path: str = "GM_utils/w1_256.pth",
        gm_w2_path: str = "GM_utils/w2.pth",
        gm_use_gnr: bool = True,
        gm_gnr_path: str = "GM_utils/GNR_bits256/model_final.pth",
        gm_model_nf: int = 128,
        gm_classifier_type: int = 0,
        gm_use_classifier: bool = True,
        gm_classifier_path: str = "GM_utils/sd21_cls2.pkl",
        gm_classifier_threshold: float = 0.9371684180856926,
        w_seed: int = 999999,
        w_channel: int = 3,
        w_pattern: str = "ring",
        w_mask_shape: str = "circle",
        w_radius: int = 4,
        w_measurement: str = "l1_complex",
        w_injection: str = "complex",
        w_pattern_const: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tuple(self.latent_shape)[1:] != (4, 64, 64):
            raise ValueError(
                "GM currently supports SD-style 512px latents with shape "
                f"(B, 4, 64, 64), got {tuple(self.latent_shape)}."
            )

        self.ch = gm_channel_copy
        self.w = gm_w_copy
        self.h = gm_h_copy
        self.threshold = gm_detection_threshold
        self.ring_weight = gm_ring_weight
        self.utils_dir = self._resolve_path(gm_utils_dir)
        self.w1_path = self._resolve_path(gm_w1_path) if gm_w1_path else self.utils_dir / "w1_256.pth"
        self.w2_path = self._resolve_path(gm_w2_path) if gm_w2_path else self.utils_dir / "w2.pth"
        self.use_gnr = gm_use_gnr
        self.gnr_path = self._resolve_path(gm_gnr_path) if gm_gnr_path else self.utils_dir / "GNR_bits256" / "model_final.pth"
        self.model_nf = gm_model_nf
        self.classifier_type = gm_classifier_type
        self.use_classifier = gm_use_classifier
        self.classifier_path = self._resolve_path(gm_classifier_path) if gm_classifier_path else self.utils_dir / "sd21_cls2.pkl"
        self.classifier_threshold = gm_classifier_threshold
        if self.use_classifier and not self.use_gnr:
            raise ValueError(
                "GM classifier mode follows GaussMarker's ensemble detector and "
                "requires --gm_use_gnr so the bit feature is GNR-restored."
            )

        self.w_seed = w_seed
        self.w_channel = w_channel
        self.w_pattern = w_pattern
        self.w_mask_shape = w_mask_shape
        self.w_radius = w_radius
        self.w_measurement = w_measurement
        self.w_injection = w_injection
        self.w_pattern_const = w_pattern_const

        if 4 % self.ch != 0 or 64 % self.w != 0 or 64 % self.h != 0:
            raise ValueError("GM copy factors must divide latent dimensions 4x64x64.")

        self.watermark = None
        self.m = None
        self.key = None
        self.nonce = None
        utils.set_random_seed(self.w_seed)
        self._load_w1_state()
        self.gt_patch = self._load_or_create_w2_patch()
        self.watermarking_mask = self._get_watermarking_mask(self.latent_shape)
        self.gnr = self._load_gnr() if self.use_gnr else None
        self.classifier = self._load_classifier() if self.use_classifier else None

    def get_wm_type(self) -> str:
        return "GM"

    @staticmethod
    def _provider_root():
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_path(cls, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return cls._provider_root() / path

    @staticmethod
    def apply_arg_defaults(args, argv):
        for arg_name, default_value in GM_ARG_DEFAULTS.items():
            cli_name = f"--{arg_name}"
            if cli_name not in argv:
                setattr(args, arg_name, default_value)

    def _create_watermark_bits(self):
        if self.watermark is not None and self.m is not None:
            return
        watermark_shape = (
            self.batch_size,
            4 // self.ch,
            64 // self.w,
            64 // self.h,
        )
        self.watermark = torch.randint(0, 2, watermark_shape, device=self.device)
        self.m = self.watermark.repeat(1, self.ch, self.w, self.h).to(torch.int64)

    def _load_w1_state(self):
        if not self.w1_path.exists():
            return
        w_info = torch.load(self.w1_path, map_location=self.device, weights_only=False)
        watermark = w_info.get("w", w_info.get("watermark"))
        m = w_info.get("m")
        if watermark is not None:
            self.watermark = torch.as_tensor(watermark, device=self.device).to(torch.int64)
            if self.watermark.shape[0] != self.batch_size:
                self.watermark = self.watermark[:1].repeat(self.batch_size, 1, 1, 1)
        if m is not None:
            self.m = torch.as_tensor(m, device=self.device).reshape(1, 4, 64, 64).to(torch.int64)
            if self.m.shape[0] != self.batch_size:
                self.m = self.m[:1].repeat(self.batch_size, 1, 1, 1)
        self.key = w_info.get("key")
        self.nonce = w_info.get("nonce")

    def _load_or_create_w2_patch(self):
        if self.w2_path.exists():
            return torch.load(self.w2_path, map_location=self.device, weights_only=False).to(self.device)
        return self._get_watermarking_pattern()

    def _load_gnr(self):
        if not self.gnr_path.exists():
            raise FileNotFoundError(
                f"GM GNR is enabled but model file was not found: {self.gnr_path}. "
                "Place model_final.pth under GM_utils/GNR_bits256/ or pass --gm_gnr_path."
            )
        in_channels = 8 if self.classifier_type == 1 else 4
        model = GmUNet(in_channels, 4, nf=self.model_nf).to(self.device)
        state_dict = torch.load(self.gnr_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_classifier(self):
        import warnings
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"GM classifier is enabled but file was not found: {self.classifier_path}. "
                "Place sd21_cls2.pkl under GM_utils/ or pass --gm_classifier_path."
            )
        
        try:
            import joblib
            from sklearn.exceptions import InconsistentVersionWarning
        except ImportError as exc:
            raise ImportError(
                "GM classifier mode requires joblib. Install joblib or run without "
                "--gm_use_classifier."
            ) from exc

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(self.classifier_path)


    @staticmethod
    def _trunc_sampling(message_bits):
        message_np = message_bits.detach().cpu().numpy().reshape(-1)
        z = np.zeros_like(message_np, dtype=np.float32)
        ppf = [norm.ppf(0.0), norm.ppf(0.5), norm.ppf(1.0)]
        for i, bit in enumerate(message_np):
            bit = int(bit)
            if bit in (0, 1):
                z[i] = np.float32(truncnorm.rvs(ppf[bit], ppf[bit + 1]))
            else:
                dec_mes = int(reduce(lambda a, b: 2 * a + b, message_np[i : i + 1]))
                z[i] = np.float32(truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1]))
        return torch.from_numpy(z)

    def _gaussian_latents_from_m(self):
        latents = []
        for batch_idx in range(self.batch_size):
            latent = self._trunc_sampling(self.m[batch_idx]).reshape(4, 64, 64)
            latents.append(latent)
        return torch.stack(latents, dim=0).to(device=self.device, dtype=self.dtype)

    def _get_watermarking_mask(self, latent_shape):
        watermarking_mask = torch.zeros(latent_shape, dtype=torch.bool, device=self.device)
        if self.w_mask_shape == "circle":
            np_mask = circle_mask(latent_shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask, device=self.device)
            if self.w_channel == -1:
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == "square":
            anchor_p = latent_shape[-1] // 2
            if self.w_channel == -1:
                watermarking_mask[:, :, anchor_p - self.w_radius : anchor_p + self.w_radius, anchor_p - self.w_radius : anchor_p + self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p - self.w_radius : anchor_p + self.w_radius, anchor_p - self.w_radius : anchor_p + self.w_radius] = True
        else:
            raise NotImplementedError(f"w_mask_shape: {self.w_mask_shape}")
        return watermarking_mask

    def _get_watermarking_pattern(self):
        utils.set_random_seed(self.w_seed)
        gt_init = torch.randn(*self.latent_shape, device=self.device)
        if "zeros" in self.w_pattern:
            return torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        if "const" in self.w_pattern:
            return torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0 + self.w_pattern_const
        if "rand" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
            return gt_patch
        if "ring" in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = torch.tensor(circle_mask(gt_init.shape[-1], r=i), device=self.device)
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
            return gt_patch
        raise NotImplementedError(f"w_pattern: {self.w_pattern}")

    def _inject_frequency_watermark(self, latents):
        if self.w_injection == "seed":
            latents_w = latents.clone()
            latents_w[self.watermarking_mask] = self.gt_patch.real.to(latents_w.dtype)[self.watermarking_mask].clone()
            return latents_w

        if self.w_injection != "complex":
            raise NotImplementedError(f"w_injection: {self.w_injection}")

        latents_w_fft = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
        return torch.fft.ifft2(torch.fft.ifftshift(latents_w_fft, dim=(-1, -2))).real.to(latents.dtype)

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        self._create_watermark_bits()
        gaussian_latents = self._gaussian_latents_from_m()
        watermarked_latents = self._inject_frequency_watermark(gaussian_latents)

        return {
            "zT_clean_torch": gaussian_latents,
            "zT_clean_PIL": torch_to_PIL(gaussian_latents),
            "zT_clean": torch_to_PIL(gaussian_latents),
            "zT_torch": watermarked_latents,
            "zT_PIL": torch_to_PIL(watermarked_latents),
            "zT": torch_to_PIL(watermarked_latents),
            "gm_watermark_torch": self.watermark,
            "gm_m_torch": self.m,
            "gm_w1_path": str(self.w1_path),
            "gm_w2_path": str(self.w2_path),
        }

    def generate(
        self,
        pipe_provider_target,
        prompts,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if latents is None:
            latents = self.get_wm_latents()["zT_torch"]
        return pipe_provider_target.generate(
            prompts=prompts,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def _vote_watermark(self, recovered_m):
        ch_stride = 4 // self.ch
        w_stride = 64 // self.w
        h_stride = 64 // self.h
        split_dim1 = torch.cat(torch.split(recovered_m, [ch_stride] * self.ch, dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, [w_stride] * self.w, dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, [h_stride] * self.h, dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0, keepdim=True)
        threshold = 1 if self.ch == 1 and self.w == 1 and self.h == 1 else self.ch * self.w * self.h // 2
        return (vote > threshold).to(torch.int64)

    def _decrypt_m_if_needed(self, recovered_m):
        if self.key is None or self.nonce is None:
            return recovered_m
        try:
            from Crypto.Cipher import ChaCha20
        except ImportError as exc:
            raise ImportError(
                "GM chacha watermark detection requires pycryptodome. Install it "
                "or use a w1 file without key/nonce encryption."
            ) from exc

        decrypted = []
        for sample in recovered_m.detach().cpu().numpy().astype(np.uint8):
            cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
            sd_byte = cipher.decrypt(np.packbits(sample.reshape(-1)).tobytes())
            sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
            decrypted.append(torch.from_numpy(sd_bit).reshape(4, 64, 64))
        return torch.stack(decrypted, dim=0).to(recovered_m.device, dtype=torch.int64)

    def _pred_watermark_from_m(self, recovered_m):
        recovered_sd = self._decrypt_m_if_needed(recovered_m)
        return self._vote_watermark(recovered_sd)

    def _frequency_metrics(self, latents):
        if "complex" in self.w_measurement:
            latents_eval = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-1, -2))
        elif "seed" in self.w_measurement:
            latents_eval = latents.float()
        else:
            raise NotImplementedError(f"w_measurement: {self.w_measurement}")
        ring_l1 = torch.abs(
            latents_eval[self.watermarking_mask] - self.gt_patch[self.watermarking_mask]
        ).mean().item()
        ring_score = -ring_l1
        ring_conf = 1.0 / (1.0 + ring_l1)
        return ring_l1, ring_score, ring_conf

    @staticmethod
    def _gaussmarker_ring_feature(ring_l1):
        # GaussMarker's eval_watermark reports l1 * 0.01, then eval_ensemble
        # feeds the negated value to sd21_cls2.pkl.
        return -float(ring_l1) * 0.01

    def _classifier_score(self, bit_acc, ring_l1):
        if self.classifier is None:
            return None
        features = np.array([[float(bit_acc), self._gaussmarker_ring_feature(ring_l1)]])
        return float(self.classifier.predict_proba(features)[:, 1][0])

    def _restore_m_with_gnr(self, raw_m):
        if self.gnr is None:
            return raw_m, False
        gnr_input = raw_m.float()
        if self.classifier_type == 1:
            target_m = self.m[: raw_m.shape[0]].to(raw_m.device).float()
            gnr_input = torch.cat([target_m, gnr_input], dim=1)
        with torch.no_grad():
            restored_m = (torch.sigmoid(self.gnr(gnr_input)) > 0.5).to(torch.int64)
        return restored_m, True

    def get_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        if self.watermark is None or self.m is None:
            raise ValueError("GM watermark state is missing. Call get_wm_latents() before detection.")

        latents = latents.to(self.device).float()
        raw_m = (latents > 0).to(torch.int64)
        raw_watermark = self._pred_watermark_from_m(raw_m)
        restored_m, used_gnr = self._restore_m_with_gnr(raw_m)
        recovered_watermark = self._pred_watermark_from_m(restored_m)
        target_watermark = self.watermark[: recovered_watermark.shape[0]].to(recovered_watermark.device)

        raw_bit_acc = (raw_watermark == target_watermark).float().mean().item()
        bit_acc = (recovered_watermark == target_watermark).float().mean().item()
        ring_l1, ring_score, ring_conf = self._frequency_metrics(latents)
        ring_classifier_feature = self._gaussmarker_ring_feature(ring_l1)
        classifier_score = self._classifier_score(bit_acc, ring_l1)
        if classifier_score is not None:
            value = classifier_score
            detection_success = value > self.classifier_threshold
            phase = "classifier"
            threshold = self.classifier_threshold
        else:
            value = bit_acc + self.ring_weight * ring_conf
            detection_success = value > self.threshold
            phase = "gnr"
            threshold = self.threshold

        return {
            "accuracies": [bit_acc],
            "bit_accuracies": [bit_acc],
            "p_values": [0.0],
            "l1_dist": [ring_score],
            "gm_raw_bit_acc": float(raw_bit_acc),
            "gm_bit_acc": float(bit_acc),
            "gm_ring_l1": float(ring_l1),
            "gm_ring_score": float(ring_score),
            "gm_ring_conf": float(ring_conf),
            "gm_ring_classifier_feature": float(ring_classifier_feature),
            "gm_used_gnr": used_gnr,
            "gm_used_classifier": classifier_score is not None,
            "gm_classifier_score": classifier_score,
            "value": float(value),
            "detection_success": detection_success,
            "threshold": threshold,
            "gm_recovered_watermark_torch": recovered_watermark,
            "log_message": (
                f"(WM type GM) classifier_score: "
                f"{classifier_score if classifier_score is not None else 'None'}; "
                f"threshold: {threshold:.6f}; detection: {detection_success}"
            ),
        }
