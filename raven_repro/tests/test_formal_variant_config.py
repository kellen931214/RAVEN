import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image

from experiments.run_raven_formal_eval import planned_shift
from raven.eval_protocol import (
    FORMAL_ATTACK_CONFIG,
    assert_formal_debug_info,
    canonical_json_hash,
    formal_attack_config_hash,
    normalize_formal_attack_config,
    transform_config_payload,
)
from raven.inversion import partial_diffusion_inversion


def variant_config(**overrides):
    return {**FORMAL_ATTACK_CONFIG, **overrides}


def debug_payload(config):
    payload = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "steps": config["steps"],
        "strength": config["strength"],
        "guidance_scale": config["guidance_scale"],
        "inversion_mode": config["inversion_mode"],
        "exact_timestep": 121,
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "negative_prompt": "",
        "warp_mode": config["warp_mode"],
        "interpolation_mode": config["latent_sampling_mode"],
        "padding_mode": config["padding_mode"],
        "align_corners": False,
        "normalized_coordinate_formula": "x_norm = 2*x_pixel/W - 1",
        "planned_flow_dx_image_px": 27.0,
        "planned_flow_dy_image_px": -29.0,
        "effective_source_dx_latent": 3.0,
        "effective_source_dy_latent": -4.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
        "effective_visual_shift_dx_image_px": -24.0,
        "effective_visual_shift_dy_image_px": 32.0,
        "view_guided_attention": True,
        "color_transfer": True,
        "color_transfer_mode": config["color_transfer_mode"],
    }
    payload["transform_config_hash"] = canonical_json_hash(
        transform_config_payload(payload)
    )
    return payload


def test_variant_config_hash_and_debug_assertion_are_config_specific():
    config = normalize_formal_attack_config(
        variant_config(
            latent_sampling_mode="bilinear",
            inversion_mode="ddim",
            scheduler_mode="ddim",
            shift_plan_mode="formal_deterministic",
        )
    )
    payload = debug_payload(config)
    assert assert_formal_debug_info(payload, attack_config=config) == payload[
        "transform_config_hash"
    ]
    assert formal_attack_config_hash(config) != formal_attack_config_hash()


def test_zero_shift_plan_preserves_seed_and_uses_zero_effective_plan():
    baseline = normalize_formal_attack_config(
        variant_config(scheduler_mode="ddim", shift_plan_mode="formal_deterministic")
    )
    zero = normalize_formal_attack_config(
        variant_config(scheduler_mode="ddim", shift_plan_mode="zero")
    )
    base_dx, base_dy, base_seed = planned_shift(3, "7", baseline)
    zero_dx, zero_dy, zero_seed = planned_shift(3, "7", zero)
    assert (base_dx, base_dy) != (0.0, 0.0)
    assert (zero_dx, zero_dy) == (0.0, 0.0)
    assert zero_seed == base_seed == 49


def test_variant_config_rejects_mixed_scheduler_and_inversion_modes():
    import pytest

    with pytest.raises(ValueError, match="must match"):
        normalize_formal_attack_config(
            variant_config(inversion_mode="ddpm", scheduler_mode="ddim")
        )


def test_variant_config_rejects_non_ablation_protocol_drift():
    import pytest

    with pytest.raises(ValueError, match="non-variant fields"):
        normalize_formal_attack_config(
            variant_config(scheduler_mode="ddim", strength=0.20)
        )


class _Posterior:
    def mode(self):
        return torch.zeros((1, 4, 2, 2), dtype=torch.float32)


class _Vae:
    class config:
        scaling_factor = 0.18215

    def encode(self, tensor):
        class _Encoded:
            latent_dist = _Posterior()

        return _Encoded()


class _DdpmScheduler:
    class config:
        prediction_type = "epsilon"

    def set_timesteps(self, count, device):
        self.timesteps = torch.tensor([9, 4], device=device)

    def add_noise(self, latents, noise, timesteps):
        self.called = True
        return latents + noise


def test_ddpm_mode_uses_scheduler_forward_noise_without_ddim_inverse():
    scheduler = _DdpmScheduler()
    image = Image.new("RGB", (16, 16))
    result = partial_diffusion_inversion(
        _Vae(),
        scheduler,
        image,
        num_inference_steps=2,
        strength=0.5,
        generator=torch.Generator().manual_seed(1),
        device="cpu",
        dtype=torch.float32,
        mode="ddpm",
    )
    assert scheduler.called is True
    assert result.mode == "ddpm"
    assert result.inverse_scheduler == ""
    assert result.noise is not None
