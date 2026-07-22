import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image

from experiments.run_raven_formal_eval import load_immutable_source_rows, planned_shift
from raven.eval_protocol import (
    FORMAL_ATTACK_CONFIG,
    assert_formal_debug_info,
    canonical_json_hash,
    canonical_scheduler_config,
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
        "normalized_coordinate_formula": (
            "x_norm = 2*(x_pixel+0.5)/W - 1; y_norm = 2*(y_pixel+0.5)/H - 1"
            if config["warp_mode"] == "raven_paper_nfpa_gap_fill_centered"
            else "x_norm = 2*x_pixel/W - 1"
        ),
        "pixel_center_offset_image_px": (
            0.5 if config["warp_mode"] == "raven_paper_nfpa_gap_fill_centered" else 0.0
        ),
        "warp_coordinate_convention": (
            "centered_align_corners_false"
            if config["warp_mode"] == "raven_paper_nfpa_gap_fill_centered"
            else "legacy_nfpa_w_h_norm"
        ),
        "warp_implementation_version": (
            "nfpa_image_grid_w_h_norm_centered_v2"
            if config["warp_mode"] == "raven_paper_nfpa_gap_fill_centered"
            else "nfpa_image_grid_w_h_norm_v1"
        ),
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
        "attack_device_class": "cuda",
        "attack_dtype": "torch.float16",
        "scheduler_class": "DDIMScheduler",
        "scheduler_config": {"beta_start": 0.00085},
        "torch_version": "2.test",
        "diffusers_version": "0.test",
    }
    payload["scheduler_config_hash"] = canonical_json_hash(payload["scheduler_config"])
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
            shift_plan_mode="paper_random_independent_axes",
        )
    )
    payload = debug_payload(config)
    assert assert_formal_debug_info(payload, attack_config=config) == payload[
        "transform_config_hash"
    ]
    assert formal_attack_config_hash(config) != formal_attack_config_hash()


def test_scheduler_config_hash_ignores_private_metadata_order_only():
    first = {
        "_class_name": "DDPMScheduler",
        "_diffusers_version": "old-build-tag",
        "_use_default_values": ["variance_type", "timestep_spacing"],
        "beta_start": 0.00085,
        "variance_type": "fixed_small",
    }
    second = {
        **first,
        "_diffusers_version": "new-build-tag",
        "_use_default_values": ["timestep_spacing", "variance_type"],
    }
    assert canonical_json_hash(canonical_scheduler_config(first)) == canonical_json_hash(
        canonical_scheduler_config(second)
    )

    changed = {**second, "variance_type": "fixed_large"}
    assert canonical_json_hash(canonical_scheduler_config(first)) != canonical_json_hash(
        canonical_scheduler_config(changed)
    )


def test_centered_bilinear_variant_config_is_allowed_and_hash_distinct():
    config = normalize_formal_attack_config(
        variant_config(
            warp_mode="raven_paper_nfpa_gap_fill_centered",
            latent_sampling_mode="bilinear",
            inversion_mode="ddim",
            scheduler_mode="ddim",
            shift_plan_mode="paper_random_independent_axes",
            variant_name="nfpa_centered_bilinear_reflection_ddim_aligned",
        )
    )
    payload = debug_payload(config)
    assert payload["pixel_center_offset_image_px"] == 0.5
    assert payload["warp_coordinate_convention"] == "centered_align_corners_false"
    assert assert_formal_debug_info(payload, attack_config=config) == payload[
        "transform_config_hash"
    ]
    assert formal_attack_config_hash(config) != formal_attack_config_hash()

def test_immutable_source_index_requires_original_snapshot_and_metadata_sha(tmp_path):
    source = tmp_path / "metadata.csv"
    source.write_text("run_id,prompt\n0,one\n1,two\n")
    batch = tmp_path / "batch.csv"
    batch.write_text(source.read_text())
    from raven.eval_protocol import sha256_path

    index = tmp_path / "snapshot_index.jsonl"
    entry = {
        "snapshot_path": str(batch),
        "snapshot_sha256": sha256_path(batch),
        "row_count": 2,
        "source_metadata_path": str(source.resolve()),
        "source_metadata_sha256": sha256_path(source),
    }
    index.write_text(json.dumps(entry) + "\n")
    rows, source_sha = load_immutable_source_rows(
        index, source_metadata=source, expected_count=2
    )
    assert [row["run_id"] for row in rows] == ["0", "1"]
    assert source_sha == sha256_path(source)
    source.write_text("run_id,prompt\n0,changed\n1,two\n")
    import pytest
    with pytest.raises(RuntimeError, match="metadata SHA mismatch"):
        load_immutable_source_rows(index, source_metadata=source, expected_count=2)


def test_zero_shift_plan_preserves_seed_and_uses_zero_effective_plan():
    baseline = normalize_formal_attack_config(
        variant_config(scheduler_mode="ddim", shift_plan_mode="paper_random_independent_axes")
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
