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


def test_shift_magnitude_variant_can_drive_non_identity_gate():
    config = normalize_formal_attack_config(
        variant_config(
            scheduler_mode="ddim",
            shift_plan_mode="paper_random_independent_axes",
            shift_magnitudes_image_px=[2],
            variant_name="shift2_non_identity_smoke_gate",
        )
    )
    dx, dy, seed = planned_shift(0, "0", config)
    assert abs(dx) == 2.0
    assert abs(dy) == 2.0
    assert (dx, dy) != (0.0, 0.0)
    assert seed == 42
    assert formal_attack_config_hash(config) != formal_attack_config_hash()


def test_shift_magnitude_variant_rejects_non_positive_values():
    import pytest

    for magnitudes in ([], [0], [-2], [float("inf")]):
        with pytest.raises(ValueError, match="shift_magnitudes_image_px"):
            normalize_formal_attack_config(
                variant_config(shift_magnitudes_image_px=magnitudes)
            )


def test_variant_config_rejects_ddpm_inversion_with_ddim_scheduler():
    import pytest

    with pytest.raises(ValueError, match="unsupported inversion/scheduler combination"):
        normalize_formal_attack_config(
            variant_config(inversion_mode="ddpm", scheduler_mode="ddim")
        )


def test_ddim_inverse_ddpm_forward_hybrid_variant_is_allowed_and_hash_distinct():
    hybrid = normalize_formal_attack_config(
        variant_config(
            inversion_mode="ddim",
            scheduler_mode="ddpm",
            variant_name="nfpa_nearest_reflection_ddim_inverse_ddpm_aligned",
        )
    )
    assert hybrid["inversion_mode"] == "ddim"
    assert hybrid["scheduler_mode"] == "ddpm"

    pure_ddim = normalize_formal_attack_config(
        variant_config(
            inversion_mode="ddim",
            scheduler_mode="ddim",
            variant_name="nfpa_nearest_reflection_ddim_aligned",
        )
    )
    pure_ddpm = normalize_formal_attack_config(
        variant_config(
            inversion_mode="ddpm",
            scheduler_mode="ddpm",
            variant_name="nfpa_nearest_reflection_ddpm_aligned",
        )
    )
    hybrid_hash = formal_attack_config_hash(hybrid)
    assert hybrid_hash != formal_attack_config_hash(pure_ddim)
    assert hybrid_hash != formal_attack_config_hash(pure_ddpm)
    assert hybrid_hash != formal_attack_config_hash()


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


HYBRID_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "raven_ablation_configs"
    / "ddim_inverse_ddpm_reflection.json"
)


def test_ddim_ddim_and_ddpm_ddpm_pairs_are_allowed():
    ddim = normalize_formal_attack_config(
        variant_config(inversion_mode="ddim", scheduler_mode="ddim")
    )
    ddpm = normalize_formal_attack_config(
        variant_config(inversion_mode="ddpm", scheduler_mode="ddpm")
    )
    assert (ddim["inversion_mode"], ddim["scheduler_mode"]) == ("ddim", "ddim")
    assert (ddpm["inversion_mode"], ddpm["scheduler_mode"]) == ("ddpm", "ddpm")


def test_hybrid_config_file_loads_with_expected_variant_fields():
    from raven.eval_protocol import load_formal_attack_config

    config = load_formal_attack_config(HYBRID_CONFIG_PATH)
    assert config["inversion_mode"] == "ddim"
    assert config["scheduler_mode"] == "ddpm"
    assert config["latent_sampling_mode"] == "nearest"
    assert config["padding_mode"] == "reflection"
    assert config["color_transfer"] is True
    assert config["color_transfer_mode"] == "paper_exact_two_stage_aligned"
    assert config["variant_name"] == "nfpa_nearest_reflection_ddim_inverse_ddpm_aligned"
    # protocol-invariant fields must not have drifted from the formal baseline
    assert config["strength"] == FORMAL_ATTACK_CONFIG["strength"] == 0.15
    assert config["steps"] == FORMAL_ATTACK_CONFIG["steps"] == 50
    assert config["guidance_scale"] == FORMAL_ATTACK_CONFIG["guidance_scale"] == 2.5


def test_hybrid_reconstruction_scheduler_provenance_is_ddpm_and_distinct_from_ddim():
    from raven.eval_protocol import formal_runtime_provenance

    ddpm = formal_runtime_provenance(
        scheduler_mode="ddpm", device_class="cuda", attack_dtype="torch.float16"
    )
    ddim = formal_runtime_provenance(
        scheduler_mode="ddim", device_class="cuda", attack_dtype="torch.float16"
    )
    assert ddpm["scheduler_class"] == "DDPMScheduler"
    assert ddim["scheduler_class"] == "DDIMScheduler"
    # the reconstruction scheduler config hash must be method-specific
    assert ddpm["scheduler_config_hash"] != ddim["scheduler_config_hash"]
    assert (
        ddpm["scheduler_config_hash"]
        == canonical_json_hash(ddpm["scheduler_config"])
    )


class _ZeroUnet:
    def __init__(self):
        self.calls = 0

    def __call__(self, model_input, timestep, encoder_hidden_states=None, return_dict=False):
        self.calls += 1
        return (torch.zeros_like(model_input),)


def _real_ddpm_scheduler():
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
        prediction_type="epsilon",
    )


def test_hybrid_ddim_inversion_ddpm_reconstruction_runtime_provenance():
    """DDIM inversion + DDPM reconstruction must not take the DDPM add_noise path."""
    scheduler = _real_ddpm_scheduler()
    add_noise_calls = {"count": 0}
    original_add_noise = scheduler.add_noise

    def _spy_add_noise(*args, **kwargs):
        add_noise_calls["count"] += 1
        return original_add_noise(*args, **kwargs)

    scheduler.add_noise = _spy_add_noise

    unet = _ZeroUnet()
    image = Image.new("RGB", (16, 16))
    prompt_embeds = torch.zeros((2, 4, 8), dtype=torch.float32)
    result = partial_diffusion_inversion(
        _Vae(),
        scheduler,
        image,
        num_inference_steps=50,
        strength=0.15,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        dtype=torch.float32,
        mode="ddim",
        unet=unet,
        prompt_embeds=prompt_embeds,
        guidance_scale=2.5,
    )
    # inversion mode is DDIM using a real DDIM inverse scheduler
    assert result.mode == "ddim"
    assert result.inverse_scheduler == "DDIMInverseScheduler"
    # reconstruction scheduler class is DDPMScheduler
    assert result.denoise_scheduler == "DDPMScheduler"
    # DDIM inversion must NOT use the DDPM scheduler.add_noise() forward-noising path
    assert add_noise_calls["count"] == 0
    assert result.noise is None
    # DDIM inversion actually produced inverse-scheduler timestep provenance
    assert result.inversion_timesteps is not None
    assert len(result.inversion_timesteps) > 0
    assert unet.calls == len(result.inversion_timesteps)
    # the reconstruction denoise timesteps begin at the exact DDIM target timestep
    assert result.target_timestep == int(result.timesteps[0])
    assert result.target_timestep in {int(t) for t in scheduler.timesteps}
