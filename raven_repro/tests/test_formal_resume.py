import json

import pytest

from raven.eval_protocol import (
    FORMAL_ATTACK_CONFIG,
    canonical_json_hash,
    sha256_path,
    transform_config_payload,
    validate_resume_record,
)


def debug_payload(dx=27.0, dy=-29.0):
    payload = {
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "steps": 50, "strength": 0.15, "guidance_scale": 2.5,
        "inversion_mode": "ddim", "exact_timestep": 149,
        "inversion_prompt": "", "reconstruction_prompt": "", "negative_prompt": "",
        "warp_mode": "raven_paper_nfpa_gap_fill", "interpolation_mode": "nearest",
        "padding_mode": "reflection", "align_corners": False,
        "normalized_coordinate_formula": "x_norm = 2*x_pixel/W - 1",
        "pixel_center_offset_image_px": 0.0,
        "warp_coordinate_convention": "legacy_nfpa_w_h_norm",
        "warp_implementation_version": "nfpa_image_grid_w_h_norm_v1",
        "planned_flow_dx_image_px": dx, "planned_flow_dy_image_px": dy,
        "effective_source_dx_latent": 3.0, "effective_source_dy_latent": -4.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
        "effective_visual_shift_dx_image_px": -24.0,
        "effective_visual_shift_dy_image_px": 32.0,
        "view_guided_attention": True, "color_transfer": True,
        "color_transfer_mode": "paper_exact_two_stage_aligned",
        "attack_device_class": "cuda",
        "attack_dtype": "torch.float16",
        "scheduler_class": "DDIMScheduler",
        "scheduler_config": {"beta_start": 0.00085},
        "torch_version": "2.test",
        "diffusers_version": "0.test",
    }
    payload["scheduler_config_hash"] = canonical_json_hash(payload["scheduler_config"])
    payload["transform_config_hash"] = canonical_json_hash(transform_config_payload(payload))
    return payload


def test_resume_rejects_config_seed_manifest_and_debug_drift(tmp_path):
    attacked = tmp_path / "attacked.png"
    attacked.write_bytes(b"attacked")
    pre_color = tmp_path / "pre-color.png"
    pre_color.write_bytes(b"pre-color")
    debug_path = tmp_path / "debug.json"
    debug_path.write_text(json.dumps(debug_payload()))
    expected = {
        "run_id": "1", "attack_config_hash": "strength-0.15", "attack_seed": 43,
        "snapshot_sha256": "snapshot-a", "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "planned_flow_dx_image_px": 27.0, "planned_flow_dy_image_px": -29.0,
    }
    record = {
        **expected, "attacked_path": str(attacked), "attacked_sha256": sha256_path(attacked),
        "pre_color_attacked_path": str(pre_color),
        "pre_color_attacked_sha256": sha256_path(pre_color),
        "debug_info_path": str(debug_path), "debug_info_sha256": sha256_path(debug_path),
        "resolved_scheduler_config": debug_payload()["scheduler_config"],
        "resolved_scheduler_config_hash": debug_payload()["scheduler_config_hash"],
        "torch_version": debug_payload()["torch_version"],
        "diffusers_version": debug_payload()["diffusers_version"],
        "attack_device_class": "cuda",
        "attack_dtype": "torch.float16",
        "transform_config_hash": debug_payload()["transform_config_hash"],
    }
    validate_resume_record(record, expected=expected)
    for field, changed in (
        ("attack_config_hash", "strength-0.20"), ("attack_seed", 44),
        ("snapshot_sha256", "snapshot-b"), ("model_revision", "other-revision"),
    ):
        mismatch = {**expected, field: changed}
        with pytest.raises(RuntimeError, match="resume mismatch"):
            validate_resume_record(record, expected=mismatch)
    tampered = debug_payload()
    tampered["warp_mode"] = "integer"
    debug_path.write_text(json.dumps(tampered))
    record["debug_info_sha256"] = sha256_path(debug_path)
    with pytest.raises(RuntimeError, match="formal debug mismatch"):
        validate_resume_record(record, expected=expected)



def test_resume_rejects_pre_color_dtype_and_scheduler_drift(tmp_path):
    attacked = tmp_path / "attacked.png"
    attacked.write_bytes(b"attacked")
    pre_color = tmp_path / "pre.png"
    pre_color.write_bytes(b"pre")
    debug = debug_payload()
    debug_path = tmp_path / "debug.json"
    debug_path.write_text(json.dumps(debug))
    expected = {
        "run_id": "1",
        "planned_flow_dx_image_px": 27.0,
        "planned_flow_dy_image_px": -29.0,
        "attack_dtype": "torch.float16",
        "scheduler_config_hash": "selector-a",
    }
    record = {
        **expected,
        "attacked_path": str(attacked),
        "attacked_sha256": sha256_path(attacked),
        "pre_color_attacked_path": str(pre_color),
        "pre_color_attacked_sha256": sha256_path(pre_color),
        "debug_info_path": str(debug_path),
        "debug_info_sha256": sha256_path(debug_path),
        "resolved_scheduler_config": debug["scheduler_config"],
        "resolved_scheduler_config_hash": debug["scheduler_config_hash"],
        "torch_version": debug["torch_version"],
        "diffusers_version": debug["diffusers_version"],
        "attack_device_class": "cuda",
        "transform_config_hash": debug["transform_config_hash"],
    }
    validate_resume_record(record, expected=expected)
    pre_color.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="pre_color_attacked_sha256 mismatch"):
        validate_resume_record(record, expected=expected)
    pre_color.write_bytes(b"pre")
    with pytest.raises(RuntimeError, match="resume mismatch for attack_dtype"):
        validate_resume_record(
            record, expected={**expected, "attack_dtype": "torch.bfloat16"}
        )
    with pytest.raises(RuntimeError, match="resume mismatch for scheduler_config_hash"):
        validate_resume_record(
            record, expected={**expected, "scheduler_config_hash": "selector-b"}
        )
