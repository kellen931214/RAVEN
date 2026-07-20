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
        "planned_flow_dx_image_px": dx, "planned_flow_dy_image_px": dy,
        "effective_source_dx_latent": 3.0, "effective_source_dy_latent": -4.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
        "effective_visual_shift_dx_image_px": -24.0,
        "effective_visual_shift_dy_image_px": 32.0,
        "view_guided_attention": True, "color_transfer": True,
        "color_transfer_mode": "paper_exact_two_stage_aligned",
    }
    payload["transform_config_hash"] = canonical_json_hash(transform_config_payload(payload))
    return payload


def test_resume_rejects_config_seed_manifest_and_debug_drift(tmp_path):
    attacked = tmp_path / "attacked.png"
    attacked.write_bytes(b"attacked")
    debug_path = tmp_path / "debug.json"
    debug_path.write_text(json.dumps(debug_payload()))
    expected = {
        "run_id": "1", "attack_config_hash": "strength-0.15", "attack_seed": 43,
        "snapshot_sha256": "snapshot-a", "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "planned_flow_dx_image_px": 27.0, "planned_flow_dy_image_px": -29.0,
    }
    record = {
        **expected, "attacked_path": str(attacked), "attacked_sha256": sha256_path(attacked),
        "debug_info_path": str(debug_path), "debug_info_sha256": sha256_path(debug_path),
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
