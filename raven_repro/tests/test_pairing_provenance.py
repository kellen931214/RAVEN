import hashlib
from pathlib import Path

import pytest
from PIL import Image

from raven.pairing_provenance import (
    PAIRING_PROTOCOL,
    assert_attack_pair_config_match,
    audit_pairing_rows,
    build_attack_config_sha256,
    build_pairing_sha256,
    sha256_path,
)


def _image(path: Path, rgb: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), rgb).save(path)


def _pair(tmp_path: Path, run_id: int) -> dict:
    clean = tmp_path / f"clean_{run_id}.png"
    watermarked = tmp_path / f"wm_{run_id}.png"
    _image(clean, (run_id + 1, 2, 3))
    _image(watermarked, (run_id + 1, 2, 4))
    prompt = f"prompt {run_id}"
    row = {
        "protocol": PAIRING_PROTOCOL,
        "dataset": "unit",
        "run_id": str(run_id),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "base_latent_seed": 42 + run_id,
        "base_latent_sha256": f"base-{run_id}",
        "clean_base_latent_sha256": f"base-{run_id}",
        "watermarked_base_latent_sha256": f"base-{run_id}",
        "watermarked_latent_sha256": f"wm-latent-{run_id}",
        "watermark_target_sha256": "target",
        "watermark_mask_sha256": "mask",
        "generation_config_sha256": "generation",
        "watermark_config_sha256": "watermark",
        "clean_path": str(clean),
        "clean_sha256": sha256_path(clean),
        "watermarked_path": str(watermarked),
        "watermarked_sha256": sha256_path(watermarked),
        "model_id": "model",
        "model_revision": "revision",
    }
    row["pairing_sha256"] = build_pairing_sha256(row)
    return row


def _attack(pairing_sha256: str, *, seed: int = 42) -> dict:
    record = {
        "seed": seed,
        "flow_dx_image_px": 28.0,
        "flow_dy_image_px": -26.0,
        "exact_ddim_timestep": 121,
        "steps": 50,
        "strength": 0.15,
        "guidance_scale": 2.5,
        "inversion_mode": "ddim",
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "sampling_mode": "nearest",
        "padding_mode": "reflection",
        "normalization_formula": "formula",
        "color_transfer_mode": "paper_exact_two_stage_aligned",
        "model_id": "model",
        "model_revision": "revision",
        "pairing_sha256": pairing_sha256,
    }
    record["attack_config_sha256"] = build_attack_config_sha256(record)
    return record


def test_pairing_audit_accepts_unique_paired_latents(tmp_path):
    rows = [_pair(tmp_path, 0), _pair(tmp_path, 1)]
    audit = audit_pairing_rows(rows, expected_count=2)
    assert audit["passed"] is True
    assert audit["unique_base_latent_hashes"] == 2
    assert audit["duplicate_base_latent_hashes"] == 0


def test_pairing_hash_survives_csv_scalar_round_trip(tmp_path):
    row = _pair(tmp_path, 0)
    csv_row = {key: str(value) for key, value in row.items()}
    assert build_pairing_sha256(csv_row) == row["pairing_sha256"]
    assert audit_pairing_rows([csv_row], expected_count=1)["passed"] is True


def test_pairing_audit_rejects_shared_latent(tmp_path):
    rows = [_pair(tmp_path, 0), _pair(tmp_path, 1)]
    rows[1]["base_latent_sha256"] = rows[0]["base_latent_sha256"]
    rows[1]["clean_base_latent_sha256"] = rows[0]["base_latent_sha256"]
    rows[1]["watermarked_base_latent_sha256"] = rows[0]["base_latent_sha256"]
    rows[1]["pairing_sha256"] = build_pairing_sha256(rows[1])
    with pytest.raises(ValueError, match="duplicate base latent hash"):
        audit_pairing_rows(rows, expected_count=2)


def test_pairing_audit_rejects_missing_or_broken_pairing(tmp_path):
    row = _pair(tmp_path, 0)
    row.pop("pairing_sha256")
    with pytest.raises(ValueError, match="missing pairing_sha256"):
        audit_pairing_rows([row], expected_count=1)


def test_attack_pair_requires_identical_full_config_and_pairing(tmp_path):
    pair = _pair(tmp_path, 0)
    clean = _attack(pair["pairing_sha256"])
    watermarked = _attack(pair["pairing_sha256"])
    assert assert_attack_pair_config_match(clean, watermarked, "0") == clean["attack_config_sha256"]

    watermarked = _attack(pair["pairing_sha256"], seed=43)
    with pytest.raises(ValueError, match="config mismatch"):
        assert_attack_pair_config_match(clean, watermarked, "0")
