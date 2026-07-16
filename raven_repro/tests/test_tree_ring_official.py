import sys
import pathlib

RAVEN_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAVEN_ROOT))
import json
import math

import numpy as np
import pytest
import torch
from sklearn import metrics

from raven.tree_ring_official import (
    fail_on_nonfinite,
    inject_complex_watermark,
    make_rand_watermark_target,
    make_watermark_mask,
    official_complex_l1,
    official_roc,
    set_official_random_seed,
    stable_tensor_hash,
)
from scripts import tree_ring_official_raven_eval as driver


def test_per_sample_base_latents_are_unique_and_reproducible():
    hashes = []
    for seed in range(12):
        set_official_random_seed(seed)
        value = torch.randn(1, 4, 64, 64)
        hashes.append(stable_tensor_hash(value))
    assert len(set(hashes)) == len(hashes)
    set_official_random_seed(7)
    first = stable_tensor_hash(torch.randn(1, 4, 64, 64))
    set_official_random_seed(7)
    second = stable_tensor_hash(torch.randn(1, 4, 64, 64))
    assert first == second


def test_clean_and_watermarked_preinject_hashes_match():
    base = torch.randn(1, 4, 64, 64)
    assert stable_tensor_hash(base.clone()) == stable_tensor_hash(base.clone())


def test_fixed_target_and_mask_are_reproducible():
    set_official_random_seed(999999)
    first = torch.randn(1, 4, 64, 64)
    set_official_random_seed(999999)
    second = torch.randn(1, 4, 64, 64)
    target_a = make_rand_watermark_target(first)
    target_b = make_rand_watermark_target(second)
    mask_a = make_watermark_mask(first, channel=0, radius=10)
    mask_b = make_watermark_mask(second, channel=0, radius=10)
    assert torch.equal(target_a, target_b)
    assert torch.equal(mask_a, mask_b)


def test_complex_fft_injection_only_changes_mask_region():
    base = torch.randn(1, 4, 64, 64)
    target = make_rand_watermark_target(torch.randn_like(base))
    mask = make_watermark_mask(base, channel=0, radius=10)
    before_fft = torch.fft.fftshift(torch.fft.fft2(base), dim=(-1, -2))
    assigned_fft = before_fft.clone()
    assigned_fft[mask] = target[mask]
    assert torch.equal(assigned_fft[~mask], before_fft[~mask])
    assert torch.equal(assigned_fft[mask], target[mask])
    injected = inject_complex_watermark(base.clone(), mask, target)
    expected = torch.fft.ifft2(torch.fft.ifftshift(assigned_fft, dim=(-1, -2))).real
    assert torch.equal(injected, expected)


def test_official_complex_l1_score_direction():
    latent = torch.randn(1, 4, 64, 64)
    target = torch.fft.fftshift(torch.fft.fft2(latent), dim=(-1, -2))
    mask = make_watermark_mask(latent, channel=0, radius=10)
    matching = official_complex_l1(latent, mask, target)
    unrelated = official_complex_l1(torch.randn_like(latent), mask, target)
    assert matching == pytest.approx(0.0, abs=1e-6)
    assert unrelated > matching


def test_official_roc_matches_repository_semantics():
    clean = np.linspace(10.0, 20.0, 1000)
    watermarked = np.linspace(0.0, 12.0, 1000)
    report = official_roc(clean, watermarked)
    predictions = np.concatenate([-clean, -watermarked])
    labels = np.concatenate([np.zeros(1000), np.ones(1000)])
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    index = np.where(fpr < 0.01)[0][-1]
    assert report["actual_fpr"] == pytest.approx(fpr[index])
    assert report["tpr_at_1pct_fpr"] == pytest.approx(tpr[index])
    assert report["decision_threshold_negative_l1"] == pytest.approx(thresholds[index])
    assert report["false_positives"] == round(fpr[index] * 1000)


def test_nonfinite_scores_fail_fast():
    with pytest.raises(ValueError, match="non-finite"):
        fail_on_nonfinite([1.0, math.nan], ["ok", "run_id=9"])
    with pytest.raises(ValueError, match="non-finite"):
        official_roc([1.0, math.inf], [0.0, 0.1])


def test_attack_debug_requires_paired_formal_config():
    debug = {
        "inversion_mode": "ddim",
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "padding_mode": "reflection",
        "interpolation_mode": "nearest",
        "color_transfer_mode": "paper_exact_two_stage",
        "warp_input_stage": "ddim_inversion.noisy_latents_z_tau",
        "warp_input_is_inversion_noisy_latents": True,
        "decoded_output_branch": "view_branch_index_1",
        "attention_processor_count": 32,
        "timesteps": list(range(7)),
        "attention_debug": {
            "self_processor_count": 16,
            "processors_with_calls": 16,
            "total_calls": 112,
        },
    }
    driver.assert_attack_debug(debug, 0)
    debug["warp_input_stage"] = "z0"
    with pytest.raises(RuntimeError, match="config drift"):
        driver.assert_attack_debug(debug, 0)


def test_resume_hash_validation_and_no_overwrite(tmp_path):
    path = tmp_path / "record.json"
    driver.write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        driver.write_json(path, {"ok": False})
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"first")
    digest = driver.sha256_path(image_path)
    driver.validate_existing_image(image_path, digest)
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        driver.validate_existing_image(image_path, digest)


def test_protocol_hash_includes_shift_plan_and_attack_config():
    plan = [driver.build_shift(0, driver.PLAN_SEED)]
    first = driver.protocol_hash(plan)
    changed = json.loads(json.dumps(plan))
    changed[0]["flow_dx_image_px"] *= -1
    assert first != driver.protocol_hash(changed)
