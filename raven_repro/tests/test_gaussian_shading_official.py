import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from Crypto.Cipher import ChaCha20
from PIL import Image
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_bench_wm"))
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.pairing_provenance import (
    GS_PAIRING_PROTOCOL,
    audit_pairing_rows,
    build_pairing_sha256,
    canonical_json_sha256,
    sha256_path,
    tensor_sha256,
)
from utils.wm.gs_provider import GsProvider


def load_generator_module():
    path = REPO / "experiments" / "generate_watermarked_images.py"
    spec = importlib.util.spec_from_file_location("generate_watermarked_images", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator_module = load_generator_module()


def official_provider(*, secret_index=0, sampling_seed=1234, message=None, key=None, nonce=None):
    return GsProvider(
        latent_shape=(1, 4, 64, 64),
        dtype=torch.float32,
        device=torch.device("cpu"),
        gs_protocol_mode="official_compatible",
        gs_channel_copy=1,
        gs_hw_copy=8,
        offset=secret_index,
        gs_secret_index=secret_index,
        gs_sampling_seed=sampling_seed,
        message=message,
        key=key,
        nonce=nonce,
    )


# For l=1, norm.ppf((u + bit) / 2) is distribution-equivalent to
# sampling the corresponding negative or positive truncated half
# of a standard Gaussian. It is not RNG bit-exact with the
# upstream scipy.stats.truncnorm.rvs implementation.
def independent_official_reference(message, key, nonce, sampling_seed):
    payload = torch.from_numpy(np.unpackbits(np.frombuffer(message, dtype=np.uint8)).copy()).reshape(
        1, 4, 8, 8
    )
    diffused = payload.repeat(1, 1, 8, 8)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    encrypted = cipher.encrypt(np.packbits(diffused.flatten().numpy()).tobytes())
    encrypted_bits = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))
    uniforms = np.random.default_rng(sampling_seed).uniform(0.0, 1.0, size=encrypted_bits.size)
    latent = torch.tensor(
        norm.ppf((uniforms + encrypted_bits.astype(np.float64)) / 2.0).reshape(1, 4, 64, 64),
        dtype=torch.float32,
    )
    return payload, encrypted, uniforms, latent


def test_official_layout_cipher_decode_and_inverse_cdf_reference():
    message = bytes(range(32))
    key = bytes(range(32, 64))
    nonce = bytes(range(12))
    sampling_seed = 8675309
    payload, encrypted, uniforms, expected = independent_official_reference(
        message, key, nonce, sampling_seed
    )
    provider = official_provider(
        message=message,
        key=key,
        nonce=nonce,
        sampling_seed=sampling_seed,
    )
    result = provider.get_wm_latents()

    assert torch.equal(result["barcodes_torch"], payload)
    assert torch.equal(result["zT_torch"], expected)
    assert hashlib.sha256(encrypted).hexdigest() == "95f2e9a9e05f6ffee8762cc655c49dbf86275f4d64d42bee48b999f46158ecb8"
    assert tensor_sha256(expected) == "9cfe485741db158986956e4307c95b400288e52471ab5e4a8fe8c076cf594e1a"
    assert result["sampling_uniform_sha256_list"] == [
        hashlib.sha256(uniforms.astype(np.float64).tobytes()).hexdigest()
    ]
    assert provider.get_accuracies(result["zT_torch"])["bit_accuracies"] == [1.0]


def test_official_mode_is_deterministic_and_unique_per_run():
    first_hashes = []
    second_hashes = []
    secret_hashes = []
    uniform_hashes = []
    for run_id in range(10):
        seed = generator_module.deterministic_gs_sampling_seed(42, run_id)
        first = official_provider(secret_index=run_id, sampling_seed=seed).get_wm_latents()
        second = official_provider(secret_index=run_id, sampling_seed=seed).get_wm_latents()
        first_hashes.append(tensor_sha256(first["zT_torch"]))
        second_hashes.append(tensor_sha256(second["zT_torch"]))
        secret_hashes.append(first["secret_provenance_list"][0]["secret_bundle_sha256"])
        uniform_hashes.append(first["sampling_uniform_sha256_list"][0])
    assert first_hashes == second_hashes
    assert len(set(first_hashes)) == 10
    assert len(set(secret_hashes)) == 10
    assert len(set(uniform_hashes)) == 10


def test_gs_sampling_seed_matches_tree_ring_schedule():
    seed = generator_module.deterministic_gs_sampling_seed
    # per-row GS seed == base_seed + run_id, identical to the TR schedule
    assert seed(42, 0) == 42
    assert seed(42, 1) == 43
    assert seed(42, 1000) == 1042
    seeds = [seed(42, run_id) for run_id in range(1001)]
    assert seeds == list(range(42, 1043))
    assert len(set(seeds)) == 1001  # all unique
    # reproducible for the same base seed and run_id
    assert [seed(42, run_id) for run_id in range(1001)] == seeds


def test_official_direct_decode_and_random_clean_baseline():
    provider = official_provider(secret_index=7, sampling_seed=77)
    watermarked = provider.get_wm_latents()["zT_torch"]
    assert provider.get_accuracies(watermarked)["bit_accuracies"] == [1.0]

    random_latent = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(99))
    random_accuracy = provider.get_accuracies(random_latent)["bit_accuracies"][0]
    assert 0.35 <= random_accuracy <= 0.65


def test_official_majority_vote_ties_decode_zero():
    provider = official_provider(secret_index=0, sampling_seed=1)
    tie = torch.zeros((1, 4, 64, 64), dtype=torch.uint8)
    tie[:, :, :32, :] = 1
    assert torch.count_nonzero(provider._official_majority_vote(tie)) == 0
    majority = tie.clone()
    majority[:, :, 32:40, :] = 1
    assert torch.all(provider._official_majority_vote(majority) == 1)


def test_secret_metadata_contains_only_index_and_hashes():
    provider = official_provider(secret_index=5, sampling_seed=5)
    provenance = provider.secret_provenance()
    assert set(provenance) == {
        "secret_index",
        "message_sha256",
        "key_sha256",
        "nonce_sha256",
        "secret_bundle_sha256",
    }
    extractor_path = REPO / "raven_repro" / "scripts" / "extract_verification_scores.py"
    extractor_spec = importlib.util.spec_from_file_location("extract_verification_scores", extractor_path)
    extractor = importlib.util.module_from_spec(extractor_spec)
    assert extractor_spec.loader is not None
    extractor_spec.loader.exec_module(extractor)
    assert "key_hex" not in extractor.FIELDNAMES
    assert "nonce_hex" not in extractor.FIELDNAMES
    assert "ground_truth_bits" not in extractor.FIELDNAMES


def make_gs_pairing_row(root: Path, run_id: int) -> dict:
    clean = root / f"clean-{run_id}.png"
    watermarked = root / f"watermarked-{run_id}.png"
    Image.new("RGB", (8, 8), (run_id, 1, 2)).save(clean)
    Image.new("RGB", (8, 8), (run_id, 1, 3)).save(watermarked)
    prompt = f"prompt {run_id}"
    row = {
        "protocol": GS_PAIRING_PROTOCOL,
        "dataset": "diffusiondb",
        "wm_type": "GS",
        "run_id": str(run_id),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "base_latent_seed": str(1000 + run_id),
        "base_latent_sha256": f"base-{run_id}",
        "clean_base_latent_sha256": f"base-{run_id}",
        "watermarked_base_latent_sha256": f"base-{run_id}",
        "watermarked_latent_sha256": f"gs-latent-{run_id}",
        "watermark_target_sha256": f"target-{run_id}",
        "watermark_mask_sha256": "gs-mask-not-applicable",
        "generation_config_sha256": "generation",
        "watermark_config_sha256": "watermark",
        "clean_path": str(clean),
        "clean_sha256": sha256_path(clean),
        "watermarked_path": str(watermarked),
        "watermarked_sha256": sha256_path(watermarked),
        "model_id": "model",
        "model_revision": "revision",
        "gs_protocol_mode": "official_compatible",
        "gs_secret_index": str(run_id),
        "gs_message_sha256": f"message-{run_id}",
        "gs_key_sha256": f"key-{run_id}",
        "gs_nonce_sha256": f"nonce-{run_id}",
        "gs_secret_bundle_sha256": f"secret-{run_id}",
        "gs_sampling_seed": str(2000 + run_id),
        "gs_sampling_uniform_sha256": f"uniform-{run_id}",
        "gs_payload_layout": "channel_spatial_repeat",
        "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
    }
    row["pairing_sha256"] = build_pairing_sha256(row)
    return row


def test_gs_pairing_audit_requires_unique_secret_sampling_target_and_latent(tmp_path):
    rows = [make_gs_pairing_row(tmp_path, run_id) for run_id in range(3)]
    audit = audit_pairing_rows(rows, expected_count=3, verify_files=True)
    assert audit["method"] == "GS"
    assert audit["unique_gs_secret_indexes"] == 3
    assert audit["unique_gs_sampling_uniform_hashes"] == 3
    assert audit["unique_watermark_target_hashes"] == 3

    rows[1]["gs_sampling_uniform_sha256"] = rows[0]["gs_sampling_uniform_sha256"]
    rows[1]["pairing_sha256"] = build_pairing_sha256(rows[1])
    with pytest.raises(ValueError, match="duplicate GS sampling uniforms"):
        audit_pairing_rows(rows, expected_count=3, verify_files=False)


def test_gs_resume_accepts_identical_and_rejects_sampling_drift():
    run_id = 4
    sampling_seed = generator_module.deterministic_gs_sampling_seed(42, run_id)
    result = official_provider(secret_index=run_id, sampling_seed=sampling_seed).get_wm_latents()
    secret = result["secret_provenance_list"][0]
    stored = {
        "gs_secret_index": str(run_id),
        "gs_sampling_seed": str(sampling_seed),
        "gs_sampling_uniform_sha256": result["sampling_uniform_sha256_list"][0],
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "watermarked_latent_sha256": tensor_sha256(result["zT_torch"]),
    }
    generator_module.validate_gs_resume_provenance(
        stored, run_id=run_id, sampling_seed=sampling_seed, wm_results=result
    )
    stored["gs_sampling_seed"] = str(sampling_seed + 1)
    with pytest.raises(RuntimeError, match="gs_sampling_seed"):
        generator_module.validate_gs_resume_provenance(
            stored, run_id=run_id, sampling_seed=sampling_seed, wm_results=result
        )
