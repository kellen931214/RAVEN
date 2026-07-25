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


# ---------------------------------------------------------------------------
# Default-path (official) coverage — added 2026-07-25 when GS default flipped
# from legacy to official_compatible and detection defaulted to official tau.
# ---------------------------------------------------------------------------

import argparse

from utils.wm.gs_provider import (
    OFFICIAL_REPRODUCTION_MODEL_ID,
    OFFICIAL_REPRODUCTION_SCHEDULER,
    OFFICIAL_REPRODUCTION_REVISION,
    apply_official_reproduction_defaults,
    parser as gs_parser,
)


def _default_provider(**overrides):
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_size=1,
    )
    kwargs.update(overrides)
    return GsProvider(**kwargs)


def test_gs_default_protocol_mode_is_official_compatible():
    # Parser default.
    assert gs_parser.parse_known_args([])[0].gs_protocol_mode == "official_compatible"
    # Provider constructor default (no gs_protocol_mode passed).
    assert _default_provider().gs_protocol_mode == "official_compatible"


def test_gs_default_detection_mode_is_official_not_legacy():
    assert gs_parser.parse_known_args([])[0].gs_detection_mode == "official_onebit"
    provider = _default_provider()
    assert provider.gs_detection_mode == "official_onebit"
    info = provider.active_detection_threshold()
    assert info["detection_mode"] == "official_onebit"
    assert info["threshold_type"] == "official_beta_tail_tau_onebit"
    assert info["threshold_type"] != "legacy_default_threshold"
    assert info["comparison_operator"] == ">="


def test_gs_official_tau_onebit_actually_drives_detection_success():
    provider = _default_provider()
    official = provider.official_thresholds()
    tau = official["tau_onebit"]
    # The threshold used for success is the official tau, not legacy GS_THRESHOLDS.
    from utils.utils import GS_THRESHOLDS

    assert provider.active_detection_threshold()["threshold"] == tau
    assert tau != GS_THRESHOLDS
    # Success is decided by value >= tau (official), so a value between tau and the
    # (higher) legacy threshold succeeds under official but would fail under legacy.
    between = (tau + GS_THRESHOLDS) / 2.0
    assert tau < between < GS_THRESHOLDS
    assert provider.is_detection_successful(between) is True
    assert provider.is_detection_successful(tau) is True  # >= boundary
    assert provider.is_detection_successful(tau - 1e-6) is False


def test_gs_traceability_and_legacy_detection_modes_are_explicit_and_distinct():
    trace = _default_provider(gs_detection_mode="official_traceability")
    t_info = trace.active_detection_threshold()
    assert t_info["threshold"] == trace.official_thresholds()["tau_bits"]
    assert t_info["threshold_type"] == "official_beta_tail_tau_bits"
    assert t_info["comparison_operator"] == ">="

    legacy = _default_provider(gs_detection_mode="legacy_default")
    l_info = legacy.active_detection_threshold()
    from utils.utils import GS_THRESHOLDS

    assert l_info["threshold"] == GS_THRESHOLDS
    assert l_info["threshold_type"] == "legacy_default_threshold"
    assert l_info["comparison_operator"] == ">"
    # Legacy uses strict '>': exactly at the threshold is NOT a success.
    assert legacy.is_detection_successful(GS_THRESHOLDS) is False
    assert legacy.is_detection_successful(GS_THRESHOLDS + 1e-6) is True

    with pytest.raises(ValueError, match="gs_detection_mode"):
        _default_provider(gs_detection_mode="not_a_mode")


def test_legacy_protocol_still_works_when_explicitly_requested():
    provider = _default_provider(gs_protocol_mode="legacy", gs_hw_copy=8)
    assert provider.gs_protocol_mode == "legacy"
    # Legacy encode/decode round-trips (byte-replication path still functional).
    result = provider.get_wm_latents()
    accuracies = provider.get_accuracies(result["zT_torch"])
    assert accuracies["bit_accuracies"][0] == 1.0


def test_official_reproduction_defaults_apply_only_to_gs_standalone_runners():
    # GS with no explicit model/scheduler -> official upstream generation defaults.
    ns = argparse.Namespace(
        wm_type="GS", modelid_target="placeholder", scheduler_target="DDIM", model_revision=None
    )
    applied = apply_official_reproduction_defaults(ns, argv=["run_watermark.py"])
    assert ns.modelid_target == OFFICIAL_REPRODUCTION_MODEL_ID == "stabilityai/stable-diffusion-2-1-base"
    assert ns.scheduler_target == OFFICIAL_REPRODUCTION_SCHEDULER == "DPM"
    assert ns.model_revision == OFFICIAL_REPRODUCTION_REVISION == "fp16"
    assert applied["modelid_target"] == OFFICIAL_REPRODUCTION_MODEL_ID

    # Explicit user flags are respected (not overridden).
    ns2 = argparse.Namespace(
        wm_type="GS", modelid_target="mine", scheduler_target="Euler", model_revision="rev"
    )
    apply_official_reproduction_defaults(
        ns2, argv=["run_watermark.py", "--modelid_target", "mine", "--scheduler_target", "Euler", "--model_revision", "rev"]
    )
    assert (ns2.modelid_target, ns2.scheduler_target, ns2.model_revision) == ("mine", "Euler", "rev")

    # Non-GS methods are never touched.
    ns3 = argparse.Namespace(
        wm_type="TR", modelid_target="RedbeardNZ/stable-diffusion-2-1-base",
        scheduler_target="DDIM", model_revision="c6a5e9",
    )
    applied3 = apply_official_reproduction_defaults(ns3, argv=["run_watermark.py"])
    assert applied3 == {}
    assert ns3.modelid_target == "RedbeardNZ/stable-diffusion-2-1-base"
    assert ns3.scheduler_target == "DDIM"
    assert ns3.model_revision == "c6a5e9"


def test_reproduction_defaults_explicit_official_model_gets_dpm_and_fp16():
    # GS + explicit official model, no scheduler/revision -> inject DPM + fp16.
    ns = argparse.Namespace(
        wm_type="GS", modelid_target=OFFICIAL_REPRODUCTION_MODEL_ID,
        scheduler_target="DDIM", model_revision=None,
    )
    apply_official_reproduction_defaults(
        ns, argv=["run_watermark.py", "--modelid_target", OFFICIAL_REPRODUCTION_MODEL_ID]
    )
    assert ns.scheduler_target == OFFICIAL_REPRODUCTION_SCHEDULER == "DPM"
    assert ns.model_revision == OFFICIAL_REPRODUCTION_REVISION == "fp16"


def test_reproduction_defaults_explicit_nonofficial_model_gets_no_fp16_no_dpm():
    # GS + explicit NON-official model, no scheduler/revision -> must NOT attach
    # fp16 (a model-specific weight branch) nor force DPM.
    ns = argparse.Namespace(
        wm_type="GS", modelid_target="RedbeardNZ/stable-diffusion-2-1-base",
        scheduler_target="DDIM", model_revision=None,
    )
    applied = apply_official_reproduction_defaults(
        ns, argv=["run_watermark.py", "--modelid_target", "RedbeardNZ/stable-diffusion-2-1-base"]
    )
    assert ns.modelid_target == "RedbeardNZ/stable-diffusion-2-1-base"
    assert ns.scheduler_target == "DDIM"  # not forced to DPM
    assert ns.model_revision is None  # no fp16 injected
    assert applied.get("using_official_model") is False
    assert "model_revision" not in applied
    assert "scheduler_target" not in applied


def test_reproduction_defaults_do_not_override_explicit_scheduler_or_revision():
    ns = argparse.Namespace(
        wm_type="GS", modelid_target=OFFICIAL_REPRODUCTION_MODEL_ID,
        scheduler_target="Euler", model_revision="rev123",
    )
    apply_official_reproduction_defaults(
        ns,
        argv=[
            "run_watermark.py", "--modelid_target", OFFICIAL_REPRODUCTION_MODEL_ID,
            "--scheduler_target", "Euler", "--model_revision", "rev123",
        ],
    )
    assert ns.scheduler_target == "Euler"
    assert ns.model_revision == "rev123"


def test_run_watermark_parser_accepts_model_revision():
    import run_watermark
    args = run_watermark.build_parser().parse_known_args(
        ["--wm_type", "GS", "--num", "1", "--model_revision", "fp16"]
    )[0]
    assert args.model_revision == "fp16"
    assert args.wm_type == "GS"
    # default is None when omitted
    args2 = run_watermark.build_parser().parse_known_args(["--wm_type", "GS", "--num", "1"])[0]
    assert args2.model_revision is None


def test_run_removal_parser_accepts_model_revision():
    import run_removal
    args = run_removal.build_parser().parse_args(
        ["--wm_type", "GS", "--num", "1", "--model_revision", "fp16"]
    )
    assert args.model_revision == "fp16"
    assert args.wm_type == "GS"
    args2 = run_removal.parse_args(["--wm_type", "GS", "--num", "1"])
    assert args2.model_revision is None


def test_formal_generator_keeps_shared_cohort_model_scheduler_defaults():
    # The formal generator must NOT adopt the official upstream model/scheduler;
    # it keeps the shared matched cohort (RedbeardNZ + DDIM) so GS stays comparable.
    # Inspect the argparse defaults via a fresh parser build with empty argv.
    import sys as _sys

    argv_backup = _sys.argv
    try:
        _sys.argv = ["generate_watermarked_images.py"]
        args = generator_module.parse_args()
    finally:
        _sys.argv = argv_backup
    assert args.modelid_target == "RedbeardNZ/stable-diffusion-2-1-base"
    assert args.scheduler_target == "DDIM"
    # But GS defaults there are still official.
    assert args.gs_protocol_mode == "official_compatible"
    assert args.gs_detection_mode == "official_onebit"


def test_detection_fields_excluded_from_provenance_hashes():
    # Detection is a scoring-time policy; changing it must never perturb the
    # embedding-config hash or the pairing hash (so generation_config_sha256 /
    # watermark_config_sha256 / pairing_sha256 stay stable across detection changes).
    from raven.eval_protocol import PROVIDER_FIELDS_BY_METHOD
    from raven.pairing_provenance import PAIRING_HASH_FIELDS, GS_REQUIRED_FIELDS

    detection_fields = {
        "gs_detection_mode",
        "detection_threshold",
        "detection_threshold_type",
        "detection_threshold_comparison_operator",
        "threshold_calibrated_from_current_clean_negatives",
        "watermark_implementation_protocol",
        "generation_benchmark_protocol",
        "upstream_official_reproduction_runner",
        "before_detection_successful",
        "gs_official_tau_onebit",
        "gs_official_tau_bits",
    }
    assert not (detection_fields & set(PROVIDER_FIELDS_BY_METHOD["GS"]))
    assert not (detection_fields & set(PAIRING_HASH_FIELDS))
    assert not (detection_fields & set(GS_REQUIRED_FIELDS))


# ---------------------------------------------------------------------------
# Migration tests (experiments/migrate_gs_detection_metadata.py)
# ---------------------------------------------------------------------------

import csv as _csv
import json as _json


def _load_migration_module():
    path = REPO / "experiments" / "migrate_gs_detection_metadata.py"
    spec = importlib.util.spec_from_file_location("migrate_gs_detection_metadata", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


migration_module = _load_migration_module()


def _make_gs_cohort(tmp_path, n=2, *, legacy_style=True):
    """Build a small, self-contained, audit-valid official-compatible GS cohort.

    Returns (metadata_csv_path, rows). PNGs are real (tiny) files whose SHA-256 is
    recorded, so file verification and the pairing audit both pass.
    """
    method_dir = tmp_path / "GS"
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "watermark_config.json").write_text(
        _json.dumps({"message_width_in_bytes": 32, "channel_copy": 1, "hw_copy": 8})
    )
    gen_cfg_sha = canonical_json_sha256({"model_id": "RedbeardNZ/stable-diffusion-2-1-base", "steps": 50})
    wm_cfg_sha = canonical_json_sha256({"wm_type": "GS", "mode": "official_compatible"})
    mask_sha = canonical_json_sha256({"method": "GS", "mask": "not_applicable", "version": 1})

    rows = []
    for run_id in range(n):
        seed = 42 + run_id
        provider = official_provider(secret_index=run_id, sampling_seed=seed)
        res = provider.get_wm_latents()
        clean_zt = res["zT_clean_torch"]
        wm_zt = res["zT_torch"]
        base_sha = tensor_sha256(clean_zt)
        secret = res["secret_provenance_list"][0]

        clean_path = method_dir / f"{run_id:06d}_clean.png"
        wm_path = method_dir / f"{run_id:06d}_wm.png"
        Image.new("RGB", (8, 8), (run_id % 256, 0, 0)).save(clean_path)
        Image.new("RGB", (8, 8), (0, run_id % 256, 7)).save(wm_path)

        prompt = f"prompt number {run_id}"
        row = {
            "protocol": GS_PAIRING_PROTOCOL,
            "dataset_name": "unit", "dataset": "unit",
            "run_id": run_id, "prompt_id": run_id, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source": "unit", "wm_type": "GS", "wm_name": "Gaussian Shading",
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
            "scheduler_target": "DDIM", "resolution": 512,
            "threshold_calibrated_from_current_clean_negatives": "False",
            "before_detection_metric_value": "1.0",
            "base_latent_seed": seed, "generation_seed": seed,
            "base_latent_sha256": base_sha,
            "clean_base_latent_sha256": base_sha,
            "watermarked_base_latent_sha256": base_sha,
            "watermarked_latent_sha256": tensor_sha256(wm_zt),
            "watermark_target_sha256": tensor_sha256(res["barcodes_torch"]),
            "watermark_mask_sha256": mask_sha,
            "generation_config_sha256": gen_cfg_sha,
            "watermark_config_sha256": wm_cfg_sha,
            "clean_path": str(clean_path.resolve()),
            "clean_sha256": sha256_path(clean_path),
            "watermarked_path": str(wm_path.resolve()),
            "watermarked_image_path": str(wm_path.resolve()),
            "watermarked_sha256": sha256_path(wm_path),
            "gs_protocol_mode": "official_compatible",
            "gs_secret_index": run_id,
            "gs_message_sha256": secret["message_sha256"],
            "gs_key_sha256": secret["key_sha256"],
            "gs_nonce_sha256": secret["nonce_sha256"],
            "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
            "gs_sampling_seed": seed,
            "gs_sampling_uniform_sha256": res["sampling_uniform_sha256_list"][0],
            "gs_payload_layout": "channel_spatial_repeat",
            "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
            "gs_official_tau_onebit": "0.6484375",
            "gs_official_tau_bits": "0.71484375",
            "before_bit_accuracy": "1.0",
        }
        if legacy_style:
            row["detection_threshold"] = "0.70703125"
            row["detection_threshold_type"] = "legacy_default_threshold"
            row["before_detection_successful"] = "True"
        row["pairing_sha256"] = build_pairing_sha256(row)
        rows.append(row)

    csv_path = method_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path, rows


def _read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(_csv.DictReader(handle))


def test_migration_upgrades_normal_legacy_rows(tmp_path):
    csv_path, _ = _make_gs_cohort(tmp_path, n=2)
    png_before = {
        p.name: sha256_path(p) for p in csv_path.parent.glob("*.png")
    }
    summary = migration_module.migrate_metadata_file(csv_path)
    assert summary["changed"] is True
    assert summary["pairing_audit_passed"] is True
    rows = _read_rows(csv_path)
    for row in rows:
        assert row["gs_detection_mode"] == "official_onebit"
        assert row["detection_threshold"] == "0.6484375"
        assert row["detection_threshold_type"] == "official_beta_tail_tau_onebit"
        assert row["detection_threshold_comparison_operator"] == ">="
        assert row["watermark_implementation_protocol"] == "official_compatible"
        assert row["generation_benchmark_protocol"] == "shared_formal_cohort_redbeardnz_ddim"
        assert row["upstream_official_reproduction_runner"].startswith("stabilityai/")
        assert row["before_detection_successful"] == "True"  # 1.0 >= 0.6484375
    # PNGs untouched.
    png_after = {p.name: sha256_path(p) for p in csv_path.parent.glob("*.png")}
    assert png_before == png_after


def test_migration_is_idempotent(tmp_path):
    csv_path, _ = _make_gs_cohort(tmp_path, n=2)
    migration_module.migrate_metadata_file(csv_path)
    first = csv_path.read_bytes()
    summary2 = migration_module.migrate_metadata_file(csv_path)
    assert summary2["already_current"] is True
    assert summary2["changed"] is False
    assert csv_path.read_bytes() == first
    # No second backup created on the idempotent re-run.
    backups = list(csv_path.parent.glob("*.premigration.*.bak"))
    assert len(backups) == 1


def test_migration_preserves_immutable_and_pairing_hashes(tmp_path):
    csv_path, before_rows = _make_gs_cohort(tmp_path, n=2)
    before = {r["run_id"]: r for r in before_rows}
    migration_module.migrate_metadata_file(csv_path)
    for row in _read_rows(csv_path):
        b = before[int(row["run_id"])] if int(row["run_id"]) in before else before[row["run_id"]]
        for field in (
            "watermarked_latent_sha256", "generation_config_sha256",
            "watermark_config_sha256", "clean_sha256", "watermarked_sha256",
            "pairing_sha256",
        ):
            assert row[field] == str(b[field]), field


def test_migration_rejects_image_hash_mismatch(tmp_path):
    csv_path, rows = _make_gs_cohort(tmp_path, n=1)
    data = _read_rows(csv_path)
    data[0]["watermarked_sha256"] = "0" * 64
    _rewrite(csv_path, data)
    with pytest.raises(migration_module.MigrationError, match="SHA mismatch"):
        migration_module.migrate_metadata_file(csv_path)


def test_migration_rejects_secret_provenance_mismatch(tmp_path):
    csv_path, rows = _make_gs_cohort(tmp_path, n=1)
    data = _read_rows(csv_path)
    data[0]["gs_message_sha256"] = "a" * 64
    _rewrite(csv_path, data)
    with pytest.raises(migration_module.MigrationError, match="gs_message_sha256"):
        migration_module.migrate_metadata_file(csv_path)


def test_migration_rejects_latent_target_mismatch(tmp_path):
    csv_path, rows = _make_gs_cohort(tmp_path, n=1)
    data = _read_rows(csv_path)
    data[0]["watermark_target_sha256"] = "b" * 64
    # keep pairing hash consistent with the tampered value so we exercise the
    # target-provenance check rather than the pairing check.
    data[0]["pairing_sha256"] = build_pairing_sha256(data[0])
    _rewrite(csv_path, data)
    with pytest.raises(migration_module.MigrationError, match="watermark_target_sha256"):
        migration_module.migrate_metadata_file(csv_path)


def test_migration_rejects_missing_bit_accuracy(tmp_path):
    csv_path, rows = _make_gs_cohort(tmp_path, n=1)
    data = _read_rows(csv_path)
    data[0]["before_bit_accuracy"] = ""
    data[0]["before_detection_metric_value"] = ""
    _rewrite(csv_path, data)
    with pytest.raises(migration_module.MigrationError, match="no usable bit-accuracy"):
        migration_module.migrate_metadata_file(csv_path)


def test_migration_rejects_inconsistent_bit_accuracy(tmp_path):
    csv_path, rows = _make_gs_cohort(tmp_path, n=1)
    data = _read_rows(csv_path)
    data[0]["before_bit_accuracy"] = "1.0"
    data[0]["before_detection_metric_value"] = "0.5"
    _rewrite(csv_path, data)
    with pytest.raises(migration_module.MigrationError, match="inconsistent bit-accuracy"):
        migration_module.migrate_metadata_file(csv_path)


def test_migration_unifies_partial_cohort_schema(tmp_path):
    # Row 0 already migrated (has new columns), row 1 old-style (missing them),
    # written with a superset header -> migration must unify without error.
    csv_path, rows = _make_gs_cohort(tmp_path, n=2)
    data = _read_rows(csv_path)
    # Simulate row 0 already carrying the new detection columns.
    data[0]["gs_detection_mode"] = "official_onebit"
    data[0]["detection_threshold"] = "0.6484375"
    data[0]["detection_threshold_type"] = "official_beta_tail_tau_onebit"
    data[0]["detection_threshold_comparison_operator"] = ">="
    data[0]["watermark_implementation_protocol"] = "official_compatible"
    data[0]["generation_benchmark_protocol"] = "shared_formal_cohort_redbeardnz_ddim"
    data[0]["upstream_official_reproduction_runner"] = (
        "stabilityai/stable-diffusion-2-1-base+DPMSolverMultistepScheduler+fp16"
    )
    data[0]["before_detection_successful"] = "True"
    fieldnames = list(dict.fromkeys([*data[0].keys(), *data[1].keys()]))
    _rewrite(csv_path, data, fieldnames=fieldnames)
    summary = migration_module.migrate_metadata_file(csv_path)
    assert summary["pairing_audit_passed"] is True
    out = _read_rows(csv_path)
    header = set(out[0].keys())
    for field in migration_module.DETECTION_FIELDS:
        assert field in header
    for row in out:
        assert row["gs_detection_mode"] == "official_onebit"


def test_migration_dry_run_does_not_write(tmp_path):
    csv_path, _ = _make_gs_cohort(tmp_path, n=2)
    before = csv_path.read_bytes()
    summary = migration_module.migrate_metadata_file(csv_path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["changed"] is False
    assert csv_path.read_bytes() == before
    assert not list(csv_path.parent.glob("*.premigration.*.bak"))


def _rewrite(csv_path, rows, fieldnames=None):
    fieldnames = fieldnames or list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
