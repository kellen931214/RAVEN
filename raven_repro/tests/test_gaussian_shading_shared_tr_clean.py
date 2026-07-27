"""Shared-clean V2: GS embeds from the canonical Tree-Ring clean latent.

Covers latent reconstruction from the real TR cohort, the deterministic CDF
uniform derivation, the external-uniform provider path, the V2 pairing/metadata
rules and the cross-method shared-clean audit. No GPU and no image generation.
"""

import copy
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_bench_wm"))
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.pairing_provenance import (  # noqa: E402
    GS_PAIRING_PROTOCOL,
    GS_SHARED_TR_CLEAN_MODE,
    GS_SHARED_TR_CLEAN_PROTOCOL,
    GS_SHARED_CLEAN_V2_FIELDS,
    GS_UNIFORM_DERIVATION,
    GS_V2_REQUIRED_FIELDS,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    TR_PAIRING_PROTOCOL,
    audit_pairing_rows,
    audit_tr_gs_shared_clean,
    build_pairing_sha256,
    canonical_json_sha256,
    gs_fields_for_protocol,
    sha256_path,
    tensor_sha256,
)
from utils.wm import gs_provider as gs_provider_module  # noqa: E402
from utils.wm.gs_provider import GsProvider  # noqa: E402

LATENT_SHAPE = (1, 4, 64, 64)
REAL_TR_METADATA = REPO / "data" / "tr" / "diffusiondb" / "TR" / "metadata.csv"


def load_shared_clean_module():
    path = REPO / "experiments" / "generate_gs_from_tr_shared_clean.py"
    spec = importlib.util.spec_from_file_location("generate_gs_from_tr_shared_clean", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


shared_clean_module = load_shared_clean_module()


def tr_base_latent(seed: int) -> torch.Tensor:
    """The canonical Tree-Ring base-latent procedure, reproduced verbatim."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32, device="cpu")


def cdf_uniforms(base_cpu: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        norm.cdf(base_cpu.numpy().astype(np.float64)), dtype=np.float64
    )


def shared_clean_provider(secret_index: int) -> GsProvider:
    return GsProvider(
        latent_shape=LATENT_SHAPE,
        dtype=torch.float32,
        device=torch.device("cpu"),
        gs_protocol_mode=GS_SHARED_TR_CLEAN_MODE,
        gs_channel_copy=1,
        gs_hw_copy=8,
        offset=secret_index,
        gs_secret_index=secret_index,
    )


def official_seeded_provider(secret_index: int, sampling_seed: int) -> GsProvider:
    return GsProvider(
        latent_shape=LATENT_SHAPE,
        dtype=torch.float32,
        device=torch.device("cpu"),
        gs_protocol_mode="official_compatible",
        gs_channel_copy=1,
        gs_hw_copy=8,
        offset=secret_index,
        gs_secret_index=secret_index,
        gs_sampling_seed=sampling_seed,
    )


# --------------------------------------------------------------------------- #
# 1. Latent reconstruction from the real Tree-Ring cohort
# --------------------------------------------------------------------------- #


def read_real_tr_rows(limit=None):
    if not REAL_TR_METADATA.is_file():
        pytest.skip(f"real TR cohort not present: {REAL_TR_METADATA}")
    with REAL_TR_METADATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def test_rebuilt_latent_matches_real_tr_metadata_sha():
    for row in read_real_tr_rows(limit=5):
        base = tr_base_latent(int(row["base_latent_seed"]))
        actual = tensor_sha256(base)
        assert actual == row["base_latent_sha256"]
        assert actual == row["clean_base_latent_sha256"]


def test_rebuild_helper_rejects_wrong_seed():
    row = dict(read_real_tr_rows(limit=1)[0])
    row["base_latent_seed"] = str(int(row["base_latent_seed"]) + 1)
    with pytest.raises(shared_clean_module.SharedCleanError, match="base_latent_sha256"):
        shared_clean_module.rebuild_shared_clean_latent(
            torch, row, resolution=512, device=torch.device("cpu"), dtype=torch.float32
        )


def test_real_tr_clean_images_match_recorded_sha():
    for row in read_real_tr_rows(limit=3):
        assert shared_clean_module.verify_source_clean_image(row) == Path(row["clean_path"])


# --------------------------------------------------------------------------- #
# 2. CDF uniform derivation
# --------------------------------------------------------------------------- #


def test_cdf_uniforms_are_reproducible_and_hash_stably():
    base = tr_base_latent(42)
    first = cdf_uniforms(base)
    second = cdf_uniforms(tr_base_latent(42))
    assert np.array_equal(first, second)
    assert first.dtype == np.float64
    digest = hashlib.sha256(first.reshape(-1).tobytes(order="C")).hexdigest()
    assert digest == hashlib.sha256(second.reshape(-1).tobytes(order="C")).hexdigest()
    # Different run_ids must not share uniforms.
    other = cdf_uniforms(tr_base_latent(43))
    assert not np.array_equal(first, other)


def test_derive_uniforms_are_strictly_inside_unit_interval():
    uniforms = shared_clean_module.derive_uniforms(np, tr_base_latent(42))
    assert np.isfinite(uniforms).all()
    assert (uniforms > 0.0).all() and (uniforms < 1.0).all()


def test_provider_rejects_out_of_range_or_wrong_dtype_uniforms():
    base = tr_base_latent(42)
    provider = shared_clean_provider(0)
    uniforms = cdf_uniforms(base)

    broken = uniforms.copy()
    broken.reshape(-1)[0] = 0.0
    with pytest.raises(ValueError, match="strictly greater than 0"):
        provider.get_wm_latents_from_uniforms(broken, base)

    broken = uniforms.copy()
    broken.reshape(-1)[0] = 1.0
    with pytest.raises(ValueError, match="strictly less than 1"):
        provider.get_wm_latents_from_uniforms(broken, base)

    broken = uniforms.copy()
    broken.reshape(-1)[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        provider.get_wm_latents_from_uniforms(broken, base)

    with pytest.raises(ValueError, match="float64"):
        provider.get_wm_latents_from_uniforms(uniforms.astype(np.float32), base)


# --------------------------------------------------------------------------- #
# 3. External-uniform provider path
# --------------------------------------------------------------------------- #


def test_external_uniform_path_uses_no_numpy_rng(monkeypatch):
    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("shared-clean GS embedding must not consult numpy RNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "uniform", forbidden)
    monkeypatch.setattr(gs_provider_module.np.random, "default_rng", forbidden)
    monkeypatch.setattr(gs_provider_module.np.random, "uniform", forbidden)

    base = tr_base_latent(42)
    result = shared_clean_provider(0).get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    assert result["gs_protocol_mode"] == GS_SHARED_TR_CLEAN_MODE
    assert result["uniform_source"] == "externally_supplied"


def test_shared_clean_mode_refuses_the_self_sampling_entrypoint():
    with pytest.raises(RuntimeError, match="get_wm_latents_from_uniforms"):
        shared_clean_provider(0).get_wm_latents()


def test_clean_latent_is_returned_byte_identical():
    base = tr_base_latent(42)
    result = shared_clean_provider(0).get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    returned = result["zT_clean_torch"]
    assert torch.equal(returned, base)
    assert tensor_sha256(returned) == tensor_sha256(base)
    # Byte-identity here is structural, not numerical: the provider hands back the
    # very storage it was given, so no float behaviour can affect it.
    assert returned.data_ptr() == base.data_ptr()


def test_clean_latent_does_not_depend_on_the_cdf_ppf_round_trip():
    """Reconstruction via norm.ppf is NOT the contract, even where it agrees.

    Measured on this cohort the float32 CDF->PPF round trip happens to be exact
    (0 differing elements across seeds 42..141, 1.6M elements). That is an
    incidental property of float32 at these magnitudes and is not guaranteed, so
    the provider still returns the supplied latent verbatim rather than rebuilding
    it.
    """
    base = tr_base_latent(42)
    round_trip = torch.tensor(
        norm.ppf(cdf_uniforms(base)).reshape(LATENT_SHAPE), dtype=torch.float32
    )
    incidentally_equal = tensor_sha256(round_trip) == tensor_sha256(base)
    returned = shared_clean_provider(0).get_wm_latents_from_uniforms(
        cdf_uniforms(base), base
    )["zT_clean_torch"]
    # Whatever the round trip does, the returned latent is the input itself.
    assert returned.data_ptr() == base.data_ptr()
    assert tensor_sha256(returned) == tensor_sha256(base)
    assert isinstance(incidentally_equal, bool)


def test_clean_latent_dtype_mismatch_fails_closed():
    base = tr_base_latent(42)
    with pytest.raises(ValueError, match="refusing to cast"):
        shared_clean_provider(0).get_wm_latents_from_uniforms(
            cdf_uniforms(base), base.to(torch.float64)
        )


def test_secret_payload_and_cipher_match_the_official_compatible_path():
    """Only the uniforms differ; every official element must be identical."""
    base = tr_base_latent(42)
    shared = shared_clean_provider(7).get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    seeded = official_seeded_provider(7, 1234).get_wm_latents()

    assert shared["message_bits_str_list"] == seeded["message_bits_str_list"]
    assert shared["secret_provenance_list"] == seeded["secret_provenance_list"]
    assert torch.equal(shared["barcodes_torch"], seeded["barcodes_torch"])
    assert tensor_sha256(shared["barcodes_torch"]) == tensor_sha256(seeded["barcodes_torch"])
    # ...and the watermarked latents differ, because the uniforms differ.
    assert not torch.equal(shared["zT_torch"], seeded["zT_torch"])


def test_shared_clean_embedding_is_the_official_quantile_partition():
    base = tr_base_latent(42)
    provider = shared_clean_provider(3)
    uniforms = cdf_uniforms(base)
    result = provider.get_wm_latents_from_uniforms(uniforms, base)

    message, key, nonce = provider._secret_bytes(3)
    _, diffused = provider._official_payload(message)
    encrypted_bits = provider._official_encrypt_bits(diffused, key, nonce)
    expected = torch.tensor(
        norm.ppf((uniforms.reshape(-1) + encrypted_bits.astype(np.float64)) / 2.0).reshape(
            LATENT_SHAPE
        ),
        dtype=torch.float32,
    )
    assert torch.equal(result["zT_torch"], expected)
    # The sign of every element decodes back to the encrypted bit it carries.
    decoded = (result["zT_torch"] > 0).to(torch.uint8).flatten().numpy()
    assert np.array_equal(decoded, encrypted_bits)


def test_shared_clean_detector_recovers_the_payload_exactly():
    base = tr_base_latent(42)
    provider = shared_clean_provider(5)
    result = provider.get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    accuracies = provider.get_accuracies(result["zT_torch"])
    assert accuracies["bit_accuracies"] == [1.0]
    assert provider.is_detection_successful(1.0)
    info = provider.active_detection_threshold()
    assert info["threshold_type"] == "official_beta_tail_tau_onebit"
    assert info["comparison_operator"] == ">="
    assert info["calibrated_from_current_clean_negatives"] is False


def test_uniform_sha_uses_float64_c_order_bytes():
    base = tr_base_latent(42)
    result = shared_clean_provider(0).get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    expected = hashlib.sha256(
        cdf_uniforms(base).reshape(-1).tobytes(order="C")
    ).hexdigest()
    assert result["sampling_uniform_sha256_list"] == [expected]


# --------------------------------------------------------------------------- #
# 4. Synthetic TR + GS V2 cohort
# --------------------------------------------------------------------------- #


def _generation_config():
    return {
        "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
        "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
        "scheduler": "DDIM",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "resolution": 512,
        "dtype": "torch.float32",
    }


def make_cohort(tmp_path, n=2):
    """Build a matched TR cohort + shared-clean GS V2 cohort on disk."""
    clean_dir = tmp_path / "clean"
    tr_dir = tmp_path / "tr"
    gs_dir = tmp_path / "gs"
    for path in (clean_dir, tr_dir, gs_dir):
        path.mkdir(parents=True, exist_ok=True)

    gen_sha = canonical_json_sha256(_generation_config())
    tr_mask_sha = canonical_json_sha256({"method": "TR", "mask": "unit", "version": 1})
    tr_target_sha = canonical_json_sha256({"method": "TR", "target": "unit"})
    tr_cfg_sha = canonical_json_sha256({"wm_type": "TR", "unit": True})
    gs_mask_sha = canonical_json_sha256(
        {"method": "GS", "mask": "not_applicable", "version": 1}
    )
    gs_cfg_sha = canonical_json_sha256(
        {"wm_type": "GS", "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE}
    )

    tr_rows, gs_rows = [], []
    tr_metadata_path = tr_dir / "metadata.csv"
    for run_id in range(n):
        seed = 42 + run_id
        base = tr_base_latent(seed)
        base_sha = tensor_sha256(base)
        prompt = f"shared clean prompt {run_id}"
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        clean_path = clean_dir / f"{run_id:06d}.png"
        Image.new("RGB", (8, 8), (run_id % 256, 1, 2)).save(clean_path)
        tr_wm_path = tr_dir / f"{run_id:06d}_tr.png"
        Image.new("RGB", (8, 8), (3, run_id % 256, 4)).save(tr_wm_path)
        gs_wm_path = gs_dir / f"{run_id:06d}_gs.png"
        Image.new("RGB", (8, 8), (5, 6, run_id % 256)).save(gs_wm_path)
        clean_sha = sha256_path(clean_path)

        common = {
            "dataset_name": "unit", "dataset": "unit",
            "run_id": run_id, "prompt_id": run_id, "prompt": prompt,
            "prompt_sha256": prompt_sha, "source": "unit",
            "num_shards": 1, "shard_index": 0,
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
            "scheduler_target": "DDIM", "resolution": 512,
            "base_latent_seed": seed, "generation_seed": seed,
            "base_latent_sha256": base_sha,
            "clean_base_latent_sha256": base_sha,
            "watermarked_base_latent_sha256": base_sha,
            "generation_config_sha256": gen_sha,
            "clean_path": str(clean_path), "clean_sha256": clean_sha,
        }

        tr_row = {
            "protocol": TR_PAIRING_PROTOCOL, "wm_type": "TR", "wm_name": "Tree-Ring",
            **common,
            "watermarked_latent_sha256": canonical_json_sha256({"tr_wm": run_id}),
            "watermark_target_sha256": tr_target_sha,
            "watermark_mask_sha256": tr_mask_sha,
            "watermark_config_sha256": tr_cfg_sha,
            "watermarked_path": str(tr_wm_path),
            "watermarked_sha256": sha256_path(tr_wm_path),
        }
        tr_row["pairing_sha256"] = build_pairing_sha256(tr_row)
        tr_rows.append(tr_row)

        provider = shared_clean_provider(run_id)
        result = provider.get_wm_latents_from_uniforms(cdf_uniforms(base), base)
        secret = result["secret_provenance_list"][0]
        gs_row = {
            "protocol": GS_SHARED_TR_CLEAN_PROTOCOL, "wm_type": "GS",
            "wm_name": "Gaussian Shading", **common,
            "watermarked_latent_sha256": tensor_sha256(result["zT_torch"]),
            "watermark_target_sha256": tensor_sha256(result["barcodes_torch"]),
            "watermark_mask_sha256": gs_mask_sha,
            "watermark_config_sha256": gs_cfg_sha,
            "watermarked_path": str(gs_wm_path),
            "watermarked_sha256": sha256_path(gs_wm_path),
            "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
            "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
            "shared_clean_source_metadata_path": str(tr_metadata_path),
            "shared_clean_source_metadata_sha256": "0" * 64,
            "shared_clean_sample_sha256": base_sha,
            "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
            "tr_base_latent_sha256": base_sha,
            "tr_clean_path": str(clean_path),
            "tr_clean_sha256": clean_sha,
            "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
            "gs_secret_index": run_id,
            "gs_message_sha256": secret["message_sha256"],
            "gs_key_sha256": secret["key_sha256"],
            "gs_nonce_sha256": secret["nonce_sha256"],
            "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
            "gs_sampling_uniform_sha256": result["sampling_uniform_sha256_list"][0],
            "gs_payload_layout": "channel_spatial_repeat",
            "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
        }
        gs_row["pairing_sha256"] = build_pairing_sha256(gs_row)
        gs_rows.append(gs_row)

    with tr_metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tr_rows[0]))
        writer.writeheader()
        writer.writerows(tr_rows)
    return tr_rows, gs_rows, tr_metadata_path


def test_v2_cohort_passes_the_pairing_audit(tmp_path):
    _, gs_rows, _ = make_cohort(tmp_path)
    audit = audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=True)
    assert audit["protocol"] == GS_SHARED_TR_CLEAN_PROTOCOL
    assert audit["shared_clean_source_method"] == "TR"
    assert audit["gs_uniform_derivation"] == GS_UNIFORM_DERIVATION
    assert audit["unique_gs_sampling_uniform_hashes"] == len(gs_rows)
    # V2 carries no RNG sampling seed at all.
    assert audit["unique_gs_sampling_seeds"] == 0
    assert "gs_sampling_seed" not in GS_V2_REQUIRED_FIELDS


def test_tr_and_gs_share_latent_clean_path_and_clean_sha(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    for tr_row, gs_row in zip(tr_rows, gs_rows):
        assert tr_row["base_latent_sha256"] == gs_row["base_latent_sha256"]
        assert tr_row["clean_base_latent_sha256"] == gs_row["clean_base_latent_sha256"]
        assert tr_row["clean_path"] == gs_row["clean_path"]
        assert tr_row["clean_sha256"] == gs_row["clean_sha256"]
        assert tr_row["base_latent_seed"] == gs_row["base_latent_seed"]
        assert tr_row["prompt_sha256"] == gs_row["prompt_sha256"]
        assert tr_row["generation_config_sha256"] == gs_row["generation_config_sha256"]
        # ...but the watermarked images are genuinely different artifacts.
        assert tr_row["watermarked_sha256"] != gs_row["watermarked_sha256"]


def test_cross_method_audit_passes_and_reports_identity(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path, n=3)
    report = audit_tr_gs_shared_clean(tr_rows, gs_rows, verify_files=True)
    assert report["passed"] is True
    assert report["gs_rows_checked"] == 3
    assert report["unique_clean_sha256"] == 3
    assert report["unique_base_latent_sha256"] == 3
    assert report["unique_gs_watermarked_sha256"] == 3
    assert report["shared_clean_source_method"] == "TR"


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("base_latent_seed", 999, "base_latent_seed"),
        ("base_latent_sha256", "f" * 64, "base_latent_sha256"),
        ("clean_base_latent_sha256", "f" * 64, "clean_base_latent_sha256"),
        ("prompt_sha256", "f" * 64, "prompt_sha256"),
        ("clean_path", "/nonexistent/other.png", "clean_path"),
        ("clean_sha256", "f" * 64, "clean_sha256"),
        ("generation_config_sha256", "f" * 64, "generation_config_sha256"),
    ],
)
def test_cross_method_audit_catches_drift(tmp_path, field, value, match):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    broken = copy.deepcopy(gs_rows)
    broken[1][field] = value
    with pytest.raises(ValueError, match=match):
        audit_tr_gs_shared_clean(tr_rows, broken, verify_files=True)


def test_cross_method_audit_rejects_unmatched_run_id(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    with pytest.raises(ValueError, match="no matching TR source row"):
        audit_tr_gs_shared_clean(tr_rows[:1], gs_rows, verify_files=True)


def test_cross_method_audit_rejects_v1_cohort(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    gs_rows[0]["protocol"] = GS_PAIRING_PROTOCOL
    with pytest.raises(ValueError, match="requires"):
        audit_tr_gs_shared_clean(tr_rows, gs_rows, verify_files=False)


def test_cross_method_audit_detects_missing_or_tampered_files(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    Path(gs_rows[0]["watermarked_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA drift"):
        audit_tr_gs_shared_clean(tr_rows, gs_rows, verify_files=True)


@pytest.mark.parametrize("field", GS_SHARED_CLEAN_V2_FIELDS)
def test_v2_missing_shared_field_fails_closed(tmp_path, field):
    _, gs_rows, _ = make_cohort(tmp_path)
    for row in gs_rows:
        row[field] = ""
    with pytest.raises(ValueError, match=field):
        audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=False)


def test_v2_self_certification_is_impossible(tmp_path):
    """Editing only the GS-side mirror fields must not produce a passing audit."""
    _, gs_rows, _ = make_cohort(tmp_path)
    gs_rows[0]["tr_clean_sha256"] = "f" * 64
    gs_rows[0]["pairing_sha256"] = build_pairing_sha256(gs_rows[0])
    with pytest.raises(ValueError, match="clean image SHA mismatch"):
        audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=False)


def test_v2_pairing_hash_binds_shared_clean_identity(tmp_path):
    _, gs_rows, _ = make_cohort(tmp_path)
    row = gs_rows[0]
    baseline = build_pairing_sha256(row)
    for field in (
        "shared_clean_protocol",
        "shared_clean_source_metadata_sha256",
        "shared_clean_sample_sha256",
        "gs_uniform_derivation",
        "tr_base_latent_sha256",
        "tr_clean_sha256",
    ):
        mutated = dict(row)
        mutated[field] = "mutated"
        assert build_pairing_sha256(mutated) != baseline, field


def test_v2_pairing_hash_is_path_independent(tmp_path):
    """A canonical-layout move must not invalidate an audited cohort."""
    _, gs_rows, _ = make_cohort(tmp_path)
    row = dict(gs_rows[0])
    baseline = build_pairing_sha256(row)
    row["clean_path"] = "/moved/clean.png"
    row["tr_clean_path"] = "/moved/clean.png"
    row["shared_clean_source_metadata_path"] = "/moved/metadata.csv"
    row["watermarked_path"] = "/moved/wm.png"
    assert build_pairing_sha256(row) == baseline


def test_v2_requires_the_shared_clean_provider_mode(tmp_path):
    _, gs_rows, _ = make_cohort(tmp_path)
    for row in gs_rows:
        row["gs_protocol_mode"] = "official_compatible"
        row["pairing_sha256"] = build_pairing_sha256(row)
    with pytest.raises(ValueError, match="gs_protocol_mode"):
        audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=False)


def test_protocol_field_sets_are_distinct_and_fail_closed():
    assert gs_fields_for_protocol(GS_PAIRING_PROTOCOL) != gs_fields_for_protocol(
        GS_SHARED_TR_CLEAN_PROTOCOL
    )
    assert "gs_sampling_seed" in gs_fields_for_protocol(GS_PAIRING_PROTOCOL)
    with pytest.raises(ValueError, match="unsupported GS pairing protocol"):
        gs_fields_for_protocol("gaussian_shading_made_up_v9")


def test_provider_and_protocol_agree_on_the_mode_name():
    assert gs_provider_module.GS_SHARED_TR_CLEAN_MODE == GS_SHARED_TR_CLEAN_MODE


# --------------------------------------------------------------------------- #
# 5. Generator behaviour: no clean regeneration, resume, fail-closed metadata
# --------------------------------------------------------------------------- #


def test_generator_never_writes_clean_images():
    """The script has no clean-image generation or copy path at all."""
    source = (REPO / "experiments" / "generate_gs_from_tr_shared_clean.py").read_text()
    for forbidden in ("generate_clean", "shutil.copy", "clean_image.save", "clean_output_dir"):
        assert forbidden not in source
    assert "clean_images_generated" in source


def test_generator_refuses_to_overwrite_existing_metadata(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("run_id\n0\n", encoding="utf-8")
    with pytest.raises(shared_clean_module.SharedCleanError, match="--resume"):
        shared_clean_module.existing_completed_rows(path, resume=False)


def test_generator_resume_rejects_provenance_drift(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    tr_row, gs_row = tr_rows[0], gs_rows[0]
    base = tr_base_latent(int(tr_row["base_latent_seed"]))
    provider = shared_clean_provider(int(tr_row["run_id"]))
    result = provider.get_wm_latents_from_uniforms(cdf_uniforms(base), base)

    shared_clean_module.validate_resume_row(
        gs_row,
        run_id=int(tr_row["run_id"]),
        tr_row=tr_row,
        wm_results=result,
        base_latent_sha256=tensor_sha256(base),
    )
    tampered = dict(gs_row)
    tampered["gs_sampling_uniform_sha256"] = "f" * 64
    with pytest.raises(shared_clean_module.SharedCleanError, match="resume mismatch"):
        shared_clean_module.validate_resume_row(
            tampered,
            run_id=int(tr_row["run_id"]),
            tr_row=tr_row,
            wm_results=result,
            base_latent_sha256=tensor_sha256(base),
        )


def test_generator_resume_rejects_missing_or_changed_output(tmp_path):
    tr_rows, gs_rows, _ = make_cohort(tmp_path)
    tr_row, gs_row = tr_rows[0], gs_rows[0]
    base = tr_base_latent(int(tr_row["base_latent_seed"]))
    result = shared_clean_provider(0).get_wm_latents_from_uniforms(cdf_uniforms(base), base)
    Path(gs_row["watermarked_path"]).write_bytes(b"changed")
    with pytest.raises(shared_clean_module.SharedCleanError, match="SHA drift"):
        shared_clean_module.validate_resume_row(
            gs_row,
            run_id=0,
            tr_row=tr_row,
            wm_results=result,
            base_latent_sha256=tensor_sha256(base),
        )


def test_shard_selection_is_disjoint_and_stable(tmp_path):
    tr_rows, _, _ = make_cohort(tmp_path, n=4)
    shards = [
        [
            int(row["run_id"])
            for row in shared_clean_module.select_rows(
                tr_rows, num_shards=2, shard_index=index, run_ids=None, limit=None
            )
        ]
        for index in (0, 1)
    ]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == [0, 1, 2, 3]


def test_run_id_selection_fails_closed_on_missing_id(tmp_path):
    tr_rows, _, _ = make_cohort(tmp_path, n=2)
    with pytest.raises(shared_clean_module.SharedCleanError, match="not present"):
        shared_clean_module.select_rows(
            tr_rows, num_shards=1, shard_index=0, run_ids=[99], limit=None
        )


def test_append_row_rejects_schema_drift(tmp_path):
    path = tmp_path / "metadata.csv"
    shared_clean_module.append_row(path, {"a": 1, "b": 2})
    with pytest.raises(shared_clean_module.SharedCleanError, match="schema mismatch"):
        shared_clean_module.append_row(path, {"a": 1, "c": 3})


# --------------------------------------------------------------------------- #
# 6. V1 cohorts stay valid and unrelabelled
# --------------------------------------------------------------------------- #


def test_real_v1_gs_cohort_still_audits_as_v1():
    path = REPO / "data" / "gs" / "gs_diffusiondb_1001_match_tr" / "GS" / "metadata.csv"
    if not path.is_file():
        pytest.skip(f"V1 GS cohort not present: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(rows, expected_count=len(rows), verify_files=False)
    assert audit["protocol"] == GS_PAIRING_PROTOCOL
    assert audit["unique_gs_sampling_seeds"] == len(rows)
    assert "shared_clean_protocol" not in audit
    for row in rows[:5]:
        assert row["gs_protocol_mode"] == "official_compatible"
        assert row["pairing_sha256"] == build_pairing_sha256(row)


def test_real_tr_cohort_pairing_hashes_are_unchanged():
    for row in read_real_tr_rows(limit=25):
        assert row["pairing_sha256"] == build_pairing_sha256(row)


def test_mixed_protocol_cohort_is_rejected(tmp_path):
    _, gs_rows, _ = make_cohort(tmp_path)
    gs_rows[0]["protocol"] = GS_PAIRING_PROTOCOL
    with pytest.raises(ValueError):
        audit_pairing_rows(gs_rows, expected_count=len(gs_rows), verify_files=False)
