"""Focused tests for the GM / T2S ``shared_tr_clean_v2`` runners (Issue #9).

These prove exactly the six facts Issue #9 asks for and nothing else:

1. GM consumes the supplied canonical base latent and draws no replacement.
2. T2S consumes the supplied canonical base latent and draws no replacement.
3. A wrong seed / latent SHA / prompt SHA / clean SHA / generation config fails closed.
4. The canonical clean bytes and SHA are unchanged before and after generation.
5. Resume rejects source, bundle/state, configuration or output drift without
   modifying prior artifacts.
6. The cross-method audit validates TR/GS/GM/T2S and detects missing, duplicate
   and drifted rows.

Everything runs on CPU with a stub diffusion pipeline and a synthetic two-row
Tree-Ring cohort. No network access and no GPU are required.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO / "eval_bench_wm", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from raven.pairing_provenance import (  # noqa: E402
    GM_SHARED_TR_CLEAN_MODE,
    GM_SHARED_TR_CLEAN_PROTOCOL,
    GM_UNIFORM_DERIVATION,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    T2S_SHARED_TR_CLEAN_MODE,
    T2S_SHARED_TR_CLEAN_PROTOCOL,
    TR_PAIRING_PROTOCOL,
    audit_pairing_rows,
    audit_shared_clean_cohorts,
    build_pairing_sha256,
    canonical_json_sha256,
    sha256_path,
    tensor_sha256,
)

import shared_clean_tr  # noqa: E402
from shared_clean_tr import (  # noqa: E402
    CleanImageGuard,
    SharedCleanError,
    rebuild_shared_clean_latent,
    verify_generation_config,
    verify_source_clean_image,
    verify_source_prompt,
)

gm_runner = importlib.import_module("generate_gm_from_tr_shared_clean")
t2s_runner = importlib.import_module("generate_t2s_from_tr_shared_clean")

MODEL_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
RESOLUTION = 512
LATENT_SHAPE = (1, 4, RESOLUTION // 8, RESOLUTION // 8)
GENERATION_CONFIG = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "scheduler": "DDIM",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "resolution": RESOLUTION,
    "dtype": str(torch.float32),
}
GENERATION_CONFIG_SHA256 = canonical_json_sha256(GENERATION_CONFIG)


# --------------------------------------------------------------------------- #
# Synthetic canonical Tree-Ring cohort
# --------------------------------------------------------------------------- #

def _deterministic_image(tag: str) -> Image.Image:
    rng = np.random.RandomState(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
    return Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8))


def _base_latent(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32, device="cpu")


def build_tr_cohort(root: Path, seeds=(42, 43)) -> Path:
    """A minimal but fully valid TR cohort that ``audit_pairing_rows`` accepts."""
    clean_dir = root / "clean"
    tr_dir = root / "tr"
    clean_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)

    # One cohort-wide Tree-Ring target/mask, as the real TR protocol has.
    target_sha256 = canonical_json_sha256({"tr": "ring", "version": 1})
    mask_sha256 = canonical_json_sha256({"tr": "circle", "version": 1})
    watermark_config_sha256 = canonical_json_sha256({"wm_type": "TR", "version": 1})

    rows = []
    for run_id, seed in enumerate(seeds):
        base = _base_latent(seed)
        base_sha = tensor_sha256(base)
        prompt = f"a synthetic prompt for run {run_id}"
        clean_path = clean_dir / f"{run_id:06d}.png"
        _deterministic_image(f"clean-{run_id}").save(clean_path)
        item_dir = tr_dir / f"{run_id:06d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        wm_path = item_dir / "watermarked.png"
        _deterministic_image(f"tr-wm-{run_id}").save(wm_path)

        row = {
            "protocol": TR_PAIRING_PROTOCOL,
            "dataset_name": "synthetic",
            "dataset": "synthetic",
            "run_id": run_id,
            "prompt_id": run_id,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source": "synthetic",
            "wm_type": "TR",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "scheduler_target": "DDIM",
            "num_inference_steps_target": 50,
            "guidance_scale_target": 7.5,
            "resolution": RESOLUTION,
            "base_latent_seed": seed,
            "base_latent_sha256": base_sha,
            "clean_base_latent_sha256": base_sha,
            "watermarked_base_latent_sha256": base_sha,
            "watermarked_latent_sha256": tensor_sha256(base + 1.0),
            "watermark_target_sha256": target_sha256,
            "watermark_mask_sha256": mask_sha256,
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "watermark_config_sha256": watermark_config_sha256,
            "clean_path": str(clean_path.resolve()),
            "clean_sha256": sha256_path(clean_path),
            "watermarked_path": str(wm_path.resolve()),
            "watermarked_image_path": str(wm_path.resolve()),
            "watermarked_sha256": sha256_path(wm_path),
        }
        row["pairing_sha256"] = build_pairing_sha256(row)
        rows.append(row)

    metadata = tr_dir / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return metadata


def read_rows(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Stub pipeline
# --------------------------------------------------------------------------- #

class StubPipeProvider:
    """Deterministic stand-in for the diffusers pipeline.

    The image is a function of the watermarked latent, so two different latents
    always yield two different images — which is what the uniqueness gates in the
    audit actually test.
    """

    def __init__(self):
        self.calls = []

    def get_latent_shape(self):
        return LATENT_SHAPE

    def get_dtype(self):
        return torch.float32

    def generate(self, prompts, latents, num_inference_steps, guidance_scale):
        digest = tensor_sha256(latents)
        self.calls.append({"prompt": prompts, "latent_sha256": digest})
        return {"images_PIL": [_deterministic_image(digest)]}

    def stash_pipe(self):
        return None


class DummyGuard:
    def check(self, _label):
        return None


@pytest.fixture
def stub_pipeline(monkeypatch):
    from utils.pipe import pipe_utils

    provider = StubPipeProvider()
    monkeypatch.setattr(pipe_utils, "get_pipe_provider", lambda **kwargs: provider)
    return provider


# --------------------------------------------------------------------------- #
# Runner drivers
# --------------------------------------------------------------------------- #

def gm_args(tmp_path: Path, tr_metadata: Path, **overrides):
    argv = [
        "--tr-metadata", str(tr_metadata),
        "--output-dir", str(tmp_path / "gm"),
        "--dataset-name", "synthetic",
        "--device", "cpu",
        "--gm-bundle-dir", str(overrides.pop("bundle_dir", tmp_path / "gm_bundle")),
        "--create-bundle", str(overrides.pop("create_bundle", True)),
        "--gm-watermark-bits-seed", "7",
        "--smoke-only", "true",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return gm_runner.parse_args(argv)


def t2s_args(tmp_path: Path, tr_metadata: Path, **overrides):
    argv = [
        "--tr-metadata", str(tr_metadata),
        "--output-dir", str(tmp_path / "t2s"),
        "--dataset-name", "synthetic",
        "--device", "cpu",
        "--smoke-only", "true",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return t2s_runner.parse_args(argv)


def run_gm(tmp_path, tr_metadata, **overrides):
    args = gm_args(tmp_path, tr_metadata, **overrides)
    return gm_runner.run(args, DummyGuard(), torch.device("cpu"))


def run_t2s(tmp_path, tr_metadata, **overrides):
    args = t2s_args(tmp_path, tr_metadata, **overrides)
    return t2s_runner.run(args, DummyGuard(), torch.device("cpu"))


@pytest.fixture
def cohort(tmp_path):
    return build_tr_cohort(tmp_path / "source")


# --------------------------------------------------------------------------- #
# 1. GM consumes the supplied canonical base latent
# --------------------------------------------------------------------------- #

def test_gm_provider_consumes_supplied_latent_and_draws_no_replacement(monkeypatch, tmp_path):
    from utils.wm.gm_provider import GmProvider

    provider = GmProvider(
        latent_shape=LATENT_SHAPE,
        device=torch.device("cpu"),
        gm_profile="legacy",
        gm_bundle_dir=str(tmp_path / "bundle"),
        gm_create_bundle=True,
        gm_allow_in_memory_state=False,
        gm_torch_dtype="float32",
        gm_watermark_bits_seed=7,
        gm_use_gnr=False,
        gm_gnr_path=None,
        gm_use_classifier=False,
        gm_classifier_path=None,
        modelid_target=MODEL_ID,
        model_revision=MODEL_REVISION,
        scheduler_target="DDIM",
        resolution=RESOLUTION,
    )

    base = _base_latent(42)
    base_sha = tensor_sha256(base)

    # Any RNG draw at all would be a replacement of the supplied latent.
    from scipy.stats import truncnorm

    monkeypatch.setattr(
        truncnorm, "rvs", lambda *a, **k: pytest.fail("GM drew a replacement latent")
    )
    monkeypatch.setattr(
        np.random, "RandomState", lambda *a, **k: pytest.fail("GM consulted the legacy RNG")
    )

    result = provider.get_wm_latents_from_base_latent(base)

    # Same storage, same SHA: the clean side is literally the supplied tensor.
    assert result["zT_clean_torch"].data_ptr() == base.data_ptr()
    assert tensor_sha256(result["zT_clean_torch"]) == base_sha
    assert result["gm_protocol_mode"] == GM_SHARED_TR_CLEAN_MODE
    assert result["gm_uniform_derivation"] == GM_UNIFORM_DERIVATION
    assert tensor_sha256(result["zT_torch"]) != base_sha

    # The pre-injection latent is the quantile partition of norm.cdf(base).
    from scipy.stats import norm

    pre = result["gm_pre_frequency_latent"].numpy().astype(np.float64)
    bits = provider.m_flat.astype(np.float64).reshape(pre.shape)
    assert np.array_equal((pre > 0).astype(np.float64), bits)
    recovered = 2.0 * norm.cdf(pre) - bits
    expected = norm.cdf(base.numpy().astype(np.float64))
    assert float(np.max(np.abs(recovered - expected))) < gm_runner.UNIFORM_ROUNDTRIP_TOLERANCE

    # The bound really is an identity check: an unrelated latent is orders of
    # magnitude away from it, so it can never be satisfied by accident.
    unrelated = norm.cdf(_base_latent(43).numpy().astype(np.float64))
    assert float(np.max(np.abs(recovered - unrelated))) > 0.1

    # A different base latent must produce a different watermarked latent.
    other = provider.get_wm_latents_from_base_latent(_base_latent(43))
    assert tensor_sha256(other["zT_torch"]) != tensor_sha256(result["zT_torch"])


def test_gm_rejects_wrong_shape_dtype_and_non_finite_latents(tmp_path):
    from utils.wm.gm_provider import GmProvider

    provider = GmProvider(
        latent_shape=LATENT_SHAPE,
        device=torch.device("cpu"),
        gm_profile="legacy",
        gm_bundle_dir=str(tmp_path / "bundle"),
        gm_create_bundle=True,
        gm_allow_in_memory_state=False,
        gm_torch_dtype="float32",
        gm_watermark_bits_seed=7,
        gm_use_gnr=False,
        gm_gnr_path=None,
        gm_use_classifier=False,
        gm_classifier_path=None,
        modelid_target=MODEL_ID,
        model_revision=MODEL_REVISION,
        scheduler_target="DDIM",
        resolution=RESOLUTION,
    )
    with pytest.raises(ValueError, match="expected"):
        provider.get_wm_latents_from_base_latent(torch.randn(1, 4, 32, 32))
    with pytest.raises(ValueError, match="float32"):
        provider.get_wm_latents_from_base_latent(_base_latent(42).half())
    broken = _base_latent(42).clone()
    broken[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        provider.get_wm_latents_from_base_latent(broken)


# --------------------------------------------------------------------------- #
# 2. T2S consumes the supplied canonical base latent
# --------------------------------------------------------------------------- #

def test_t2s_provider_consumes_supplied_latent_and_draws_no_replacement(monkeypatch):
    from utils.wm import t2s_provider as t2s_module
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(
        latent_shape=LATENT_SHAPE,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    base = _base_latent(42)
    base_sha = tensor_sha256(base)
    second_base = _base_latent(43)  # built before torch.randn is disarmed

    # torch.randn is exactly the "draw a replacement latent" call.
    monkeypatch.setattr(
        t2s_module.torch, "randn", lambda *a, **k: pytest.fail("T2S drew a replacement latent")
    )

    latents, state = provider.new_sample(sample_seed=42, base_latent=base)

    assert state.base_latent_sha256 == base_sha
    assert tensor_sha256(latents) != base_sha
    # T2S only reorders and re-signs its source's magnitudes.
    assert t2s_module.abs_magnitude_multiset_sha256(latents) == (
        t2s_module.abs_magnitude_multiset_sha256(base)
    )

    other, _ = provider.new_sample(sample_seed=43, base_latent=second_base)
    assert tensor_sha256(other) != tensor_sha256(latents)


def test_t2s_rejects_wrong_shape_and_a_substituted_latent(monkeypatch):
    from utils.wm import t2s_provider as t2s_module
    from utils.wm.t2s_provider import T2SProvider

    provider = T2SProvider(
        latent_shape=LATENT_SHAPE, dtype=torch.float32, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="expected"):
        provider.new_sample(sample_seed=1, base_latent=torch.randn(1, 4, 32, 32))

    # If the encoder silently ignored the supplied z, the magnitude gate fires.
    original_encode = t2s_module.T2SMark.encode

    def ignoring_encode(self, bits, key, generator=None, z=None):
        return original_encode(self, bits, key, generator=generator, z=None)

    monkeypatch.setattr(t2s_module.T2SMark, "encode", ignoring_encode)
    with pytest.raises(ValueError, match="did not consume the supplied base latent"):
        provider.new_sample(sample_seed=1, base_latent=_base_latent(42))


# --------------------------------------------------------------------------- #
# 3. Wrong seed / SHA / prompt / clean / generation config fails closed
# --------------------------------------------------------------------------- #

def test_rebuild_rejects_a_wrong_seed_or_latent_sha(cohort):
    row = read_rows(cohort)[0]

    wrong_seed = dict(row, base_latent_seed="999")
    with pytest.raises(SharedCleanError, match="base_latent_sha256"):
        rebuild_shared_clean_latent(
            torch, wrong_seed, resolution=RESOLUTION, device=torch.device("cpu"),
            dtype=torch.float32,
        )

    wrong_sha = dict(row, clean_base_latent_sha256="0" * 64)
    with pytest.raises(SharedCleanError, match="clean_base_latent_sha256"):
        rebuild_shared_clean_latent(
            torch, wrong_sha, resolution=RESOLUTION, device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_prompt_clean_and_generation_config_fail_closed(cohort, tmp_path):
    rows = read_rows(cohort)
    row = rows[0]

    with pytest.raises(SharedCleanError, match="prompt hash mismatch"):
        verify_source_prompt(dict(row, prompt="a different prompt"))

    with pytest.raises(SharedCleanError, match="clean image SHA drift"):
        verify_source_clean_image(dict(row, clean_sha256="0" * 64))

    missing = tmp_path / "nope.png"
    with pytest.raises(SharedCleanError, match="clean image missing"):
        verify_source_clean_image(dict(row, clean_path=str(missing)))

    with pytest.raises(SharedCleanError, match="generation config does not match"):
        verify_generation_config(rows, dict(GENERATION_CONFIG, guidance_scale=3.0))
    # The matching configuration is accepted and returns the TR hash.
    assert verify_generation_config(rows, GENERATION_CONFIG) == GENERATION_CONFIG_SHA256


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_runner_fails_closed_on_a_drifted_source_row(cohort, tmp_path, stub_pipeline, runner):
    """A tampered clean SHA in the source metadata stops the run before writing."""
    rows = read_rows(cohort)
    rows[0]["clean_sha256"] = "0" * 64
    tampered = tmp_path / "tampered_metadata.csv"
    with tampered.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    driver = run_gm if runner == "gm" else run_t2s
    with pytest.raises(Exception):
        driver(tmp_path, tampered)
    assert not (tmp_path / runner / "metadata.csv").exists()


# --------------------------------------------------------------------------- #
# 4. Canonical clean bytes and SHA are unchanged before/after generation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_generation_leaves_the_canonical_clean_images_untouched(
    cohort, tmp_path, stub_pipeline, runner
):
    tr_rows = read_rows(cohort)
    before = {
        row["clean_path"]: (
            sha256_path(Path(row["clean_path"])),
            Path(row["clean_path"]).stat().st_mtime_ns,
            Path(row["clean_path"]).read_bytes(),
        )
        for row in tr_rows
    }
    tr_metadata_before = sha256_path(cohort)

    summary = (run_gm if runner == "gm" else run_t2s)(tmp_path, cohort)

    assert summary["clean_images_generated"] == 0
    assert summary["clean_images_copied"] == 0
    assert summary["clean_images_verified_unchanged"] == len(tr_rows)
    for path, (sha, mtime, data) in before.items():
        current = Path(path)
        assert sha256_path(current) == sha
        assert current.stat().st_mtime_ns == mtime
        assert current.read_bytes() == data
    # The TR metadata and TR watermarked images are read-only too.
    assert sha256_path(cohort) == tr_metadata_before
    for row in tr_rows:
        assert sha256_path(Path(row["watermarked_path"])) == row["watermarked_sha256"]

    # Only method-specific watermarked images were produced.
    produced = sorted(p.name for p in (tmp_path / runner).rglob("*.png"))
    assert produced == ["watermarked.png", "watermarked.png"]


def test_clean_image_guard_detects_modification(tmp_path):
    path = tmp_path / "clean.png"
    _deterministic_image("guard").save(path)
    guard = CleanImageGuard()
    guard.snapshot(path, expected_sha256=sha256_path(path))
    guard.assert_unchanged(path)

    _deterministic_image("guard-modified").save(path)
    with pytest.raises(SharedCleanError, match="was modified"):
        guard.assert_unchanged(path)


# --------------------------------------------------------------------------- #
# Shared-clean identities on the produced rows
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "runner,protocol,mode_field,mode",
    [
        ("gm", GM_SHARED_TR_CLEAN_PROTOCOL, "gm_protocol_mode", GM_SHARED_TR_CLEAN_MODE),
        ("t2s", T2S_SHARED_TR_CLEAN_PROTOCOL, "t2s_protocol_mode", T2S_SHARED_TR_CLEAN_MODE),
    ],
)
def test_rows_bind_the_mandatory_tr_identities(
    cohort, tmp_path, stub_pipeline, runner, protocol, mode_field, mode
):
    (run_gm if runner == "gm" else run_t2s)(tmp_path, cohort)
    tr_by_id = {row["run_id"]: row for row in read_rows(cohort)}
    rows = read_rows(tmp_path / runner / "metadata.csv")
    assert len(rows) == len(tr_by_id)

    for row in rows:
        tr = tr_by_id[row["run_id"]]
        assert row["protocol"] == protocol
        assert row[mode_field] == mode
        assert row["shared_clean_protocol"] == SHARED_CLEAN_PROTOCOL
        assert row["shared_clean_source_method"] == SHARED_CLEAN_SOURCE_METHOD
        # Issue #9 mandatory identities.
        assert row["prompt_sha256"] == tr["prompt_sha256"]
        assert row["base_latent_seed"] == tr["base_latent_seed"]
        assert row["base_latent_sha256"] == tr["base_latent_sha256"]
        assert row["clean_base_latent_sha256"] == tr["clean_base_latent_sha256"]
        assert row["watermark_pre_injection_base_latent_sha256"] == tr["base_latent_sha256"]
        assert row["clean_path"] == tr["clean_path"]
        assert row["clean_sha256"] == tr["clean_sha256"]
        assert row["generation_config_sha256"] == tr["generation_config_sha256"]
        # The watermarked artifacts must differ from the clean/base ones.
        assert row["watermarked_latent_sha256"] != tr["base_latent_sha256"]
        assert row["watermarked_sha256"] != tr["clean_sha256"]
        assert row["watermarked_sha256"] != tr["watermarked_sha256"]
        assert row["smoke_only"] == "True"
        assert row["formal_output_eligible"] == "False"

    # Independent samples: distinct latents and distinct images.
    assert len({row["watermarked_latent_sha256"] for row in rows}) == len(rows)
    assert len({row["watermarked_sha256"] for row in rows}) == len(rows)


def test_gm_and_t2s_use_the_authoritative_providers(cohort, tmp_path, stub_pipeline):
    from utils.wm import gm_provider, t2s_provider

    run_gm(tmp_path, cohort)
    run_t2s(tmp_path, cohort)
    gm_row = read_rows(tmp_path / "gm" / "metadata.csv")[0]
    t2s_row = read_rows(tmp_path / "t2s" / "metadata.csv")[0]

    assert gm_row["gm_provider_entrypoint_sha256"] == sha256_path(Path(gm_provider.__file__))
    assert t2s_row["t2s_provider_entrypoint_sha256"] == sha256_path(
        Path(t2s_provider.__file__)
    )
    assert Path(gm_row["gm_provider_entrypoint_path"]).name == "gm_provider.py"
    assert Path(t2s_row["t2s_provider_entrypoint_path"]).name == "t2s_provider.py"


def test_t2s_writes_a_portable_state_artifact(cohort, tmp_path, stub_pipeline):
    from utils.wm.t2s_provider import T2SWatermarkState

    run_t2s(tmp_path, cohort)
    for row in read_rows(tmp_path / "t2s" / "metadata.csv"):
        state = T2SWatermarkState.load(Path(row["t2s_state_path"]))
        assert state.state_sha256() == row["t2s_state_sha256"]
        assert state.base_latent_sha256 == row["base_latent_sha256"]
        assert state.image_sha256 == row["watermarked_sha256"]
        assert state.rng_mode == row["t2s_rng_mode"]


def test_gm_requires_a_persisted_bundle(cohort, tmp_path, stub_pipeline):
    from utils.wm.gm_bundle import GmBundleError

    with pytest.raises(GmBundleError, match="no GM bundle"):
        run_gm(tmp_path, cohort, create_bundle=False)


# --------------------------------------------------------------------------- #
# 5. Resume is deterministic and fail closed
# --------------------------------------------------------------------------- #

def _artifact_state(root: Path):
    return {
        str(p.relative_to(root)): sha256_path(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_resume_verifies_and_skips_without_rewriting(cohort, tmp_path, stub_pipeline, runner):
    driver = run_gm if runner == "gm" else run_t2s
    first = driver(tmp_path, cohort)
    assert first["rows_written_this_run"] == 2
    before = _artifact_state(tmp_path / runner)
    rows_before = read_rows(tmp_path / runner / "metadata.csv")

    extra = {"bundle_dir": tmp_path / "gm_bundle", "create_bundle": False} if runner == "gm" else {}
    args = (gm_args if runner == "gm" else t2s_args)(tmp_path, cohort, **extra)
    args.resume = True
    second = (gm_runner if runner == "gm" else t2s_runner).run(
        args, DummyGuard(), torch.device("cpu")
    )

    assert second["rows_written_this_run"] == 0
    assert second["rows_verified_and_skipped"] == 2
    # Nothing was regenerated: the metadata and every produced image are byte
    # identical (only the run's own report JSONs may be rewritten).
    assert read_rows(tmp_path / runner / "metadata.csv") == rows_before
    after = _artifact_state(tmp_path / runner)
    for name, digest in before.items():
        if name.endswith("watermarked.png") or name.endswith("metadata.csv"):
            assert after[name] == digest


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_resume_rejects_output_drift(cohort, tmp_path, stub_pipeline, runner):
    driver = run_gm if runner == "gm" else run_t2s
    driver(tmp_path, cohort)
    victim = sorted((tmp_path / runner).rglob("watermarked.png"))[0]
    _deterministic_image("tampered").save(victim)

    extra = {"bundle_dir": tmp_path / "gm_bundle", "create_bundle": False} if runner == "gm" else {}
    args = (gm_args if runner == "gm" else t2s_args)(tmp_path, cohort, **extra)
    args.resume = True
    with pytest.raises(Exception, match="SHA drift"):
        (gm_runner if runner == "gm" else t2s_runner).run(
            args, DummyGuard(), torch.device("cpu")
        )


def test_resume_rejects_a_different_gm_bundle(cohort, tmp_path, stub_pipeline):
    """An existing cohort may never trigger bundle creation, and a foreign bundle
    may never be substituted for the one the cohort was generated with."""
    from utils.wm.gm_bundle import GmBundleError

    run_gm(tmp_path, cohort)
    rows_before = read_rows(tmp_path / "gm" / "metadata.csv")

    # 1. Resuming into a different, non-existent bundle path stops before any
    #    file is written — --create-bundle is ignored once a cohort exists.
    other = tmp_path / "gm_bundle_other"
    args = gm_args(tmp_path, cohort, bundle_dir=other, create_bundle=True)
    args.resume = True
    with pytest.raises(GmBundleError, match="no GM bundle"):
        gm_runner.run(args, DummyGuard(), torch.device("cpu"))
    assert not other.exists()
    assert read_rows(tmp_path / "gm" / "metadata.csv") == rows_before

    # 2. A complete but *different* bundle (different ChaCha20 state and ring
    #    target) is rejected by the run-manifest gate, which binds the bundle
    #    config SHA, before any row is regenerated.
    foreign = tmp_path / "gm_bundle_foreign"
    build = gm_args(tmp_path / "throwaway", cohort, bundle_dir=foreign, create_bundle=True)
    build.gm_watermark_bits_seed = 999
    gm_runner.run(build, DummyGuard(), torch.device("cpu"))

    args = gm_args(tmp_path, cohort, bundle_dir=foreign, create_bundle=False)
    args.resume = True
    with pytest.raises(SharedCleanError, match="Nothing was modified"):
        gm_runner.run(args, DummyGuard(), torch.device("cpu"))
    assert read_rows(tmp_path / "gm" / "metadata.csv") == rows_before


def test_resume_rejects_a_different_t2s_profile(cohort, tmp_path, stub_pipeline):
    run_t2s(tmp_path, cohort)
    rows_before = read_rows(tmp_path / "t2s" / "metadata.csv")

    # The provider config SHA is bound into the run manifest, so a different RNG
    # profile is rejected before any row is regenerated.
    args = t2s_args(tmp_path, cohort, t2s_rng_mode="raven_deterministic")
    args.resume = True
    with pytest.raises(SharedCleanError, match="Nothing was modified"):
        t2s_runner.run(args, DummyGuard(), torch.device("cpu"))
    assert read_rows(tmp_path / "t2s" / "metadata.csv") == rows_before


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_a_second_run_without_resume_refuses_to_touch_an_existing_cohort(
    cohort, tmp_path, stub_pipeline, runner
):
    driver = run_gm if runner == "gm" else run_t2s
    driver(tmp_path, cohort)
    before = _artifact_state(tmp_path / runner)
    extra = {"bundle_dir": tmp_path / "gm_bundle", "create_bundle": False} if runner == "gm" else {}
    with pytest.raises(SharedCleanError, match="pass --resume"):
        driver(tmp_path, cohort, **extra)
    assert _artifact_state(tmp_path / runner) == before


# --------------------------------------------------------------------------- #
# 6. Cross-method audit over TR / GS / GM / T2S
# --------------------------------------------------------------------------- #

def _gs_like_rows(tr_metadata: Path, tmp_path: Path):
    """A minimal valid GS V2 cohort, so the audit is exercised over four methods."""
    from raven.pairing_provenance import (
        GS_SHARED_TR_CLEAN_MODE,
        GS_SHARED_TR_CLEAN_PROTOCOL,
        GS_UNIFORM_DERIVATION,
    )

    tr_rows = read_rows(tr_metadata)
    metadata_sha = sha256_path(tr_metadata)
    gs_dir = tmp_path / "gs"
    gs_dir.mkdir(parents=True, exist_ok=True)
    mask_sha = canonical_json_sha256({"method": "GS", "mask": "not_applicable", "version": 1})
    config_sha = canonical_json_sha256({"wm_type": "GS", "version": 1})

    rows = []
    for tr in tr_rows:
        run_id = int(tr["run_id"])
        item = gs_dir / f"{run_id:06d}"
        item.mkdir(parents=True, exist_ok=True)
        path = item / "watermarked.png"
        _deterministic_image(f"gs-wm-{run_id}").save(path)
        row = {
            "protocol": GS_SHARED_TR_CLEAN_PROTOCOL,
            "dataset": tr["dataset"],
            "run_id": run_id,
            "wm_type": "GS",
            "prompt": tr["prompt"],
            "prompt_sha256": tr["prompt_sha256"],
            "model_id": tr["model_id"],
            "model_revision": tr["model_revision"],
            "base_latent_seed": tr["base_latent_seed"],
            "base_latent_sha256": tr["base_latent_sha256"],
            "clean_base_latent_sha256": tr["clean_base_latent_sha256"],
            "watermarked_base_latent_sha256": tr["base_latent_sha256"],
            "watermarked_latent_sha256": f"{run_id:064d}",
            "watermark_target_sha256": hashlib.sha256(f"gs-target-{run_id}".encode()).hexdigest(),
            "watermark_mask_sha256": mask_sha,
            "generation_config_sha256": tr["generation_config_sha256"],
            "watermark_config_sha256": config_sha,
            "clean_path": tr["clean_path"],
            "clean_sha256": tr["clean_sha256"],
            "watermarked_path": str(path.resolve()),
            "watermarked_sha256": sha256_path(path),
            "gs_protocol_mode": GS_SHARED_TR_CLEAN_MODE,
            "gs_secret_index": run_id,
            "gs_message_sha256": hashlib.sha256(f"m{run_id}".encode()).hexdigest(),
            "gs_key_sha256": hashlib.sha256(f"k{run_id}".encode()).hexdigest(),
            "gs_nonce_sha256": hashlib.sha256(f"n{run_id}".encode()).hexdigest(),
            "gs_secret_bundle_sha256": hashlib.sha256(f"b{run_id}".encode()).hexdigest(),
            "gs_sampling_uniform_sha256": hashlib.sha256(f"u{run_id}".encode()).hexdigest(),
            "gs_payload_layout": "channel_spatial_repeat",
            "gs_cipher": "PyCryptodome_ChaCha20_32byte_key_12byte_nonce",
            "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
            "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
            "shared_clean_source_metadata_path": str(tr_metadata),
            "shared_clean_source_metadata_sha256": metadata_sha,
            "shared_clean_sample_sha256": tr["base_latent_sha256"],
            "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
            "tr_base_latent_sha256": tr["base_latent_sha256"],
            "tr_clean_path": tr["clean_path"],
            "tr_clean_sha256": tr["clean_sha256"],
        }
        row["pairing_sha256"] = build_pairing_sha256(row)
        rows.append(row)
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return rows


def test_cross_method_audit_validates_tr_gs_gm_t2s(cohort, tmp_path, stub_pipeline):
    run_gm(tmp_path, cohort)
    run_t2s(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    cohorts = {
        "GS": _gs_like_rows(cohort, tmp_path),
        "GM": read_rows(tmp_path / "gm" / "metadata.csv"),
        "T2S": read_rows(tmp_path / "t2s" / "metadata.csv"),
    }
    result = audit_shared_clean_cohorts(
        tr_rows, cohorts, verify_files=True, require_methods=("GS", "GM", "T2S")
    )
    assert result["passed"] is True
    assert result["methods"] == ["GM", "GS", "T2S"]
    assert result["rows_checked"] == {"GM": 2, "GS": 2, "T2S": 2}
    assert result["cross_method_run_ids"] == ["0", "1"]
    # One clean image per run_id, shared by all three methods.
    assert result["unique_clean_sha256"] == {"GM": 2, "GS": 2, "T2S": 2}


def test_cross_method_audit_detects_a_missing_row(cohort, tmp_path, stub_pipeline):
    run_gm(tmp_path, cohort)
    gm_rows = read_rows(tmp_path / "gm" / "metadata.csv")
    tr_rows = read_rows(cohort)
    # A GM row whose run_id has no TR source row.
    orphan = dict(gm_rows[0], run_id="99")
    with pytest.raises(ValueError, match="no matching TR source row"):
        audit_shared_clean_cohorts(tr_rows, {"GM": [orphan]}, verify_files=False)


def test_cross_method_audit_detects_duplicate_rows(cohort, tmp_path, stub_pipeline):
    run_gm(tmp_path, cohort)
    gm_rows = read_rows(tmp_path / "gm" / "metadata.csv")
    tr_rows = read_rows(cohort)
    with pytest.raises(ValueError, match="duplicate GM run_id"):
        audit_shared_clean_cohorts(
            tr_rows, {"GM": [gm_rows[0], dict(gm_rows[0])]}, verify_files=False
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("prompt_sha256", "0" * 64, "field=prompt_sha256"),
        ("clean_sha256", "0" * 64, "field=clean_sha256"),
        ("base_latent_sha256", "0" * 64, "field=base_latent_sha256"),
        ("generation_config_sha256", "0" * 64, "field=generation_config_sha256"),
        ("tr_clean_path", "/nowhere.png", "tr_clean_path mismatch"),
    ],
)
def test_cross_method_audit_detects_drift(
    cohort, tmp_path, stub_pipeline, field, value, message
):
    run_t2s(tmp_path, cohort)
    rows = read_rows(tmp_path / "t2s" / "metadata.csv")
    tr_rows = read_rows(cohort)
    drifted = dict(rows[0])
    drifted[field] = value
    with pytest.raises(ValueError, match=message):
        audit_shared_clean_cohorts(tr_rows, {"T2S": [drifted]}, verify_files=False)


def test_cross_method_audit_rejects_two_methods_sharing_one_watermarked_image(
    cohort, tmp_path, stub_pipeline
):
    run_gm(tmp_path, cohort)
    run_t2s(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    gm_rows = read_rows(tmp_path / "gm" / "metadata.csv")
    t2s_rows = read_rows(tmp_path / "t2s" / "metadata.csv")
    collided = dict(t2s_rows[0])
    collided["watermarked_sha256"] = gm_rows[0]["watermarked_sha256"]
    with pytest.raises(ValueError, match="identical watermarked image"):
        audit_shared_clean_cohorts(
            tr_rows, {"GM": [gm_rows[0]], "T2S": [collided]}, verify_files=False
        )


def test_pairing_audit_rejects_a_protocol_or_mode_swap(cohort, tmp_path, stub_pipeline):
    run_gm(tmp_path, cohort)
    rows = read_rows(tmp_path / "gm" / "metadata.csv")

    swapped_mode = [dict(row, gm_protocol_mode="official_compatible") for row in rows]
    with pytest.raises(ValueError, match="gm_protocol_mode"):
        audit_pairing_rows(swapped_mode, expected_count=2, verify_files=False)

    relabelled = [dict(row, protocol=T2S_SHARED_TR_CLEAN_PROTOCOL) for row in rows]
    with pytest.raises(ValueError, match="unsupported pairing protocol"):
        audit_pairing_rows(relabelled, expected_count=2, verify_files=False)


def test_pairing_audit_rejects_a_shared_gm_bundle_row_with_a_repeated_latent(
    cohort, tmp_path, stub_pipeline
):
    run_gm(tmp_path, cohort)
    rows = read_rows(tmp_path / "gm" / "metadata.csv")
    # Two rows claiming the same pre-injection latent means one sample was reused.
    rows[1]["gm_pre_injection_latent_sha256"] = rows[0]["gm_pre_injection_latent_sha256"]
    with pytest.raises(ValueError, match="duplicate gm_pre_injection_latent_sha256"):
        audit_pairing_rows(rows, expected_count=2, verify_files=False)


def test_pairing_audit_counts_t2s_session_key_collisions_without_failing(
    cohort, tmp_path, stub_pipeline
):
    """A repeated 16-bit session key is a birthday collision, not shared state.

    ``--t2s_key_length`` is 16 bits by default (upstream ``run.py``), so a cohort
    of n samples has about n*(n-1)/2 / 2**16 colliding pairs — ~7.6 at n=1001.
    Asserting global uniqueness there fails a correct cohort, so the audit
    records the collisions instead.
    """
    run_t2s(tmp_path, cohort)
    rows = read_rows(tmp_path / "t2s" / "metadata.csv")
    rows[1]["t2s_session_key_sha256"] = rows[0]["t2s_session_key_sha256"]
    # The session key is bound by the pairing hash, so a collided cohort carries
    # a pairing hash computed over the repeated key, not a stale one.
    rows[1]["pairing_sha256"] = build_pairing_sha256(rows[1])

    audit = audit_pairing_rows(rows, expected_count=2, verify_files=False)

    assert audit["passed"] is True
    stats = audit["collision_counted_field_stats"]["t2s_session_key_sha256"]
    assert stats == {"distinct_values": 1, "colliding_pairs": 1, "max_repeat": 2}


def test_pairing_audit_still_rejects_duplicate_t2s_per_sample_state(
    cohort, tmp_path, stub_pipeline
):
    """Dropping the session key must not weaken the fields that are per-sample."""
    run_t2s(tmp_path, cohort)
    rows = read_rows(tmp_path / "t2s" / "metadata.csv")
    for field in ("t2s_watermark_id", "t2s_state_sha256", "t2s_abs_magnitude_sha256"):
        drifted = [dict(row) for row in rows]
        drifted[1][field] = drifted[0][field]
        drifted[1]["pairing_sha256"] = build_pairing_sha256(drifted[1])
        with pytest.raises(ValueError, match=f"duplicate {field}"):
            audit_pairing_rows(drifted, expected_count=2, verify_files=False)


def test_audit_script_runs_end_to_end(cohort, tmp_path, stub_pipeline):
    import subprocess

    run_gm(tmp_path, cohort)
    run_t2s(tmp_path, cohort)
    output = tmp_path / "audit.json"
    script = REPO / "raven_repro" / "scripts" / "audit_shared_clean_cohorts.py"
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--tr-metadata", str(cohort),
            "--gm-metadata", str(tmp_path / "gm" / "metadata.csv"),
            "--t2s-metadata", str(tmp_path / "t2s" / "metadata.csv"),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["cross_method"]["rows_checked"] == {"GM": 2, "T2S": 2}


# --------------------------------------------------------------------------- #
# 7. Run-id coverage, source-SHA binding and method-artifact verification
# --------------------------------------------------------------------------- #

def test_audit_fails_when_a_method_is_missing_an_expected_run_id(
    cohort, tmp_path, stub_pipeline
):
    run_gm(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    gm_rows = read_rows(tmp_path / "gm" / "metadata.csv")

    # The full set passes; dropping one row must not.
    audit_shared_clean_cohorts(
        tr_rows, {"GM": gm_rows}, verify_files=False, expected_run_ids=[0, 1]
    )
    with pytest.raises(ValueError, match="does not cover the expected run_ids"):
        audit_shared_clean_cohorts(
            tr_rows, {"GM": gm_rows[:1]}, verify_files=False, expected_run_ids=[0, 1]
        )
    # An extra run_id outside the expected set is rejected too.
    with pytest.raises(ValueError, match="does not cover the expected run_ids"):
        audit_shared_clean_cohorts(
            tr_rows, {"GM": gm_rows}, verify_files=False, expected_run_ids=[0]
        )


def test_audit_fails_on_a_wrong_recorded_tr_metadata_sha(cohort, tmp_path, stub_pipeline):
    run_t2s(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    rows = read_rows(tmp_path / "t2s" / "metadata.csv")

    # Without the source path the recorded SHA is unchecked; with it, it is.
    audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=False)
    audit_shared_clean_cohorts(
        tr_rows, {"T2S": rows}, verify_files=False, tr_metadata_path=cohort
    )
    lying = [dict(row, shared_clean_source_metadata_sha256="0" * 64) for row in rows]
    with pytest.raises(ValueError, match="shared_clean_source_metadata_sha256 drift"):
        audit_shared_clean_cohorts(
            tr_rows, {"T2S": lying}, verify_files=False, tr_metadata_path=cohort
        )


def test_audit_detects_gm_bundle_artifact_drift(cohort, tmp_path, stub_pipeline):
    run_gm(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    rows = read_rows(tmp_path / "gm" / "metadata.csv")
    audit_shared_clean_cohorts(tr_rows, {"GM": rows}, verify_files=True)

    bundle = Path(rows[0]["gm_bundle_dir"])
    w1 = bundle / "w1.pth"
    original = w1.read_bytes()

    w1.write_bytes(original + b"drift")
    with pytest.raises(ValueError, match="w1.pth SHA drift"):
        audit_shared_clean_cohorts(tr_rows, {"GM": rows}, verify_files=True)

    w1.unlink()
    with pytest.raises(FileNotFoundError, match="GM bundle w1 missing"):
        audit_shared_clean_cohorts(tr_rows, {"GM": rows}, verify_files=True)

    w1.write_bytes(original)
    audit_shared_clean_cohorts(tr_rows, {"GM": rows}, verify_files=True)

    manifest = bundle / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["bundle_config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="GM bundle config SHA drift"):
        audit_shared_clean_cohorts(tr_rows, {"GM": rows}, verify_files=True)


def test_audit_detects_t2s_state_artifact_drift(cohort, tmp_path, stub_pipeline):
    run_t2s(tmp_path, cohort)
    tr_rows = read_rows(cohort)
    rows = read_rows(tmp_path / "t2s" / "metadata.csv")
    audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=True)

    state_path = Path(rows[0]["t2s_state_path"])
    original = state_path.read_text()

    # An edited payload breaks the self-signature.
    payload = json.loads(original)
    payload["tau"] = 0.5
    state_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="T2S state signature invalid"):
        audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=True)

    # A correctly re-signed but different state no longer matches the row.
    payload = json.loads(original)
    payload["tau"] = 0.5
    payload.pop("state_sha256")
    payload["state_sha256"] = canonical_json_sha256(payload)
    state_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="T2S state SHA drift"):
        audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=True)

    state_path.unlink()
    with pytest.raises(FileNotFoundError, match="T2S state artifact missing"):
        audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=True)

    state_path.write_text(original)
    audit_shared_clean_cohorts(tr_rows, {"T2S": rows}, verify_files=True)


# --------------------------------------------------------------------------- #
# 8. Run manifest: an incompatible resume must not create or modify anything
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_incompatible_resume_modifies_no_artifact(cohort, tmp_path, stub_pipeline, runner):
    driver = run_gm if runner == "gm" else run_t2s
    driver(tmp_path, cohort)

    root = tmp_path / runner
    manifest_path = root / "run_manifest.json"
    assert manifest_path.is_file()
    before_root = _artifact_state(root)
    before_bundle = _artifact_state(tmp_path / "gm_bundle") if runner == "gm" else {}
    manifest_before = json.loads(manifest_path.read_text())

    # A different selection is a different run: the manifest gate must reject it
    # before a pipeline is loaded or a GM bundle is created.
    if runner == "gm":
        extra = {"bundle_dir": tmp_path / "gm_bundle_new", "create_bundle": True}
    else:
        extra = {}
    args = (gm_args if runner == "gm" else t2s_args)(tmp_path, cohort, **extra)
    args.resume = True
    args.run_ids = [0]
    with pytest.raises(SharedCleanError, match="Nothing was modified"):
        (gm_runner if runner == "gm" else t2s_runner).run(
            args, DummyGuard(), torch.device("cpu")
        )

    assert _artifact_state(root) == before_root
    assert json.loads(manifest_path.read_text()) == manifest_before
    if runner == "gm":
        assert not (tmp_path / "gm_bundle_new").exists()
        assert _artifact_state(tmp_path / "gm_bundle") == before_bundle


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_compatible_resume_keeps_the_original_manifest(
    cohort, tmp_path, stub_pipeline, runner
):
    driver = run_gm if runner == "gm" else run_t2s
    driver(tmp_path, cohort)
    manifest_path = tmp_path / runner / "run_manifest.json"
    before = json.loads(manifest_path.read_text())

    extra = {"bundle_dir": tmp_path / "gm_bundle", "create_bundle": False} if runner == "gm" else {}
    args = (gm_args if runner == "gm" else t2s_args)(tmp_path, cohort, **extra)
    args.resume = True
    (gm_runner if runner == "gm" else t2s_runner).run(
        args, DummyGuard(), torch.device("cpu")
    )

    after = json.loads(manifest_path.read_text())
    assert after == before
    # The creation time is not rewritten by a resume.
    assert after["created_utc"] == before["created_utc"]
    assert after["run_config_sha256"] == before["run_config_sha256"]


# --------------------------------------------------------------------------- #
# 9. Smoke labelling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_smoke_rows_are_labelled_incomplete(cohort, tmp_path, stub_pipeline, runner):
    summary = (run_gm if runner == "gm" else run_t2s)(tmp_path, cohort)
    assert summary["incomplete"] is True
    assert summary["smoke_only"] is True
    assert summary["formal_output_eligible"] is False
    for row in read_rows(tmp_path / runner / "metadata.csv"):
        assert row["incomplete"] == "True"
        assert row["smoke_only"] == "True"
        assert row["formal_output_eligible"] == "False"


@pytest.mark.parametrize("runner", ["gm", "t2s"])
def test_a_formal_run_is_not_labelled_incomplete(cohort, tmp_path, stub_pipeline, runner):
    args = (gm_args if runner == "gm" else t2s_args)(tmp_path, cohort)
    args.smoke_only = False
    summary = (gm_runner if runner == "gm" else t2s_runner).run(
        args, DummyGuard(), torch.device("cpu")
    )
    assert summary["incomplete"] is False
    assert summary["formal_output_eligible"] is True
    for row in read_rows(tmp_path / runner / "metadata.csv"):
        assert row["incomplete"] == "False"
        assert row["formal_output_eligible"] == "True"


def test_provider_and_protocol_mode_names_agree():
    """raven/ duplicates the provider mode names; they must never drift apart."""
    from utils.wm.gm_provider import GM_SHARED_TR_CLEAN_MODE as gm_mode
    from utils.wm.gm_provider import GM_UNIFORM_DERIVATION as gm_derivation
    from utils.wm.t2s_provider import T2S_SHARED_TR_CLEAN_MODE as t2s_mode

    assert gm_mode == GM_SHARED_TR_CLEAN_MODE
    assert gm_derivation == GM_UNIFORM_DERIVATION
    assert t2s_mode == T2S_SHARED_TR_CLEAN_MODE
