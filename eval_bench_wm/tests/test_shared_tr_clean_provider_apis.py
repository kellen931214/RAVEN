"""Provider-level supplied-latent gates for ``shared_tr_clean_v2``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "eval_bench_wm") not in sys.path:
    sys.path.insert(0, str(REPO / "eval_bench_wm"))


def _base():
    g = torch.Generator(device="cpu").manual_seed(123)
    return torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float32)


def test_hsqr_shared_clean_api_uses_supplied_latent_and_draws_no_replacement(monkeypatch):
    from utils.wm import sfw_bundle
    from utils.wm.hsqr_provider import HSQRProvider

    provider = HSQRProvider(
        latent_shape=(1, 4, 64, 64),
        device=torch.device("cpu"),
        hsqr_profile="official_sfwmark_sd21",
        hsqr_key_index=0,
        hsqr_torch_dtype="float32",
        modelid_target="RedbeardNZ/stable-diffusion-2-1-base",
        model_revision="test",
        scheduler_target="DDIM",
    )
    monkeypatch.setattr(provider, "sample_base_latent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replacement latent drawn")))
    base = _base()
    out = provider.get_wm_latents_from_base_latent(base)
    assert out["hsqr_pre_injection_latent_sha256"] == sfw_bundle.sha256_tensor(base)
    assert out["hsqr_post_injection_latent_sha256"] != out["hsqr_pre_injection_latent_sha256"]
    with pytest.raises(Exception, match="float32"):
        provider.get_wm_latents_from_base_latent(base.to(torch.float64))


def test_hstr_shared_clean_api_uses_supplied_latent_and_draws_no_replacement(monkeypatch):
    from utils.wm import sfw_bundle
    from utils.wm.hstr_provider import HSTRProvider

    provider = HSTRProvider(
        latent_shape=(1, 4, 64, 64),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hstr_profile="official_sfwmark_sd21",
        hstr_key_index=1,
        hstr_rng_device="cpu",
        modelid_target=None,
        model_revision="test",
        scheduler_target=None,
    )
    monkeypatch.setattr(provider, "sample_base_latent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replacement latent drawn")))
    base = _base()
    out = provider.get_wm_latents_from_base_latent(base)
    assert out["hstr_pre_injection_latent_sha256"] == sfw_bundle.sha256_tensor(base)
    assert out["hstr_post_injection_latent_sha256"] != out["hstr_pre_injection_latent_sha256"]
    with pytest.raises(Exception, match="shape mismatch"):
        provider.get_wm_latents_from_base_latent(base[:, :, :32, :32])


def test_rid_shared_clean_api_uses_supplied_latent_and_draws_no_replacement(monkeypatch):
    from utils.wm import rid_bundle
    from utils.wm.ringid_provider import RingIDProvider

    provider = RingIDProvider(
        latent_shape=(1, 4, 64, 64),
        device=torch.device("cpu"),
        rid_profile="legacy",
        rid_key_index=0,
        rid_torch_dtype="float32",
        modelid_target="test",
        model_revision="test",
        scheduler_target="DDIM",
    )
    monkeypatch.setattr(provider, "sample_clean_latent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replacement latent drawn")))
    base = _base()
    out = provider.get_wm_latents_from_base_latent(base)
    assert out["rid_pre_injection_latent_sha256"] == rid_bundle.sha256_tensor(base)
    assert out["rid_post_injection_latent_sha256"] != out["rid_pre_injection_latent_sha256"]
    bad = base.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(Exception, match="NaN|Inf"):
        provider.get_wm_latents_from_base_latent(bad)
