"""Generate HSTR fixtures by executing frozen official SFWMark source."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for item in (str(ROOT), str(TESTS)):
    if item not in sys.path:
        sys.path.insert(0, item)

import official_sfwmark_source as official_src  # noqa: E402
from utils.wm import sfw_bundle  # noqa: E402
from utils.wm.hstr_provider import (  # noqa: E402
    HSTR_SCORE_DEFINITION,
    HSTR_SCORE_DIRECTION,
    HSTRProvider,
    OFFICIAL_BASE_KEY_SEED,
    OFFICIAL_HSTR_PROFILE,
)

REQUIRED_KEYS = (0, 1, 1024, 2047)
LATENT_SHAPE = (1, 4, 64, 64)
BATCH_LATENT_SHAPE = (2, 4, 64, 64)
BASE_LATENT_SEED = 314159


class _FakeUnet:
    in_channels = 4
    dtype = torch.float32


class _FakePipe:
    device = torch.device("cpu")
    unet = _FakeUnet()

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        return torch.randn(
            (batch_size, num_channels_latents, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=dtype,
        )


def pack_tensor(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "data_b64": base64.b64encode(tensor.numpy().tobytes()).decode("ascii"),
        "sha256": sfw_bundle.sha256_tensor(tensor),
    }


def unpack_tensor(payload: dict) -> torch.Tensor:
    import numpy as np

    dtype_map = {
        "torch.float32": np.float32,
        "torch.complex64": np.complex64,
        "torch.bool": np.bool_,
    }
    array = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype=dtype_map[payload["dtype"]])
    return torch.from_numpy(array.copy()).reshape(tuple(payload["shape"]))


def base_latents() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(BASE_LATENT_SEED)
    return torch.randn(BATCH_LATENT_SHAPE, generator=generator, dtype=torch.float32)


def provider(key_index: int) -> HSTRProvider:
    return HSTRProvider(
        latent_shape=LATENT_SHAPE,
        device=torch.device("cpu"),
        hstr_profile=OFFICIAL_HSTR_PROFILE,
        hstr_key_index=key_index,
        hstr_rng_device="cpu",
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DDIM",
        resolution=512,
    )


def official_pattern(utils, key_index: int) -> torch.Tensor:
    return utils.make_Fourier_treering_pattern(
        _FakePipe(),
        LATENT_SHAPE,
        w_seed=OFFICIAL_BASE_KEY_SEED + key_index,
        resolution=512,
        hs=True,
        center=True,
        heter=True,
    ).detach().cpu()


def official_inject_and_score(utils, pattern: torch.Tensor, latents: torch.Tensor) -> tuple[torch.Tensor, list[float], list[float]]:
    pattern_batch = pattern.repeat(latents.shape[0], 1, 1, 1)
    masks = utils.tree_masks.clone().cpu()
    masks[:, utils.HETER_WATERMARK_CHANNEL] = utils.single_channel_heter_watermark_mask.cpu()
    injected, _ = utils.inject_wm(
        latents.clone(),
        pattern_batch,
        masks,
        cut_real=False,
        center=True,
        device="cpu",
    )
    distances = []
    for item in injected:
        target_fft = torch.zeros_like(item.unsqueeze(0), dtype=torch.complex64)
        target_fft[utils.center_slice] = utils.fft(item.unsqueeze(0)[utils.center_slice])
        distances.append(float(utils.get_distance(
            pattern,
            target_fft,
            mask=utils.watermark_region_mask_hstr,
            channel=utils.RINGID_WATERMARK_CHANNEL,
            p=1,
            mode="complex",
            channel_min=True,
            center=True,
        )))
    return injected.detach().cpu(), distances, [-value for value in distances]


def build_fixtures() -> dict:
    official_src.reset_stub_usage()
    utils = official_src.load_official_utils()
    for name in (
        "tree_masks",
        "ringid_masks",
        "watermark_region_mask_hstr",
        "single_channel_heter_watermark_mask",
        "single_channel_tree_watermark_mask",
    ):
        value = getattr(utils, name, None)
        if isinstance(value, torch.Tensor):
            setattr(utils, name, value.cpu())
    official_src.reset_stub_usage()
    latents = base_latents()
    fixture = {
        "schema": "hstr_official_fixtures_v1",
        "derivation": "executed frozen official SFWMark src/utils.py; not transcribed",
        "provenance": {
            "official_repo": official_src.OFFICIAL_SFWMARK_REPO,
            "official_commit": official_src.OFFICIAL_SFWMARK_COMMIT,
            "official_utils_sha256": official_src.OFFICIAL_UTILS_SHA256,
        },
        "official_constants": {
            "base_key_seed": OFFICIAL_BASE_KEY_SEED,
            "wm_capacity": 2048,
            "center_slice": [10, 54],
            "radius": 14,
            "radius_cutoff": 3,
            "watermark_channels": [3],
            "heterogeneous_channels": [0],
            "score_definition": HSTR_SCORE_DEFINITION,
            "score_direction": HSTR_SCORE_DIRECTION,
        },
        "base_latents": pack_tensor(latents),
        "keys": {},
    }
    patterns = {}
    for key_index in REQUIRED_KEYS:
        pattern = official_pattern(utils, key_index)
        injected, distances, scores = official_inject_and_score(utils, pattern, latents)
        prov = provider(key_index)
        if not torch.equal(pattern, prov.gt_patch.cpu()):
            diff = (pattern - prov.gt_patch.cpu()).abs().max().item()
            raise RuntimeError(f"key {key_index} provider pattern differs from official max_abs={diff}")
        rav_injected = prov.get_wm_latents(latents_clean=latents.clone())['zT_torch'].cpu()
        if not torch.equal(injected, rav_injected):
            diff = (injected - rav_injected).abs().max().item()
            raise RuntimeError(f"key {key_index} provider injection differs from official max_abs={diff}")
        rav_scores = prov.get_accuracies(injected)
        if rav_scores['hstr_channel_min_l1'] != distances:
            raise RuntimeError(f"key {key_index} distance mismatch {rav_scores['hstr_channel_min_l1']} != {distances}")
        patterns[key_index] = pattern
        fixture["keys"][str(key_index)] = {
            "key_seed": OFFICIAL_BASE_KEY_SEED + key_index,
            "selected_pattern": pack_tensor(pattern),
            "injected_latent": pack_tensor(injected),
            "detector_target_sha256": sfw_bundle.sha256_tensor(pattern),
            "per_item_raw_distance": distances,
            "canonical_score": scores,
        }
    # Identification among the required candidate keys for the first injected item.
    for key_index in REQUIRED_KEYS:
        injected = unpack_tensor(fixture["keys"][str(key_index)]["injected_latent"])[0:1]
        candidate_distances = {}
        for candidate, pattern in patterns.items():
            target_fft = torch.zeros_like(injected, dtype=torch.complex64)
            target_fft[utils.center_slice] = utils.fft(injected[utils.center_slice])
            candidate_distances[str(candidate)] = float(utils.get_distance(
                pattern,
                target_fft,
                mask=utils.watermark_region_mask_hstr,
                channel=utils.RINGID_WATERMARK_CHANNEL,
                p=1,
                mode="complex",
                channel_min=True,
                center=True,
            ))
        fixture["keys"][str(key_index)]["key_identification_result"] = {
            "candidate_keys": list(REQUIRED_KEYS),
            "identified_key_index": min(candidate_distances, key=candidate_distances.get),
            "candidate_raw_distances": candidate_distances,
        }
    fixture["stub_usage_during_hstr_calls"] = official_src.stub_usage()
    if fixture["stub_usage_during_hstr_calls"]:
        raise RuntimeError(f"official HSTR fixture generation touched dependency stubs: {fixture['stub_usage_during_hstr_calls']}")
    return fixture


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(TESTS / "fixtures" / "hstr_official_fixtures.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    fixture = build_fixtures()
    out = Path(args.out)
    payload = sfw_bundle.canonical_json(fixture) + "\n"
    if args.check:
        existing = out.read_text(encoding="utf-8")
        if existing != payload:
            raise SystemExit(f"{out} does not match regenerated official HSTR fixtures")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
