"""Generate the HSQR parity fixtures from the *real* frozen official SFWMark code.

This replaces the previous spec-derived fixtures. Every value written here is
produced by executing ``src/utils.py`` from

    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb

via :mod:`tests.official_sfwmark_source`, which hash-pins the official file.
The script additionally performs the element-by-element comparison against
``HSQRProvider`` and refuses to write fixtures unless every required key
matches exactly, so a fixture file can never record a disagreement.

Usage::

    export SFWMARK_OFFICIAL_SRC=/path/to/SFWMark   # frozen commit
    python eval_bench_wm/tools/generate_hsqr_official_fixtures.py

Add ``--check`` to compare without rewriting the fixture file.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / "eval_bench_wm"
for _path in (str(_PKG_ROOT), str(_PKG_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import official_sfwmark_source as official_src  # noqa: E402
from utils.wm.hsqr_provider import HSQRProvider  # noqa: E402

FIXTURE_SCHEMA = "hsqr_official_fixtures_v1"
FIXTURE_PATH = _PKG_ROOT / "tests" / "fixtures" / "hsqr_official_fixtures.json"

#: The four keys Issue #5 requires: first, second, a middle key and the last.
REQUIRED_KEYS = (0, 1, 1024, 2047)

#: Deterministic base latent used for the injection / distance fixtures. The
#: digest is stored in the fixture file, so a change in torch's RNG is reported
#: as an explicit fixture mismatch instead of silently shifting every value.
BASE_LATENT_SEED = 20250729
BASE_LATENT_SHAPE = (4, 4, 64, 64)
BASE_LATENT_RECIPE = (
    "torch.randn(4, 4, 64, 64, dtype=torch.float32, "
    f"generator=torch.Generator('cpu').manual_seed({BASE_LATENT_SEED}))"
)

#: Injection and detection are float32 FFT round trips evaluated by the same
#: torch build, so provider and official output are expected to agree bitwise.
#: These tolerances exist only to give a useful failure message.
LATENT_ATOL = 0.0
DISTANCE_ATOL = 0.0


def build_base_latent() -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(BASE_LATENT_SEED)
    return torch.randn(*BASE_LATENT_SHAPE, dtype=torch.float32, generator=generator)


def pack_bool_tensor(tensor: torch.Tensor) -> str:
    """Base64 of the bit-packed booleans, so fixtures pin every element."""
    array = tensor.detach().cpu().numpy().astype(bool)
    return base64.b64encode(np.packbits(array.reshape(-1))).decode("ascii")


def unpack_bool_tensor(blob: str, shape) -> torch.Tensor:
    raw = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
    count = int(np.prod(shape))
    bits = np.unpackbits(raw)[:count].astype(bool).reshape(shape)
    return torch.from_numpy(bits.copy())


def sha256_tensor(tensor: torch.Tensor) -> str:
    import hashlib

    tensor = tensor.detach().cpu().contiguous()
    header = f"torch|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def official_pattern(utils, key_index: int) -> torch.Tensor:
    """Official ``(1,42,42)`` pattern, using the official key mapping."""
    # generate.py: w_seed_list = [*range(w_seed, w_seed + wm_capacity)]
    #              make_hsqr_pattern(idx=this_w_seed)
    w_seed_list = [*range(utils.w_seed, utils.w_seed + utils.wm_capacity)]
    return utils.make_hsqr_pattern(idx=w_seed_list[key_index])


def official_inject(utils, base_latent: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Official center-mode injection, exactly as generate.py calls it."""
    batch = base_latent.shape[0]
    qr = pattern.unsqueeze(0).repeat(batch, 1, 1, 1)  # (N,c_wm,42,42)
    return utils.inject_hsqr(base_latent, qr, center=True, device="cpu")


def official_distances(utils, latents: torch.Tensor, pattern: torch.Tensor) -> list[float]:
    """Official per-item distance, exactly as detect.py computes it.

    detect.py builds a zero (N,4,64,64) complex canvas, writes the full complex
    FFT of the 44x44 centre into it, then calls ``get_distance_hsqr`` on one
    ``(1,4,64,64)`` slice at a time.
    """
    spectrum = torch.zeros_like(latents, dtype=torch.complex64)
    spectrum[utils.center_slice] = utils.fft(latents[utils.center_slice])
    return [
        float(utils.get_distance_hsqr(pattern, spectrum[i][None, ...], center=True))
        for i in range(latents.shape[0])
    ]


def build_provider(key_index: int) -> HSQRProvider:
    return HSQRProvider(
        latent_shape=(1, 4, 64, 64),
        device="cpu",
        hsqr_profile="official_sfwmark_sd21",
        hsqr_base_key_seed=7433,
        hsqr_key_index=key_index,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        model_revision=None,
        scheduler_target="DDIM",
        resolution=512,
    )


def compare_key(utils, key_index: int, base_latent: torch.Tensor) -> dict:
    """Element-by-element official-vs-provider comparison for one key."""
    provider = build_provider(key_index)

    # --- key derivation ---------------------------------------------------
    w_seed_list = [*range(utils.w_seed, utils.w_seed + utils.wm_capacity)]
    off_seed = w_seed_list[key_index]
    off_payload = f"HSQR{off_seed % 10000}"
    assert provider.key_seed(key_index) == off_seed, (
        f"key {key_index}: seed {provider.key_seed(key_index)} != official {off_seed}"
    )
    assert provider.payload_text(key_index) == off_payload, (
        f"key {key_index}: payload {provider.payload_text(key_index)!r} != {off_payload!r}"
    )

    # --- QR pattern, every element ---------------------------------------
    official_src.reset_stub_usage()
    off_pattern = official_pattern(utils, key_index)
    mine_pattern = provider.make_pattern(key_index)
    assert off_pattern.shape == mine_pattern.shape, (
        f"key {key_index}: shape {tuple(mine_pattern.shape)} != {tuple(off_pattern.shape)}"
    )
    assert off_pattern.dtype == mine_pattern.dtype == torch.bool
    mismatches = int((off_pattern != mine_pattern).sum())
    assert mismatches == 0, f"key {key_index}: {mismatches} QR elements differ from official"

    # --- injection, every element ----------------------------------------
    off_latent = official_inject(utils, base_latent.clone(), off_pattern)
    mine_latent = provider.inject(base_latent.clone(), pattern=mine_pattern)
    mine_latent = mine_latent.detach().cpu().to(torch.float32)
    assert off_latent.shape == mine_latent.shape, (
        f"key {key_index}: latent shape {tuple(mine_latent.shape)} != {tuple(off_latent.shape)}"
    )
    max_abs = float((off_latent - mine_latent).abs().max())
    assert max_abs <= LATENT_ATOL, (
        f"key {key_index}: watermarked latent differs from official by {max_abs} "
        f"(tolerance {LATENT_ATOL})"
    )

    # --- detector distance and canonical score ---------------------------
    off_wm_dist = official_distances(utils, off_latent, off_pattern)
    mine_wm_dist = provider.l1_distances(mine_latent, pattern=mine_pattern)
    off_clean_dist = official_distances(utils, base_latent, off_pattern)
    mine_clean_dist = provider.l1_distances(base_latent, pattern=mine_pattern)

    for label, off_vals, mine_vals in (
        ("watermarked", off_wm_dist, mine_wm_dist),
        ("clean", off_clean_dist, mine_clean_dist),
    ):
        assert len(off_vals) == len(mine_vals) == base_latent.shape[0], (
            f"key {key_index} ({label}): expected one distance per batch item"
        )
        for i, (a, b) in enumerate(zip(off_vals, mine_vals)):
            assert abs(a - b) <= DISTANCE_ATOL, (
                f"key {key_index} ({label}) item {i}: distance {b} != official {a}"
            )

    # score sign: detect.py scores with ``-func(...)``
    for off_d, mine_d in zip(off_wm_dist, mine_wm_dist):
        assert provider.score_from_distance(mine_d) == -off_d, (
            f"key {key_index}: canonical score does not match official -distance"
        )

    # Nothing in the official HSQR path may have touched a stubbed dependency.
    touched = official_src.stub_usage()
    assert not touched, f"key {key_index}: official HSQR path touched stubs {touched}"

    return {
        "key_index": key_index,
        "key_seed": off_seed,
        "payload": off_payload,
        "pattern_shape": list(off_pattern.shape),
        "pattern_ones": int(off_pattern.sum()),
        "pattern_sha256": sha256_tensor(off_pattern),
        "pattern_bits_b64": pack_bool_tensor(off_pattern),
        "watermarked_latent_sha256": sha256_tensor(off_latent),
        "watermarked_l1_distances": [float(v) for v in off_wm_dist],
        "watermarked_scores": [float(-v) for v in off_wm_dist],
        "clean_l1_distances": [float(v) for v in off_clean_dist],
        "clean_scores": [float(-v) for v in off_clean_dist],
    }


def build_fixtures() -> dict:
    utils = official_src.load_official_utils()

    # Freeze the official constants themselves, read off the official module.
    constants = {
        "base_key_seed": int(utils.w_seed),
        "wm_capacity": int(utils.wm_capacity),
        "qr_version": int(utils.qr_version),
        "box_size": int(utils.box_size),
        "border": 0,  # QRCodeGenerator(..., border=0) in utils.py
        "error_correction": "H",
        "delta": int(utils.delta),
        "watermark_channel": list(utils.HSQR_WATERMARK_CHANNEL),
        "center_start": int(utils.start),
        "center_end": int(utils.end),
        "latent_hw": int(utils.hw_latent),
        "target_magnitude": 45.0,
        "score_definition": "negative_mean_complex_l1_distance",
    }

    base_latent = build_base_latent()
    keys = {}
    for key_index in REQUIRED_KEYS:
        keys[str(key_index)] = compare_key(utils, key_index, base_latent)

    return {
        "schema": FIXTURE_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eval_bench_wm/tools/generate_hsqr_official_fixtures.py",
        "derivation": "executed the frozen official src/utils.py; not transcribed",
        "provenance": official_src.official_provenance(),
        "official_constants": constants,
        "base_latent": {
            "recipe": BASE_LATENT_RECIPE,
            "seed": BASE_LATENT_SEED,
            "shape": list(BASE_LATENT_SHAPE),
            "dtype": "float32",
            "sha256": sha256_tensor(base_latent),
        },
        "tolerances": {
            "watermarked_latent_max_abs": LATENT_ATOL,
            "distance_abs": DISTANCE_ATOL,
            "qr_pattern": "exact (bitwise)",
        },
        "keys": keys,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed fixtures without rewriting")
    parser.add_argument("--out", default=str(FIXTURE_PATH))
    args = parser.parse_args(argv)

    fixtures = build_fixtures()
    out_path = Path(args.out)

    if args.check:
        if not out_path.is_file():
            print(f"FAIL: {out_path} does not exist")
            return 1
        committed = json.loads(out_path.read_text(encoding="utf-8"))
        volatile = {"generated_utc"}
        a = {k: v for k, v in committed.items() if k not in volatile}
        b = {k: v for k, v in fixtures.items() if k not in volatile}
        if a != b:
            print("FAIL: regenerated fixtures differ from the committed ones")
            return 1
        print(f"OK: committed fixtures reproduce from the official source ({out_path})")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixtures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    for key, entry in sorted(fixtures["keys"].items(), key=lambda kv: int(kv[0])):
        print(f"  key {key:>4}  seed={entry['key_seed']}  payload={entry['payload']}  "
              f"ones={entry['pattern_ones']}  sha={entry['pattern_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
