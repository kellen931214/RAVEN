"""Element-wise inversion parity between RAVEN's SFWMark front-end and the official code.

Issue #5 asks for frozen official inversion parity evidence comparing:

    preprocessed input tensor
    VAE latent
    inverse scheduler timesteps
    selected intermediate latents
    final recovered latent
    HSQR L1 distance
    canonical score

This script produces exactly that, by running both implementations against the
*same* loaded pipeline:

* ``official`` — ``ddim_invert`` / ``pil2latent`` / ``transform_img`` executed
  from the frozen official ``src/utils.py`` (hash-pinned, see
  ``tests/official_sfwmark_source``). The official path drives the whole
  diffusers pipeline with a swapped-in ``DDIMInverseScheduler``.
* ``raven`` — ``utils/wm/sfw_inversion``, which runs its own DDIM inversion loop
  over the repository's vendored ``DDIMInverseScheduler``.

Intermediates are captured with a forward pre-hook on the UNet, so both paths
are observed identically and neither has to be modified.

Scope of the claim
------------------
This establishes **implementation parity against the official code**. It is not
a claim of numerical reproduction of the published results, which would require
the official ``stabilityai/stable-diffusion-2-1-base`` weights. That repository
is currently delisted from the Hugging Face Hub (HTTP 404 for the whole
``stabilityai`` SD-2 family, with a valid token), so a mirror must be requested
explicitly with ``--allow-non-official-model``. The evidence file always records
``official_model_used`` so a mirror run can never be mistaken for an official one.

Usage::

    export SFWMARK_OFFICIAL_SRC=/path/to/SFWMark
    python eval_bench_wm/tools/hsqr_inversion_parity.py \
        --model stabilityai/stable-diffusion-2-1-base --device cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / "eval_bench_wm"
for _path in (str(_PKG_ROOT), str(_PKG_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import official_sfwmark_source as official_src  # noqa: E402
from utils.wm import sfw_inversion  # noqa: E402
from utils.wm.hsqr_provider import (  # noqa: E402
    HSQRProvider, OFFICIAL_BASE_KEY_SEED, OFFICIAL_PROFILE_NAME,
)
from utils.wm.runner_common import gpu_preflight  # noqa: E402

OFFICIAL_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
EVIDENCE_PATH = _PKG_ROOT / "tests" / "fixtures" / "hsqr_inversion_parity_evidence.json"
EVIDENCE_SCHEMA = "hsqr_inversion_parity_v1"

#: Which UNet steps to record as "selected intermediate latents".
INTERMEDIATE_STEPS = (0, 1, 24, 48, 49)

#: Image the parity run inverts: a real watermarked sample (key 0), produced by
#: injecting the official pattern into a deterministic latent and VAE-decoding it.
IMAGE_LATENT_SEED = 20250729


class UNetTap:
    """Record ``(timestep, incoming latent)`` for every UNet call."""

    def __init__(self, unet):
        self.unet = unet
        self.timesteps: list[float] = []
        self.latents: list[torch.Tensor] = []
        self._handle = None

    def __enter__(self):
        def hook(_module, args, _kwargs):
            if args:
                self.latents.append(args[0].detach().to("cpu", torch.float32).clone())
                if len(args) > 1:
                    step = args[1]
                    self.timesteps.append(
                        float(step.item()) if torch.is_tensor(step) else float(step)
                    )
            return None

        self._handle = self.unet.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
        return False


def tensor_stats(tensor: torch.Tensor) -> dict:
    import hashlib

    flat = tensor.detach().to("cpu", torch.float32).contiguous()
    header = f"torch|{flat.dtype}|{tuple(flat.shape)}|".encode("utf-8")
    return {
        "shape": list(flat.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(header + flat.numpy().tobytes()).hexdigest(),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Element-wise comparison of two tensors."""
    a = a.detach().to("cpu", torch.float32)
    b = b.detach().to("cpu", torch.float32)
    entry = {
        "artifact": name,
        "official": tensor_stats(a),
        "raven": tensor_stats(b),
        "shape_match": list(a.shape) == list(b.shape),
    }
    if entry["shape_match"]:
        diff = (a - b).abs()
        denom = a.abs().max().clamp_min(1e-12)
        entry.update({
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "max_rel_diff": float((diff.max() / denom)),
            "bitwise_identical": bool(torch.equal(a, b)),
            "elements_compared": int(a.numel()),
        })
    return entry


def build_pipe(model_id: str, device: str, dtype: torch.dtype):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_provider(device: str, model_id: str) -> HSQRProvider:
    return HSQRProvider(
        latent_shape=(1, 4, 64, 64),
        device=device,
        hsqr_profile=OFFICIAL_PROFILE_NAME,
        hsqr_base_key_seed=OFFICIAL_BASE_KEY_SEED,
        hsqr_key_index=0,
        modelid_target=model_id,
        model_revision=None,
        scheduler_target="DDIM",
        resolution=512,
    )


@torch.no_grad()
def make_watermarked_image(pipe, provider, device, dtype):
    """Decode a watermarked latent into the PIL image both paths will invert."""
    generator = torch.Generator("cpu").manual_seed(IMAGE_LATENT_SEED)
    base = torch.randn(1, 4, 64, 64, dtype=torch.float32, generator=generator)
    watermarked = provider.inject(base.clone()).to(device, dtype)
    image = pipe.vae.decode(
        watermarked / pipe.vae.config.scaling_factor
    ).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    array = (image[0].permute(1, 2, 0).to("cpu", torch.float32).numpy() * 255)
    from PIL import Image
    import numpy as np

    return Image.fromarray(array.round().astype(np.uint8)), watermarked


class _ProviderTarget:
    """Minimal ``pipe_provider_target`` for :mod:`utils.wm.sfw_inversion`."""

    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device
        self.scheduler = pipe.scheduler


@torch.no_grad()
def run_parity(args) -> dict:
    utils = official_src.load_official_utils()
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    preflight = gpu_preflight(torch.device(device))

    pipe = build_pipe(args.model, device, dtype)
    provider = build_provider(device, args.model)
    image, source_latent = make_watermarked_image(pipe, provider, device, dtype)

    # ---- 1. preprocessed input tensor -----------------------------------
    official_src.reset_stub_usage()
    official_pre = utils.transform_img(image, resolution=512)
    raven_pre = sfw_inversion.transform_img(image, target_size=512)
    comparisons = [compare("preprocessed_input_tensor", official_pre, raven_pre)]

    # ---- 2. VAE latent ---------------------------------------------------
    official_z0 = utils.pil2latent(pipe, [image])
    raven_pre_batched = raven_pre.unsqueeze(0).to(device, dtype)
    raven_z0 = sfw_inversion.encode_image_latents(
        pipe, raven_pre_batched, float(pipe.vae.config.scaling_factor)
    )
    comparisons.append(compare("vae_latent", official_z0, raven_z0))

    # ---- 3-5. inversion: timesteps, intermediates, final latent ---------
    base_scheduler = pipe.scheduler
    with UNetTap(pipe.unet) as official_tap:
        official_zT = utils.ddim_invert(pipe, [image], invert_prompt="", invert_guidance=0)
    pipe.scheduler = base_scheduler

    target = _ProviderTarget(pipe, device)
    with UNetTap(pipe.unet) as raven_tap:
        raven_result = sfw_inversion.invert_pil_image(
            image, target, resolution=512, inversion_prompt="",
            guidance_scale=0.0, num_inference_steps=50,
        )
    raven_zT = raven_result["zT_torch"]

    timesteps_match = official_tap.timesteps == raven_tap.timesteps
    intermediates = []
    for step in INTERMEDIATE_STEPS:
        if step < len(official_tap.latents) and step < len(raven_tap.latents):
            entry = compare(
                f"intermediate_latent_step_{step}",
                official_tap.latents[step], raven_tap.latents[step],
            )
            entry["unet_step"] = step
            entry["official_timestep"] = official_tap.timesteps[step]
            entry["raven_timestep"] = raven_tap.timesteps[step]
            intermediates.append(entry)

    comparisons.append(compare("final_recovered_latent", official_zT, raven_zT))

    # ---- 6-7. HSQR distance and canonical score -------------------------
    pattern = provider.make_pattern(0)
    official_dist = float(provider.l1_distances(
        official_zT.to("cpu", torch.float32), pattern=pattern)[0])
    raven_dist = float(provider.l1_distances(
        raven_zT.to("cpu", torch.float32), pattern=pattern)[0])

    stub_touched = official_src.stub_usage()

    return {
        "schema": EVIDENCE_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eval_bench_wm/tools/hsqr_inversion_parity.py",
        "claim": (
            "implementation parity of utils/wm/sfw_inversion against the frozen "
            "official ddim_invert on identical weights"
        ),
        "not_claimed": (
            "numerical reproduction of the published SFWMark results, which needs "
            "the official stabilityai/stable-diffusion-2-1-base weights"
        ),
        "provenance": official_src.official_provenance(),
        "model": {
            "model_id": args.model,
            "official_model_id": OFFICIAL_MODEL_ID,
            "official_model_used": args.model == OFFICIAL_MODEL_ID,
            "dtype": args.dtype,
            "scheduler": "DDIM",
            "resolution": 512,
            "inversion_steps": 50,
            "inversion_prompt": "",
            "inversion_guidance": 0.0,
        },
        "gpu_preflight": preflight,
        "source_watermarked_latent": tensor_stats(source_latent),
        "official_stubs_touched": stub_touched,
        "timesteps": {
            "official": official_tap.timesteps,
            "raven": raven_tap.timesteps,
            "match": timesteps_match,
            "count_official": len(official_tap.timesteps),
            "count_raven": len(raven_tap.timesteps),
        },
        "comparisons": comparisons,
        "intermediate_latents": intermediates,
        "hsqr_detection": {
            "official_l1_distance": official_dist,
            "raven_l1_distance": raven_dist,
            "l1_abs_diff": abs(official_dist - raven_dist),
            "official_score": -official_dist,
            "raven_score": -raven_dist,
            "score_abs_diff": abs(official_dist - raven_dist),
            "score_definition": "negative_mean_complex_l1_distance",
        },
    }


def summarize(evidence: dict) -> None:
    print(f"model                 : {evidence['model']['model_id']}")
    print(f"official model used   : {evidence['model']['official_model_used']}")
    print(f"timesteps match       : {evidence['timesteps']['match']} "
          f"({evidence['timesteps']['count_official']} steps)")
    for entry in evidence["comparisons"] + evidence["intermediate_latents"]:
        print(f"  {entry['artifact']:<34} max_abs_diff="
              f"{entry.get('max_abs_diff', float('nan')):.3e} "
              f"bitwise={entry.get('bitwise_identical')}")
    det = evidence["hsqr_detection"]
    print(f"  HSQR L1 official={det['official_l1_distance']:.6f} "
          f"raven={det['raven_l1_distance']:.6f} diff={det['l1_abs_diff']:.3e}")
    print(f"  score official={det['official_score']:.6f} raven={det['raven_score']:.6f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=OFFICIAL_MODEL_ID)
    parser.add_argument("--allow-non-official-model", action="store_true",
                        help="permit a mirror; the evidence records it as non-official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--out", default=str(EVIDENCE_PATH))
    args = parser.parse_args(argv)

    if args.model != OFFICIAL_MODEL_ID and not args.allow_non_official_model:
        parser.error(
            f"--model {args.model} is not the official {OFFICIAL_MODEL_ID}. "
            "Pass --allow-non-official-model to record an explicitly non-official run."
        )

    evidence = run_parity(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    summarize(evidence)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
