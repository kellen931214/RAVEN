"""Standalone watermark verification.

Verifies suspect images against a saved portable watermark state in a fresh
process. It does not need the generation provider instance, the original latent,
or any earlier ``get_wm_latents()`` call.

This runner only does CLI parsing, state/image pairing, inversion dispatch and
provider invocation. All detector logic lives in
``utils/wm/t2s_provider.detect_from_reversed_latents``; all inversion logic lives
in ``utils/wm/t2s_inversion``.

Examples
--------
Explicit pairs (never inferred from filename or row order)::

    python run_verification.py --wm_type T2S \\
        --pair out/t2s_state/sd21-t2s-00000.json=out/images/00000.png \\
        --pair out/t2s_state/sd21-t2s-00001.json=out/images/00001.png \\
        --modelid_target stabilityai/stable-diffusion-2-1-base

Content-addressed pairing, matching each state's recorded ``image_sha256``::

    python run_verification.py --wm_type T2S \\
        --state_dir out/t2s_state --image_dir out/images \\
        --modelid_target stabilityai/stable-diffusion-2-1-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import PIL.Image
import torch

from utils.canonical import sha256_path
from utils.pipe import pipe_utils
from utils.wm.t2s_inversion import invert_image
from utils.wm.t2s_provider import (
    T2S_INVERSION_MODES,
    T2SWatermarkState,
    detect_from_reversed_latents,
)


def require_gpu() -> torch.device:
    """AGENTS.md GPU hard stop: never silently fall back to CPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print(
            "STOPPED: GPU unavailable inside Docker. No further work was performed.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    device = torch.device("cuda")
    torch.zeros(1, device=device) + 1  # basic allocation + kernel check
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="standalone watermark verification")
    parser.add_argument("--wm_type", type=str, default="T2S", choices=["T2S"],
                        help="Only methods with a registered standalone verifier are accepted")

    parser.add_argument("--pair", action="append", default=[], metavar="STATE=IMAGE",
                        help="Explicit state/image pair; repeatable")
    parser.add_argument("--state_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)

    parser.add_argument("--modelid_target", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default=None,
                        help="Defaults to the scheduler recorded in the state")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Defaults to the resolution recorded in the state")
    parser.add_argument("--inversion_mode", type=str, default=None,
                        choices=list(T2S_INVERSION_MODES),
                        help="Defaults to the mode recorded in the state")
    parser.add_argument("--num_inversion_steps", type=int, default=None,
                        help="Defaults to the step count recorded in the state")

    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--strict_image_sha", action="store_true", default=False,
                        help="Fail when a suspect image does not match the state's recorded image_sha256")
    return parser


def resolve_pairs(args) -> list:
    """Return [(state, image_path)] without relying on name or row ordering."""
    pairs = []

    for item in args.pair:
        if "=" not in item:
            raise SystemExit(f"--pair expects STATE=IMAGE, got {item!r}")
        state_path, image_path = item.split("=", 1)
        pairs.append((T2SWatermarkState.load(state_path), Path(image_path)))

    if args.state_dir or args.image_dir:
        if not (args.state_dir and args.image_dir):
            raise SystemExit("--state_dir and --image_dir must be given together")
        states = [T2SWatermarkState.load(p) for p in sorted(Path(args.state_dir).glob("*.json"))]
        # Content-addressed: index the images by digest, then look each state up
        # by the image_sha256 it recorded at generation time.
        by_digest = {}
        for image_path in sorted(Path(args.image_dir).iterdir()):
            if image_path.is_file():
                by_digest.setdefault(sha256_path(image_path), image_path)
        for state in states:
            if state.image_sha256 is None:
                raise SystemExit(
                    f"state {state.watermark_id} has no image_sha256; "
                    "use explicit --pair instead of directory matching"
                )
            image_path = by_digest.get(state.image_sha256)
            if image_path is None:
                raise SystemExit(
                    f"no image in {args.image_dir} matches image_sha256 of state {state.watermark_id}"
                )
            pairs.append((state, image_path))

    if not pairs:
        raise SystemExit("nothing to verify: pass --pair or --state_dir/--image_dir")
    return pairs


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    device = require_gpu()

    pairs = resolve_pairs(args)

    # Fail closed before spending time loading a model.
    image_digests = {}
    for state, image_path in pairs:
        if not image_path.is_file():
            raise SystemExit(f"suspect image not found: {image_path}")
        image_digests[image_path] = sha256_path(image_path)
        if args.strict_image_sha and state.image_sha256 not in (None, image_digests[image_path]):
            raise SystemExit(
                f"image {image_path} sha256={image_digests[image_path]} does not match "
                f"state {state.watermark_id} image_sha256={state.image_sha256}"
            )

    reference_state = pairs[0][0]

    resolution = args.resolution or reference_state.resolution or 512
    scheduler = args.scheduler_target or reference_state.scheduler or "DDIM"

    pipe_provider = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=resolution,
        device=device,
        schedulers_name=scheduler,
        disable_tqdm=True,
        revision=args.model_revision or reference_state.model_revision,
    )

    records = []
    for state, image_path in pairs:
        inversion_mode = args.inversion_mode or state.inversion_mode
        num_inversion_steps = args.num_inversion_steps or state.num_inversion_steps

        image_sha256 = image_digests[image_path]
        image = PIL.Image.open(image_path).convert("RGB")
        with torch.no_grad():
            reversed_latents = invert_image(
                pipe_provider,
                image,
                inversion_mode=inversion_mode,
                num_inversion_steps=num_inversion_steps,
                benchmark_num_inference_steps=state.num_inference_steps,
            )

        result = detect_from_reversed_latents(state, reversed_latents)
        result.update({
            "wm_type": args.wm_type,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "image_sha256_matches_state": (
                None if state.image_sha256 is None else state.image_sha256 == image_sha256
            ),
            "inversion_mode": inversion_mode,
            "num_inversion_steps": num_inversion_steps,
            "model_id": args.modelid_target,
            "scheduler": scheduler,
        })
        records.append(result)

        key_acc = result["key_accuracy"]
        msg_acc = result["message_accuracy"]
        print(
            f"[{result['watermark_id']}] detected={result['detection_success']} "
            f"score_true_key={result['score_true_key']:.5f} "
            f"score_control_key={result['score_control_key']:.5f} "
            f"margin={result['score_margin']:.5f} "
            f"key_acc={'N/A' if key_acc is None else f'{key_acc:.5f}'} "
            f"msg_acc={'N/A' if msg_acc is None else f'{msg_acc:.5f}'} "
            f"[{result['decision_rule']}; {result['score_direction']}; not TPR@1%FPR]"
        )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
