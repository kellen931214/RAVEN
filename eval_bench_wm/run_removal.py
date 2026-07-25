"""
Run watermark removal attacks and evaluate detection before/after removal.
"""

import argparse
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

from utils.bit_accuracy import format_staged_bit_accuracy_rows
from utils.imprint_utils import validate
from utils.logger import get_logger
from utils.pipe import pipe_utils
from utils.prompt_utils import get_text_prompts
from utils.rm import get_remover
from utils.utils import check_if_detection_successful, describe_legacy_detection_threshold, get_detection_threshold, seed_everything, set_random_seed
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.hstr_provider import parser as hstr_parser
from utils.wm.hsqr_provider import parser as hsqr_parser
from utils.wm.prc_provider import parser as prc_parser
from utils.wm.ringid_provider import parser as ringid_parser
from utils.wm.tag_provider import parser as tag_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.sph_provider import parser as sph_parser
from utils.wm.t2s_provider import parser as t2s_parser
from utils.wm.maxsive_provider import parser as maxsive_parser
from utils.wm.shallow_provider import parser as shallow_parser
from utils.wm.gm_provider import parser as gm_parser
from utils.wm.wm_utils import WmProviders


MODEL_IDS = sorted(
    set(pipe_utils.PIPE_PROVIDERS.keys())
    | {
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-3-medium-diffusers",
    }
)

EAGER_LOAD_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
]

MODEL_NAME_MAPPING = {
    "CompVis/stable-diffusion-v1-4": "sd14",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-2-1-base": "sd21",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "pixart",
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers": "sana",
}

METRIC_MAP = {
    "PRC": "value",
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
    "GS": "bit_accuracy",
    "TAG": "bit_accuracy",
    "SPH": "bit_accuracy",
    "T2S": "bit_accuracy",
    "MAXSIVE": "value",
    "SHALLOW": "value",
    "GM": "value",
}

def build_parser():
    """Construct the run_removal argument parser (side-effect free, testable)."""
    parent_parsers = [
        gs_parser,
        tr_parser,
        prc_parser,
        tag_parser,
        ringid_parser,
        hstr_parser,
        hsqr_parser,
        t2s_parser,
        sph_parser,
        maxsive_parser,
        shallow_parser,
        gm_parser,
    ]
    parser = argparse.ArgumentParser(description="watermark-removal", parents=parent_parsers)
    parser.add_argument("--out_dir", type=str, default="out/removal/")
    parser.add_argument("--modelid_target", type=str, default="stabilityai/stable-diffusion-2-1-base", choices=MODEL_IDS)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--wm_type", type=str, default="GS", choices=[wm.name for wm in WmProviders])
    parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logger", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save_images", action="store_true", default=False)

    parser.add_argument("--rm_method", type=str, default="nfpa", choices=["nfpa"])
    parser.add_argument("--rm_modelid", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--rm_steps", type=int, default=10)
    parser.add_argument("--rm_xy", type=int, default=40)
    parser.add_argument("--rm_z", type=int, default=0)
    parser.add_argument("--rm_guidance_scale", type=float, default=7.5)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def detection_value(results, wm_type):
    return results[METRIC_MAP[wm_type]]


def compact_results(results):
    return {
        "bit_accuracy": results["bit_accuracy"],
        "bit_accuracy_status": results["bit_accuracy_status"],
        "bit_accuracy_error": results["bit_accuracy_error"],
        "bit_accuracy_error_count": results["bit_accuracy_error_count"],
        "metric_schema_version": results["metric_schema_version"],
        "p_value": results["p_value"],
        "value": results["value"],
        "l1_dist": results["l1_dist"],
        "detection_success": results["detection_success"],
        "log_message": results["log_message"],
        "prc_threshold": results["prc_threshold"],
        "psnr": results["psnr"],
        "ssim": results["ssim"],
        "msssim": results["msssim"],
        "lpips": results["lpips"],
    }


def main():
    args = parse_args()

    wm_provider_cls = WmProviders[args.wm_type].value
    if args.wm_type == "GS" and int(args.num) != 1:
        raise RuntimeError(
            "Generic GS runners allow only an explicitly single-sample legacy/debug run; "
            "use experiments/generate_watermarked_images.py for auditable per-run GS state"
        )
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    # Standalone GS reproduction runner: adopt official upstream generation
    # defaults (stabilityai SD2.1-base + DPM + fp16 weight revision + an
    # official-inspired inversion approximation) unless the user specified them
    # explicitly. The official-inspired inversion note is only emitted when actually
    # on the official model + DPM (see apply_official_reproduction_defaults). Other
    # methods and the formal generator are unaffected.
    if args.wm_type == "GS":
        from utils.wm.gs_provider import apply_official_reproduction_defaults
        gs_official_defaults = apply_official_reproduction_defaults(args, sys.argv)
        print(f"[GS] official reproduction defaults applied: {gs_official_defaults}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)

    out_dir = os.path.join(
        args.out_dir,
        MODEL_NAME_MAPPING.get(args.modelid_target, "target"),
        f"wm-{args.wm_type}",
        f"rm-{args.rm_method}",
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = None
    if args.logger:
        logger = get_logger(out_dir, f"{MODEL_NAME_MAPPING.get(args.modelid_target, 'target')}_{args.wm_type}_{args.rm_method}")

    detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)
    legacy_threshold_metadata = describe_legacy_detection_threshold(args.wm_type, args.modelid_target)
    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)

    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=True if args.modelid_target in EAGER_LOAD_MODELS else False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
        revision=getattr(args, "model_revision", None),
    )

    wm_provider = wm_provider_cls(
        latent_shape=pipe_provider_target.get_latent_shape(),
        dtype=pipe_provider_target.get_dtype(),
        device=device,
        **vars(args),
    )

    # GS detection uses the official beta-tail threshold family (default
    # official_onebit, ">="); legacy_default must be requested explicitly.
    gs_detection_info = (
        wm_provider.active_detection_threshold() if args.wm_type == "GS" else None
    )

    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]
    message_bits_str_initial = (
        wm_initial_results["message_bits_str_list"][0]
        if "message_bits_str_list" in wm_initial_results
        else None
    )

    remover = None
    all_rows = []
    for run_id, target_prompt in tqdm(enumerate(target_prompts), total=len(target_prompts), desc="Removal"):
        if args.wm_type in ["PRC"]:
            seed_everything(args.seed)

        if hasattr(wm_provider, "generate"):
            generated = wm_provider.generate(
                pipe_provider_target=pipe_provider_target,
                prompts=target_prompt,
                latents=wm_zT,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
        else:
            generated = pipe_provider_target.generate(
                prompts=target_prompt,
                latents=wm_zT,
                num_inference_steps=args.num_inference_steps_target,
                guidance_scale=args.guidance_scale_target,
            )
        watermarked_image = generated["images_PIL"][0]

        before = validate(
            out_dir=out_dir,
            image_to_verify_PIL=watermarked_image,
            original_PIL=watermarked_image,
            wm_provider=wm_provider,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            step=-1,
            message_bits_str_initial=message_bits_str_initial,
            do_psnr=False,
            do_ssim=False,
            do_msssim=False,
            do_lpips=False,
        )
        if args.wm_type in ["SPH", "T2S"]:
            before_successful = None
        elif args.wm_type in ["PRC", "MAXSIVE", "SHALLOW", "GM"]:
            before_successful = before["detection_success"]
        elif args.wm_type == "GS":
            before_successful = wm_provider.is_detection_successful(
                detection_value(before, args.wm_type)
            )
        else:
            before_successful = check_if_detection_successful(
                wm_type=args.wm_type,
                threshold=detection_threshold,
                value=detection_value(before, args.wm_type),
            )

        pipe_provider_target.stash_pipe()
        if remover is None:
            remover = get_remover(
                args.rm_method,
                model_id=args.rm_modelid,
                device=device,
                resolution=args.resolution,
                seed=args.seed,
            )
        removed_image = remover.remove(
            watermarked_image,
            num_inference_steps=args.rm_steps,
            xy=args.rm_xy,
            z=args.rm_z,
            guidance_scale=args.rm_guidance_scale,
        )

        after = validate(
            out_dir=out_dir,
            image_to_verify_PIL=removed_image,
            original_PIL=watermarked_image,
            wm_provider=wm_provider,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            step=run_id,
            message_bits_str_initial=message_bits_str_initial,
            do_psnr=True,
            do_ssim=True,
            do_msssim=True,
            do_lpips=False,
        )
        if args.wm_type in ["SPH", "T2S"]:
            after_successful = None
        elif args.wm_type in ["PRC", "MAXSIVE", "SHALLOW", "GM"]:
            after_successful = after["detection_success"]
        elif args.wm_type == "GS":
            after_successful = wm_provider.is_detection_successful(
                detection_value(after, args.wm_type)
            )
        else:
            after_successful = check_if_detection_successful(
                wm_type=args.wm_type,
                threshold=detection_threshold,
                value=detection_value(after, args.wm_type),
            )

        if args.save_images:
            image_dir = os.path.join(out_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            watermarked_image.save(os.path.join(image_dir, "watermarked.png"))
            removed_image.save(os.path.join(image_dir, "removed.png"))

        before_compact = compact_results(before)
        after_compact = compact_results(after)
        row = {
            "run_id": run_id,
            "prompt": target_prompt,
            "wm_type": args.wm_type,
            "target_model": args.modelid_target,
            "rm_method": args.rm_method,
            "rm_modelid": args.rm_modelid,
            "rm_steps": args.rm_steps,
            "rm_xy": args.rm_xy,
            "rm_z": args.rm_z,
            "before_detection_successful": before_successful,
            "after_detection_successful": after_successful,
            "detection_threshold": (
                gs_detection_info["threshold"]
                if gs_detection_info is not None
                else legacy_threshold_metadata["threshold"]
            ),
            "detection_threshold_type": (
                gs_detection_info["threshold_type"]
                if gs_detection_info is not None
                else legacy_threshold_metadata["threshold_type"]
            ),
            "detection_threshold_nominal_fpr": (
                gs_detection_info["nominal_fpr"]
                if gs_detection_info is not None
                else legacy_threshold_metadata["nominal_fpr"]
            ),
            "threshold_calibrated_from_current_clean_negatives": False,
        }
        if gs_detection_info is not None:
            row.update({
                "gs_detection_mode": gs_detection_info["detection_mode"],
                "gs_official_tau_onebit": gs_detection_info["official_tau_onebit"],
                "gs_official_tau_bits": gs_detection_info["official_tau_bits"],
                "detection_threshold_comparison_operator": gs_detection_info["comparison_operator"],
            })
        row.update({f"before_{key}": value for key, value in before_compact.items()})
        row.update({f"after_{key}": value for key, value in after_compact.items()})
        all_rows.append(row)

        message = (
            f"run={run_id + 1}/{len(target_prompts)} "
            f"before={before_successful} after={after_successful} "
            f"before_metric={detection_value(before, args.wm_type):.5f} "
            f"after_metric={detection_value(after, args.wm_type):.5f}"
        )
        if logger:
            logger.info(message)
            if before["log_message"]:
                logger.info(f"before: {before['log_message']}")
            if after["log_message"]:
                logger.info(f"after: {after['log_message']}")
        else:
            print(message)
            if before["log_message"]:
                print(f"before: {before['log_message']}")
            if after["log_message"]:
                print(f"after: {after['log_message']}")

        if args.save:
            pd.DataFrame(format_staged_bit_accuracy_rows(all_rows)).to_csv(
                os.path.join(out_dir, f"{args.rm_method}_{args.wm_type}_removal.csv"),
                index=False,
            )

    if args.save:
        pd.DataFrame(format_staged_bit_accuracy_rows(all_rows)).to_csv(
            os.path.join(out_dir, f"{args.rm_method}_{args.wm_type}_removal.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
