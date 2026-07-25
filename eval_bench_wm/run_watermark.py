"""
Script to run the watermark.
"""

import PIL
import PIL.Image
import torch
import torch.nn.functional as F

import pandas as pd

from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.prc_provider import parser as prc_parser
from utils.wm.tag_provider import parser as tag_parser
from utils.wm.ringid_provider import parser as ringid_parser
from utils.wm.hstr_provider import parser as hstr_parser
from utils.wm.hsqr_provider import parser as hsqr_parser
from utils.wm.sph_provider import parser as sph_parser
from utils.wm.t2s_provider import parser as t2s_parser
from utils.wm.maxsive_provider import parser as maxsive_parser
from utils.wm.shallow_provider import parser as shallow_parser
from utils.wm.gm_provider import parser as gm_parser

from utils.pipe import pipe_utils
from utils.imprint_utils import invert_image, validate
from utils.image_utils import distort_images, check_flag
from utils.bit_accuracy import format_bit_accuracy_value
# from utils.finger_utils import decode_tensors, decode_tensors_reverse#, spd_dist_latents, matched_spd_distance
from utils.utils import get_detection_threshold, check_if_detection_successful
from utils.utils import set_random_seed, seed_everything

from utils.prompt_utils import get_text_prompts

import os
import sys
from tqdm import tqdm
from utils.logger import get_logger


model_id = ["CompVis/stable-diffusion-v1-4",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            "cagliostrolab/animagine-xl-3.0",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            "THUDM/CogView4-6B"]

model_flux = ["black-forest-labs/FLUX.1-dev",
              "stabilityai/stable-diffusion-3-medium-diffusers",
              "stabilityai/stable-diffusion-3.5-medium",
              "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
              "THUDM/CogView4-6B"]

model_name_mapping = {
    "CompVis/stable-diffusion-v1-4": "sd14",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "stabilityai/stable-diffusion-2-1-base": "sd21",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "pixart",
    "PixArt-alpha/PixArt-XL-2-512x512": "pixart-xl",
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "stabilityai/stable-diffusion-3.5-medium": "sd35m",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers": "sana",
    "THUDM/CogView4-6B": "cogview4",
}

parent_parsers = [
    gs_parser, tr_parser, prc_parser,
    tag_parser, ringid_parser, hstr_parser,
    hsqr_parser, sph_parser, t2s_parser,
    maxsive_parser, shallow_parser,
    gm_parser,
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
import argparse


def build_parser():
    """Construct the run_watermark argument parser.

    Kept side-effect free (no model loading) so tests can build and parse a
    command line without importing/running the full generation body.
    """
    parser = argparse.ArgumentParser(description="test_watermark", parents=parent_parsers)

    parser.add_argument("--out_dir", type=str, default="out/watermark_gen/")
    parser.add_argument("--target_prompt", type=str, default="cat standing on a rock in front of a crowd of cats, backlighting, digital art, trending on pixiv, fanart")

    # target model
    parser.add_argument("--modelid_target",
                        type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        choices=[model for model in model_id])
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 20 for FLUX, 28 for SD3, 20 for Pix
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3, 4.5 for Pix

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--wm_type",
                        type=str,
                        default="GS",
                        choices=[wm.name for wm in WmProviders])
    parser.add_argument("--distort", action="store_true", default=False)

    # dataset
    parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--logger", action="store_true", default=False)
    return parser


def value_log_label(wm_type):
    if wm_type == "PRC":
        return "PRC value"
    if wm_type == "T2S":
        return "T2S norm1_w"
    if wm_type == "MAXSIVE":
        return "MAXSIVE value"
    if wm_type == "GM":
        return "GM value"
    return "value"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args, unknown_args = build_parser().parse_known_args(argv)

    if args.modelid_target == "stabilityai/stable-diffusion-3.5-medium":
        if "--num_inference_steps_target" not in sys.argv:
            args.num_inference_steps_target = 40
        if "--guidance_scale_target" not in sys.argv:
            args.guidance_scale_target = 4.5

    wm_provider_cls = WmProviders[args.wm_type].value
    if args.wm_type == "GS" and int(args.num) != 1:
        raise RuntimeError(
            "Generic GS runners allow only an explicitly single-sample legacy/debug run; "
            "use experiments/generate_watermarked_images.py for auditable per-run GS state"
        )
    if hasattr(wm_provider_cls, "apply_arg_defaults"):
        wm_provider_cls.apply_arg_defaults(args, sys.argv)

    # Standalone GS reproduction runner: adopt official-inspired upstream generation
    # defaults (stabilityai SD2.1-base + DPM + fp16 weight revision + official-inspired
    # inversion configuration) unless the user specified them explicitly. This does NOT
    # affect other methods or the formal generator (which keeps the shared
    # RedbeardNZ + DDIM cohort).
    if args.wm_type == "GS":
        from utils.wm.gs_provider import apply_official_reproduction_defaults
        _gs_official_defaults = apply_official_reproduction_defaults(args, argv)
        print(f"[GS] official reproduction defaults applied: {_gs_official_defaults}", flush=True)

    # set seeds
    set_random_seed(args.seed)

    # logger
    logger = None
    if args.logger:
        log_name = f"{model_name_mapping[args.modelid_target]}_{args.wm_type}"
        logger = get_logger(args.out_dir, log_name)

    # retrieve the detection threshold for the settings
    detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)
    target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
    # target_prompts = target_prompts[1:]

    # pipe_provider used by the target model (SDXL, PixArt, FLUX)
    pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                        resolution=args.resolution,
                                                        device=DEVICE,
                                                        eager_loading=True if args.modelid_target in model_flux else False,
                                                        schedulers_name=args.scheduler_target,
                                                        disable_tqdm=True,
                                                        revision=getattr(args, "model_revision", None),)

    # generate a watermarked latent zT
    # This way like it is done here is a simple way to obtain a watermark provider for a simple test run.
    # If you want to do mass experiments and have batch_sizes > 1, plz have look at the utils.wm_provider.WmProvider.generate_providers method
    wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), device=DEVICE, **vars(args))
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]

    # for Gaussian Shading, we also get an initial message
    message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

    metric_map = {
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
        "SHALLOW": "l1_dist",
        "GM": "value",
    }

    for id, (target_prompt) in tqdm(enumerate(target_prompts), total=len(target_prompts), desc="Generating images"):
        print(f"\n--- Starting run {id+1}/{len(target_prompts)} ---")
        # print(f"Target prompt: {target_prompt}")
        if args.logger:
            logger.info("Test single watermarked image")

        if args.wm_type in ["PRC", "SPH"]:
            seed_everything(args.seed)

        # generate a watermarked image with the target model
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

        generated_PIL_list = generated["images_PIL"]
        benign_image = generated_PIL_list[0]
        benign_image.save("watermarked_image.png")

        # from PIL import Image
        # benign_image = Image.open("sana_vae_TR.png").convert("RGB")

        if args.distort:
            benign_image = distort_images(benign_image, jpeg_ratio=20)
            benign_image.save("distort_image.png")

        # distort param:
        # r_degree=(0, 150), jpeg_ratio=(10, 90), sp_prob_fixed=(0.05, 0.4), crop_scale_TR=(0.9 raw, 0.5),
        # random_crop_ratio=(0.1-0.5), random_drop_ratio=(0.1, 0.8), gaussian_std_fixed=(0.05, 0.4), :done
        # median_blur_k=(1,3,5,...,17), gaussian_blur_r=(2,4,6,8,10), resize_ratio=(0.1, 0.9),
        # brightness_factor=(2,16), contrast_factor,
        # vertical_shift_ratio=(0.1, 0.8), horizontal_shift_ratio=(0.1, 0.8), flip_ratio=1,

        if args.wm_type != "SHALLOW":
            with torch.no_grad(): # retrieve zT
                zT_retrieved = pipe_provider_target.invert_images(benign_image, num_inference_steps=args.num_inference_steps_target)["zT_torch"]
            # mse = F.mse_loss(wm_zT, zT_retrieved)
            # print("mse:", mse.item())

        # benign_image = PIL.Image.open("watermarked_image.png")

        rows = []
        results = validate(
                out_dir=args.out_dir,
                image_to_verify_PIL=benign_image,
                original_PIL=benign_image,
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

        # check if detection was successfull
        if args.wm_type in ["SPH", "T2S"]:
            detection_successful = None
        elif args.wm_type in ["PRC", "MAXSIVE", "SHALLOW", "GM"]:
            detection_successful = results["detection_success"]
        elif args.wm_type == "GS":
            # Official Gaussian Shading detection: default gs_detection_mode is
            # official_onebit (beta-tail tau_onebit, ">="); legacy_default must be
            # requested explicitly. Never uses the legacy fixed GS_THRESHOLDS by default.
            detection_successful = wm_provider.is_detection_successful(results[metric_map[args.wm_type]])
        else:
            detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                                threshold=detection_threshold,
                                                                value=results[metric_map[args.wm_type]])
        results["detection_successful"] = detection_successful

        rows.append({
                    "bit_accuracy": results["bit_accuracy"],
                    "p_value": results["p_value"],
                    "value": results["value"],
                    "detection_success": results["detection_success"],
                    "log_message": results["log_message"],
                    "prc_threshold": results["prc_threshold"],
                    "detection_successful": results["detection_successful"],
                    })
        value_label = value_log_label(args.wm_type)
        bit_accuracy_display = format_bit_accuracy_value(
            results["bit_accuracy"],
            results["bit_accuracy_status"],
            results["bit_accuracy_error"],
        )

        if args.logger:
            logger.info(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display}, p_value: {results['p_value']}, {value_label}: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
            if results["log_message"]:
                logger.info(results["log_message"])
        else:
            print(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display}, p_value: {results['p_value']}, {value_label}: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
            if results["log_message"]:
                print(results["log_message"])


if __name__ == "__main__":
    main()
