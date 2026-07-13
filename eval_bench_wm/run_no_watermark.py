"""
Script to run no watermark.
"""

import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from utils.pipe import pipe_utils
from utils.utils import set_random_seed
from utils.prompt_utils import get_text_prompts

import argparse
from tqdm import tqdm
import pandas as pd
from utils.logger import get_logger

model_id = ["CompVis/stable-diffusion-v1-4",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", 
            "cagliostrolab/animagine-xl-3.0",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "THUDM/CogView4-6B"]

model_flux = ["black-forest-labs/FLUX.1-dev",
              "stabilityai/stable-diffusion-3-medium-diffusers",
              "THUDM/CogView4-6B"]

model_name_mapping = {
    "CompVis/stable-diffusion-v1-4": "sd14",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "THUDM/CogView4-6B": "cogview4",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "pixart",
    "PixArt-alpha/PixArt-XL-2-512x512": "pixart-xl",
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-2-1-base": "sd21",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
parser = argparse.ArgumentParser(description="test_no_watermark")

parser.add_argument("--out_dir", type=str, default="out/watermark_gen/")
parser.add_argument("--target_prompt", type=str, default="cat standing on a rock in front of a crowd of cats, backlighting, digital art, trending on pixiv, fanart")

# target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0",
                    choices=[model for model in model_id])
parser.add_argument("--scheduler_target", type=str, default="DDIM")
parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 20 for FLUX, 28 for SD3
parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3

parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=100)
parser.add_argument("--logger", action="store_true", default=False)
parser.add_argument("--save", action="store_true", default=False)

# dataset
parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])

args, unknown_args = parser.parse_known_args()

# set seeds
set_random_seed(args.seed)

# logger
if args.logger:
    log_name = f"{model_name_mapping[args.modelid_target]}_no_wm"
    logger = get_logger(args.out_dir, log_name)

target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
# target_prompts = target_prompts[1:]

# pipe_provider used by the target model (SDXL, PixArt, FLUX)
pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                    resolution=args.resolution,
                                                    device=DEVICE,
                                                    eager_loading=True if args.modelid_target in model_flux else False,
                                                    disable_tqdm=True,)

os.makedirs(args.out_dir, exist_ok=True)

zT_list = []
ret_list = []

all_results = []
for id, (target_prompt) in tqdm(enumerate(target_prompts), total=len(target_prompts), desc="Generating images"):
    print(f"\n--- Starting run {id+1}/{len(target_prompts)} ---")
    # print(f"Target prompt: {target_prompt}")
    if args.logger:
        logger.info("Test unwatermarked image")

    # generate a watermarked image with the target model
    zT = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    generated_PIL_list = pipe_provider_target.generate(prompts=target_prompt,
                                                    latents=zT,
                                                    num_inference_steps=args.num_inference_steps_target,
                                                    guidance_scale=args.guidance_scale_target,
                                                    )["images_PIL"]
    benign_image = generated_PIL_list[0]
    
    with torch.no_grad(): # retrieve zT
        zT_retrieved = pipe_provider_target.invert_images(benign_image, num_inference_steps=args.num_inference_steps_target)["zT_torch"]
    
    if args.save:
        save_path = os.path.join(args.out_dir, f"benign_{id}.png")
        benign_image.save(save_path)

    # zT_cpu = zT.detach().cpu().to(torch.float32)          # 移到 CPU，避免占 GPU
    # zT_ret_cpu = zT_retrieved.detach().cpu().to(torch.float32)

    # zT_list.append(zT_cpu)
    # ret_list.append(zT_ret_cpu)

    # zT_latents  = torch.cat(zT_list,  dim=0)   # [N * B, C, H, W]
    # ret_latents = torch.cat(ret_list, dim=0)   # [N * B, C, H, W]

    # torch.save(
    #     {
    #         "zT_latents": zT_latents,
    #         "ret_latents": ret_latents,
    #     },
    #     "gs_zT_latents_pairs.pt",
    # )