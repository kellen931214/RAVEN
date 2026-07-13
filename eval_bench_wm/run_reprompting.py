"""
Script to run the reprompting attack.
"""

import os
import torch
import pandas as pd
import argparse
import torch.nn.functional as F
from utils.bit_accuracy import format_bit_accuracy_value
from utils.imprint_utils import validate
from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.prc_provider import parser as prc_parser
from utils.wm.tag_provider import parser as tag_parser
from utils.wm.ringid_provider import parser as ringid_parser
from utils.wm.hstr_provider import parser as hstr_parser
from utils.wm.hsqr_provider import parser as hsqr_parser
from utils.wm.sph_provider import parser as sph_parser
# from utils.wm.lwe_provider import parser as lwe_parser

from utils.utils import get_detection_threshold, check_if_detection_successful

from utils.pipe import pipe_utils

from utils.prompt_utils import PROMPTS_SD_LIST, PROMPTS_I2P_LIST

from utils.utils import set_random_seed, seed_everything
from utils.prompt_utils import get_text_prompts, get_harmful_prompts_from_huggingface
from tqdm import tqdm

from utils.logger import get_logger
from utils.image_utils import check_flag
from utils.finger_utils import decode_tensors, decode_tensors_reverse
# , spd_dist_latents


def bit_accuracy_display(results):
    return format_bit_accuracy_value(
        results["bit_accuracy"],
        results["bit_accuracy_status"],
        results["bit_accuracy_error"],
    )

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
    "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning": "sd21",
}

model_id = ["CompVis/stable-diffusion-v1-4",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", 
            "black-forest-labs/FLUX.1-dev",
            "THUDM/CogView4-6B",
            "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning", ]

model_flux = ["black-forest-labs/FLUX.1-dev",
              "stabilityai/stable-diffusion-3-medium-diffusers",
              "THUDM/CogView4-6B"
            ]

parent_parsers = [
    gs_parser, tr_parser, prc_parser, 
    tag_parser, ringid_parser, hstr_parser,
    hsqr_parser, sph_parser,
]

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# args
parser = argparse.ArgumentParser(description="reprompt", parents=parent_parsers)

parser.add_argument("--out_dir", type=str, default="out/reprompt/")
parser.add_argument("--num", type=int, default=100)

# prompts
parser.add_argument("--target_prompt_index", type=int, default=100)
parser.add_argument("--target_prompt", type=str, default=None)
parser.add_argument("--attacker_prompt_index", type=int, default=100)
parser.add_argument("--attacker_prompt", type=str, default=None)

# target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0",
                    choices=[model for model in model_id])
parser.add_argument("--scheduler_target", type=str, default="DDIM")
parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3, 4.5 for Pix
parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 20 for FLUX, 28 for SD3, 20 for Pix

# attacker model
parser.add_argument("--modelid_attacker", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--scheduler_attacker", type=str, default="DDIM")
parser.add_argument("--guidance_scale_attacker", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3, 4.5 for Pix
parser.add_argument("--num_inference_steps_attacker", type=int, default=50)  # 20 for FLUX, 28 for SD3, 20 for Pix

parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])

parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--wm_type",
                    type=str,
                    default="GS",
                    choices=[wm.name for wm in WmProviders])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resample", action="store_true", default=False)
parser.add_argument("--logger", action="store_true", default=False)
parser.add_argument("--save", action="store_true", default=False)

args = parser.parse_args()

# args.target_prompt_index = int(args.num)
# args.attacker_prompt_index = int(args.num)

# save outputin a subfolder defined by the index of the prompt we're using if explicit prompt is not given
# out_dir = os.path.join(args.out_dir,
#                        f"target_prompt_index={args.target_prompt_index if args.target_prompt is None else 'custom'}",
#                        f"attacker_prompt_index={args.attacker_prompt_index if args.attacker_prompt is None else 'custom'}")
out_dir = os.path.join(args.out_dir, model_name_mapping[args.modelid_target], 
                       model_name_mapping[args.modelid_attacker],
                       f"wm-{args.wm_type}")

# prompts are taken from a predefined list of SD-Prompts (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
# target_prompts = PROMPTS_SD_LIST[args.target_prompt_index] if args.target_prompt is None else args.target_prompt
target_prompts = get_text_prompts(num_prompts = args.num, dataset_id=args.dataset_id)

# prompts are taken from a predefined list of I2P (https://huggingface.co/datasets/AIML-TUDA/i2p)
# attacker_prompts = PROMPTS_I2P_LIST[args.attacker_prompt_index] if args.attacker_prompt is None else args.attacker_prompt
attacker_prompts = get_text_prompts(num_prompts = args.num, from_end=True, dataset_id=args.dataset_id)

# target_prompts = target_prompts[509:]
# attacker_prompts = attacker_prompts[509:]
# attacker_prompts = get_harmful_prompts_from_huggingface(args.attacker_prompt_index)
# add full prompt datasets here if you like

# attacker model
pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                    resolution=args.resolution,
                                                    schedulers_name=args.scheduler_target,
                                                    unet_id_or_checkpoint_dir=None,
                                                    lora_checkpoint_dir=None,
                                                    device=DEVICE,
                                                    eager_loading=True if args.modelid_target in model_flux else False,
                                                    disable_tqdm=True
                                                    )  # finetuned model
pipe_provider_attacker = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_attacker,
                                                      resolution=args.resolution,
                                                      device=DEVICE,
                                                      eager_loading=True if args.modelid_attacker in model_flux else False,
                                                      disable_tqdm=True
                                                      )  # base model


if args.logger:
    log_name = f"{model_name_mapping[args.modelid_target]}_{model_name_mapping[args.modelid_attacker]}_{args.wm_type}"
    logger = get_logger(out_dir, log_name)

    logger.info(f"Starting reprompting run")
    logger.info(f"WM Type: {args.wm_type}")
    logger.info(f"Target Model: {args.modelid_target}")
    logger.info(f"Attacker Model: {args.modelid_attacker}")
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Total prompts: {args.num}")
            
# set seeds
set_random_seed(args.seed)

# retrieve the detection threshold for the settings
detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)

metric_map = {
    "PRC": "value",
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
    "GS": "bit_accuracy",
    # "LWE": "bit_accuracy",
    "TAG": "bit_accuracy",
}

wm_list = []
ret_list = []

results_data = []
# header = ['max', 'mean', 'std', 'topk_mean']
# header = ["inter_max", "inter_mean", "inter_std", "inter_topk_mean", "intra_max", "intra_mean", "intra_std", ]
all_results = []

for id, (target_prompt, attacker_prompt) in tqdm(enumerate(zip(target_prompts, attacker_prompts)), total=len(target_prompts), desc="Generating images"):
    print(f"\n--- Starting run {id+1}/{len(target_prompts)} ---")
    rows = [] # A new list for each run's metrics
    with torch.no_grad():
        
        # --------------------------------------------------------------- PHASE 1 ----------------------------------------------------------------------
        # if args.logger == True:
        #     logger.info("phase 1: generate target image")
        if args.wm_type in ["PRC", "LWE"]:
            seed_everything(args.seed)

        # generate a watermarked latent zT
        wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
        wm_initial_results = wm_provider.get_wm_latents()
        wm_zT = wm_initial_results["zT_torch"]
        
        # generate a benign image
        res_1 = pipe_provider_target.generate(prompts=target_prompt,
                                              num_inference_steps=args.num_inference_steps_target,
                                              guidance_scale=args.guidance_scale_target,
                                              latents=wm_zT,
                                            #   callback_on_step_end=decode_tensors,
                                            #   callback_on_step_end_tensor_inputs=["latents"]
                                              )
                                              
        benign_image = res_1["images_PIL"][0]
            
        # for Gaussian Shading, we also get an initial message
        message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

        # collect metrics
        results = validate(
            out_dir=out_dir,
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
      
        detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                             threshold=detection_threshold,
                                                             value=results[metric_map[args.wm_type]])
    
        results["detection_successful"] = detection_successful
        # rows.append(results)
        
        # log
        if args.logger:
            logger.info(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display(results)}, p_value: {results['p_value']}, PRC value: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
        else:
            print(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display(results)}, p_value: {results['p_value']}, PRC value: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
        
        # --------------------------------------------------------------- PHASE 2 ----------------------------------------------------------------------
        # print("phase 2: invert using attacker model")
        
        pipe_provider_target.stash_pipe()
        res_2 = pipe_provider_attacker.invert_images(images=res_1["images_torch"], num_inference_steps=args.num_inference_steps_attacker)

        # --------------------------------------------------------------- PHASE 3 ----------------------------------------------------------------------
        # print("phase 3: generate attacker image")

        # resample strategy, used in Reprompt+ attack, but only for GS
        # For TR, there is no resampling, but merely trying out multiple attacker prompts and choosing the best performing sample
        if args.resample:
            recovered_zT = wm_provider.wiggle_latents(res_2["zT_torch"].clone())
            recovered_zT = recovered_zT.to(dtype=pipe_provider_attacker.get_dtype())
        else:
            recovered_zT = res_2["zT_torch"].clone()

        # generate a harmful image
        res_3 = pipe_provider_attacker.generate(prompts=attacker_prompt,
                                                num_inference_steps=args.num_inference_steps_attacker,
                                                guidance_scale=args.guidance_scale_attacker,
                                                latents=recovered_zT,
                                                # callback_on_step_end=decode_tensors,
                                                # callback_on_step_end_tensor_inputs=["latents"]
                                                )
        harmful_image = res_3["images_PIL"][0]
        harmful_image.save("reprompt_img.png")
        # --------------------------------------------------------------- PHASE 4 ----------------------------------------------------------------------
        # print("phase 4: invert using target model and verify watermark")

        pipe_provider_attacker.stash_pipe()
        
        # collect metrics
        results_harmful = validate(
            out_dir=out_dir,
            image_to_verify_PIL=harmful_image,
            original_PIL=benign_image,
            wm_provider=wm_provider,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            step=1,
            message_bits_str_initial=message_bits_str_initial,
            do_psnr=False,
            do_ssim=False,
            do_msssim=False,
            do_lpips=False,
            # callback_on_step_end=decode_tensors_reverse,
            # callback_on_step_end_tensor_inputs=["latents"]
            )
        
        with torch.no_grad(): # retrieve zT
            zT_retrieved = pipe_provider_target.invert_images(harmful_image, num_inference_steps=args.num_inference_steps_target)["zT_torch"]
        
        # mse = F.mse_loss(wm_zT, zT_retrieved)
        # print(mse.item())

        # dist = spd_dist_latents(wm_zT, zT_retrieved, use_airm=True)
        # print("SPD distance:", dist)

        # from utils.spd_utils import spd_dist_latents, matched_spd_distance
        # d_corr = spd_dist_latents(wm_zT, zT_retrieved)
        # # print("SPD distance (corr):", d_corr)
        # d_auto, sigma = matched_spd_distance(wm_zT, zT_retrieved)
        # d_final = min(d_corr, d_auto)
        
        # if args.logger:
        #     logger.info(f"SPD distance (corr): {d_final}")
        # else:
        #     print("SPD distance (corr):", d_final)
            # print("Gap:", spread_gap)
        # # logger.info(f"SPD distance (corr): {d_final}")

        # wm_zT_cpu = wm_zT.detach().cpu().to(torch.float32)          # 移到 CPU，避免占 GPU
        # zT_ret_cpu = zT_retrieved.detach().cpu().to(torch.float32)

        # wm_list.append(wm_zT_cpu)
        # ret_list.append(zT_ret_cpu)

        # from utils.spd_utils import multi_spd_airm_dist_latents
        # d2 = multi_spd_airm_dist_latents(wm_zT, zT_retrieved, n_parts=4)
        # print("multi-patch SPD dist:", d2)
        

        # from utils.finger_utils import spectral_similarity
        # sim = spectral_similarity(wm_zT, zT_retrieved)
        # print("Spectral similarity:", sim)

        # from utils.spd_utils import compare_curvature_entropy
        # result_ce = compare_curvature_entropy(wm_zT, zT_retrieved)
        # print(result_ce['K_A'])
        # print(result_ce['K_B'])

        # from utils.spd_utils import bures_distance
        # d_bures = bures_distance(wm_zT, zT_retrieved)
        # print(d_bures)

        # from utils.spd_utils import spectral_features, compute_S_spec,topk_interpatch_airm, intra_latent_patch_airm
        # # f_wm = spectral_features(wm_zT)
        # # f_wm_retrieved = spectral_features(zT_retrieved)
        # # print("Reverse spectral: ", f_wm_retrieved)
        # # results_data.append([d_final, f_wm_retrieved['condition_number']])
        # # inter_score = topk_interpatch_airm(wm_zT, zT_retrieved, grid_size=4, topk=3)
        # intra_score = intra_latent_patch_airm(zT_retrieved, grid_size=8)
        # print(intra_score)
        # if intra_score['patch_mean_std'] < 0.125:
        #     if d_final < 0.37:
        #         result = "No Detection"
        #     else:
        #         result = "Detect Forgery"
        # else:
        #     result = "Detect Forgery"
            
        # if args.logger:
        #     logger.info(f"score={intra_score['patch_mean_std']}")
        #     logger.info(f"Detection: {result}")
        # else:
        #     print(f"score={intra_score['patch_mean_std']}")
        #     print(f"Detection: {result}")
        # data = {
        #     "inter_max": inter_score["max"],
        #     "inter_mean": inter_score["mean"],
        #     "inter_std": inter_score["std"],
        #     "inter_topk_mean": inter_score["topk_mean"],

        #     "intra_max": intra_score["max"],
        #     "intra_mean": intra_score["mean"],
        #     "intra_std": intra_score["std"],
        # }

        # results_data.append(data)

        # S, z = compute_S_spec(f_wm_retrieved)
        # print(S)
        # if S > 3:
        #     print("Detect Forgery")
        # else:
        #     print("No Forgery Detected")
        
        # from utils.spd_utils import local_spd_features, local_airm_score
        # feat_w = local_spd_features(wm_zT)
        # feat_r = local_spd_features(zT_retrieved)
        # # score, ratio  = local_airm_score(feat_w, feat_r)
        # score = local_airm_score(feat_w, feat_r)
        # print("Local AIRM score:", score)
    
        # base_path = "./latent_hash"
        # tmp_flag_path = os.path.join(base_path, "tmp_flag.txt")
        # flag_status = check_flag(tmp_flag_path)
        # pool_path = os.path.join(base_path, "hash_pool.npy")
        # os.remove(pool_path)
        # if not flag_status:
            # import sys
            # results["detection_successful"] = False
            # sys.exit(0) 
    
        if args.wm_type == "PRC":
            log_message = results_harmful["log_message"]
            rows.append(log_message)

        # check if detection was successfull
        detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                             threshold=detection_threshold,
                                                             value=results_harmful[metric_map[args.wm_type]])
        harmful_image_summary = {
            "run_id": id + 1,
            "log_message": f"(Harmful image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display(results_harmful)}, p_value: {results_harmful['p_value']}. PRC value: {results_harmful['value']:.5f}, l1_dist: {-results_harmful['l1_dist']:.5f}"
        }
    
        rows.append(harmful_image_summary)
        
        # log
        if args.logger:
            logger.info(f"(Harmful image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display(results_harmful)}, p_value: {results_harmful['p_value']}. PRC value: {results_harmful['value']:.5f}, l1_dist: {-results_harmful['l1_dist']:.5f}")
        else:
            print(f"(Harmful image) detection_success: {detection_successful}, bit accuracy: {bit_accuracy_display(results_harmful)}, p_value: {results_harmful['p_value']}. PRC value: {results_harmful['value']:.5f}, l1_dist: {-results_harmful['l1_dist']:.5f}")

    all_results.extend(rows)

# wm_latents  = torch.cat(wm_list,  dim=0)   # [N * B, C, H, W]
# ret_latents = torch.cat(ret_list, dim=0)   # [N * B, C, H, W]

# torch.save(
#     {
#         "wm_latents": wm_latents,
#         "ret_latents": ret_latents,
#     },
#     f"{args.wm_type}_forged_latents_sd21_sd21.pt",
# )

if args.logger:
    logger.info(f"--- Run {id+1}/{len(target_prompts)} completed. ---\n")
else:
    print(f"--- Run {id+1}/{len(target_prompts)} completed. ---\n")
 
# import csv
# filename = f"dual_{args.wm_type}_num_{len(target_prompts)}_{args.dataset_id}.csv"
# with open(os.path.join(out_dir, filename), 'w', newline='', encoding='utf-8') as file:
#     # writer = csv.writer(file)
#     # writer.writerow(header)
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(results_data)
# print(f"\n save csv {filename}")

if args.save:
    df = pd.DataFrame(all_results)
    filename = f"{args.wm_type}_num_{len(target_prompts)}_{args.dataset_id}.csv"
    df.to_csv(os.path.join(out_dir, filename))
