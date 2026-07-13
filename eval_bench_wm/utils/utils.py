import random

import torch
import numpy as np

import PIL.Image
import os

# Gaussian Shading FPR=10-6
# caluclated using get_GS_thresholds, but you need mp math and it is not easy to install via pip
# This is the threshold for the multi-bit setting with 100k users
GS_THRESHOLDS = 0.70703125
SPH_THRESHOLD = GS_THRESHOLDS
# single:
#         Threshold for fpri 1e-06     :0.6484375                     
#         Threshold for fpri 1e-16     :0.75                          
#         Threshold for fpri 1e-32     :0.8515625                     
#         Threshold for fpri 1e-64     :0.97265625                    
# multi:
#         Threshold for fpri 1e-06     :0.70703125                    
#         Threshold for fpri 1e-16     :0.7890625                     
#         Threshold for fpri 1e-32     :0.875                         
#         Threshold for fpri 1e-64     :0.984375


# Tree-Ring FPR=1%
# empricially measured with 5000 watermarked vs 5000 non-watermarked images on each model
TR_THRESHOLD_SD2 =  0.0244251404325773 # (10^-3: 0.006185474165907805, 10^-4: 0.0008937217604252055)
TR_THRESHOLD_SDXL =  0.0261008646426459
TR_THRESHOLD_PIXART = 0.015917123872865434
TR_THRESHOLD_FLUX = 0.02185009852518841
TR_THRESHOLD_FOR_MODEL = {
    "stabilityai/stable-diffusion-2-1-base": TR_THRESHOLD_SD2,
    "RedbeardNZ/stable-diffusion-2-1-base": TR_THRESHOLD_SD2,
    "stabilityai/stable-diffusion-xl-base-1.0": TR_THRESHOLD_SDXL,
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": TR_THRESHOLD_PIXART,
    "black-forest-labs/FLUX.1-dev": TR_THRESHOLD_FLUX
}

# PRC FPR=10-5
PRC_THRESHOLD_SD = -10925.489511465277
PRC_THRESHOLD_FLUX = -44986
PRC_THRESHOLD_SD3 = -45004.72577242192
PRC_THRESHOLD_FOR_MODEL = {
    "stabilityai/stable-diffusion-xl-base-1.0": PRC_THRESHOLD_SD,
    # "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": TR_THRESHOLD_PIXART,
    "black-forest-labs/FLUX.1-dev": PRC_THRESHOLD_FLUX,
    "stabilityai/stable-diffusion-3-medium-diffusers": PRC_THRESHOLD_SD3
}

# RID FPR=10-3 # (10^-2: 73.19385528564453)
RID_THRESHOLD = -70.95758819580078

# HSTR FPR=10-3 # (10^-2: 43.953006744384766)
HSTR_THRESHOLD = -40.98629379272461 # -40

# HSQR FPR=10-3 # (10^-2: 68.04405975341797)
HSQR_THRESHOLD = -65.86233520507812

# SHALLOW FPR=10-3 # (10^-2: xx)
SHALLOW_THRESHOLD = -58.96875    

# GM phase-2 dual score threshold. Recalibrate after ROC experiments.
GM_THRESHOLD = 0.9

# These are legacy benchmark operating points, not thresholds calibrated from
# the current experiment's clean-negative distribution.  Their nominal FPRs
# are intentionally explicit so callers cannot label all resulting detection
# rates as TPR@1%FPR.
LEGACY_THRESHOLD_NOMINAL_FPR = {
    "GS": 1e-6,
    "TR": 1e-2,
    "RID": 1e-3,
    "HSTR": 1e-3,
    "HSQR": 1e-3,
    "SHALLOW": 1e-3,
}

def get_detection_threshold(wm_type: str,
                            model_name: str = None) -> float:
    """
    Get a legacy/default detection threshold for the given model.

    This function does *not* calibrate a threshold from clean negatives in the
    current run. Use ``raven_repro/scripts/eval_reproduction.py`` for
    protocol-correct TPR@1%FPR evaluation.

    @param wm_type: str
    @param model_name: str

    @return: float
    """
    if wm_type == "GS":
        # print(get_GS_thresholds())
        return GS_THRESHOLDS
    elif wm_type == "TR":
        assert model_name, "Model name must be provided for TR"
        return TR_THRESHOLD_FOR_MODEL.get(model_name, TR_THRESHOLD_SDXL)
    elif wm_type == "RID":
        assert model_name, "Model name must be provided for RID"
        return RID_THRESHOLD
    elif wm_type == "HSTR":
        assert model_name, "Model name must be provided for HSTR"
        return HSTR_THRESHOLD
    elif wm_type == "HSQR":
        assert model_name, "Model name must be provided for HSQR"
        return HSQR_THRESHOLD
    elif wm_type == "TAG":
        return GS_THRESHOLDS
    elif wm_type == "SPH":
        return SPH_THRESHOLD
    elif wm_type == "T2S":
        return 0.0
    elif wm_type == "MAXSIVE":
        return 0.0
    elif wm_type == "SHALLOW":
        assert model_name, "Model name must be provided for SHALLOW"
        return SHALLOW_THRESHOLD
    elif wm_type == "GM":
        return GM_THRESHOLD
    elif wm_type == "PRC":
        assert model_name, "Model name must be provided for PRC"
        return PRC_THRESHOLD_FOR_MODEL.get(model_name, PRC_THRESHOLD_SD)
    else:
        raise ValueError("Unknown watermark type")


def describe_legacy_detection_threshold(wm_type: str, model_name: str = None) -> dict:
    """Return a labeled legacy threshold record suitable for result metadata."""
    method = wm_type.upper()
    return {
        "method": method,
        "threshold": get_detection_threshold(method, model_name),
        "threshold_type": "legacy_default_threshold",
        "nominal_fpr": LEGACY_THRESHOLD_NOMINAL_FPR.get(method),
        "calibrated_from_current_clean_negatives": False,
    }
    

def check_if_detection_successful(wm_type: str, threshold: float, value: float) -> bool:
    """
    Check if the detection is successful

    @param wm_type: str
    @param threshold: float
    @param value: float

    @return: bool
    """
    if wm_type == "GS":
        return value > threshold
    elif wm_type == "TR":
        return value < threshold
    elif wm_type == "RID":
        return (-value) > threshold
    elif wm_type == "HSTR":
        return (-value) > threshold
    elif wm_type == "HSQR":
        return (-value) > threshold
    elif wm_type == "TAG":
        return value > threshold
    elif wm_type == "SPH":
        return value > threshold
    elif wm_type == "T2S":
        return value > threshold
    elif wm_type == "MAXSIVE":
        return value > threshold
    elif wm_type == "SHALLOW":
        return (-value) > threshold
    elif wm_type == "GM":
        return value > threshold
    elif wm_type == "PRC":
        return value >= threshold
    else:
        raise ValueError("Unknown watermark type")


def set_random_seed(seed=0):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

##############################################################################################################################################################################################

def get_GS_thresholds(num_bits=256, NUM_USERS=100*1000):
    """
    This implements the calculation of theoretical thresholds for different FPRS in Gaussian Shading as per the instructions in their appendix.
    You do not need it though, as the threshold is provided above for our used settings.

    Need to install mpmath. Did not manage to do it via pip.
    """
    INTERESTING_FPRs_GS = [10**-i for i in [6, 16, 32, 64]]
    from scipy.special import betainc
    import mpmath as mp
    mp.mp.dps = 100


    # Number of bits in the WM message
    K = num_bits
    N = NUM_USERS
    
    def beta_func(τ, k=K):
        a = τ + 1
        b = k - τ
        return betainc(a, b, 0.5)

    # Generating thresholds and their corresponding single-user FPR
    thresholds = range(K//2, K + 1)
    single_user_FPRs = [beta_func(τ) for τ in thresholds]

    # Calculating multi-user FPR
    single_user_FPRs = np.array(single_user_FPRs)  # Ensure it's a numpy array if not already

    # Convert single_user_FPRs to mpmath floats for higher precision
    single_user_FPRs_mp = [mp.mpf(fpr) for fpr in single_user_FPRs]

    # Compute the result with high precision
    multi_user_FPRs = [1 - mp.exp(-N * fpr) for fpr in single_user_FPRs_mp]
    
    
    def find_first_index_below_threshold(values, a):
        return next((i for i, x in enumerate(values) if x < a), None)

    # zero-bit / detection-only scenario
    FPRs_GS = [beta_func(τ) for τ in thresholds]
    THRESHOLD_INDEXES_FOR_FPRs_GS = {fpri: find_first_index_below_threshold(FPRs_GS, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS.items()}
    
    print("single:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")
        
    # multi-bit / detection & attribution scenario
    FPRs_GS_MULTI = multi_user_FPRs
    THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI = {fpri: find_first_index_below_threshold(FPRs_GS_MULTI, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS_MULTI = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS_MULTI.items()}
    print("multi:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")

    return {"INTERESTING_FPRs_GS": INTERESTING_FPRs_GS,
            "THRESHOLD_INDEXES_FOR_FPRs_GS": THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FOR_FPRs_GS": THRESHOLD_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FLOAT_FOR_FPRs_GS": THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI # tau_bits, 多使用者追蹤
            }


if __name__ == "__main__":
    print(get_GS_thresholds())
