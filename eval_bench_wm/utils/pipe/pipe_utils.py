
import torch

from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from .schedulers.scheduling_maxsive_dpm import MaxsiveDPMSolverMultistepScheduler
from .schedulers.scheduling_local_dpm import LocalDPMSolverMultistepScheduler

from diffusers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from .SD_provider import SDPipeProvider
from .SDXL_provider import SDXLPipeProvider, AnimePipeProvider
from .SDXL_refiner_provider import SDXLRefinerPipeProvider
from .PixArt_provider import PixArtPipeProvider
from .Sana_provider import SanaPipeProvider



# Map model id onto pipe. Add more if needed
PIPE_PROVIDERS = {
       'CompVis/stable-diffusion-v1-4': SDPipeProvider,
       'stable-diffusion-v1-5/stable-diffusion-v1-5': SDPipeProvider,
       'stabilityai/stable-diffusion-2-1-base': SDPipeProvider,
       'RedbeardNZ/stable-diffusion-2-1-base': SDPipeProvider,
       'danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning': SDPipeProvider,
       'stabilityai/stable-diffusion-xl-base-1.0': SDXLPipeProvider,
       'stabilityai/stable-diffusion-xl-refiner-1.0': SDXLRefinerPipeProvider,
       'PixArt-alpha/PixArt-Sigma-XL-2-512-MS': PixArtPipeProvider,
       'Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers': SanaPipeProvider,
       'cagliostrolab/animagine-xl-3.0': AnimePipeProvider,
    # FLUX see below
    }


SCHEDULER_CLASSES = {
     "DDIM": (DDIMScheduler, DDIMInverseScheduler),
     "DPM": (DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler),
     "DPM_DDIM_INV": (DPMSolverMultistepScheduler, DDIMInverseScheduler),
     "MAXSIVE_DPM": (MaxsiveDPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler),
     "Euler": (None, None),  # special case for Flux
     # LocalDPMSolverMultistepScheduler
}



def get_pipe_provider(pretrained_model_name_or_path: str,
                      resolution: int,  # only square image
                      unet_id_or_checkpoint_dir: str = None,
                      lora_checkpoint_dir: str = None,
                      vae_id: str = None,
                      zero_unet: bool = False,
                      device: torch.device = torch.device("cuda"),
                      eager_loading: bool = False,
                      schedulers_name: str = "DDIM",
                      **kwargs):
        """
        Get correct pipe provider

        @param pretrained_model_name_or_path: str
        @param resolution: int
        @param unet_id_or_checkpoint_dir: str
        @param lora_checkpoint_dir: str
        @param vae_id: str
        @param zero_unet: bool
        @param device: torch.device
        @param eager_loading: bool
        @param schedulers_name: str
        @param kwargs: dict

        @return: PipeProvider
        """

        if "FLUX" in pretrained_model_name_or_path:
             from .Flux_provider import FluxPipeProvider
             PIPE_PROVIDERS['black-forest-labs/FLUX.1-dev'] = FluxPipeProvider

        if "stable-diffusion-3" in pretrained_model_name_or_path:
             from .SD3_provider import SD3PipeProvider
             PIPE_PROVIDERS[pretrained_model_name_or_path] = SD3PipeProvider
        
        if "Efficient-Large-Model" in pretrained_model_name_or_path:
             from .Sana_provider import SanaPipeProvider
             PIPE_PROVIDERS['Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers'] = SanaPipeProvider

        if "stable-diffusion-xl-refiner" in pretrained_model_name_or_path:
             PIPE_PROVIDERS[pretrained_model_name_or_path] = SDXLRefinerPipeProvider
             
     #    if "CogView4" in pretrained_model_name_or_path:
     #         from .CogView4_provider import CogView4PipeProvider
     #         PIPE_PROVIDERS["THUDM/CogView4-6B"] = CogView4PipeProvider    
     
        # get correct pipe with all possible arguments. Args are then disseminated in the subclasses
        if pretrained_model_name_or_path not in PIPE_PROVIDERS:
             supported = ", ".join(sorted(PIPE_PROVIDERS.keys()))
             raise KeyError(
                  f"Unknown model id: {pretrained_model_name_or_path}. "
                  f"Please register it in PIPE_PROVIDERS. Supported ids: {supported}"
             )
        pipe_provider = PIPE_PROVIDERS[pretrained_model_name_or_path]
        return pipe_provider(pretrained_model_name_or_path=pretrained_model_name_or_path,
                             resolution=resolution,
                             unet_id_or_checkpoint_dir=unet_id_or_checkpoint_dir,
                             lora_checkpoint_dir=lora_checkpoint_dir,
                             vae_id=vae_id,
                             zero_unet=zero_unet,
                             device=device,
                             eager_loading=eager_loading,
                             scheduler_classes=SCHEDULER_CLASSES[schedulers_name],
                             **kwargs)
