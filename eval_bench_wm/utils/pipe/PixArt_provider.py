import torch
import typing
from .pipe_provider import PipeProvider

from utils.image_utils import torch_to_PIL, torch_to_PIL

from diffusers import PixArtSigmaPipeline, DiffusionPipeline, Transformer2DModel, AutoencoderKL

#from diffusers import DDIMScheduler, DDIMInverseScheduler
from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler


ROOTNAME = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

def latent_to_pil(latents, height, width, pipe):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


def pil_to_latent(image, height, width, pipe):
    image = pipe.image_processor.preprocess(image).to(dtype=torch.bfloat16,
                                                      device="cuda"
                                                      )
    latents = pipe.vae.encode(image, return_dict=False)[0].sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = pipe._pack_latents(latents,
                                 1,
                                 4,
                                 height // pipe.vae_scale_factor * 2,
                                 width // pipe.vae_scale_factor * 2)
    return latents


class PixArtPipeProvider(PipeProvider):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call_from_pretrained(self,
                             pretrained_model_name_or_path: str,
                             **kwargs) -> DiffusionPipeline:
        """
        Call Diffusers "from_pretrained"

        @param pretrained_model_name_or_path: str
        @param kwargs: dict

        @return: DiffusionPipeline
        """

        pipe_class = self.get_diffusers_pipe_class()

        # load transformer seperately. PixArt needs this
        transformer = self.get_unet(pretrained_model_name_or_path)

        # pipe must be loaded from ROOTNAME
        pipe = pipe_class.from_pretrained(pretrained_model_name_or_path=ROOTNAME,
                                          transformer=transformer,
                                          scheduler=self.scheduler,
                                          safety_checker=None,
                                          torch_dtype=self.get_dtype(),
                                          **kwargs)
        
        del transformer

        return pipe
    
    # -------------------------------------------------------- GENERATE + INVERSION --------------------------------------------------------

    # def generate(self,
    #              prompts: typing.List[str],
    #              num_inference_steps: int = 50,
    #              guidance_scale: float = 7.5,
    #              latents: typing.Optional[torch.Tensor] = None,
    #              callback_on_step_end=None,
    #              callback_on_step_end_tensor_inputs=None,
    #              return_latents: bool = True) -> dict:
    #     """
    #     Generate dict with

    #     @param prompt: str
    #     @param num_inference_steps: int
    #     @param guidance_scale: float
    #     @param latents: torch.Tensor with batch dim
    #     @param return_latents: bool

    #     @return dict
    #     """

    #     if not isinstance(prompts, list):
    #         prompts = [prompts]

    #     zT = latents 

    #     zT_shape = self.get_latent_shape(batch_size=1)
    #     zT_prepared = zT.view(zT_shape[0] * zT_shape[1] * zT_shape[2], zT_shape[3])

    #     z0_prepared = self.pipe(
    #         prompts,
    #         height=self.resolution,
    #         width=self.resolution,
    #         guidance_scale=guidance_scale,
    #         num_inference_steps=num_inference_steps,
    #         max_sequence_length=512,
    #         generator=torch.Generator("cpu").manual_seed(42),
    #         output_type="latent",
    #         latents=zT_prepared[None],
    #         callback_on_step_end=callback_on_step_end,
    #         callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    #     ).images[0]
        
    #     with torch.no_grad():
    #         image = latent_to_pil(z0_prepared[None], height=self.resolution, width=self.resolution, pipe=self.pipe)[0]
    
    #     # collect results
    #     images_PIL = [image] # list of PIL image
    #     images_torch = self.PIL_to_torch([image], dtype=torch.float32)

    #     if return_latents:
    #         z0_torch = z0_prepared.view(zT_shape)
    #         z0_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))

    #         zT_torch = zT.to(self.device) if zT is not None else torch.zeros(z0_torch.shape).to(self.device)
    #         zT_PIL = torch_to_PIL(torch.zeros((1, 4, self.resolution // 8, self.resolution // 8)))
        
    #     return {
    #         # prompts
    #         "prompts": prompts,  # is list
    #         # images
    #         'images_torch': images_torch,
    #         'images_PIL': images_PIL,
    #         'images': images_PIL,
    #         # z0
    #         'z0_torch': z0_torch if return_latents else None,
    #         'z0_PIL': z0_PIL if return_latents else None,
    #         'z0': z0_PIL if return_latents else None,
    #         # zT
    #         'zT_torch': zT_torch if return_latents else None,
    #         'zT_PIL': zT_PIL if return_latents else None,
    #         'zT': zT_PIL if return_latents else None,}
    
    # def invert_z0(self,
    #               latents: torch.Tensor,
    #               num_inference_steps: int = 50,
    #               callback_on_step_end=None,
    #               callback_on_step_end_tensor_inputs=None) -> torch.tensor:
    #     """
    #     Do DDIM inversion on given latents z0

    #     @param latents: torch tensor with batch dim
    #     @param num_inference_steps: int

    #     @return: zT torch tensor with batch dim on self.device

    #     LATENTS HAS SHAPE 1, 1024, 64
    #     """

    #     # invert
    #     self.pipe.scheduler.__class__ = self.scheduler_classes[1].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())
    #     zT_inv = self.pipe(
    #         "",
    #         latents=latents,
    #         height=self.resolution,
    #         width=self.resolution,
    #         guidance_scale=1,
    #         num_inference_steps=num_inference_steps,
    #         max_sequence_length=512,
    #         #generator=torch.Generator("cpu").manual_seed(0),
    #         output_type="latent",
    #         callback_on_step_end=callback_on_step_end,
    #         callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
    #     ).images[0]
        
    #     zT_shape = self.get_latent_shape(batch_size=1)
    #     zT_inv = zT_inv.view(zT_shape)
    #     zT_inv = zT_inv.to(dtype=torch.float32)
        
    #     # reset back to default scheduler
    #     self.pipe.scheduler.__class__ = self.scheduler_classes[0].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())

    #     #return out
    #     return zT_inv
    
    # # # -------------------------------------------------------- UTILS --------------------------------------------------------

    # def imgs_to_latents(self, images: torch.Tensor) -> torch.Tensor:
    #     return self.pipe.vae.encode(images, return_dict=False)[0]
    
    # def get_latent_shape(self, batch_size=1):
    #     return (batch_size, 4, self.resolution // 8, self.resolution // 8)
    
    # def get_random_latents(self, batch_size=1) -> torch.Tensor:
    #     return torch.randn(*self.get_latent_shape(batch_size=batch_size),
    #                        device=self.device,
    #                        dtype=self.get_dtype())
    
    def get_diffusers_pipe_class(self):
        return PixArtSigmaPipeline

    def get_scheduler(self):
        return self.scheduler_classes[0].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())

    def get_inverse_scheduler(self):
        return self.scheduler_classes[1].from_pretrained(ROOTNAME, subfolder='scheduler', torch_dtype=self.get_dtype())

    def get_unet(self, unet_id_or_checkpoint_dir: str):
        return Transformer2DModel.from_pretrained(unet_id_or_checkpoint_dir, subfolder='transformer', torch_dtype=self.get_dtype(), device=self.device)

    def get_vae(self, *args, **kwargs):
        return AutoencoderKL.from_pretrained(ROOTNAME, subfolder="vae", torch_dtype=self.get_dtype(), device=self.device)

    def get_vae_params(self):
        return None
    
    def get_unet_params(self):
        return None
