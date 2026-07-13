import torch
from diffusers import StableDiffusionXLPipeline

from .pipe_provider import PipeProvider
import typing
from utils.image_utils import torch_to_PIL

class SDXLPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(cls):
        return StableDiffusionXLPipeline
    
    def get_random_latents(self, batch_size=1) -> torch.Tensor:
        return torch.randn(*self.get_latent_shape(batch_size=batch_size),
                           device=self.device,
                           dtype=self.get_dtype())
    

class AnimePipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(cls):
        return StableDiffusionXLPipeline
    
    def get_random_latents(self, batch_size=1) -> torch.Tensor:
        return torch.randn(*self.get_latent_shape(batch_size=batch_size),
                           device=self.device,
                           dtype=self.get_dtype())
    def __load_pipe(self):
        """Load pipe and push to device"""
        if self.pipe is None:
            self.pipe = self.load_diffusers_pipe(self.pretrained_model_name_or_path, **self.kwargs)
        self.pipe = self.pipe.to(self.device) if self.allow_device() else self.pipe
    def generate(self,
                 prompts: typing.List[str],
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 latents: typing.Optional[torch.Tensor] = None,
                 num_images_per_prompt: int = 1,
                 return_latents: bool = True) -> dict:
        """
        Generate dict with

        @param prompt: str
        @param num_inference_steps: int
        @param guidance_scale: float
        @param latents: torch.Tensor with batch dim
        @param num_images_per_prompt: int
        @param return_latents: bool

        @return dict
        """
        self.__load_pipe()
        self.set_scheduler()

        # make sure latents are on device
        if latents is not None:
            latents = latents.to(self.device)

        # generate
        out = self.pipe(
             prompts,
             num_images_per_prompt=num_images_per_prompt,
             guidance_scale=guidance_scale,
             num_inference_steps=num_inference_steps,
             height=self.resolution,
             width=self.resolution,
             negative_prompt=["nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"],
             latents=latents,
             )
    
        # collect results
        images_PIL = out.images # list of PIL image
        images_torch = self.PIL_to_torch(out.images)

        if return_latents:
            z0_torch = self.imgs_to_latents(out.images)
            z0_PIL = torch_to_PIL(z0_torch)

            zT_torch = latents.to(self.device) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
            zT_PIL = torch_to_PIL(latents) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
        
        return {
            # prompts
            "prompts": prompts,  # is list
            # images
            'images_torch': images_torch,
            'images_PIL': images_PIL,
            'images': images_PIL,
            # z0
            'z0_torch': z0_torch if return_latents else None,
            'z0_PIL': z0_PIL if return_latents else None,
            'z0': z0_PIL if return_latents else None,
            # zT
            'zT_torch': zT_torch if return_latents else None,
            'zT_PIL': zT_PIL if return_latents else None,
            'zT': zT_PIL if return_latents else None,}