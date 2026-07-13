from diffusers import StableDiffusionXLImg2ImgPipeline

from .pipe_provider import PipeProvider


class SDXLRefinerPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(self):
        return StableDiffusionXLImg2ImgPipeline
