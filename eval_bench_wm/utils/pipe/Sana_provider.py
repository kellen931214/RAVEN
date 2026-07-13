from .pipe_provider import PipeProvider


class SanaPipeProvider(PipeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(self):
        try:
            from diffusers import SanaPipeline
        except ImportError as exc:
            raise ImportError(
                "SanaPipeline is not available in this diffusers installation. "
                "Please install a diffusers version that supports Sana before using "
                "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers."
            ) from exc
        return SanaPipeline
