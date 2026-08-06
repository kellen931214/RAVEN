"""RAVEN reproduction package.

``RavenPipeline`` is imported lazily so importing ``raven.metrics`` or
``raven.experiment_io`` does not pull in ``diffusers`` / ``torch``.
"""

__all__ = ["RavenPipeline"]

_RAVEN_PIPELINE = None


def __getattr__(name: str):
    if name == "RavenPipeline":
        global _RAVEN_PIPELINE
        if _RAVEN_PIPELINE is None:
            from .pipeline_raven import RavenPipeline as _RavenPipeline
            _RAVEN_PIPELINE = _RavenPipeline
        return _RAVEN_PIPELINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
