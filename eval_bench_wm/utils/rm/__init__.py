from .nfpa import NFPAWatermarkRemover


RM_METHODS = {
    "nfpa": NFPAWatermarkRemover,
}


def get_remover(method: str, **kwargs):
    method = method.lower()
    if method not in RM_METHODS:
        raise ValueError(f"Unknown removal method: {method}")
    return RM_METHODS[method](**kwargs)
