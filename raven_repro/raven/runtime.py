"""Attack-runtime utilities: image I/O, RNG seeding, CPU thread limiting."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


# --------------------------------------------------------------------------- #
# CPU thread limiting
# --------------------------------------------------------------------------- #
def limit_cpu_threads(num_threads: int = 1) -> None:
    """Keep CPU-side libraries from fanning out across the shared server."""
    value = str(int(num_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(int(num_threads))
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# RNG seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #
def save_image(image: Image.Image, path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def tensor_to_image(tensor) -> Image.Image:
    """Convert a tensor in [0, 1] or [-1, 1] to a PIL RGB image."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError("tensor_to_image requires numpy and torch") from exc

    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def image_to_tensor(image: Image.Image, device: Optional[str] = None, dtype: Optional[Any] = None):
    """Convert a PIL image to a BCHW tensor in [-1, 1]."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError("image_to_tensor requires numpy and torch") from exc

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


def image_size_divisible_by_8(image: Image.Image) -> Tuple[int, int]:
    if image.width % 8 or image.height % 8:
        raise ValueError(f"Image dimensions must be divisible by 8, got {image.size}")
    return image.size
