"""Utility helpers for the RAVEN reproduction."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from PIL import Image, ImageOps


def parse_bool(value: bool | str) -> bool:
    """Parse CLI booleans from true/false style strings."""
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def prepare_output_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_image(path: str | os.PathLike[str], size: Optional[int] = 512) -> Image.Image:
    """Load an RGB image and center-crop/resize to a square size divisible by 8."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if size is not None:
        if size % 8 != 0:
            raise ValueError(f"Image size must be divisible by 8, got {size}")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    elif image.width % 8 != 0 or image.height % 8 != 0:
        new_w = image.width - image.width % 8
        new_h = image.height - image.height % 8
        if new_w <= 0 or new_h <= 0:
            raise ValueError(f"Image is too small after enforcing divisibility by 8: {image.size}")
        image = ImageOps.fit(image, (new_w, new_h), method=Image.Resampling.LANCZOS)
    return image


def save_image(image: Image.Image, path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


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


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


def iter_image_files(input_dir: str | os.PathLike[str]) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for path in sorted(Path(input_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def image_size_divisible_by_8(image: Image.Image) -> Tuple[int, int]:
    if image.width % 8 or image.height % 8:
        raise ValueError(f"Image dimensions must be divisible by 8, got {image.size}")
    return image.size
