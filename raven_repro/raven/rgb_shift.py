"""Explicit RGB-space translations for pre-inversion attack cohorts."""

from __future__ import annotations

import numpy as np
from PIL import Image


def translate_rgb_zero_padding(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Translate an RGB image without wrapping or edge-value replication.

    Positive ``dx`` moves content right; positive ``dy`` moves content down.
    Pixels shifted outside the canvas are discarded and newly exposed pixels
    are exactly RGB ``(0, 0, 0)``.
    """
    if not isinstance(dx, int) or not isinstance(dy, int):
        raise TypeError("dx and dy must be integer pixel offsets")

    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, channels = source.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB image, got shape {source.shape}")

    translated = np.zeros_like(source)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    if source_x0 < source_x1 and source_y0 < source_y1:
        target_x0, target_y0 = max(0, dx), max(0, dy)
        target_x1 = target_x0 + (source_x1 - source_x0)
        target_y1 = target_y0 + (source_y1 - source_y0)
        translated[target_y0:target_y1, target_x0:target_x1] = source[
            source_y0:source_y1, source_x0:source_x1
        ]
    return Image.fromarray(translated, mode="RGB")
