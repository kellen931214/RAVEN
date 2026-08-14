"""Small deterministic checks for the RGB zero-padding shift helper.

Run directly with ``python raven_repro/tests/test_rgb_zero_padding.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.rgb_shift import translate_rgb_zero_padding


def expected_translation(source: np.ndarray, dx: int, dy: int) -> np.ndarray:
    expected = np.zeros_like(source)
    height, width = source.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx0 < sx1 and sy0 < sy1:
        tx0, ty0 = max(0, dx), max(0, dy)
        expected[ty0:ty0 + sy1 - sy0, tx0:tx0 + sx1 - sx0] = source[sy0:sy1, sx0:sx1]
    return expected


def main() -> None:
    # Unique non-zero RGB values make both accidental wrapping and padding
    # replication unambiguous.
    source = np.arange(1, 5 * 4 * 3 + 1, dtype=np.uint8).reshape(4, 5, 3)
    image = Image.fromarray(source, mode="RGB")
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1)):
        actual = np.asarray(translate_rgb_zero_padding(image, dx, dy))
        expected = expected_translation(source, dx, dy)
        assert actual.shape == source.shape, (dx, dy, actual.shape)
        assert np.array_equal(actual, expected), (dx, dy)
        assert np.array_equal(actual[expected == 0], np.zeros_like(actual[expected == 0])), (dx, dy)
    print("RGB zero-padding shift sanity checks: PASS (5 cases)")


if __name__ == "__main__":
    main()
