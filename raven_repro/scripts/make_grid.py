#!/usr/bin/env python
"""Create a simple image grid from files or folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def collect_images(paths):
    images = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for name in ("input.png", "view_guided_output.png", "final_color_corrected.png", "final.png"):
                candidate = path / name
                if candidate.exists():
                    images.append(candidate)
        elif path.exists():
            images.append(path)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a grid image.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--thumb", type=int, default=256)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    paths = collect_images(args.images)
    if not paths:
        raise FileNotFoundError("No images found for grid")
    thumbs = [ImageOps.fit(Image.open(path).convert("RGB"), (args.thumb, args.thumb)) for path in paths]
    rows = (len(thumbs) + args.cols - 1) // args.cols
    grid = Image.new("RGB", (args.cols * args.thumb, rows * args.thumb), "white")
    for idx, image in enumerate(thumbs):
        x = (idx % args.cols) * args.thumb
        y = (idx // args.cols) * args.thumb
        grid.paste(image, (x, y))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


if __name__ == "__main__":
    main()
