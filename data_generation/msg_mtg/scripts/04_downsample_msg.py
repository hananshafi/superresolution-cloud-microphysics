#!/usr/bin/env python3
"""Bicubically resize 16-bit MSG patches while preserving their radiometric values.

The original resize script was not retained. PIL BICUBIC was recovered by testing
candidate resize implementations against the saved 128x128 files pixel-for-pixel.
"""

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {args.input_dir}")

    written = 0
    skipped = 0
    for source in paths:
        destination = args.output_dir / source.name
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(source) as image:
            resized = image.resize(
                (args.width, args.height), resample=Image.Resampling.BICUBIC
            )
            resized.save(destination)
        written += 1
    print(f"Wrote {written} image(s); skipped {skipped} existing image(s).")


if __name__ == "__main__":
    main()
