#!/usr/bin/env python3
"""Convert each image independently from its source range to uint8 [0, 255]."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    constant = 0
    for source in paths:
        destination = args.output_dir / source.name
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(source) as image:
            array = np.asarray(image)
        value_min = array.min()
        value_max = array.max()
        if value_max == value_min:
            print(f"[SKIP] Constant-valued image: {source.name}")
            constant += 1
            continue
        normalized = (array - value_min) / (value_max - value_min)
        output = (normalized * 255).astype(np.uint8)
        Image.fromarray(output).save(destination)
        written += 1
    print(
        f"Wrote {written} image(s); skipped {skipped} existing and "
        f"{constant} constant-valued image(s)."
    )


if __name__ == "__main__":
    main()
