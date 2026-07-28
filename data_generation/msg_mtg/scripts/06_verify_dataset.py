#!/usr/bin/env python3
"""Check the retained MSG/MTG directory layout, identifiers, sizes, and dtypes."""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DirectorySpec:
    name: str
    prefix: str
    first_index: int
    last_index: int
    size: tuple[int, int]
    dtype: np.dtype


SPECS = (
    DirectorySpec("train_HR", "mtg", 1, 2250, (512, 512), np.dtype("uint16")),
    DirectorySpec("train_LR", "msg", 1, 2250, (512, 512), np.dtype("uint16")),
    DirectorySpec("test_HR", "mtg", 2251, 2750, (512, 512), np.dtype("uint16")),
    DirectorySpec("test_LR", "msg", 2251, 2750, (512, 512), np.dtype("uint16")),
    DirectorySpec("train_LR_128", "msg", 1, 2250, (128, 128), np.dtype("uint16")),
    DirectorySpec("test_LR_128", "msg", 2251, 2750, (128, 128), np.dtype("uint16")),
    DirectorySpec("test_HR_converted", "mtg", 2251, 2750, (512, 512), np.dtype("uint8")),
    DirectorySpec("test_LR_128_converted", "msg", 2251, 2750, (128, 128), np.dtype("uint8")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--full", action="store_true", help="Read every image instead of five samples per directory"
    )
    return parser.parse_args()


def file_index(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.png", path.name)
    return int(match.group(1)) if match else None


def select_samples(paths: list[Path], full: bool) -> list[Path]:
    if full or len(paths) <= 5:
        return paths
    positions = np.linspace(0, len(paths) - 1, num=5, dtype=int)
    return [paths[int(position)] for position in positions]


def main() -> None:
    args = parse_args()
    errors = []
    observed_ids: dict[str, set[int]] = {}
    for spec in SPECS:
        directory = args.data_root / spec.name
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")
            continue
        paths = sorted(directory.glob("*.png"), key=lambda path: file_index(path, spec.prefix) or -1)
        ids = {index for path in paths if (index := file_index(path, spec.prefix)) is not None}
        expected = set(range(spec.first_index, spec.last_index + 1))
        observed_ids[spec.name] = ids
        if ids != expected:
            missing = sorted(expected - ids)
            extra = sorted(ids - expected)
            errors.append(
                f"{spec.name}: ID mismatch (missing={missing[:10]}, extra={extra[:10]})"
            )
        if len(paths) != len(expected):
            errors.append(f"{spec.name}: expected {len(expected)} PNGs, found {len(paths)}")

        for path in select_samples(paths, args.full):
            with Image.open(path) as image:
                array = np.asarray(image)
                if image.size != spec.size:
                    errors.append(f"{path}: expected size {spec.size}, found {image.size}")
                if array.dtype != spec.dtype:
                    errors.append(f"{path}: expected dtype {spec.dtype}, found {array.dtype}")
        print(
            f"{spec.name:24s} count={len(paths):4d} size={spec.size[0]}x{spec.size[1]} "
            f"dtype={spec.dtype}"
        )

    for left, right in (
        ("train_HR", "train_LR"),
        ("test_HR", "test_LR"),
        ("train_LR", "train_LR_128"),
        ("test_LR", "test_LR_128"),
        ("test_HR", "test_HR_converted"),
        ("test_LR_128", "test_LR_128_converted"),
    ):
        if left in observed_ids and right in observed_ids and observed_ids[left] != observed_ids[right]:
            errors.append(f"Pairing mismatch: {left} and {right}")

    if errors:
        print("\nVerification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("\nDataset layout verified successfully.")


if __name__ == "__main__":
    main()
