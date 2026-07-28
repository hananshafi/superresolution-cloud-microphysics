#!/usr/bin/env python3
"""Remove the Google Drive "Copy of " prefix from downloaded raw files."""

import argparse
from pathlib import Path


def rename_files(root: Path, prefix: str) -> tuple[int, int]:
    renamed = 0
    skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        destination = path.with_name(path.name[len(prefix) :])
        if destination.exists():
            print(f"[SKIP] {path} -> {destination} (target exists)")
            skipped += 1
            continue
        print(f"[RENAME] {path} -> {destination}")
        path.rename(destination)
        renamed += 1
    return renamed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path("Comparison"), Path("Comparison_Fine")],
        help="Directories to scan recursively (default: Comparison Comparison_Fine)",
    )
    parser.add_argument("--prefix", default="Copy of ", help="Filename prefix to remove")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_renamed = 0
    total_skipped = 0
    for directory in args.directories:
        if not directory.is_dir():
            print(f"[WARN] Directory not found: {directory}")
            continue
        print(f"Processing directory: {directory}")
        renamed, skipped = rename_files(directory, args.prefix)
        total_renamed += renamed
        total_skipped += skipped
    print(f"Renamed {total_renamed} file(s); skipped {total_skipped} file(s).")


if __name__ == "__main__":
    main()
