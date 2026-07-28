#!/usr/bin/env python3
"""Split paired MTG/MSG patches by their sequential numeric identifiers."""

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=2251)
    parser.add_argument("--end-index", type=int, default=2750)
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Use move to reproduce the retained dataset; copy is the safer default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_hr = args.data_root / "train_HR"
    train_lr = args.data_root / "train_LR"
    test_hr = args.data_root / "test_HR"
    test_lr = args.data_root / "test_LR"
    test_hr.mkdir(parents=True, exist_ok=True)
    test_lr.mkdir(parents=True, exist_ok=True)

    operations = []
    missing = []
    conflicts = []
    for index in range(args.start_index, args.end_index + 1):
        for source_dir, destination_dir, prefix in (
            (train_hr, test_hr, "mtg"),
            (train_lr, test_lr, "msg"),
        ):
            source = source_dir / f"{prefix}_{index}.png"
            destination = destination_dir / source.name
            if not source.exists():
                missing.append(source)
            elif destination.exists() and not args.overwrite:
                conflicts.append(destination)
            else:
                operations.append((source, destination))

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} source file(s):\n{preview}")
    if conflicts:
        preview = "\n".join(f"  {path}" for path in conflicts[:10])
        raise FileExistsError(
            f"Found {len(conflicts)} destination conflict(s); use --overwrite if intended:\n{preview}"
        )

    operation = shutil.move if args.mode == "move" else shutil.copy2
    for source, destination in operations:
        if destination.exists():
            destination.unlink()
        operation(source, destination)
    pair_count = (args.end_index - args.start_index) + 1
    print(f"{args.mode.title()}d {pair_count} paired samples into test_HR/test_LR.")


if __name__ == "__main__":
    main()
