#!/usr/bin/env python3
"""Rank checkpoint metrics using the journal's balanced selection rule."""

import argparse
import csv
import json
from pathlib import Path


CRITERIA = (
    ("psnr_mean", True),
    ("ssim_mean", True),
    ("lpips_mean", False),
    ("gradient_error", False),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "Bicubic":
                continue
            metrics = {
                key: float(value)
                for key, value in row.items()
                if key.endswith(("_mean", "_std"))
            }
            metrics["gradient_error"] = abs(metrics["gradient_ratio_mean"] - 1.0)
            rows.append({"method": row["method"], "metrics": metrics})
    if not rows:
        raise ValueError(f"No checkpoint rows found in {path}")
    return rows


def assign_ranks(rows, metric, descending):
    ordered = sorted(rows, key=lambda row: row["metrics"][metric], reverse=descending)
    for rank, row in enumerate(ordered, start=1):
        row.setdefault("metric_ranks", {})[metric] = rank


def main():
    args = parse_args()
    rows = load_rows(args.input_csv)
    for metric, descending in CRITERIA:
        assign_ranks(rows, metric, descending)

    for row in rows:
        row["mean_rank"] = sum(row["metric_ranks"].values()) / len(CRITERIA)
    rows.sort(key=lambda row: (row["mean_rank"], -row["metrics"]["psnr_mean"]))

    output = {
        "source_csv": str(args.input_csv),
        "selection_rule": (
            "Lowest mean ordinal rank over PSNR, SSIM, LPIPS, and absolute "
            "gradient-ratio error from 1.0; RMSE is excluded as redundant with PSNR."
        ),
        "selected_method": rows[0]["method"],
        "ranked_checkpoints": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {rows[0]['method']} (mean rank {rows[0]['mean_rank']:.2f})")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
