#!/usr/bin/env python3
"""Generate aligned 16-bit MTG/MSG patch pairs from FCI and SEVIRI data.

This is a cleaned, path-configurable transcription of the executed first cell in
Gen_MTG_MSG_paired_data.ipynb. Scientific defaults match the retained dataset.
"""

import argparse
import gc
import glob
import json
import os
import re
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import dask
import imageio.v2 as imageio
import numpy as np
from satpy import Scene

warnings.filterwarnings("ignore")
dask.config.set(scheduler="single-threaded")


@dataclass(frozen=True)
class ChannelTask:
    mtg_channel: str
    msg_channel: str
    calibration: str


STANDARD_TASKS = (
    ChannelTask("vis_08", "VIS008", "reflectance"),
    ChannelTask("nir_16", "IR_016", "reflectance"),
    ChannelTask("wv_63", "WV_062", "brightness_temperature"),
    ChannelTask("wv_73", "WV_073", "brightness_temperature"),
    ChannelTask("ir_87", "IR_087", "brightness_temperature"),
    ChannelTask("ir_97", "IR_097", "brightness_temperature"),
    ChannelTask("ir_123", "IR_120", "brightness_temperature"),
    ChannelTask("ir_133", "IR_134", "brightness_temperature"),
)

FINE_TASKS = (
    ChannelTask("vis_06", "VIS006", "reflectance"),
    # The notebook listed nir_22 with no MSG counterpart, so it produced no pair.
    ChannelTask("ir_38", "IR_039", "brightness_temperature"),
    ChannelTask("ir_105", "IR_108", "brightness_temperature"),
)


def extract_time_from_filename(filename: str, satellite: str) -> datetime:
    basename = os.path.basename(filename)
    try:
        if satellite == "MSG":
            for part in basename.split("-"):
                if len(part) >= 14 and part[:14].isdigit():
                    return datetime.strptime(part[:14], "%Y%m%d%H%M%S")
        if satellite == "MTG":
            matches = re.findall(r"(\d{14})", basename)
            if matches:
                return datetime.strptime(matches[0], "%Y%m%d%H%M%S")
    except (TypeError, ValueError):
        pass
    return datetime.min


def calculate_roi_timerange(
    start_time: datetime, roi_bbox: tuple[float, float, float, float], duration_min: float
) -> tuple[datetime, datetime]:
    lat_min, lat_max = roi_bbox[1], roi_bbox[3]
    frac_start = np.clip((lat_min + 81.0) / 162.0, 0, 1)
    frac_end = np.clip((lat_max + 81.0) / 162.0, 0, 1)
    return (
        start_time + timedelta(minutes=duration_min * float(frac_start)),
        start_time + timedelta(minutes=duration_min * float(frac_end)),
    )


def get_all_pairs(
    mtg_dir: Path,
    msg_dir: Path,
    group_gap_min: float = 12.0,
    max_msg_diff_min: float = 15.0,
) -> list[tuple[list[str], str, datetime]]:
    mtg_files = sorted(glob.glob(str(mtg_dir / "**" / "*.nc"), recursive=True))
    mtg_files = [path for path in mtg_files if "TRAIL" not in path]
    msg_files = sorted(glob.glob(str(msg_dir / "**" / "*.nat"), recursive=True))
    if not mtg_files or not msg_files:
        return []

    mtg_timed = sorted(
        (extract_time_from_filename(path, "MTG"), path) for path in mtg_files
    )
    groups: list[tuple[list[str], datetime]] = []
    current_group = [mtg_timed[0][1]]
    current_start = mtg_timed[0][0]
    for timestamp, path in mtg_timed[1:]:
        if (timestamp - current_start).total_seconds() < group_gap_min * 60.0:
            current_group.append(path)
        else:
            groups.append((current_group, current_start))
            current_group = [path]
            current_start = timestamp
    groups.append((current_group, current_start))

    msg_timed = [(extract_time_from_filename(path, "MSG"), path) for path in msg_files]
    pairs = []
    for group_files, group_start in groups:
        msg_time, msg_file = min(
            msg_timed, key=lambda item: abs((group_start - item[0]).total_seconds())
        )
        difference = abs((group_start - msg_time).total_seconds())
        if difference <= max_msg_diff_min * 60.0:
            pairs.append((group_files, msg_file, group_start))
    return pairs


def get_limits(array: np.ndarray, calibration: str) -> tuple[float, float]:
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return 0.0, 1.0
    value_min = float(np.percentile(valid, 2))
    value_max = float(np.percentile(valid, 98))
    if calibration == "reflectance":
        value_min = max(value_min, 0.0)
        value_max = max(value_max, 0.6)
    else:
        value_min = min(value_min, 200.0)
        value_max = max(value_max, 300.0)
    if value_max - value_min < 0.01:
        value_max = value_min + 0.1
    return value_min, value_max


def scale_to_uint16(array: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    data = np.array(array, copy=True)
    invalid = ~np.isfinite(data)
    if value_max <= value_min:
        return np.zeros_like(data, dtype=np.uint16)
    data = np.clip(data, value_min, value_max)
    normalized = (data - value_min) / (value_max - value_min + 1e-12)
    output = (normalized * 65535).round().astype(np.uint16)
    output[invalid] = 0
    return output


class PatchWriter:
    def __init__(
        self,
        output_root: Path,
        patch_size: int,
        patches_per_channel: int,
        nodata_fraction_max: float,
        black_fraction_max: float,
    ) -> None:
        self.output_root = output_root
        self.hr_dir = output_root / "train_HR"
        self.lr_dir = output_root / "train_LR"
        self.patch_size = patch_size
        self.patches_per_channel = patches_per_channel
        self.nodata_fraction_max = nodata_fraction_max
        self.black_fraction_max = black_fraction_max
        self.counter = 1
        self.hr_dir.mkdir(parents=True, exist_ok=True)
        self.lr_dir.mkdir(parents=True, exist_ok=True)
        if any(self.hr_dir.glob("*.png")) or any(self.lr_dir.glob("*.png")):
            raise RuntimeError(
                f"Output directories under {output_root} are not empty. "
                "Use a new output root to avoid mixing dataset generations."
            )
        self.manifest_path = output_root / "generation_manifest.jsonl"
        self.manifest = self.manifest_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self.manifest.close()

    def save_random_patches(
        self,
        mtg_array: np.ndarray,
        msg_array: np.ndarray,
        value_min: float,
        value_max: float,
        metadata: dict,
    ) -> int:
        if mtg_array is None or msg_array is None or mtg_array.shape != msg_array.shape:
            print(f"[WARN] Missing or mismatched arrays for {metadata['pair_name']}; skipping")
            return 0

        height, width = mtg_array.shape
        if height < self.patch_size or width < self.patch_size:
            print(f"[WARN] Scene is too small for {self.patch_size}px patches; skipping")
            return 0

        valid_mask = np.isfinite(mtg_array) & np.isfinite(msg_array)
        rows = np.where(valid_mask.any(axis=1))[0]
        columns = np.where(valid_mask.any(axis=0))[0]
        if rows.size == 0 or columns.size == 0:
            print(f"[WARN] No overlapping valid pixels for {metadata['pair_name']}")
            return 0

        y_min, y_max = int(rows.min()), int(rows.max()) + 1
        x_min, x_max = int(columns.min()), int(columns.max()) + 1
        if y_max - y_min < self.patch_size or x_max - x_min < self.patch_size:
            print(f"[WARN] Valid overlap is too small for {metadata['pair_name']}")
            return 0

        saved = 0
        attempts = 0
        max_attempts = self.patches_per_channel * 10
        while saved < self.patches_per_channel and attempts < max_attempts:
            attempts += 1
            y = int(np.random.randint(y_min, y_max - self.patch_size + 1))
            x = int(np.random.randint(x_min, x_max - self.patch_size + 1))
            mtg_patch = mtg_array[y : y + self.patch_size, x : x + self.patch_size]
            msg_patch = msg_array[y : y + self.patch_size, x : x + self.patch_size]

            nodata_fraction = max(
                float(np.mean(~np.isfinite(mtg_patch))),
                float(np.mean(~np.isfinite(msg_patch))),
            )
            if nodata_fraction > self.nodata_fraction_max:
                continue

            mtg_uint16 = scale_to_uint16(mtg_patch, value_min, value_max)
            msg_uint16 = scale_to_uint16(msg_patch, value_min, value_max)
            black_fraction = max(
                float(np.mean(mtg_uint16 == 0)), float(np.mean(msg_uint16 == 0))
            )
            if black_fraction > self.black_fraction_max:
                continue

            index = self.counter
            hr_name = f"mtg_{index}.png"
            lr_name = f"msg_{index}.png"
            imageio.imwrite(self.hr_dir / hr_name, mtg_uint16)
            imageio.imwrite(self.lr_dir / lr_name, msg_uint16)

            record = {
                "index": index,
                "hr_file": f"train_HR/{hr_name}",
                "lr_file": f"train_LR/{lr_name}",
                "crop_x": x,
                "crop_y": y,
                "patch_size": self.patch_size,
                "value_min": value_min,
                "value_max": value_max,
                "nodata_fraction": nodata_fraction,
                "black_fraction": black_fraction,
                **metadata,
            }
            self.manifest.write(json.dumps(record, sort_keys=True) + "\n")
            self.manifest.flush()
            self.counter += 1
            saved += 1

        if saved < self.patches_per_channel:
            print(
                f"[WARN] Saved {saved}/{self.patches_per_channel} patches for "
                f"{metadata['pair_name']} after {attempts} attempts"
            )
        return saved


def process_phase(
    phase_name: str,
    mtg_dir: Path,
    msg_dir: Path,
    tasks: tuple[ChannelTask, ...],
    writer: PatchWriter,
    args: argparse.Namespace,
) -> None:
    print(f"\n=== {phase_name} ===")
    pairs = get_all_pairs(mtg_dir, msg_dir)
    if not pairs:
        print(f"No MTG/MSG pairs found under {mtg_dir} and {msg_dir}")
        return
    print(f"Found {len(pairs)} raw scene pair(s).")

    for pair_index, (mtg_files, msg_file, mtg_start) in enumerate(pairs):
        mtg_roi_start, mtg_roi_end = calculate_roi_timerange(
            mtg_start, args.roi, args.mtg_duration
        )
        msg_end = extract_time_from_filename(msg_file, "MSG")
        msg_start = msg_end - timedelta(minutes=args.msg_duration)
        msg_roi_start, msg_roi_end = calculate_roi_timerange(
            msg_start, args.roi, args.msg_duration
        )
        mtg_midpoint = mtg_roi_start + (mtg_roi_end - mtg_roi_start) / 2
        msg_midpoint = msg_roi_start + (msg_roi_end - msg_roi_start) / 2
        lag_minutes = (mtg_midpoint - msg_midpoint).total_seconds() / 60.0
        print(
            f"Pair {pair_index + 1}/{len(pairs)}: {len(mtg_files)} MTG chunks, "
            f"{Path(msg_file).name}, lag={lag_minutes:+.1f} min"
        )

        for task in tasks:
            print(
                f"  MTG={task.mtg_channel}, MSG={task.msg_channel}, "
                f"calibration={task.calibration}"
            )
            mtg_scene = None
            msg_scene = None
            try:
                mtg_scene = Scene(filenames=mtg_files, reader="fci_l1c_nc")
                mtg_scene.load([task.mtg_channel], calibration=task.calibration)
                msg_scene = Scene(filenames=[msg_file], reader="seviri_l1b_native")
                msg_scene.load([task.msg_channel], calibration=task.calibration)

                if args.align_mode == "MSG_TO_MTG":
                    destination = mtg_scene[task.mtg_channel].attrs.get("area")
                    if destination is None:
                        destination = mtg_scene.finest_area()
                    msg_scene = msg_scene.resample(destination, resampler="nearest")
                elif args.align_mode == "MTG_TO_MSG":
                    destination = msg_scene[task.msg_channel].attrs.get("area")
                    if destination is None:
                        destination = msg_scene.finest_area()
                    mtg_scene = mtg_scene.resample(destination, resampler="nearest")

                if args.crop:
                    mtg_scene = mtg_scene.crop(ll_bbox=args.roi)
                    msg_scene = msg_scene.crop(ll_bbox=args.roi)

                mtg_array = mtg_scene[task.mtg_channel].compute().values.astype(np.float32)
                msg_array = msg_scene[task.msg_channel].compute().values.astype(np.float32)
                if task.calibration == "reflectance" and np.nanmax(mtg_array) > 2:
                    mtg_array /= 100.0
                if task.calibration == "reflectance" and np.nanmax(msg_array) > 2:
                    msg_array /= 100.0

                value_min, value_max = get_limits(mtg_array, task.calibration)
                timestamp = mtg_start.strftime("%Y%m%d%H%M%S")
                pair_name = (
                    f"{phase_name}_{task.mtg_channel}_{task.msg_channel}_"
                    f"{timestamp}_pair{pair_index:03d}"
                )
                writer.save_random_patches(
                    mtg_array,
                    msg_array,
                    value_min,
                    value_max,
                    {
                        "phase": phase_name,
                        "pair_index": pair_index,
                        "pair_name": pair_name,
                        "mtg_channel": task.mtg_channel,
                        "msg_channel": task.msg_channel,
                        "calibration": task.calibration,
                        "mtg_scan_start": mtg_start.isoformat(),
                        "msg_file": str(Path(msg_file).resolve()),
                        "mtg_files": [str(Path(path).resolve()) for path in mtg_files],
                        "alignment": args.align_mode,
                    },
                )
            except Exception as error:
                print(f"[ERROR] {task.mtg_channel}/{task.msg_channel}: {error}")
                traceback.print_exc()
            finally:
                del mtg_scene, msg_scene
                gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--std-dir", type=Path, default=Path("Comparison"))
    parser.add_argument("--fine-dir", type=Path, default=Path("Comparison_Fine"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("MTG_MSG_regenerated")
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-channel", type=int, default=250)
    parser.add_argument("--nodata-fraction-max", type=float, default=0.01)
    parser.add_argument("--black-fraction-max", type=float, default=0.70)
    parser.add_argument(
        "--align-mode",
        choices=("MSG_TO_MTG", "MTG_TO_MSG", "NONE"),
        default="MSG_TO_MTG",
    )
    parser.add_argument("--crop", action="store_true", help="Crop to --roi after alignment")
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=(-10.0, -10.0, 10.0, 10.0),
    )
    parser.add_argument("--mtg-duration", type=float, default=9.5)
    parser.add_argument("--msg-duration", type=float, default=12.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional NumPy seed. The historical generation did not set one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    writer = PatchWriter(
        args.output_root,
        args.patch_size,
        args.patches_per_channel,
        args.nodata_fraction_max,
        args.black_fraction_max,
    )
    try:
        process_phase("Standard_Resolution", args.std_dir, args.std_dir, STANDARD_TASKS, writer, args)
        process_phase("High_Resolution", args.fine_dir, args.std_dir, FINE_TASKS, writer, args)
    finally:
        writer.close()
    print(f"Generated {writer.counter - 1} aligned pairs under {args.output_root}.")
    print(f"Crop manifest: {writer.manifest_path}")


if __name__ == "__main__":
    main()
