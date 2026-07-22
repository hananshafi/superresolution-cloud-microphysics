#!/usr/bin/env python3
"""Evaluate journal super-resolution results with one consistent protocol."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("seviri-viirs", "msg-mtg"))
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--bicubic-dir", required=True, type=Path)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Prediction name and directory. May be supplied more than once.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_prediction(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=DIR, received: {value}")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError(f"Expected NAME=DIR, received: {value}")
    return name, Path(directory)


def sample_id(path):
    match = re.search(r"_(\d+)(?:_|$)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract a sample ID from {path.name}")
    return int(match.group(1))


def index_images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    indexed = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = sample_id(path)
        if image_id in indexed:
            raise ValueError(f"Duplicate sample ID {image_id} in {directory}")
        indexed[image_id] = path

    if not indexed:
        raise ValueError(f"No supported images found in {directory}")
    return indexed


def validate_ids(gt_index, prediction_index, name):
    gt_ids = set(gt_index)
    prediction_ids = set(prediction_index)
    missing = sorted(gt_ids - prediction_ids)
    extra = sorted(prediction_ids - gt_ids)
    if missing or extra:
        raise ValueError(
            f"{name} does not match the ground truth: "
            f"{len(missing)} missing IDs and {len(extra)} extra IDs"
        )


def load_rgb(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Unsupported image shape {array.shape} for {path}")

    if array.dtype == np.uint8:
        scale = 255.0
    elif array.dtype == np.uint16:
        scale = 65535.0
    else:
        raise ValueError(f"Unsupported image dtype {array.dtype} for {path}")
    return np.ascontiguousarray(array.astype(np.float32) / scale)


def gradient_ratio(prediction, target):
    # Luminance follows the RGB convention used by the prior paper evaluation.
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    prediction_gray = prediction @ weights
    target_gray = target @ weights

    pred_dx = cv2.Sobel(prediction_gray, cv2.CV_32F, 1, 0, ksize=3)
    pred_dy = cv2.Sobel(prediction_gray, cv2.CV_32F, 0, 1, ksize=3)
    target_dx = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    target_dy = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)

    pred_magnitude = cv2.magnitude(pred_dx, pred_dy)
    target_magnitude = cv2.magnitude(target_dx, target_dy)
    return float(pred_magnitude.mean() / (target_magnitude.mean() + 1e-8))


def pixel_metrics(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")

    mse = float(np.mean(np.square(prediction - target), dtype=np.float64))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ssim = structural_similarity(target, prediction, channel_axis=2, data_range=1.0)
    return {
        "psnr": psnr,
        "rmse": rmse,
        "ssim": float(ssim),
        "gradient_ratio": gradient_ratio(prediction, target),
    }


@torch.inference_mode()
def lpips_batch(model, predictions, targets, device):
    prediction_tensor = torch.from_numpy(np.stack(predictions)).permute(0, 3, 1, 2)
    target_tensor = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2)
    prediction_tensor = prediction_tensor.to(device=device, non_blocking=True) * 2.0 - 1.0
    target_tensor = target_tensor.to(device=device, non_blocking=True) * 2.0 - 1.0
    values = model(prediction_tensor, target_tensor).flatten()
    return values.detach().cpu().numpy().astype(np.float64).tolist()


def evaluate_method(name, prediction_index, gt_index, lpips_model, device, batch_size):
    per_image = []
    ids = sorted(gt_index)
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        predictions = [load_rgb(prediction_index[image_id]) for image_id in batch_ids]
        targets = [load_rgb(gt_index[image_id]) for image_id in batch_ids]
        lpips_values = lpips_batch(lpips_model, predictions, targets, device)

        for image_id, prediction, target, lpips_value in zip(
            batch_ids, predictions, targets, lpips_values
        ):
            metrics = pixel_metrics(prediction, target)
            metrics["id"] = image_id
            metrics["lpips"] = lpips_value
            per_image.append(metrics)

        batch_number = start // batch_size + 1
        processed = min(start + batch_size, len(ids))
        if batch_number == 1 or batch_number % 10 == 0 or processed == len(ids):
            print(f"{name}: {processed}/{len(ids)}", flush=True)
    return per_image


def summarize(per_image, bicubic_lpips):
    metric_names = ("psnr", "rmse", "ssim", "gradient_ratio", "lpips")
    summary = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in per_image], dtype=np.float64)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    lpips_values = np.asarray([row["lpips"] for row in per_image], dtype=np.float64)
    ratios = lpips_values / np.maximum(bicubic_lpips, 1e-12)
    summary["perceptual_ratio"] = {
        "mean": float(ratios.mean()),
        "std": float(ratios.std(ddof=0)),
    }
    return summary


def write_csv(path, benchmark, sample_count, methods):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "method",
        "prediction_dir",
        "sample_count",
        "psnr_mean",
        "psnr_std",
        "rmse_mean",
        "rmse_std",
        "ssim_mean",
        "ssim_std",
        "gradient_ratio_mean",
        "gradient_ratio_std",
        "lpips_mean",
        "lpips_std",
        "perceptual_ratio_mean",
        "perceptual_ratio_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, method in methods.items():
            row = {
                "benchmark": benchmark,
                "method": name,
                "prediction_dir": method["prediction_dir"],
                "sample_count": sample_count,
            }
            for metric_name, values in method["metrics"].items():
                row[f"{metric_name}_mean"] = values["mean"]
                row[f"{metric_name}_std"] = values["std"]
            writer.writerow(row)


def main():
    args = parse_args()
    prediction_specs = [("Bicubic", args.bicubic_dir)]
    prediction_specs.extend(parse_prediction(value) for value in args.prediction)
    names = [name for name, _ in prediction_specs]
    if len(names) != len(set(names)):
        raise ValueError("Prediction names must be unique")

    gt_index = index_images(args.gt_dir)
    prediction_indices = {}
    for name, directory in prediction_specs:
        prediction_index = index_images(directory)
        validate_ids(gt_index, prediction_index, name)
        prediction_indices[name] = prediction_index

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    per_method = {}
    per_method["Bicubic"] = evaluate_method(
        "Bicubic",
        prediction_indices["Bicubic"],
        gt_index,
        lpips_model,
        device,
        args.batch_size,
    )
    bicubic_lpips = np.asarray(
        [row["lpips"] for row in per_method["Bicubic"]], dtype=np.float64
    )

    for name, _ in prediction_specs[1:]:
        per_method[name] = evaluate_method(
            name,
            prediction_indices[name],
            gt_index,
            lpips_model,
            device,
            args.batch_size,
        )

    methods = {}
    directories = dict(prediction_specs)
    for name, rows in per_method.items():
        methods[name] = {
            "prediction_dir": str(directories[name].resolve()),
            "metrics": summarize(rows, bicubic_lpips),
        }

    output = {
        "benchmark": args.benchmark,
        "ground_truth_dir": str(args.gt_dir.resolve()),
        "sample_count": len(gt_index),
        "lpips_backbone": "alex",
        "std_definition": "population",
        "rmse_units": "normalized_[0,1]",
        "methods": methods,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, args.benchmark, len(gt_index), methods)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
