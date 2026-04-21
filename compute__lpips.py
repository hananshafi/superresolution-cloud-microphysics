#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import math

import numpy as np
import torch
import lpips
from PIL import Image


def load_image_as_tensor(path: Path, device: torch.device):
    """
    Load image as float tensor in [-1, 1], shape (1, 3, H, W).
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # [0,1], HWC
    # to CHW
    arr = np.transpose(arr, (2, 0, 1))  # 3xHxW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # 1x3xHxW
    tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1] for LPIPS
    return tensor


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor):
    """
    Compute PSNR between two images in [-1,1], shape (1,3,H,W) or (3,H,W).
    Returns PSNR in dB.
    """
    # Convert to [0,1] for PSNR computation
    x = (img1 + 1.0) / 2.0
    y = (img2 + 1.0) / 2.0

    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def main(seviri_dir, viirs_dir, resize_to_seviri=True):
    seviri_dir = Path(seviri_dir)
    viirs_dir = Path(viirs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    seviri_paths = sorted(seviri_dir.glob("seviri_*.*"))

    total_lpips = 0.0
    total_psnr = 0.0
    count = 0

    for seviri_path in seviri_paths:
        # Extract numeric ID after "seviri_"
        stem = seviri_path.stem  # e.g., seviri_123
        if not stem.startswith("seviri_"):
            continue
        image_id = stem.split("seviri_")[1]  # "123"

        # Assume same extension, but you can relax this if needed
        ext = seviri_path.suffix  # .png, .jpg, ...
        viirs_path = viirs_dir / f"viirs_{image_id}{ext}"
        if not viirs_path.exists():
            print(f"[WARN] No matching VIIRS image for {seviri_path.name}")
            continue

        # Load images as PIL first to handle size
        sev_img = Image.open(seviri_path).convert("RGB")
        viirs_img = Image.open(viirs_path).convert("RGB")

        # Optionally resize VIIRS to SEVIRI size (common for SR evaluation)
        if resize_to_seviri and sev_img.size != viirs_img.size:
            viirs_img = viirs_img.resize(sev_img.size, Image.BICUBIC)

        # Convert back to tensors
        sev_arr = np.array(sev_img).astype(np.float32) / 255.0
        viirs_arr = np.array(viirs_img).astype(np.float32) / 255.0

        sev_arr = np.transpose(sev_arr, (2, 0, 1))  # 3xHxW
        viirs_arr = np.transpose(viirs_arr, (2, 0, 1))

        sev_tensor = torch.from_numpy(sev_arr).unsqueeze(0).to(device)  # 1x3xHxW
        viirs_tensor = torch.from_numpy(viirs_arr).unsqueeze(0).to(device)

        # Map to [-1,1] for LPIPS
        sev_tensor_lpips = sev_tensor * 2.0 - 1.0
        viirs_tensor_lpips = viirs_tensor * 2.0 - 1.0

        with torch.no_grad():
            lpips_val = loss_fn(sev_tensor_lpips, viirs_tensor_lpips).item()
        psnr_val = compute_psnr(sev_tensor_lpips, viirs_tensor_lpips)

        total_lpips += lpips_val
        total_psnr += psnr_val
        count += 1

        print(f"{seviri_path.name} <-> {viirs_path.name} | LPIPS: {lpips_val:.4f}, PSNR: {psnr_val:.2f} dB")

    if count == 0:
        print("No valid pairs found.")
        return

    avg_lpips = total_lpips / count
    avg_psnr = total_psnr / count

    print("\n==================== Summary ====================")
    print(f"Number of pairs: {count}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Average PSNR:  {avg_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average LPIPS and PSNR for SEVIRI–VIIRS pairs.")
    parser.add_argument("--seviri_dir", type=str,default='results_100k',
                        help="Folder containing seviri_{num} images.")
    parser.add_argument("--viirs_dir", type=str,default='/share/data/drive_3/hanan/climate_data/train_data/test/viirs',
                        help="Folder containing viirs_{num} images.")
    parser.add_argument("--no_resize", action="store_true",
                        help="If set, do NOT resize VIIRS to SEVIRI size.")
    args = parser.parse_args()

    main(args.seviri_dir, args.viirs_dir, resize_to_seviri=not args.no_resize)
