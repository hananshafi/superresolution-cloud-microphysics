#!/usr/bin/env python3
"""Benchmark Stage 2 journal inference on one GPU."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sd-path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--chopping-size", type=int, default=128)
    parser.add_argument("--chopping-bs", type=int, default=8)
    parser.add_argument("--color-fix", default="")
    return parser.parse_args()


def inference_args(args):
    return SimpleNamespace(
        in_path=str(args.in_dir),
        out_path="",
        bs=1,
        chopping_bs=args.chopping_bs,
        timesteps=None,
        num_steps=args.num_steps,
        start_step=250,
        cfg_path=str(args.project_root / "configs" / "sample-sd-turbo.yaml"),
        sd_path=str(args.sd_path) if args.sd_path is not None else "",
        started_ckpt_path=str(args.checkpoint),
        tiled_vae=True,
        color_fix=args.color_fix,
        chopping_size=args.chopping_size,
    )


def timed_call(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def summary(values):
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "images_per_second": 1000.0 / statistics.mean(values),
    }


def main():
    args = parse_args()
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "src"))

    from inference_sr import get_configs
    from sampler_sr import SuperResolutionSampler
    from utils import util_image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    image_paths = sorted(args.in_dir.glob("*.png"))[: args.num_images]
    if len(image_paths) != args.num_images:
        raise RuntimeError(f"Expected {args.num_images} PNG inputs, found {len(image_paths)}")

    load_start = time.perf_counter()
    configs = get_configs(inference_args(args))
    sampler = SuperResolutionSampler(configs)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    tensors = []
    for path in image_paths:
        image = util_image.imread(path, chn="rgb", dtype="float32")
        tensors.append(util_image.img2tensor(image).cuda())

    def stage2_inference(tensor):
        return sampler.sample_func(tensor, return_tensor=True)

    for tensor in tensors[: args.warmup]:
        stage2_inference(tensor)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    inference_times = []
    for tensor in tensors:
        _, elapsed = timed_call(lambda tensor=tensor: stage2_inference(tensor))
        inference_times.append(elapsed)

    output = {
        "device": torch.cuda.get_device_name(0),
        "num_images": len(tensors),
        "num_steps_per_pass": args.num_steps,
        "input_shape": list(tensors[0].shape),
        "model_load_seconds": load_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "stage2_checkpoint": str(args.checkpoint.resolve()),
        "stage2_inference": summary(inference_times),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
