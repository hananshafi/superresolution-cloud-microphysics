#!/usr/bin/env python3
"""Benchmark single-pass and cascaded journal inference on one GPU."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
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
        started_ckpt_path=str(args.stage1_checkpoint),
        tiled_vae=True,
        color_fix=args.color_fix,
        chopping_size=args.chopping_size,
    )


def load_predictor(configs, checkpoint, util_common, util_net):
    model_config = configs.model_start
    params = model_config.get("params", {})
    model = util_common.get_obj_from_str(model_config.target)(**params).cuda()
    state = torch.load(checkpoint, map_location="cuda")
    if "state_dict" in state:
        state = state["state_dict"]
    util_net.reload_model(model, state)
    return model.eval()


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
    from utils import util_common, util_image, util_net

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    image_paths = sorted(args.in_dir.glob("*.png"))[: args.num_images]
    if len(image_paths) != args.num_images:
        raise RuntimeError(f"Expected {args.num_images} PNG inputs, found {len(image_paths)}")

    load_start = time.perf_counter()
    configs = get_configs(inference_args(args))
    sampler = SuperResolutionSampler(configs)
    stage1_predictor = sampler.sd_pipe.start_noise_predictor
    stage2_predictor = load_predictor(configs, args.stage2_checkpoint, util_common, util_net)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    tensors = []
    for path in image_paths:
        image = util_image.imread(path, chn="rgb", dtype="float32")
        tensors.append(util_image.img2tensor(image).cuda())

    def single_pass(tensor):
        sampler.sd_pipe.start_noise_predictor = stage2_predictor
        return sampler.sample_func(tensor, return_tensor=True)

    def cascaded_pass(tensor):
        sampler.sd_pipe.start_noise_predictor = stage1_predictor
        stage1_output = sampler.sample_func(tensor, return_tensor=True)
        stage2_input = F.interpolate(
            stage1_output,
            size=tensor.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        sampler.sd_pipe.start_noise_predictor = stage2_predictor
        return sampler.sample_func(stage2_input, return_tensor=True)

    for tensor in tensors[: args.warmup]:
        single_pass(tensor)
        cascaded_pass(tensor)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    single_times = []
    cascade_times = []
    for tensor in tensors:
        _, elapsed = timed_call(lambda tensor=tensor: single_pass(tensor))
        single_times.append(elapsed)
        _, elapsed = timed_call(lambda tensor=tensor: cascaded_pass(tensor))
        cascade_times.append(elapsed)

    output = {
        "device": torch.cuda.get_device_name(0),
        "num_images": len(tensors),
        "num_steps_per_pass": args.num_steps,
        "input_shape": list(tensors[0].shape),
        "model_load_seconds": load_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "single_pass": summary(single_times),
        "cascaded": summary(cascade_times),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
