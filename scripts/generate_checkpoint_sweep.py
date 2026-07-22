#!/usr/bin/env python3
"""Generate predictions for every checkpoint in a journal checkpoint sweep."""

import argparse
import importlib
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", help="Optional checkpoint steps to run")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sd-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inference-module", default="inference_sr")
    parser.add_argument("--sampler-module", default="sampler_sr")
    parser.add_argument("--sampler-class", default="SuperResolutionSampler")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chopping-size", type=int, default=128)
    parser.add_argument("--chopping-bs", type=int, default=8)
    parser.add_argument("--color-fix", default="", choices=("", "rgb", "wavelet", "ycbcr"))
    return parser.parse_args()


def checkpoint_step(path):
    match = re.fullmatch(r"model_(\d+)\.pth", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def inference_args(args, first_checkpoint):
    return SimpleNamespace(
        in_path=str(args.input_dir),
        out_path="",
        bs=args.batch_size,
        chopping_bs=args.chopping_bs,
        timesteps=None,
        num_steps=args.num_steps,
        start_step=250,
        cfg_path=str(args.project_root / "configs" / "sample-sd-turbo.yaml"),
        sd_path=str(args.sd_path) if args.sd_path is not None else "",
        started_ckpt_path=str(first_checkpoint),
        tiled_vae=True,
        color_fix=args.color_fix,
        chopping_size=args.chopping_size,
    )


def image_count(directory):
    return sum(path.suffix.lower() == ".png" for path in directory.iterdir())


def load_checkpoint(model, checkpoint, util_net):
    state = torch.load(checkpoint, map_location="cuda")
    if "state_dict" in state:
        state = state["state_dict"]
    util_net.reload_model(model, state)
    model.eval()


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def main():
    args = parse_args()
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "src"))

    from utils import util_net

    inference_module = importlib.import_module(args.inference_module)
    sampler_module = importlib.import_module(args.sampler_module)
    get_configs = inference_module.get_configs
    sampler_class = getattr(sampler_module, args.sampler_class)

    checkpoints = sorted(args.checkpoint_dir.glob("model_*.pth"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    if args.steps:
        requested_steps = set(args.steps)
        checkpoints = [path for path in checkpoints if checkpoint_step(path) in requested_steps]
        found_steps = {checkpoint_step(path) for path in checkpoints}
        if found_steps != requested_steps:
            missing = sorted(requested_steps - found_steps)
            raise FileNotFoundError(f"Requested checkpoint steps not found: {missing}")
    expected_images = image_count(args.input_dir)
    if expected_images == 0:
        raise FileNotFoundError(f"No PNG inputs found in {args.input_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = get_configs(inference_args(args, checkpoints[0]))
    if args.sd_path is not None and (args.sd_path / "model_index.json").exists():
        configs.sd_pipe.params.pretrained_model_name_or_path = str(args.sd_path)
    if args.local_files_only:
        configs.sd_pipe.params.local_files_only = True
        configs.sd_pipe.params.token = False
    sampler = sampler_class(configs)
    sampler.sd_pipe.set_progress_bar_config(disable=True)
    predictor = sampler.sd_pipe.start_noise_predictor
    # Match standalone inference: its seed is set before model construction, so
    # each checkpoint must begin from the same post-construction RNG state.
    inference_rng_state = capture_rng_state()

    completed = []
    for index, checkpoint in enumerate(checkpoints):
        step = checkpoint_step(checkpoint)
        output_dir = args.output_root / f"step_{step:06d}"
        if output_dir.is_dir() and image_count(output_dir) == expected_images:
            print(f"Skipping complete checkpoint {step}: {output_dir}", flush=True)
        else:
            if index > 0:
                load_checkpoint(predictor, checkpoint, util_net)
            restore_rng_state(inference_rng_state)
            print(f"Generating checkpoint {step}: {checkpoint}", flush=True)
            sampler.inference(args.input_dir, output_dir, bs=args.batch_size)
            generated = image_count(output_dir)
            if generated != expected_images:
                raise RuntimeError(
                    f"Checkpoint {step} generated {generated}/{expected_images} images"
                )
        completed.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "prediction_dir": str(output_dir.resolve()),
                "sample_count": expected_images,
            }
        )

    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "chopping_size": args.chopping_size,
        "chopping_batch_size": args.chopping_bs,
        "seed": int(configs.seed),
        "rng_protocol": "restore_post_model_construction_state",
        "color_fix": args.color_fix or None,
        "project_root": str(args.project_root.resolve()),
        "sd_path": str(args.sd_path.resolve()) if args.sd_path is not None else None,
        "local_files_only": args.local_files_only,
        "inference_module": args.inference_module,
        "sampler_module": args.sampler_module,
        "sampler_class": args.sampler_class,
        "checkpoints": completed,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
