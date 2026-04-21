#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from sampler_sr import SuperResolutionSampler

from utils import util_common
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_SD_TURBO_DIRS = [
    PROJECT_ROOT / "checkpoints" / "sd-turbo",
    Path("/share/data/drive_3/hanan/InvSR/checkpoints/sd-turbo"),
    Path("/share/data/drive_1/hanan/InvSR/checkpoints/sd-turbo"),
    Path("/share/data/drive_3/hanan/InvSR/weights/models--stabilityai--sd-turbo"),
    Path("/share/data/drive_1/hanan/InvSR/weights/models--stabilityai--sd-turbo"),
]

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path")
    parser.add_argument("-o", "--out_path", type=str, default="", help="Output path")
    parser.add_argument("--bs", type=int, default=1, help="Batchsize for loading image")
    parser.add_argument("--chopping_bs", type=int, default=8, help="Batchsize for chopped patch")
    parser.add_argument("-t", "--timesteps", type=int, nargs="+", help="The inversed timesteps")
    parser.add_argument("-n", "--num_steps", type=int, default=1, help="Number of inference steps")
    parser.add_argument(
        "--start_step",
        type=int,
        default=250,
        help="Starting diffusion timestep used when num_steps is greater than 5",
    )
    parser.add_argument(
        "--cfg_path", type=str, default="./configs/sample-sd-turbo.yaml", help="Configuration path.",
    )
    parser.add_argument(
        "--sd_path", type=str, default="", help="Path for Stable Diffusion Model",
    )
    parser.add_argument(
        "--started_ckpt_path", type=str, default="", help="Checkpoint path for noise predictor"
    )
    parser.add_argument(
        "--tiled_vae", type=str2bool, default='true', help="Enabled tiled VAE.",
    )
    parser.add_argument(
        "--color_fix", type=str, default='', choices=['rgb', 'wavelet', 'ycbcr'], help="Fix the color shift",
    )
    parser.add_argument(
        "--chopping_size", type=int, default=128, help="Chopping size when dealing large images"
    )
    args = parser.parse_args()

    return args

def resolve_snapshot_dir(candidate: Path) -> str:
    if not candidate.exists():
        return ""

    if (candidate / "model_index.json").exists():
        return str(candidate)

    snapshots_dir = candidate / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_dirs = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        for snapshot_dir in snapshot_dirs:
            if (snapshot_dir / "model_index.json").exists():
                return str(snapshot_dir)

    return ""

def is_valid_sd_turbo_dir(path_str: str) -> bool:
    path = Path(path_str)
    model_index = path / "model_index.json"
    unet_candidates = [
        path / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        path / "unet" / "diffusion_pytorch_model.safetensors",
    ]
    text_encoder_candidates = [
        path / "text_encoder" / "model.fp16.safetensors",
        path / "text_encoder" / "model.safetensors",
    ]
    unet_weights = next((candidate for candidate in unet_candidates if candidate.exists()), None)
    text_encoder_weights = next((candidate for candidate in text_encoder_candidates if candidate.exists()), None)

    if not model_index.exists() or unet_weights is None or text_encoder_weights is None:
        return False

    # Git LFS placeholders are tiny text files; valid weights are much larger.
    return unet_weights.stat().st_size > 1_000_000 and text_encoder_weights.stat().st_size > 1_000_000

def resolve_sd_turbo_dir(sd_path: str) -> str:
    if sd_path:
        resolved = resolve_snapshot_dir(Path(sd_path))
        return resolved if resolved and is_valid_sd_turbo_dir(resolved) else sd_path

    for candidate in DEFAULT_LOCAL_SD_TURBO_DIRS:
        resolved = resolve_snapshot_dir(candidate)
        if resolved and is_valid_sd_turbo_dir(resolved):
            return resolved

    return ""

def get_configs(args):
    configs = OmegaConf.load(args.cfg_path)

    if args.timesteps is not None:
        assert len(args.timesteps) == args.num_steps
        configs.timesteps = sorted(args.timesteps, reverse=True)
    else:
        if args.num_steps == 1:
            configs.timesteps = [200,]
        elif args.num_steps == 2:
            configs.timesteps = [200, 100]
        elif args.num_steps == 3:
            configs.timesteps = [200, 100, 50]
        elif args.num_steps == 4:
            configs.timesteps = [200, 150, 100, 50]
        elif args.num_steps == 5:
            configs.timesteps = [250, 200, 150, 100, 50]
        else:
            assert args.num_steps <= 250
            configs.timesteps = np.linspace(
                start=args.start_step, stop=0, num=args.num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    # Resolve Stable Diffusion weights without requiring a copied local model repo.
    sd_turbo_dir = resolve_sd_turbo_dir(args.sd_path)
    if sd_turbo_dir:
        configs.sd_pipe.params.pretrained_model_name_or_path = sd_turbo_dir
    else:
        sd_cache_dir = str(PROJECT_ROOT / "weights")
        util_common.mkdir(sd_cache_dir, delete=False, parents=True)
        configs.sd_pipe.params.cache_dir = sd_cache_dir

    # path to save noise predictor
    if args.started_ckpt_path:
        started_ckpt_path = args.started_ckpt_path
    else:
        started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
        started_ckpt_dir = str(PROJECT_ROOT / "weights")
        util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
        started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
        if not started_ckpt_path.exists():
            load_file_from_url(
                url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
                model_dir=started_ckpt_dir,
                progress=True,
                file_name=started_ckpt_name,
            )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = args.bs
    configs.tiled_vae = args.tiled_vae
    configs.color_fix = args.color_fix
    configs.basesr.chopping.pch_size = args.chopping_size
    if args.bs > 1:
        configs.basesr.chopping.extra_bs = 1
    else:
        configs.basesr.chopping.extra_bs = args.chopping_bs

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    sampler = SuperResolutionSampler(configs)

    sampler.inference(args.in_path, out_path=args.out_path, bs=args.bs)

if __name__ == '__main__':
    main()
