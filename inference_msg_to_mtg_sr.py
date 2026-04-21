#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from inference_sr import get_configs, get_parser
from sampler_sr import SuperResolutionSampler

DEFAULT_LOCAL_MSG_TO_MTG_CKPT = Path(
    "/share/data/drive_3/hanan/InvSR/msg_mtg_init_paired/2026-01-16-07-23/ckpts/model_50000.pth"
)
DEFAULT_COLOR_FIX = "rgb"


def main():
    args = get_parser(description="MSG to MTG super-resolution inference")

    if not args.in_path:
        raise ValueError("Please provide --in_path for MSG to MTG inference.")
    if not args.out_path:
        raise ValueError("Please provide --out_path for MSG to MTG inference.")

    if not args.started_ckpt_path:
        if DEFAULT_LOCAL_MSG_TO_MTG_CKPT.exists():
            args.started_ckpt_path = str(DEFAULT_LOCAL_MSG_TO_MTG_CKPT)
            print(f"Using local MSG->MTG checkpoint: {args.started_ckpt_path}", flush=True)
        else:
            raise ValueError(
                "MSG to MTG inference requires --started_ckpt_path, or the local checkpoint "
                f"{DEFAULT_LOCAL_MSG_TO_MTG_CKPT} must exist."
            )

    if not args.color_fix:
        args.color_fix = DEFAULT_COLOR_FIX

    configs = get_configs(args)
    sampler = SuperResolutionSampler(configs)
    sampler.inference(args.in_path, out_path=args.out_path, bs=args.bs)


if __name__ == "__main__":
    main()
