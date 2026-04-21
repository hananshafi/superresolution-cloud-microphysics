#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2024-12-11 17:17:41

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
from sampler_sr import SuperResolutionSampler
import os
from tqdm import tqdm

from utils import util_common
from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(num_steps=1, chopping_size=128, seed=12345):
    configs = OmegaConf.load("./configs/sample-sd-turbo.yaml")

    if num_steps == 1:
        configs.timesteps = [200,]
    elif num_steps == 2:
        configs.timesteps = [200, 100]
    elif num_steps == 3:
        configs.timesteps = [200, 100, 50]
    elif num_steps == 4:
        configs.timesteps = [200, 150, 100, 50]
    elif num_steps == 5:
        configs.timesteps = [250, 200, 150, 100, 50]
    else:
        assert num_steps <= 250
        configs.timesteps = np.linspace(
            start=250, stop=0, num=num_steps, endpoint=False, dtype=np.int64()
        ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    configs.sd_path = "./weights"
    util_common.mkdir(configs.sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = configs.sd_path

    started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
    started_ckpt_dir = "./weights"
    util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
    started_ckpt_path = "/share/data/drive_4/hanan/InvSR_logs/2026-02-02-06-42/ckpts/model_50000.pth"
    if not os.path.exists(started_ckpt_path):
        load_file_from_url(
            url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
            model_dir=started_ckpt_dir,
            progress=True,
            file_name=started_ckpt_name,
        )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = 1
    configs.seed = seed
    configs.basesr.chopping.pch_size = chopping_size
    configs.basesr.chopping.extra_bs = 4

    return configs

def predict_single(in_path, num_steps=1, chopping_size=128, seed=12345):
    configs = get_configs(num_steps=num_steps, chopping_size=chopping_size, seed=seed)
    sampler = SuperResolutionSampler(configs)

    out_dir = Path('sr_output')
    if not out_dir.exists():
        out_dir.mkdir()
    sampler.inference(in_path, out_path=out_dir, bs=1)

    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")

    return im_sr, str(out_path)

def process_batch(input_dir, num_steps=1, chopping_size=128, seed=12345, progress=gr.Progress()):
    input_path = Path(input_dir)
    output_path = input_path / 'sr_output'
    output_path.mkdir(exist_ok=True)

    configs = get_configs(num_steps=num_steps, chopping_size=chopping_size, seed=seed)
    sampler = SuperResolutionSampler(configs)

    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
    total_files = len(image_files)

    if total_files == 0:
        return f"No image files found in {input_dir}"

    progress(0, desc="Processing images")
    for idx, img_path in enumerate(image_files):
        out_path = output_path / f"{img_path.stem}.png"
        sampler.inference(str(img_path), out_path=output_path, bs=1)
        progress((idx + 1)/total_files, desc=f"Processing image {idx + 1}/{total_files}")

    return f"Processed {total_files} images. Results saved in {output_path}"

title = "Cloud Microphysics Super-Resolution"

article = r"""
Single-stage demo for the inference code in this repository.
---
**License**
This project is licensed under <a rel="license" href="https://github.com/zsyOAOA/InvSR/blob/master/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.
"""
description = r"""
Single-stage super-resolution demo for cloud imagery using the local inference pipeline.<br>
"""

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Input: Low Quality Image")
                    num_steps = gr.Dropdown(
                        choices=[1,2,3,4,5],
                        value=1,
                        label="Number of steps",
                    )
                    chopping_size = gr.Dropdown(
                        choices=[128, 256],
                        value=128,
                        label="Chopping size",
                    )
                    seed = gr.Number(value=12345, precision=0, label="Random seed")
                    process_btn = gr.Button("Process")

                with gr.Column():
                    output_image = gr.Image(type="numpy", label="Output: High Quality Image")
                    output_file = gr.File(label="Download the output")

            process_btn.click(
                fn=predict_single,
                inputs=[input_image, num_steps, chopping_size, seed],
                outputs=[output_image, output_file]
            )

        with gr.Tab("Batch Processing"):
            input_dir = gr.Textbox(label="Input Directory Path")
            batch_num_steps = gr.Dropdown(
                choices=[1,2,3,4,5],
                value=1,
                label="Number of steps",
            )
            batch_chopping_size = gr.Dropdown(
                choices=[128, 256],
                value=128,
                label="Chopping size",
            )
            batch_seed = gr.Number(value=12345, precision=0, label="Random seed")
            batch_btn = gr.Button("Process Folder")
            output_text = gr.Textbox(label="Processing Status")

            batch_btn.click(
                fn=process_batch,
                inputs=[input_dir, batch_num_steps, batch_chopping_size, batch_seed],
                outputs=output_text
            )

    gr.Markdown(article)

demo.queue(max_size=5)
demo.launch(share=True)
