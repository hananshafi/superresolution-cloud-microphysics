#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import threading
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from inference_sr import get_configs as build_inference_configs
from sampler_sr import SuperResolutionSampler
from utils import util_image

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "sr_output"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_REPO_ID = os.environ.get("CLOUDSR_CHECKPOINT_REPO_ID", "hanangani/cloudsr-checkpoints")

TASKS = {
    "SEVIRI -> VIIRS": {
        "checkpoint_filename": os.environ.get(
            "CLOUDSR_SEVIRI_TO_VIIRS_CKPT",
            "cloudsr_seviri_to_viirs_model_50000.pth",
        ),
        "default_color_fix": "rgb",
        "summary": "SEVIRI low-resolution input to VIIRS-style super-resolution.",
    },
    "MSG -> MTG": {
        "checkpoint_filename": os.environ.get(
            "CLOUDSR_MSG_TO_MTG_CKPT",
            "cloudsr_msg_to_mtg_model_50000.pth",
        ),
        "default_color_fix": "rgb",
        "summary": "MSG low-resolution input to MTG-style super-resolution.",
    },
}

COLOR_FIX_OPTIONS = {
    "None": "",
    "RGB preserve": "rgb",
    "Wavelet": "wavelet",
    "YCbCr": "ycbcr",
}

_sampler_cache = {}
_sampler_build_lock = threading.Lock()
_inference_lock = threading.Lock()


def make_demo_args(started_ckpt_path, num_steps=1, chopping_size=128, seed=12345, color_fix="rgb"):
    return SimpleNamespace(
        bs=1,
        chopping_bs=4,
        timesteps=None,
        num_steps=int(num_steps),
        start_step=250,
        cfg_path=str(PROJECT_ROOT / "configs" / "sample-sd-turbo.yaml"),
        sd_path="",
        started_ckpt_path=str(started_ckpt_path),
        tiled_vae=True,
        color_fix=color_fix,
        chopping_size=int(chopping_size),
    )


def download_task_checkpoint(task_name):
    task = TASKS[task_name]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        repo_type="model",
        filename=task["checkpoint_filename"],
        local_dir=str(CHECKPOINT_DIR),
    )


def task_slug(task_name):
    if task_name == "SEVIRI -> VIIRS":
        return "seviri_to_viirs"
    if task_name == "MSG -> MTG":
        return "msg_to_mtg"
    return task_name.lower().replace(" ", "_")


def get_task_sampler(task_name):
    sampler = _sampler_cache.get(task_name)
    if sampler is not None:
        return sampler

    with _sampler_build_lock:
        sampler = _sampler_cache.get(task_name)
        if sampler is not None:
            return sampler

        checkpoint_path = download_task_checkpoint(task_name)
        args = make_demo_args(
            started_ckpt_path=checkpoint_path,
            color_fix=TASKS[task_name]["default_color_fix"],
        )
        configs = build_inference_configs(args)
        sampler = SuperResolutionSampler(configs)
        _sampler_cache[task_name] = sampler
        return sampler


def update_sampler_runtime_config(sampler, checkpoint_path, num_steps, chopping_size, seed, color_fix):
    runtime_args = make_demo_args(
        started_ckpt_path=checkpoint_path,
        num_steps=num_steps,
        chopping_size=chopping_size,
        seed=seed,
        color_fix=color_fix,
    )
    runtime_configs = build_inference_configs(runtime_args)
    sampler.configs.timesteps = runtime_configs.timesteps
    sampler.configs.seed = int(seed)
    sampler.configs.color_fix = runtime_configs.color_fix
    sampler.configs.basesr.chopping.pch_size = runtime_configs.basesr.chopping.pch_size
    sampler.configs.basesr.chopping.extra_bs = runtime_configs.basesr.chopping.extra_bs
    sampler.setup_seed(int(seed))


def run_demo(image_path, task_name, num_steps, chopping_size, seed, color_fix_label, progress=gr.Progress()):
    if image_path is None:
        raise gr.Error("Upload an input image first.")
    if not torch.cuda.is_available():
        raise gr.Error("This demo requires a GPU-backed environment.")

    progress(0.10, desc="Loading sampler")
    sampler = get_task_sampler(task_name)
    checkpoint_path = sampler.configs.model_start.ckpt_path
    color_fix = COLOR_FIX_OPTIONS[color_fix_label]

    unique_dir = OUTPUT_ROOT / task_slug(task_name) / uuid.uuid4().hex[:8]
    unique_dir.mkdir(parents=True, exist_ok=True)
    output_path = unique_dir / f"{Path(image_path).stem}.png"

    with _inference_lock:
        update_sampler_runtime_config(
            sampler=sampler,
            checkpoint_path=checkpoint_path,
            num_steps=num_steps,
            chopping_size=chopping_size,
            seed=seed,
            color_fix=color_fix,
        )
        progress(0.45, desc="Running super-resolution")
        sampler.inference(image_path, out_path=unique_dir, bs=1)

    if not output_path.exists():
        raise gr.Error("Super-resolution failed to produce an output image.")

    progress(1.0, desc="Done")
    image_sr = util_image.imread(output_path, chn="rgb", dtype="uint8")
    details = (
        f"Task: {task_name}\n"
        f"Checkpoint repo: {CHECKPOINT_REPO_ID}\n"
        f"Checkpoint file: {TASKS[task_name]['checkpoint_filename']}\n"
        f"Saved output: {output_path}"
    )
    return image_sr, str(output_path), details


def update_task_ui(task_name):
    task = TASKS[task_name]
    default_label = next(
        label for label, value in COLOR_FIX_OPTIONS.items() if value == task["default_color_fix"]
    )
    return gr.update(value=default_label), task["summary"]


title = "Cloud Microphysics Super-Resolution"

description = """
Single-image super-resolution demo for cloud imagery.
The Space downloads the task-specific noise predictor checkpoint from the public Hugging Face model repo
and uses `stabilityai/sd-turbo` as the diffusion backbone.
"""

article = """
**Available tasks**
- `SEVIRI -> VIIRS`
- `MSG -> MTG`

**Notes**
- This demo is intended for GPU-backed Hugging Face Spaces.
- The public checkpoint repo is `hanangani/cloudsr-checkpoints`.
- Local CLI inference remains available through `inference_sr.py` and `inference_msg_to_mtg_sr.py`.
"""


with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    task_summary = gr.Markdown(TASKS["SEVIRI -> VIIRS"]["summary"])

    with gr.Row():
        with gr.Column():
            task_name = gr.Dropdown(
                choices=list(TASKS.keys()),
                value="SEVIRI -> VIIRS",
                label="Task",
            )
            input_image = gr.Image(type="filepath", label="Input image")
            num_steps = gr.Dropdown(
                choices=[1, 2, 3, 4, 5],
                value=1,
                label="Number of steps",
            )
            chopping_size = gr.Dropdown(
                choices=[128, 256],
                value=128,
                label="Chopping size",
            )
            color_fix = gr.Dropdown(
                choices=list(COLOR_FIX_OPTIONS.keys()),
                value="RGB preserve",
                label="Color handling",
            )
            seed = gr.Number(value=12345, precision=0, label="Random seed")
            run_button = gr.Button("Run super-resolution")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Output image")
            output_file = gr.File(label="Output file")
            output_details = gr.Textbox(label="Run details", lines=4)

    task_name.change(fn=update_task_ui, inputs=task_name, outputs=[color_fix, task_summary])
    run_button.click(
        fn=run_demo,
        inputs=[input_image, task_name, num_steps, chopping_size, seed, color_fix],
        outputs=[output_image, output_file, output_details],
    )

    gr.Markdown(article)

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()
