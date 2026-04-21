# Recovering Cloud Microstructures with Cascaded Diffusion Inversion

<div align="center">

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026%20Workshop-blue)](https://openreview.net/forum?id=Xz7in1KpXr&invitationId=ICLR.cc/2026/Workshop/ML4RS_Main_Track/Submission29/-/Revision&referrer=%5BTasks%5D(%2Ftasks))
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/hanangani/cloudsr-checkpoints) [![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](#hugging-face-demo)

</div>

#### Authors: [Hanan Gani](https://hananshafi.github.io), Guy Pulik, Daniel Rosenfeld, [Duncan Watson-Parris][https://duncanwp.github.io], [Salman Khan](https://salman-h-khan.github.io/)

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>

This repository contains the single-stage inference code used for cloud microphysics super-resolution. Model weights are intentionally not stored in this repository.

## Runtime Requirements

- Python `3.10`
- PyTorch `2.4.0`
- `xformers==0.0.27.post2`
- `huggingface_hub>=0.19.3,<1.0`
- Additional dependencies from [`environment.yaml`](environment.yaml) and [`requirements.txt`](requirements.txt)

## Setup

You can create the local conda environment in either of the following ways.

Option 1: create the environment from [`environment.yaml`](environment.yaml)

```bash
conda env create -f environment.yaml
conda activate cloudsr
```

Option 2: create the environment manually and install dependencies from [`requirements.txt`](requirements.txt)

```bash
conda create -n cloudsr python=3.10
conda activate cloudsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"
pip install -r requirements.txt
```

If you already have a working local environment such as `invsr`, you can continue using it as long as the installed package versions remain compatible with the requirements above.

## External Assets Needed

Inference and evaluation require external model assets that are not checked into this repo:

- A pre-downloaded SD-Turbo model directory
- A noise predictor checkpoint passed with `--started_ckpt_path`
  Hosted checkpoints:
  - SEVIRI to VIIRS: `hanangani/cloudsr-checkpoints/cloudsr_seviri_to_viirs_model_50000.pth`
  - MSG to MTG: `hanangani/cloudsr-checkpoints/cloudsr_msg_to_mtg_model_50000.pth`

Download the checkpoints locally before running inference:

```bash
hf download hanangani/cloudsr-checkpoints cloudsr_seviri_to_viirs_model_50000.pth --repo-type model --local-dir ./checkpoints
hf download hanangani/cloudsr-checkpoints cloudsr_msg_to_mtg_model_50000.pth --repo-type model --local-dir ./checkpoints
```

On this machine, [`inference_sr.py`](inference_sr.py) will automatically use the first valid local SD-Turbo directory it finds. Some of those directories still include `InvSR` in the filesystem path because that is where the checkpoints are currently stored.

You can always override the detected location with `--sd_path`.

An additional LPIPS checkpoint is only needed for training, not inference:

- `vgg16_sdturbo_lpips.pth` in `weights/`

## Hugging Face Demo

The repository now includes a Hugging Face Space-ready Gradio app at [`app.py`](app.py).

What it does:

- Runs a single-image demo for both `SEVIRI -> VIIRS` and `MSG -> MTG`
- Downloads the task-specific checkpoint from `hanangani/cloudsr-checkpoints`
- Uses `stabilityai/sd-turbo` as the diffusion backbone

Deployment notes:

- Use a GPU-backed Hugging Face Space
- The default entry point is `app.py`
- Install dependencies from [`requirements.txt`](requirements.txt)
- Keep `huggingface_hub` below `1.0` for compatibility with the current `transformers` and `diffusers` stack

Optional Space environment variables:

- `CLOUDSR_CHECKPOINT_REPO_ID`
- `CLOUDSR_SEVIRI_TO_VIIRS_CKPT`
- `CLOUDSR_MSG_TO_MTG_CKPT`

## Inference

Run inference from the repository root:

```bash
cd /share/data/drive_3/hanan/superresolution-cloud-microphysics

CUDA_VISIBLE_DEVICES=6 python inference_sr.py \
  -i /share/data/drive_3/hanan/climate_data/train_data/test/seviri_128 \
  -o /share/data/drive_3/hanan/iclr2026_workshop/rgb_results_50k_ckpt \
  --num_steps 1 \
  --started_ckpt_path ./checkpoints/cloudsr_seviri_to_viirs_model_50000.pth
```

Useful options:

- `--sd_path` to point to a different SD-Turbo directory
- `--chopping_size 256` for larger images
- `--chopping_bs 1` if GPU memory is tight
- `--color_fix rgb` for strict palette preservation on pseudo-color cloud imagery
- `--color_fix wavelet` for the wavelet-based color correction path
- `--start_step` if you want to use `--num_steps` greater than `5`

## MSG To MTG Inference

Use [`inference_msg_to_mtg_sr.py`](inference_msg_to_mtg_sr.py) for the MSG low-resolution to MTG high-resolution workflow.

```bash
cd /share/data/drive_3/hanan/superresolution-cloud-microphysics

CUDA_VISIBLE_DEVICES=6 python inference_msg_to_mtg_sr.py \
  -i /share/data/drive_3/hanan/climate_data/MTG_MSG_train_data/test_LR_128 \
  -o /share/data/drive_3/hanan/iclr2026_workshop/msg_mtg/results_single_stage \
  --num_steps 1 \
  --started_ckpt_path ./checkpoints/cloudsr_msg_to_mtg_model_50000.pth
```

Notes:

- If `--started_ckpt_path` is omitted on this machine, the script will automatically use `/share/data/drive_3/hanan/InvSR/msg_mtg_init_paired/2026-01-16-07-23/ckpts/model_50000.pth` when it exists.
- The MSG to MTG wrapper defaults to `--color_fix rgb` unless you explicitly pass another color-fix option.

## Single-Sample Example

The following example was run on:

- Input: `/share/data/drive_3/hanan/climate_data/train_data/test/seviri_128/seviri_454.png`
- Checkpoint: `./checkpoints/cloudsr_seviri_to_viirs_model_50000.pth`
- SD-Turbo directory: `/share/data/drive_3/hanan/InvSR/weights/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2`

Run:

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n cloudsr python inference_sr.py \
  -i /share/data/drive_3/hanan/climate_data/train_data/test/seviri_128/seviri_454.png \
  -o /share/data/drive_3/hanan/iclr2026_workshop/seviri_viirs/random_sample_seviri_454/single_stage_rgb \
  --num_steps 1 \
  --color_fix rgb \
  --sd_path /share/data/drive_3/hanan/InvSR/weights/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2 \
  --started_ckpt_path ./checkpoints/cloudsr_seviri_to_viirs_model_50000.pth
```

Expected output:

- `random_sample_seviri_454/single_stage_rgb/seviri_454.png`

If you want to keep earlier no-color-fix outputs for comparison, save them into separate directories. The recommended setting for this repository is `--color_fix rgb`.

## Codebase Notes

- The repo uses the local patched `diffusers` copy under [`src/diffusers`](src/diffusers).
- The inference path was adjusted so it can run without copying SD-Turbo weights into this repository.
- Historical checkpoint and weight directories on this machine still include `InvSR` in their path names; that is just the storage location, not the project name used in this repo.
- Task-specific MSG to MTG training configs are available as [`configs/sd-turbo-sr-ldis-msg-mtg.yaml`](configs/sd-turbo-sr-ldis-msg-mtg.yaml) and [`configs/sd_turbo-sr-ldis-pairwise-msg-mtg.yaml`](configs/sd_turbo-sr-ldis-pairwise-msg-mtg.yaml).
