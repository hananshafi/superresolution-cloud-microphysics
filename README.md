# Recovering Cloud Microstructures with Cascaded Diffusion Inversion

<div align="center">

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026%20Workshop-blue)](https://openreview.net/forum?id=Xz7in1KpXr&invitationId=ICLR.cc/2026/Workshop/ML4RS_Main_Track/Submission29/-/Revision&referrer=%5BTasks%5D(%2Ftasks))
[![Webpage](https://img.shields.io/badge/Webpage-Project%20Page-0ea5a4)](https://hananshafi.github.io/superresolution-cloud-microphysics/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/hanangani/cloudsr-checkpoints)
[![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](#gradio-demo)

</div>

#### Authors: [Hanan Gani](https://hananshafi.github.io), Guy Pulik, Daniel Rosenfeld, [Duncan Watson-Parris](https://duncanwp.github.io), [Salman Khan](https://salman-h-khan.github.io/)

<div align="center" style="margin:18px 0 10px 0;">
  <img src="docs/assets/logos/mbzuai.png" alt="MBZUAI" height="52" style="margin:0 18px 10px 18px; vertical-align:middle;" />
  <img src="docs/assets/logos/ucsd-full.png" alt="University of California, San Diego" height="60" style="margin:0 18px 10px 18px; vertical-align:middle;" />
  <img src="docs/assets/logos/uaerep.png" alt="UAEREP" height="52" style="margin:0 18px 10px 18px; vertical-align:middle;" />
</div>

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>

This repository contains data-generation, training, inference, and evaluation
code for cloud microphysics super-resolution.

## Runtime Requirements

- Python `3.10`
- PyTorch `2.4.0`
- `xformers==0.0.27.post2`
- `huggingface_hub>=0.19.3,<1.0`
- Additional dependencies from [`environment.yaml`](environment.yaml) and [`requirements.txt`](requirements.txt)

This repository uses the local patched `diffusers` copy under [`src/diffusers`](src/diffusers).

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

## Data Generation

The code-only MSG/MTG preprocessing workflow is under
[`data_generation/msg_mtg`](data_generation/msg_mtg). It includes raw-product
discovery, Satpy calibration, projection alignment, paired patch extraction,
train/test splitting, 4x input resizing, and dataset verification. Raw
satellite files and generated images are not stored in Git.

Install its satellite-reader dependencies separately when the training
environment does not already provide them:

```bash
pip install -r data_generation/msg_mtg/requirements.txt
```

## Two-Stage Training

Training is controlled by explicit stage-aware configs. The loader validates
the stage, dataset type, and augmentation setting before initializing GPUs, so
the old manual comment/uncomment workflow is no longer required.

| Stage | Supervision | Dataset type | Augmentation |
|---|---|---|---|
| Stage 1 | Real matched LR/HR sensor pairs | `realesrgan_paired` | Disabled |
| Stage 2 | HR-only imagery initialized from Stage 1 | `realesrgan` | Real-ESRGAN degradation enabled |

Stage 1 preserves each matched pair. Synthetic blur, random resize, noise, JPEG,
sinc filtering, random crop, flip, and rotation are disabled. The paired loader
only performs the required deterministic 4x resize of the observed LR image.

Train Stage 1 on SEVIRI/VIIRS:

```bash
python main.py \
  --cfg_path configs/sd_turbo-sr-ldis-pairwise.yaml \
  --save_dir runs/seviri_viirs_stage1
```

Train Stage 1 on MSG/MTG:

```bash
python main.py \
  --cfg_path configs/sd_turbo-sr-ldis-pairwise-msg-mtg.yaml \
  --save_dir runs/msg_mtg_stage1
```

Stage 2 starts from a selected Stage 1 checkpoint and keeps the complete
Real-ESRGAN synthetic degradation pipeline active. Pass `--init_ckpt` to avoid
editing a machine-specific checkpoint path in the YAML.

Train Stage 2 on VIIRS HR imagery:

```bash
python main.py \
  --cfg_path configs/sd-turbo-sr-ldis.yaml \
  --init_ckpt /path/to/seviri_viirs_stage1/ckpts/model_50000.pth \
  --save_dir runs/seviri_viirs_stage2
```

Train Stage 2 on MTG HR imagery:

```bash
python main.py \
  --cfg_path configs/sd-turbo-sr-ldis-msg-mtg.yaml \
  --init_ckpt /path/to/msg_mtg_stage1/ckpts/model_50000.pth \
  --save_dir runs/msg_mtg_stage2
```

When more than one GPU is visible, launch with `torchrun` and the desired
process count. To run a single process, expose one GPU with
`CUDA_VISIBLE_DEVICES`.

The Stage 2 connection is parameter initialization from Stage 1, not an
image-output cascade. Stage 2 consumes HR images and creates its LR inputs
online using blur, resize, Gaussian/Poisson noise, JPEG, and sinc-filter
augmentation.

## External Assets Needed

Training, inference, and evaluation require external model assets that are not checked into this repo:

- A local SD-Turbo model directory
- A noise predictor checkpoint passed with `--started_ckpt_path`
  Hosted checkpoints:
  - SEVIRI to VIIRS: `hanangani/cloudsr-checkpoints/cloudsr_seviri_to_viirs_model_50000.pth`
  - MSG to MTG: `hanangani/cloudsr-checkpoints/cloudsr_msg_to_mtg_model_50000.pth`

Download the checkpoints locally before running inference:

```bash
hf download hanangani/cloudsr-checkpoints cloudsr_seviri_to_viirs_model_50000.pth --repo-type model --local-dir ./checkpoints
hf download hanangani/cloudsr-checkpoints cloudsr_msg_to_mtg_model_50000.pth --repo-type model --local-dir ./checkpoints
```

## Gradio Demo

The repository includes a local Gradio demo at [`app.py`](app.py).

What it does:

- Runs a single-image demo for both `SEVIRI -> VIIRS` and `MSG -> MTG`
- Uses `stabilityai/sd-turbo` as the diffusion backbone
- Loads the task-specific checkpoint through the same inference pipeline as the CLI scripts

Run it from the repository root:

```bash
python app.py
```

Notes:

- The demo requires a GPU-backed environment.
- By default it uses the public checkpoints from `hanangani/cloudsr-checkpoints`.
- Optional environment variables:
  - `CLOUDSR_CHECKPOINT_REPO_ID`
  - `CLOUDSR_SEVIRI_TO_VIIRS_CKPT`
  - `CLOUDSR_MSG_TO_MTG_CKPT`

## Inference

Run inference from the repository root:

```bash
cd superresolution-cloud-microphysics

python inference_sr.py \
  -i /path/to/seviri_input \
  -o /path/to/output_dir \
  --num_steps 1 \
  --sd_path /path/to/sd-turbo \
  --started_ckpt_path ./checkpoints/cloudsr_seviri_to_viirs_model_50000.pth
```

Useful options:

- `--sd_path` to point to your local SD-Turbo directory
- `--chopping_size 256` for larger images
- `--chopping_bs 1` if GPU memory is tight
- `--color_fix rgb` for strict palette preservation on pseudo-color cloud imagery
- `--color_fix wavelet` for the wavelet-based color correction path
- `--start_step` if you want to use `--num_steps` greater than `5`

## MSG To MTG Inference

Use [`inference_msg_to_mtg_sr.py`](inference_msg_to_mtg_sr.py) for the MSG low-resolution to MTG high-resolution workflow.

```bash
cd superresolution-cloud-microphysics

python inference_msg_to_mtg_sr.py \
  -i /path/to/msg_input \
  -o /path/to/output_dir \
  --num_steps 1 \
  --sd_path /path/to/sd-turbo \
  --started_ckpt_path ./checkpoints/cloudsr_msg_to_mtg_model_50000.pth
```

Notes:

- The MSG to MTG wrapper defaults to `--color_fix rgb` unless you explicitly pass another color-fix option.

## Citation

```bibtex
@inproceedings{gani2026recovering,
  title     = {Recovering Cloud Microstructures with Cascaded Diffusion Inversion},
  author    = {Gani, Hanan and Pulik, Guy and Rosenfeld, Daniel and Watson-Parris, Duncan and Khan, Salman},
  booktitle = {ICLR 2026 Workshop on Machine Learning for Remote Sensing (ML4RS)},
  year      = {2026}
}
```

## Acknowledgement

Our codebase is built on top of [InvSR](https://github.com/zsyOAOA/InvSR). We thank the InvSR authors for releasing their codebase.
