# Recovering Cloud Microstructures with Cascaded Diffusion Inversion

<div align="center">

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026%20Workshop-blue)](https://openreview.net/forum?id=Xz7in1KpXr&invitationId=ICLR.cc/2026/Workshop/ML4RS_Main_Track/Submission29/-/Revision&referrer=%5BTasks%5D(%2Ftasks))
[![Webpage](https://img.shields.io/badge/Webpage-Project%20Page-0ea5a4)](https://hananshafi.github.io/superresolution-cloud-microphysics/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/hanangani/cloudsr-checkpoints)
[![Demo](https://img.shields.io/badge/Demo-Gradio-orange)](#gradio-demo)

</div>

#### Authors: [Hanan Gani](https://hananshafi.github.io), Guy Pulik, Daniel Rosenfeld, [Duncan Watson-Parris](https://duncanwp.github.io), [Salman Khan](https://salman-h-khan.github.io/)

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>

This repository contains the inference code used for cloud microphysics super-resolution.

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

## External Assets Needed

Inference and evaluation require external model assets that are not checked into this repo:

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
