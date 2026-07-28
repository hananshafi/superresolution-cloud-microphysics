# MSG/MTG data generation

This directory contains code for reading raw MSG/SEVIRI and MTG/FCI products,
creating projection-aligned pairs, and preparing the 4x super-resolution
training layout. Raw satellite products and generated images are intentionally
excluded from Git.

## Expected raw inputs

The scripts expect:

```text
<data-root>/
|-- Comparison/
|   |-- MTG FCI FDHSI NetCDF chunks
|   `-- MSG SEVIRI native .nat file
`-- Comparison_Fine/
    `-- MTG FCI HRFI NetCDF chunks
```

All chunks from one MTG scan must be supplied together. A chunk is not an
independent scene.

Install the satellite-reader dependencies separately from the model environment
if needed:

```bash
pip install -r data_generation/msg_mtg/requirements.txt
```

## Raw-to-pair workflow

1. Discover complete MTG FDHSI and HRFI scans and the MSG native observations.
2. Read MTG with Satpy reader `fci_l1c_nc` and MSG with
   `seviri_l1b_native`.
3. Parse acquisition times, group MTG chunks by scan, and select the nearest
   MSG observation within the configured temporal tolerance.
4. Calibrate visible/NIR channels as reflectance and thermal channels as
   brightness temperature.
5. Obtain the native projection area and longitude-latitude grid for each
   loaded channel.
6. Reproject MSG onto the corresponding MTG grid. The retained dataset uses
   nearest-neighbor resampling.
7. Restrict extraction to the joint finite-data overlap and apply no-data and
   black-pixel rejection thresholds.
8. Inspect native previews, aligned side-by-side views, overlays, and
   difference maps before accepting a generation.
9. Apply identical common-grid windows to both sensors and record source files,
   channels, calibration, timestamps, resampler, crop coordinates, and output
   names in a manifest.

A matched pair consists of two arrays on the same projection and grid with the
same dimensions and pixel window. Equal native array indices do not by
themselves imply equal latitude-longitude locations.

Calibrated physical arrays and coordinate grids should be retained in NetCDF,
GeoTIFF, or NPZ when geolocation or physical values are required. PNG output is
appropriate for model input and visualization, but display scaling must not be
treated as physical calibration. The current generator writes paired uint16
PNGs and a JSONL crop manifest; it does not export per-pixel latitude-longitude
arrays.

## Channel mappings

| MTG FCI | MSG SEVIRI | Calibration |
|---|---|---|
| `vis_08` | `VIS008` | Reflectance |
| `nir_16` | `IR_016` | Reflectance |
| `wv_63` | `WV_062` | Brightness temperature |
| `wv_73` | `WV_073` | Brightness temperature |
| `ir_87` | `IR_087` | Brightness temperature |
| `ir_97` | `IR_097` | Brightness temperature |
| `ir_123` | `IR_120` | Brightness temperature |
| `ir_133` | `IR_134` | Brightness temperature |
| `vis_06` | `VIS006` | Reflectance |
| `ir_38` | `IR_039` | Brightness temperature |
| `ir_105` | `IR_108` | Brightness temperature |

## Commands

Run from the directory containing `Comparison/` and `Comparison_Fine/`.

Normalize downloaded filenames:

```bash
python /path/to/repo/data_generation/msg_mtg/scripts/01_rename_downloads.py \
  Comparison Comparison_Fine
```

Generate aligned 512x512 uint16 pairs into a new output root:

```bash
python /path/to/repo/data_generation/msg_mtg/scripts/02_generate_paired_patches.py \
  --std-dir Comparison \
  --fine-dir Comparison_Fine \
  --output-root MTG_MSG_regenerated \
  --seed 2024
```

The historical generation did not set a seed. A fixed seed and the generated
`generation_manifest.jsonl` are required for reproducible new crops.

Apply the historical ID split:

```bash
python /path/to/repo/data_generation/msg_mtg/scripts/03_split_train_test.py \
  --data-root MTG_MSG_regenerated \
  --start-index 2251 \
  --end-index 2750 \
  --mode move
```

Create the actual 128x128 MSG inputs used for 4x training:

```bash
python /path/to/repo/data_generation/msg_mtg/scripts/04_downsample_msg.py \
  --input-dir MTG_MSG_regenerated/train_LR \
  --output-dir MTG_MSG_regenerated/train_LR_128

python /path/to/repo/data_generation/msg_mtg/scripts/04_downsample_msg.py \
  --input-dir MTG_MSG_regenerated/test_LR \
  --output-dir MTG_MSG_regenerated/test_LR_128
```

Optional per-image uint8 evaluation copies can be generated with
`05_minmax_to_uint8.py`. This conversion changes the physical scale and should
not replace the uint16 source products.

Verify the retained directory layout:

```bash
python /path/to/repo/data_generation/msg_mtg/scripts/06_verify_dataset.py \
  --data-root MTG_MSG_regenerated
```

## Training handoff

Stage 1 consumes real matched pairs and does not apply random augmentation:

```text
train_HR/      MTG targets, 512x512
train_LR/      aligned MSG observations, resized to 128x128 by the paired loader
test_HR/       MTG validation targets, 512x512
test_LR_128/   MSG validation inputs, 128x128
```

Stage 2 consumes only `train_HR/` and synthesizes low-resolution observations
with the configured Real-ESRGAN blur, resize, noise, JPEG, and sinc pipeline.
See the repository root README for training commands and stage guarantees.
