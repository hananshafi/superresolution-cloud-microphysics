# Journal experiment results

These files contain the measurements reported in the Remote Sensing journal
draft. The evaluation uses strict sample-ID matching across the complete
validation sets: 1,000 SEVIRI-to-VIIRS pairs and 500 MSG-to-MTG pairs.

- `*_metrics.{json,csv}` contains bicubic, learned-baseline, and stage-ablation
  metrics.
- `*_sampling_metrics.{json,csv}` compares one-step and five-step predictions
  generated from the same checkpoint and code path.
- `*_cascade_runtime.json` contains single-pass and two-pass cascade timing on
  an NVIDIA A100 40 GB GPU.

Metrics are per-image PSNR, normalized RMSE, SSIM, luminance Sobel-gradient
ratio, and raw AlexNet LPIPS. Reported standard deviations are population
standard deviations. The manuscript reports raw LPIPS; the normalized
perceptual ratio retained in these result files is provided only for comparison
with the earlier workshop evaluation.
