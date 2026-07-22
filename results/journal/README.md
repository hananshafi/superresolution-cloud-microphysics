# Journal experiment results

These files contain the measurements reported in the Remote Sensing journal
draft. Main-comparison and selected-checkpoint metrics use strict sample-ID
matching across the complete validation sets: 1,000 SEVIRI-to-VIIRS pairs and
500 MSG-to-MTG pairs. The checkpoint-selection subset is documented below.

- `*_metrics.{json,csv}` contains bicubic, learned-baseline, and training-stage
  ablation metrics. Stage 1 is included only as an ablation reference; all main
  method results use the Stage 2 checkpoint directly.
- `*_sampling_metrics.{json,csv}` compares one-step and five-step predictions
  generated from the same Stage 2 checkpoint and code path.
- `*_stage1_checkpoint_sweep.{json,csv}` records candidate metrics for the
  Stage 1 checkpoint sweep. MSG-to-MTG uses all 500 pairs; SEVIRI-to-VIIRS uses
  the deterministic 200-ID subset `0, 5, 10, ..., 995`.
- `*_stage1_checkpoint_selection.json` records the ordered checkpoint ranks.
  Selection uses mean ordinal rank over PSNR, SSIM, LPIPS, and absolute
  gradient-ratio error from 1.0; RMSE is reported but omitted to avoid
  double-weighting pixel distortion with PSNR.
- `*_stage1_selected_metrics.{json,csv}` contains the selected checkpoint's
  final metrics on the complete validation set used in the manuscript.
- `msg_mtg_stage2_checkpoint_sweep.{json,csv}` records the full 5k--105k
  Stage 2 sweep over all 500 pairs, and
  `msg_mtg_stage2_checkpoint_selection.json` records the balanced-rank result.
  The selected 35k checkpoint supplies every MSG-to-MTG metric reported for
  the proposed method; metrics are never combined across checkpoints.
- `msg_mtg_stage2_35k_runtime_k{1,5}.json` contains uncontended A100 timing
  measurements for the selected checkpoint over 50 preloaded crops.

Metrics are per-image PSNR, normalized RMSE, SSIM, luminance Sobel-gradient
ratio, and raw AlexNet LPIPS. Reported standard deviations are population
standard deviations. The manuscript reports raw LPIPS; the normalized
perceptual ratio retained in these result files is provided only for comparison
with the earlier workshop evaluation.

MSG-to-MTG Stage 1 predictions use the original one-step inference path without
color correction. SEVIRI-to-VIIRS Stage 1 predictions use the same one-step
path with YCbCr chroma preservation applied uniformly to every checkpoint.
