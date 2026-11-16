# Overview

- **Paper ID:** 2506.22433
- **Title:** WarpRF: Multi-View Consistency for Training-Free Uncertainty Quantification and Applications in Radiance Fields
- **Domain:** Computer Vision / Neural Rendering / Active View Selection
- **TL;DR:** WarpRF estimates radiance-field uncertainty from cross-view rendering consistency and uses it to select informative new views without training an uncertainty model.

## Short Summary

The paper studies uncertainty estimation for radiance fields and its use in active view selection and active mapping. It proposes a training-free method that compares target renderings with source-view renderings warped into the target camera. The reported experiments show improved AUSE over prior uncertainty methods and improved downstream view selection for 3DGS and SVRaster, plus active mapping gains for NeRF and 3DGS.

## Key Results

- On Table 1 AUSE, 3DGS + WarpRF reports 0.337 on ScanNet++ and 0.227 on ETH3D, better than 3DGS + FisherRF at 0.355 and 0.295.
- On active view selection for Mip-NeRF360, 3DGS + WarpRF reports 20.715 PSNR, 0.6174 SSIM, and 0.3487 LPIPS.
- On SVRaster active view selection, WarpRF improves over random and farthest-view baselines on NeRF Synthetic with 20 views and Tanks and Temples.

## This Gym

A scaled, controlled version of the paper's two evaluations. The first is
per-pixel depth uncertainty (AUSE) for frozen tiny-NeRF fields trained from 3
or 8 views. The second is sequential acquisition of 2 of 16 candidate views,
scored by the final field's PSNR/SSIM/depth error on 32 held-out views drawn
from the same pose distribution as the candidates. Both run on procedurally
generated scenes, with 56 hidden test instances. See `briefing/evaluation.md`.
