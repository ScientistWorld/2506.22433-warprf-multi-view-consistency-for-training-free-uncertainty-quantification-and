# Evaluation

## Metrics

- **AUSE**: Area Under the Sparsification Error curve for depth uncertainty. Pixels are progressively removed according to predicted uncertainty, mean absolute depth error is recomputed on the remaining pixels, and the area is measured against an oracle curve that removes pixels by true depth error. Lower is better.
- **PSNR**: Peak signal-to-noise ratio for held-out novel-view synthesis. Higher is better.
- **SSIM**: Structural similarity for held-out novel-view synthesis. Higher is better.
- **LPIPS**: Learned perceptual image patch similarity for held-out novel-view synthesis. Lower is better.
- **Surface reconstruction metrics**: Accuracy and Completion are lower-is-better mesh distances; Completion Ratio and F1 score are higher-is-better reconstruction quality metrics.

## Baselines and Targets

Uncertainty calibration should be compared against paper-reported AUSE values on ScanNet++ and ETH3D for both NeRF and 3DGS backbones. Active view selection should be compared against fixed-budget selection baselines such as random, farthest-view, ActiveNeRF, Manifold, and FisherRF where available. Mapping should compare against prior active mapping or uncertainty-based view selection baselines under the same view budget.

The primary target is to match or improve the paper-reported method results in `scoring/reference.json`, while preserving the fixed-view-budget and fixed-backbone constraints described in `scoring/CONSTRAINTS.md`.

## Evaluation Protocol

Evaluate uncertainty on held-out test views with ground-truth depth. For ScanNet++, use the six scenes named in the paper supplement and train/evaluate on the first 40 images using alternating train/test images. For ETH3D, use the high-resolution scenes with alternating train/test images.

For active view selection, begin from the same small initial camera set as the reference protocol, select one view at each acquisition step from the candidate pool, retrain or continue training according to the fixed backbone schedule, and evaluate final held-out renderings using PSNR, SSIM, and LPIPS. Report averages over the same scene sets and repeated runs when the protocol uses stochastic selection.

For active mapping, keep the initial views, acquisition budget, candidate-pose generation, training iterations, and reconstruction evaluation fixed. Report both rendering quality and surface reconstruction quality so that a method cannot improve images while degrading geometry, or vice versa.
