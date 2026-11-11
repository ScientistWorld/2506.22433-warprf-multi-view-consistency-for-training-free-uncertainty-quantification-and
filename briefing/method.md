# Paper's Method

## Key Contribution

WarpRF introduces training-free uncertainty quantification for radiance fields using multi-view consistency. It treats rendered training-view images and depths as reliable references, warps them into an unseen target view using camera geometry and rendered depth, then measures disagreement with the target rendering as uncertainty.

## Approach

For pixel-level depth uncertainty (paper Eq. 8), WarpRF renders depth at a target camera and at the source training cameras. Each target pixel is back-projected with its rendered depth, moved into a source camera and projected; the source rendered depth at that location is lifted to 3D and moved back into the target camera. The uncertainty is the mean absolute difference of camera-z between the two, over the source views whose projection has positive depth; pixels with no such source take the view's maximum.

For image-level uncertainty (paper Eq. 9), WarpRF backward-warps every rendered source image into the candidate view with the candidate's rendered depth, takes the per-pixel minimum over sources of the channel-summed absolute difference to the candidate's rendering (robust to occlusions), and sums over all pixels. Candidates with the largest image-level uncertainty are acquired next for active view selection or active mapping.

Sampling follows the authors' released code: nearest sampling for depth, bilinear for colour, border padding outside the source image.

The method is applied without changing the radiance-field training objective or model internals. The paper demonstrates it on NeRF, 3D Gaussian Splatting, and SVRaster.

## Implementation in This Gym

- `method/warprf.py`: Eq. 8 and Eq. 9 on rendered RGB-D and poses (numpy).
- `method/predict.py`: the two-round prediction interface (`briefing/evaluation.md`).
- `scripts/method.sh` registers it as `outputs/warprf/predict.json`.

## Code Availability

The project page links to the official repository at
`https://github.com/sadrasafa/WarpRF`, which provides the 3DGS experiment
implementation (selection in `active/warprf.py`, AUSE evaluation in
`evaluate_uncertainty.py`).

## Main Claims

- **Primary claim**: Multi-view consistency provides a strong training-free uncertainty estimate for radiance fields, outperforming prior uncertainty methods on AUSE for ScanNet++ and ETH3D.
- **Downstream claim**: Using the uncertainty score for active view selection improves novel-view synthesis under limited view budgets on Mip-NeRF360, NeRF Synthetic, and Tanks and Temples.
- **Mapping claim**: The same uncertainty signal improves active mapping for surface reconstruction with NeRF and 3DGS, including better mesh metrics under fixed acquisition budgets.
- **Generality claim**: The method can be applied to multiple radiance-field families because it only requires rendered images, rendered depths, and camera poses.
