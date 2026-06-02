# Paper's Method

## Key Contribution

WarpRF introduces training-free uncertainty quantification for radiance fields using multi-view consistency. It treats rendered training-view images and depths as reliable references, warps them into an unseen target view using camera geometry and rendered depth, then measures disagreement with the target rendering as uncertainty.

## Approach

For pixel-level depth uncertainty, WarpRF renders depth at a target camera and at source training cameras. It projects each source depth map into the target camera and averages the absolute difference between the target depth and valid projected depths per pixel. Pixels whose rendered geometry is inconsistent across views receive higher uncertainty.

For image-level uncertainty, WarpRF renders source images and warps them into a candidate target view. It sums per-pixel photometric disagreement between the target rendering and the best matching warped source rendering, using a minimum across sources to reduce occlusion sensitivity. Candidate views with larger image-level uncertainty are selected as next views for active view selection or active mapping.

The method is applied without changing the radiance-field training objective or model internals. The paper demonstrates it on NeRF, 3D Gaussian Splatting, and SVRaster.

## Main Claims

- **Primary claim**: Multi-view consistency provides a strong training-free uncertainty estimate for radiance fields, outperforming prior uncertainty methods on AUSE for ScanNet++ and ETH3D.
- **Downstream claim**: Using the uncertainty score for active view selection improves novel-view synthesis under limited view budgets on Mip-NeRF360, NeRF Synthetic, and Tanks and Temples.
- **Mapping claim**: The same uncertainty signal improves active mapping for surface reconstruction with NeRF and 3DGS, including better mesh metrics under fixed acquisition budgets.
- **Generality claim**: The method can be applied to multiple radiance-field families because it only requires rendered images, rendered depths, and camera poses.
