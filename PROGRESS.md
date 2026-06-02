# Progress

## What Works

- Read the WarpRF paper and identified the core method: training-free
  uncertainty from multi-view RGB/depth consistency using backward warping.
- Implemented a scaled radiance-field benchmark with fixed posed views,
  analytic RGB/depth supervision, a tiny NeRF-like renderer, and WarpRF
  uncertainty computed from rendered source/target RGB-D views.
- Implemented a reusable evaluator in `eval/` that reads saved artifacts from
  `outputs/` and writes `scoring/scores.json` without importing `method/`.
- Added paper reference numbers for Table 1, Table 4, Table 5, and Table 7 in
  `scoring/reference.json`.

## Results

Job `00b6dbce-c5f` completed the scaled reproduction and passed
`python validate.py --compare`.

- `tiny_nerf_depth_uncertainty`: WarpRF AUSE 0.222 vs deterministic random
  uncertainty AUSE 0.283; WarpRF Spearman correlation with true depth error
  0.627 vs random approximately 0.000.
- `tiny_nerf_active_view_proxy`: WarpRF selected-candidate error 0.412 with
  zero regret vs random-view mean selected-candidate error 0.197 and regret
  0.214.

## Remaining

- Run the first container job to produce `outputs/warprf/warprf_outputs.npz` and
- Run the shifted camera split added after `core_claim` to test whether the
  effect holds beyond one initial-view configuration.
- If the shifted split passes, mark `core_claim_plus` and package the final
  `scores.json`.

## Issues

- No public WarpRF code release was found in the paper text or quick web search;
  the paper says code will be released. The method was implemented directly from
  the equations and protocol.
- Full ETH3D/ScanNet++/3DGS/SVRaster reproduction is too large for the first
  smoke test, so the first benchmark uses a scaled synthetic radiance-field
  scene while preserving WarpRF's core computation.
- Job `475dc534-7af` trained the tiny radiance field and wrote
  `outputs/warprf/warprf_outputs.npz`, but failed during evaluation because the
  container has NumPy 2, where `np.trapz` has been removed. The evaluator now
  computes the trapezoid integral directly.
- Job `e3c662a6-b0d` passed, but the shifted split showed that AUSE was being
  computed over synthetic background fill pixels. The paper evaluates valid
  annotated depths, so the method artifact now writes a valid-depth mask and
  the evaluator uses it for AUSE, rank correlation, and mean depth error.

## Deviations from Paper

- The paper evaluates on ScanNet++, ETH3D, Mip-NeRF360, NeRF Synthetic, and
  Tanks and Temples with NeRF, 3DGS, and SVRaster. This first runnable
  reproduction uses a tracked analytic synthetic scene and a tiny NeRF-like
  radiance field to fit the compute budget and avoid external dataset/model
  downloads.
- The paper trains large radiance-field backbones for thousands of iterations.
  This reproduction trains a smaller NeRF-like MLP for a reduced number of
  steps, but keeps the WarpRF algorithm unchanged: rendered RGB/depth maps from
  known views are backward-warped to unseen views and consistency differences
  produce uncertainty.
