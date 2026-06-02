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

No compute-container results yet. A local smoke run was attempted only to catch
imports, but the login Python lacks NumPy; dependencies are declared in
`environment/setup.sh` for the submitted container job.

## Remaining

- Run the first container job to produce `outputs/warprf/warprf_outputs.npz` and
  `scoring/scores.json`.
- Inspect whether WarpRF improves AUSE and active-view proxy metrics over the
  deterministic random baselines.
- If the core claim is supported, expand to additional synthetic scene settings
  or higher training resolution.

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
