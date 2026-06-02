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

Job `00b6dbce-c5f` completed the first scaled reproduction and passed
`python validate.py --compare`.

- `tiny_nerf_depth_uncertainty`: WarpRF AUSE 0.222 vs deterministic random
  uncertainty AUSE 0.283; WarpRF Spearman correlation with true depth error
  0.627 vs random approximately 0.000.
- `tiny_nerf_active_view_proxy`: WarpRF selected-candidate error 0.412 with
  zero regret vs random-view mean selected-candidate error 0.197 and regret
  0.214.

Job `fecc9c62-9ce` completed the two-split reproduction after valid-depth
masking and passed `python validate.py --compare`.

- `tiny_nerf_depth_uncertainty`: WarpRF AUSE 0.110 vs deterministic random
  uncertainty AUSE 0.131; WarpRF Spearman correlation with true depth error
  0.159 vs random -0.026.
- `tiny_nerf_active_view_proxy`: WarpRF again selected the highest-error
  candidate view with zero selection regret.
- `tiny_nerf_shifted_depth_uncertainty`: the shifted split still failed the
  depth-uncertainty ranking target, with WarpRF AUSE 0.180 vs random 0.137.
- `tiny_nerf_shifted_active_view_proxy`: the shifted split did support the
  active-view proxy claim, with WarpRF zero regret vs random regret 0.233.

Job `744c73e2-b00` reran the two-split reproduction after assigning high
uncertainty to pixels with no valid source reprojection. The job completed and
passed `python validate.py --compare`, but the numeric scores were unchanged,
showing that the shifted depth failure is not caused by all-invalid target
pixels.

## Remaining

- Run the expanded reproduction with a dense eight-source-view split in
  addition to the original and shifted sparse splits.
- If the dense split supports both depth uncertainty and active view selection,
  mark `core_claim_plus` and package the final `scores.json`.

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
- Job `fecc9c62-9ce` passed after valid-depth masking. The shifted active-view
  proxy supports WarpRF, but shifted depth AUSE remained worse than random.
  Inspecting the method implementation showed pixels with no valid source
  reprojection received zero uncertainty; these are now assigned a high observed
  reprojection inconsistency before rerunning the expanded benchmark.
- Job `744c73e2-b00` showed that the all-invalid reprojection fallback did not
  change the scores. The shifted sparse split remains documented as a partial
  negative result, and the next run adds a dense source-view condition to test a
  separate multi-view support axis without hiding the shifted result.

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
