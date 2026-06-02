# Scientist Constraints

## Backbones

Use the same radiance-field backbone family required by each experiment: NeRF for NeRF experiments, 3DGS or PGSR-style 3DGS for 3DGS experiments, and SVRaster for SVRaster experiments. Do not replace the backbone with a different renderer, larger model family, ensemble, or pretrained scene prior. The benchmark is about uncertainty estimation and view selection, not changing the renderer being evaluated.

## Data

Use only the datasets and scene splits specified by the benchmark protocol: ScanNet++ and ETH3D for AUSE, Mip-NeRF360 and NeRF Synthetic for 3DGS active view selection, NeRF Synthetic and Tanks and Temples for SVRaster active view selection, and the specified NeRF Synthetic or Tanks and Temples scenes for active mapping. Do not add external images, depth maps, meshes, learned priors, or manually curated views beyond the allowed candidate pools.

## View Budget

Keep the initial views, candidate pool, acquisition interval, and final number of selected training views fixed for each experiment. A method may change how it scores or selects among allowed candidate views, but it may not improve scores by using extra views, test views, denser camera trajectories, or additional pose optimization outside the protocol.

## Evaluation

Held-out test views and ground-truth depths or meshes may only be used for evaluation. They must not influence uncertainty scores, view selection, training, hyperparameter tuning for a specific scene, or early stopping. Report all listed metrics for each target so image quality, uncertainty ranking, and geometry constraints remain visible.

## Compliance Checklist

- [ ] The evaluated backbone family matches the required backbone for each experiment.
- [ ] No external scene data, labels, depths, meshes, or pretrained scene priors are used.
- [ ] Final selected view counts match the protocol budget.
- [ ] Candidate views come only from the allowed candidate pool.
- [ ] Test views and ground truth are used only for final metric computation.
- [ ] All metrics listed in `scoring/TARGETS.md` are reported for the corresponding experiment.
