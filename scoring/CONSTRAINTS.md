# Scientist Constraints

## Backbones

For the runnable scaled benchmark targets in `scoring/TARGETS.md`, use the
provided tiny NeRF-style radiance-field backbone and the same training budget
unless an experiment explicitly changes one of those quantities. Do not replace
the renderer with a larger model family, an ensemble, a pretrained scene prior,
or a direct predictor of the evaluation labels. The benchmark is about
uncertainty estimation and view selection, not changing the renderer being
evaluated.

## Data

Use only the tracked analytic posed-view scene from `data/` for the runnable
scaled benchmark targets. Do not add external images, depth maps, meshes,
learned priors, or manually curated views beyond the fixed initial, candidate,
and held-out view sets. Paper-scale reference entries in
`scoring/reference.json` document the original ScanNet++, ETH3D, NeRF
Synthetic, Mip-NeRF360, Tanks and Temples, 3DGS, and SVRaster results; those are
not claimed as reproduced by this scaled gym.

## View Budget

Keep the initial views, candidate pool, and final selected-view count fixed for
each experiment. A method may change how it scores or selects among allowed
candidate views, but it may not improve scores by using extra views, held-out
views, denser camera trajectories, or additional pose optimization outside the
protocol.

## Evaluation

Held-out test views and ground-truth depths or meshes may only be used for evaluation. They must not influence uncertainty scores, view selection, training, hyperparameter tuning for a specific scene, or early stopping. Report all listed metrics for each target so image quality, uncertainty ranking, and geometry constraints remain visible.

## Compliance Checklist

- [ ] The evaluated backbone family matches the required tiny NeRF-style
      backbone for each runnable scaled experiment.
- [ ] No external scene data, labels, depths, meshes, or pretrained scene priors are used.
- [ ] Final selected view counts match the protocol budget.
- [ ] Candidate views come only from the allowed candidate pool.
- [ ] Test views and ground truth are used only for final metric computation.
- [ ] All metrics listed in `scoring/TARGETS.md` are reported for the corresponding experiment.
