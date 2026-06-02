# Evaluation Targets

## Primary: uncertainty_ause

Tests whether predicted uncertainty ranks pixels by true depth error on held-out views.

- **Metric**: AUSE (lower is better)
- **How to run**: Train the fixed radiance-field backbone on the prescribed training images, render held-out depth and uncertainty maps, compute depth MAE sparsification curves, and average AUSE over the specified ScanNet++ and ETH3D scenes.

## Primary: active_view_nvs

Tests whether uncertainty-driven view acquisition improves final novel-view synthesis quality under a fixed image budget.

- **Metrics**: PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better)
- **How to run**: Start from the fixed initial camera set, iteratively select views from the allowed candidate pool until the target view budget is reached, train the fixed radiance-field backbone with the prescribed schedule, and evaluate held-out renderings.

## Constraint: fixed_view_budget

Tests whether gains come from better view selection rather than using more training images.

- **Metric**: selected_views (must equal the protocol budget)
- **How to run**: Record the number of unique training views used at final evaluation for each active selection experiment.

## Constraint: method_general_backbone

Tests whether the uncertainty or selection strategy can operate without changing the fixed radiance-field backbone family used in the protocol.

- **Metric**: backbone_changed (must be false)
- **How to run**: Inspect the submitted method configuration and verify that the evaluated backbone is the required NeRF, 3DGS, or SVRaster backbone for the experiment.

## Secondary: active_mapping

Tests whether active acquisition improves surface reconstruction while also reporting rendering quality.

- **Metrics**: Accuracy (lower is better), Completion (lower is better), Completion Ratio (higher is better), F1 score (higher is better), PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better)
- **How to run**: Follow the fixed active mapping protocol, reconstruct meshes after the final view budget, compare against ground-truth geometry, and evaluate held-out renderings.
