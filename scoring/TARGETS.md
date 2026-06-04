# Evaluation Targets

## Primary: tiny_nerf_depth_uncertainty

Tests whether uncertainty scores rank pixels by true held-out depth error.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/method.sh`, then `bash scripts/evaluate.sh`.

## Primary: tiny_nerf_active_view_proxy

Tests whether fixed-budget active view selection chooses a candidate view that
exposes more current model error than a random candidate.

- **Primary metric**: selected_error_lift (higher is better)
- **Constraint metrics**: view_budget and selection_regret_fraction
- **Ablation/control**: random_view, whose selected_error_lift is fixed at 1.0
  by definition of the candidate-pool mean.
- **How to run**: Run `bash scripts/method.sh`, then `bash scripts/evaluate.sh`.

## Primary: tiny_nerf_dense_depth_uncertainty

Tests the pixel uncertainty target with eight source views instead of four.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/reproduce.sh`.

## Primary: tiny_nerf_dense_active_view_proxy

Tests fixed-budget active view selection with eight source views instead of
four.

- **Primary metric**: selected_error_lift (higher is better)
- **Constraint metrics**: view_budget and selection_regret_fraction
- **Ablation/control**: random_view, whose selected_error_lift is fixed at 1.0
  by definition of the candidate-pool mean.
- **How to run**: Run `bash scripts/reproduce.sh`.

## Diagnostic: tiny_nerf_shifted_depth_uncertainty

Reports the same pixel uncertainty target on a shifted sparse camera split.
The current reproduction keeps this partial negative result visible, but it is
not used as a primary milestone target because the reproduced method is worse
than the random uncertainty control on this split.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/reproduce.sh`.

## Diagnostic: tiny_nerf_shifted_active_view_proxy

Reports the same active-view target on the shifted sparse camera split.

- **Primary metric**: selected_error_lift (higher is better)
- **Constraint metrics**: view_budget and selection_regret_fraction
- **Ablation/control**: random_view, whose selected_error_lift is fixed at 1.0
  by definition of the candidate-pool mean.
- **How to run**: Run `bash scripts/reproduce.sh`.
