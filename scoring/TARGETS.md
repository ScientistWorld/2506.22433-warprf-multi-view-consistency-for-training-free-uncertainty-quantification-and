# Evaluation Targets

## Primary: tiny_nerf_depth_uncertainty

Tests whether uncertainty scores rank pixels by true held-out depth error.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/method.sh`, then `bash scripts/evaluate.sh`.

## Primary: tiny_nerf_active_view_proxy

Tests whether fixed-budget active view selection chooses a candidate view that
exposes more current model error than a random candidate.

- **Metric**: selected_candidate_error (higher is better)
- **Constraint metric**: view_budget
- **How to run**: Run `bash scripts/method.sh`, then `bash scripts/evaluate.sh`.

## Primary: tiny_nerf_dense_depth_uncertainty

Tests the pixel uncertainty target with eight source views instead of four.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/reproduce.sh`.

## Primary: tiny_nerf_dense_active_view_proxy

Tests fixed-budget active view selection with eight source views instead of
four.

- **Metric**: selected_candidate_error (higher is better)
- **Constraint metric**: view_budget
- **How to run**: Run `bash scripts/reproduce.sh`.

## Primary: tiny_nerf_shifted_depth_uncertainty

Tests the same pixel uncertainty target on a shifted camera split.

- **Metric**: AUSE (lower is better)
- **Constraint metrics**: source_view_count and used_ground_truth_for_uncertainty
- **How to run**: Run `bash scripts/reproduce.sh`.

## Primary: tiny_nerf_shifted_active_view_proxy

Tests the same active-view target on a shifted camera split.

- **Metric**: selected_candidate_error (higher is better)
- **Constraint metric**: view_budget
- **How to run**: Run `bash scripts/reproduce.sh`.
