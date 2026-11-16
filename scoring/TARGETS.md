# Evaluation Targets

All targets are scored by `bash scripts/evaluate_train.sh` (public split) and
`bash scripts/evaluate_test.sh` (hidden split) for every method registered under
`outputs/<name>/predict.json`. Reference values are the paper's method (WarpRF)
on the hidden split, stored in `scoring/reference.json`.

## Primary: depth_uncertainty_sparse (weight 0.25)

Per-pixel depth uncertainty of a field trained on 3 views, on 6 held-out views per scene.

- **Primary metric**: `ause` (lower is better; reference 0.348, random 0.588)
- **Secondary metric**: `spearman_depth_error` (higher is better; reference 0.346, random 0.002)
- **Control**: `random_uncertainty`, computed by the evaluator.

## Primary: depth_uncertainty_dense (weight 0.25)

Same measurement for a field trained on 8 views.

- **Primary metric**: `ause` (lower is better; reference 0.358, random 0.707)
- **Secondary metric**: `spearman_depth_error` (higher is better; reference 0.230, random -0.002)
- **Control**: `random_uncertainty`, computed by the evaluator.

## Primary: active_view_selection (weight 0.50)

Sequential acquisition of 2 of 16 candidate views from a 3-view start; final
field scored on 32 held-out views drawn from the candidates' pose distribution.

- **Primary metric**: `final_psnr` (higher is better; reference 23.27 dB, random 22.44 dB, farthest 22.93 dB)
- **Benefit metric**: `final_ssim` (higher is better; reference 0.817, random 0.782, farthest 0.801)
- **Constraint metric**: `final_depth_mae` (lower is better; reference 0.335, random 0.375, farthest 0.362): image gains must not come from views that degrade the reconstructed geometry.
- **Controls**: `random_view` (exact expectation of random acquisition) and `farthest_view` (farthest-camera acquisition), computed by the evaluator.

## Constraints enforced by construction

- View budget, initial views, candidate pool, backbone and training schedule are
  fixed by the evaluator: the method only returns scores, and the evaluator
  performs the acquisitions and looks up the resulting field.
- Ground-truth depth, held-out views and unacquired candidate photographs never
  reach the method on the hidden split; the method runs isolated, with the
  hidden-test folder and the git object store hidden from it.
- Any failed, invalid or non-finite run scores the worst achievable value in
  every metric of every experiment.
