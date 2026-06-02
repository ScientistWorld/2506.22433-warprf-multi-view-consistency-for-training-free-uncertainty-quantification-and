# Experiment Overview

## tiny_nerf_depth_uncertainty

Tests whether predicted pixel uncertainty ranks held-out depth errors in a
posed-view radiance-field scene. The purpose is to verify the paper's central
uncertainty-quantification claim under a small reproducible setting with fixed
training views, fixed held-out views, and ground-truth depth available only to
the evaluator.

## tiny_nerf_active_view_proxy

Tests whether the uncertainty signal can choose a useful next camera from a
candidate pool under a fixed view budget. The selected view is scored by how
much current rendering error it exposes compared with a deterministic random
candidate baseline.

## tiny_nerf_dense_depth_uncertainty

Repeats the held-out pixel uncertainty test with eight source views instead of
four. This checks whether the uncertainty-ranking behavior holds when the same
scene has denser multi-view support.

## tiny_nerf_dense_active_view_proxy

Repeats the fixed-budget candidate-selection test with eight source views. This
checks whether the candidate ranking remains useful when the initial
reconstruction has more observed cameras.

## tiny_nerf_shifted_depth_uncertainty

Repeats the held-out pixel uncertainty test on a shifted camera split. This
is reported as a diagnostic stress test: the current sparse shifted result is a
partial negative result, so it should not be treated as evidence for the
milestone claim unless a future method improves it under the same constraints.

## tiny_nerf_shifted_active_view_proxy

Repeats the fixed-budget candidate-selection test on the shifted camera split.
This records whether the active-view conclusion survives a different set of
initial, candidate, and held-out camera poses, even though the companion depth
uncertainty diagnostic is currently negative.
