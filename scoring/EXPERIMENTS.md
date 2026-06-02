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

## tiny_nerf_shifted_depth_uncertainty

Repeats the held-out pixel uncertainty test on a shifted camera split. This
checks that the central uncertainty result is not tied to one particular choice
of initial training and held-out views.

## tiny_nerf_shifted_active_view_proxy

Repeats the fixed-budget candidate-selection test on the shifted camera split.
This checks whether the active-view conclusion survives a different set of
initial, candidate, and held-out camera poses.
