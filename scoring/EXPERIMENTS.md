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
