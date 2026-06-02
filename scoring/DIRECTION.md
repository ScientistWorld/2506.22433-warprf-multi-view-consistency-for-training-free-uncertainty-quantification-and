# Research Direction

## Research Problem

This benchmark asks whether a radiance-field system can estimate uncertainty in unseen views and use that uncertainty to acquire better training views under a limited imaging budget. Improvements should target the quality of uncertainty ranking or view selection while preserving the same renderer, data, and acquisition budget.

## Core Contribution

The paper's contribution is a training-free uncertainty signal for radiance fields and its use as a selection criterion for active view acquisition. A scientist should work on the same class of problem: uncertainty estimation and active view selection for fixed radiance-field models.

## Approach Scope

In-bounds work may redesign how uncertainty is computed from available renderings, depths, poses, colors, geometry, or candidate-view statistics. It may change aggregation rules, normalization, occlusion handling, confidence calibration, candidate scoring, or acquisition scheduling within the fixed protocol. It may also introduce lightweight computations around the fixed renderer if they do not train on extra data or alter the underlying radiance-field backbone.

## Out of Bounds

- Replacing the evaluated radiance-field backbone with a stronger renderer or pretrained reconstruction system.
- Solving the task by adding more training images, external depth supervision, external meshes, or online data not present in the benchmark.
- Using test-set ground truth to choose views, tune scene-specific parameters, or calibrate uncertainty.
- Turning the benchmark into a supervised uncertainty-learning problem with labels from the evaluation scenes unless the protocol explicitly provides such labels for training.
- Reporting only benefit metrics while omitting the fixed-budget or fixed-backbone constraints.
