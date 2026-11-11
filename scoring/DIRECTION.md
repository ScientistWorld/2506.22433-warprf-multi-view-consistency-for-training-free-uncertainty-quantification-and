# Research Direction

## Research Problem

This benchmark asks whether a radiance-field system can estimate the
reliability of what it renders from unseen viewpoints, and use that estimate to
acquire better training views under a limited imaging budget. Improvements
should target the quality of the uncertainty ranking or of the acquisition
choices, with the same renderer, data and budget.

## Core Contribution

The paper's contribution is a training-free uncertainty signal for radiance
fields that needs only renderings and camera poses, and its use as a selection
criterion for active view acquisition. A scientist should work on the same class
of problem: uncertainty estimation and view selection for fixed radiance fields.

## Approach Scope

In-bounds work may redesign how uncertainty is computed from the available
renderings, depths, captured photographs and poses: which cues are compared,
how they are aggregated across views and pixels, how occlusions, unseen regions
and view-dependent appearance are handled, how scores are normalised, and how
candidates are compared or combined across the two acquisition rounds.

## Out of Bounds

- Replacing, retraining or fine-tuning the radiance field, or rendering with a different model.
- Reading ground truth, evaluator files or anything other than the provided per-instance inputs at prediction time.
- Adding external images, depth, meshes or scene priors beyond what each instance provides.
- Changing the view budget, the candidate pool or the acquisition rule.
