# Problem Definition

## Research Question

How can a radiance-field system estimate which rendered pixels, depths, or unseen camera views are unreliable, and use that uncertainty to choose additional training views that improve novel-view synthesis and 3D reconstruction? The benchmark should evaluate uncertainty estimates for already trained radiance fields and downstream active data acquisition under limited-view settings.

## Why It Matters

Radiance fields such as NeRF, 3D Gaussian Splatting, and related renderers can fail when trained from too few or poorly placed images. A reliable uncertainty signal lets an autonomous system identify uncertain regions or viewpoints, capture the most informative new images, and improve rendering or mapping without wasting acquisition budget.

## Success Criteria

A successful solution should:

- Produce uncertainty scores that correlate with true depth/rendering error on held-out views.
- Select useful next views from a candidate pool when only a small initial set of training views is available.
- Improve held-out novel-view synthesis quality and, for mapping, reconstructed surface quality.
- Remain applicable under fixed dataset splits, view budgets, and radiance-field backbones, so gains reflect better uncertainty or selection strategy rather than more data or larger models.
