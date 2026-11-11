# Problem Definition

## Research Question

How can a radiance-field system estimate which rendered pixels, depths, or unseen camera views are unreliable, and use that uncertainty to choose additional training views that improve novel-view synthesis and 3D reconstruction? The benchmark evaluates uncertainty estimates for already trained radiance fields and downstream sequential view acquisition under limited-view settings.

## Why It Matters

Radiance fields such as NeRF, 3D Gaussian Splatting, and related renderers can fail when trained from too few or poorly placed images. A reliable uncertainty signal lets an autonomous system identify uncertain regions or viewpoints, capture the most informative new images, and improve rendering or mapping without wasting acquisition budget.

## Setting in This Benchmark

A fixed, frozen NeRF-style radiance field is trained by the benchmark on a few captured views of each scene. A method sees only what such a system would have: camera poses, the captured photographs used for training, and renderings (colour and depth) of the trained field at the views it needs. It never sees the ground truth of held-out views, and it never sees a candidate view's photograph until that view has been acquired.

## Success Criteria

A successful solution should:

- Produce per-pixel uncertainty whose ordering matches the true depth error of held-out views, both when the field was trained from very few views and from more views.
- Choose, one at a time, the next views to acquire from a fixed candidate pool so that the field trained with them renders held-out views better and with more accurate geometry.
- Work from the provided renderings and poses alone, under a fixed view budget and a fixed radiance-field backbone, so that gains reflect better uncertainty or selection rather than more data or a different model.
