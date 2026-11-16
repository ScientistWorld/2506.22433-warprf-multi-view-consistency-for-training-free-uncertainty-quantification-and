# Experiment Overview

All experiments run on procedurally generated scenes observed by a fixed,
frozen radiance field. The public train split has 16 scenes; the hidden test
split has 56 different scenes of the same kind. Every method sees exactly the
same inputs, and all ground truth stays with the evaluator.

## depth_uncertainty_sparse

Can a method tell which rendered depths of a radiance field are wrong when the
field was trained from only three captured views? For six held-out views per
scene, the method's per-pixel uncertainty is compared with the true depth
error: AUSE asks whether discarding the most uncertain pixels first removes the
largest errors, and the rank correlation asks how well the two orderings agree.
This is the regime where large parts of the scene were never observed.

## depth_uncertainty_dense

The same measurement for a field trained from eight views spread around the
scene, the regime closest to the paper's uncertainty benchmark. Errors here are
smaller and more localised, so the question is whether the uncertainty still
singles them out.

## active_view_selection

Does uncertainty pick useful views to capture next? Starting from the
three-view field, the method scores sixteen candidate cameras; the best-scored
one is captured and the field keeps training with it; the method then scores
the candidates again for the updated field and a second view is captured. The
outcome is the final field's novel-view quality (PSNR, SSIM) and geometry
(depth error) at 32 held-out views it was never shown, under a fixed two-view
budget. The candidates are drawn from the same camera distribution as the
held-out views, so a view worth capturing is one that improves the kind of view
the field will be judged on. Random and farthest-camera acquisition are the
reference points; the depth error guards against image gains bought with worse
geometry.
