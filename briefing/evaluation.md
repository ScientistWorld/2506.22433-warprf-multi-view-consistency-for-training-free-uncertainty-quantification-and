# Evaluation

## What is measured

The paper evaluates uncertainty for trained radiance fields in two ways: how
well per-pixel uncertainty ranks the true depth error of held-out views (AUSE on
ScanNet++/ETH3D), and how much better a field becomes when that uncertainty
chooses which new views to acquire (novel-view PSNR/SSIM/LPIPS and
reconstruction quality after active view selection or active mapping).

This gym measures the same two quantities in a scaled, fully controlled
setting:

| Experiment | Weight | Metrics (direction, coefficient) |
|---|---:|---|
| `depth_uncertainty_sparse` | 0.25 | `ause` (lower, 1.0), `spearman_depth_error` (higher, 0.5) |
| `depth_uncertainty_dense` | 0.25 | `ause` (lower, 1.0), `spearman_depth_error` (higher, 0.5) |
| `active_view_selection` | 0.50 | `final_psnr` (higher, 1.0), `final_ssim` (higher, 0.5), `final_depth_mae` (lower, 0.5) |

- **AUSE**: per target view, pixels with ground-truth surface depth are removed
  in decreasing order of predicted uncertainty (fractions 0, 0.01, ..., 0.99);
  the mean absolute camera-z depth error of the remaining pixels is compared with
  the oracle curve that removes pixels by true error. Both curves are divided by
  the view's full-set mean error, so a better ordering always scores lower. Ties
  in the prediction are broken by a fixed evaluator-owned key. Averaged over all
  target views.
- **Spearman**: rank correlation between predicted uncertainty and true depth
  error over the same pixels, averaged over target views.
- **Final PSNR / SSIM / depth MAE**: quality of the radiance field after the
  two acquisitions, rendered at 32 held-out evaluation views that are never
  shown to the method (PSNR in dB, SSIM with an 11x11 Gaussian window, depth MAE
  in scene units over surface pixels). Averaged over views, then scenes.

## Instances

Each instance is one procedurally generated textured scene (a floor plate and
several boxes/ellipsoids with procedural textures and shading) observed at
48x48 pixels. A fixed NeRF-style backbone recipe (`data/backbone.py`) is trained
by the benchmark itself; methods never train or modify it.

- *Sparse field*: trained on 3 captured views spread over ~150 degrees.
- *Dense field*: trained on 8 captured views spread around the scene.
- *Targets*: 6 held-out views per scene for the depth-uncertainty experiments.
- *Candidates*: 16 possible next views for acquisition, drawn independently
  from the same camera-pose distribution as the 32 held-out evaluation views
  (which are spread evenly around the scene).
- *Acquisition*: starting from the sparse field, the evaluator acquires the
  candidate with the highest score, continues training with it, asks for
  scores again from the updated field, and acquires a second candidate. The
  final quality of every possible ordered pair (240 per scene) was computed
  once, with one fixed training schedule and initialisation, when the instances
  were built.
- *Rendered depth* is the expected termination depth `sum_i w_i t_i` along each
  ray, converted to camera z, with no background term: where the field is not
  opaque the rendered depth falls short of the surface. Depth error is measured
  on this rendered depth.

The train split (`data/train_assets/`, 16 scenes) is public, including its
answers, for development. The hidden test split has 56 different scenes from the
same generator family, built and scored with the same protocol.

## Interface

A method is registered by writing `outputs/<name>/predict.json`:

```json
{"command": ["python", "-m", "my_package.predict"]}
```

The evaluators run, from the workspace root and with the workspace on
`PYTHONPATH`:

```
<command> --round 1 --input <dir> --output <dir>
<command> --round 2 --input <dir> --output <dir>
```

`<input>/manifest.json` lists the scene names; each `<input>/<scene>.npz` holds
the arrays below. The method writes `<output>/<scene>.npz` for every scene.

Conventions: OpenCV cameras (x right, y down, z forward), 4x4 camera-to-world
poses, intrinsics `K` with principal point at the image centre, pixel (row i,
col j) centred at (j + 0.5, i + 0.5). Depth is camera-space z. RGB arrays are
uint8; depth arrays are float16. `*_images` are captured photographs, `*_rgb` /
`*_depth` are renderings of the radiance field.

**Round 1 input**: `intrinsics`; `sparse_source_poses/_images/_rgb/_depth` (3
views); `target_poses` (6); `sparse_target_rgb/_depth`;
`candidate_poses` (16); `sparse_candidate_rgb/_depth`;
`dense_source_poses/_images/_rgb/_depth` (8); `dense_target_rgb/_depth`.

**Round 1 output**: `sparse_uncertainty` [6, H, W], `dense_uncertainty`
[6, H, W] (higher = less reliable depth), `candidate_scores` [16] (the highest
is acquired; ties go to the lowest index).

**Round 2 input** (field after continued training with the first acquisition):
`intrinsics`; `selected_candidates` [1]; `candidate_available` [16] bool;
`source_poses/_images/_rgb/_depth` (4 views: the 3 initial views plus the
acquired one, whose photograph is now available); `candidate_poses`;
`candidate_rgb/_depth` (16, rendered by the updated field).

**Round 2 output**: `candidate_scores` [16]; the highest-scored available
candidate is acquired.

## Rules enforced by the evaluator

- Every registered method is scored; nothing reported by a method other than the
  arrays above is read, and the evaluator derives each acquisition itself.
- The method command runs isolated (Linux user, mount and PID namespaces): the
  `eval/test/` folder and the git object store appear empty, and only the
  method's own processes are visible. It can read the rest of the workspace
  and its input folder, and write anywhere it normally could. Both evaluators
  run methods the same way.
- A run fails if the command cannot start, exits non-zero, exceeds 1800 s per
  round, or returns a missing, misshapen, non-real or non-finite array. A failed
  run receives the worst achievable value of every metric of every experiment
  (anti-oracle AUSE, Spearman -1, worst acquisition pair).
- Baselines are computed by the evaluator on the same instances:
  `random_uncertainty` (fixed pseudo-random pixel ordering), `random_view`
  (exact expectation of uniformly random sequential acquisition) and
  `farthest_view` (acquire the candidate camera farthest from all current cameras).

## Running

- `bash scripts/evaluate_train.sh` -> `scoring/scores_train.json` (train split)
- `bash scripts/evaluate_test.sh` -> `scoring/scores_test.json` (hidden split)
- `bash scripts/evaluate.sh` -> `scoring/scores.json` (train split)

All three take no arguments and score every method registered under `outputs/`.
