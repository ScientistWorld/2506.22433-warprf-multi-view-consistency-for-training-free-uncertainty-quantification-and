# Hardening report (final iteration)

This iteration started from the hardened gym of iteration 11 and the two
weaknesses `REVIEW.md` found in it:

1. WarpRF only tied farthest-camera acquisition (+0.04 dB, CI [-0.27, +0.36]),
   and no honest acquisition variant beat it, so the acquisition experiment had
   no demonstrated headroom.
2. A method that read `eval/test/assets` at prediction time scored +2.21.

A third flaw was flagged by the hardening rules rather than by the review: a
one-line rule, "uncertainty = rendered depth" (`depth_prior_hybrid`), ranked
+0.080 above the reference. Iteration 11 had disclosed it but left it open.

All three trace back to setup choices that departed from the paper, so the
instances were rebuilt rather than patched. The evaluator was isolated from the
method at the same time.

## What changed

- **Methods run isolated** (`eval/{train,test}/isolate.py`, one copy per split).
  - Each method command runs in fresh Linux user, mount and PID namespaces.
  - `eval/test/` (hidden instances, answers, seeds) and the git object store are
    covered by empty read-only tmpfs mounts.
  - `/proc` is remounted, so only the method's own processes are visible. That
    blocks `/proc/<pid>/root` access to the evaluator's view of the files.
  - The method runs without capabilities, so it cannot unmount the covers.
  - Before running any method, the evaluator checks with a probe that the hidden
    folder really appears empty. If the kernel refuses namespaces, it falls back
    to removing all permissions from the hidden instances during each run.
  - The mode used is written to the score file as `_isolation`. It was
    `namespace` in every job of this iteration.
  - Methods also run in their own session with output going to a file, so
    leftover child processes are killed and cannot hang the evaluator on
    pipes.
- **Rendered depth restored to paper Eq. 2** (`data/backbone.py`).
  - Depth is now `D = sum_i alpha_i T_i t_i`, converted to camera z.
  - The earlier backbone added `(1 - acc) * t_far`, i.e. it composited a far
    background into the depth. That made "large rendered depth" a stand-in for
    "low opacity", which is what the depth-prior rule exploited.
  - With Eq. 2 the one-line rules fail and WarpRF's AUSE moves to the paper's
    range.
- **Candidates drawn like the held-out views** (`data/scenes.py`).
  - Before, the 8 candidates came from a much wider pose law (radius 2.3-3.9,
    elevation 0-60 degrees, look-at jitter 0.35) than the held-out views
    (2.8-3.2, 15-35 degrees, 0.1).
  - As a result, the one-step gain tracked camera novelty more than rendering
    error (Spearman 0.37 vs 0.21). Selecting by the *true* candidate error
    gained only +0.04 dB over WarpRF, so a better uncertainty could not score
    higher.
  - In the paper's benchmarks, candidates and test views come from one capture.
    The gym now draws 16 candidates i.i.d. from the held-out views' own law.
- **32 held-out views instead of 8.** With 8 views, part of a candidate's value
  came from happening to sit next to one hidden view, which no method can know:
  that alignment correlated 0.24-0.27 with the gain. The paper scores on many
  test views (200 for NeRF Synthetic).
  - With 32 views, true candidate error predicts the one-step gain with
    Spearman 0.74 on train, up from 0.63.
- **More hidden scenes**: 56 test scenes (40 kept from the first rebuild plus 16
  new 128-bit seeds), and 16 public train scenes.
  - Every ordered acquisition pair (240 per scene) is trained at build time, so
    the hidden test only looks results up.
  - The build is resumable and split across two GPUs (`scripts/build_assets.sh`),
    about 3.2 h.
- **Audit extended** (`redteam/hardening/`).
  - `variants.py` adds honest depth and acquisition variants, the one-line depth
    rules, and `shortcut_escape`, which tries every route to the hidden answers.
  - `run_checks.py` adds oracle selectors and paired comparisons of the reference
    against both baselines.
  - Pure-Python analyses of the acquisition tables are in
    `redteam/hardening/analysis/`. The design study that chose the depth
    convention and candidate law is `design_study.py` / `design_study.json`.
- **Records.** The intermediate build (8 held-out views, 40 test scenes) is kept
  as `results_8eval_40scene.*`. The final audit is `results.{md,json}`.

Ranks below use the same cohort formula as before:
`sum(weight * |coef| * direction * (value - reference) / std)`, where std is
taken across the audited cases of that split. The reference is 0. CIs are 95%
scene-bootstrap intervals on paired per-scene differences.

## Do better results now score higher?

Yes for uncertainty quality; only partly for acquisition (see "What I could not
fix"). Variants before (iteration 11 test) and after (final test, 56 scenes):

| Case | What changes | Before | After (test) | After (train) |
|---|---|---:|---:|---:|
| `warprf_edge` (better) | adds rendered-depth edges to Eq. 8 | +0.094 | **+0.093** | +0.031 |
| `warprf_rgbd_edge` | adds the Eq. 9 photometric map and depth edges | +0.097 | +0.025 | -0.107 |
| `warprf_lookahead_select` | pair-aware first acquisition | -0.004 | +0.035 | +0.078 |
| `warprf_reference_copy` | identical reference | 0.000 | 0.000 | 0.000 |
| `weak_constant` (weaker) | constant predictions | -0.510 | -1.193 | -1.386 |
| `warprf_coverage_select` (weaker) | out-of-view pixels count as max error | -0.667 | -0.873 | -0.944 |
| `warprf_depth_select` (weaker) | selects by summed Eq. 8 | -1.126 | -1.010 | -0.639 |
| failed run (crash, NaN, wrong shape, broken registration) | worst value of every metric | -4.335 | -5.087 | -4.873 |

`warprf_edge` beats the reference beyond noise on both depth experiments of the
hidden test:
- sparse AUSE +0.046 [+0.019, +0.075];
- dense AUSE +0.039 [+0.015, +0.062].

On the 16-scene train split the same gains are +0.009 / +0.014 and not
significant.

Weaker methods score clearly lower: constant predictions, coverage-penalised and
depth-based selection, and every failed run.

## Major shortcuts: fix and score before → after

| Shortcut | Before | Fix | After (test rank) |
|---|---:|---|---:|
| Reading the hidden answers at prediction time (`shortcut_escape`) | **+2.211** | Namespace isolation hides `eval/test/` and `.git`, and `/proc` shows only the method's own processes. | **0.000**: on all 112 test calls, every route failed (direct read: not found; `chmod`: read-only empty tmpfs; `/proc/*/root`: one visible pid; `git show`: no repository). It falls back to WarpRF and ties the reference with no gain. |
| One-line depth rule, U = rendered depth (`depth_prior_hybrid`) | **+0.080** | Paper Eq. 2 depth: low opacity no longer pushes depth toward the far bound. | **-0.905** (AUSE worse by 0.70 / 0.70) |
| Opposite rule, U = -rendered depth (`neg_depth_prior_hybrid`) | not tested | same | **-0.335** (AUSE worse by 0.10 / 0.38) |
| Hand-written camera-coverage planners (pure-Python analysis) | farthest tied WarpRF | Candidates share the held-out views' law, so content matters. | Farthest -0.33 dB, optimal azimuth-gap planner -0.30 dB, greedy gap planner -0.41 dB vs WarpRF (test). |
| Public-seed generator oracle (`shortcut_generator_oracle`) | 0.000 test | unchanged (128-bit hidden seeds) | 0.000 test (train +1.35: the train split is public by design) |
| Fabricated metric, label and flag fields | -0.510 | unchanged (nothing a method writes besides the arrays is read) | -1.193 (equal to constant predictions) |
| Empty registry / forged or stale score files | no tie | unchanged | empty registry gets worst values; forged file overwritten; failing evaluator leaves no file |

## Shortcuts left open

- **Fallback isolation is weaker.** If a future container refused unprivileged
  user namespaces, the evaluator would only chmod the hidden instances during
  each run. A method could chmod them back, and the git store would stay
  readable. The probe makes this visible (`_isolation: "lock"` in the score
  file). Namespaces worked in every job here, and the hidden test runs in this
  same container. The bypass would be blatant code for the shortcut grader.
- **Copying hidden files at development time.** A future agent can read
  `eval/test/` in its own workspace and bake answers into its method before the
  test runs. No evaluator can prevent this. `CONSTRAINTS.md` forbids it, and it
  is left to the grader.
- **Inverting the scene family** from the captured photos (inverse graphics
  with the public generator) was not tested. It needs fitting 3-6 textured
  primitives to 3-8 views at 48x48; seed search is infeasible with 128-bit
  seeds.

## Reference score before and after

The instances changed (depth convention, candidate law, held-out views, scene
count), so the numbers are not comparable one to one.

| Experiment | Metric | Before: WarpRF / random / farthest | After: WarpRF / random / farthest | Bound |
|---|---|---|---|---|
| depth_uncertainty_sparse | AUSE ↓ | 0.463 / 0.564 / - | **0.348** / 0.588 / - | 0 |
| | Spearman ↑ | 0.148 / 0.000 / - | **0.346** / 0.002 / - | 1 |
| depth_uncertainty_dense | AUSE ↓ | 0.462 / 0.626 / - | **0.358** / 0.707 / - | 0 |
| | Spearman ↑ | 0.107 / -0.001 / - | **0.230** / -0.002 / - | 1 |
| active_view_selection | final PSNR ↑ | 23.012 / 22.594 / 22.969 | **23.267** / 22.436 / 22.933 | 24.203 (best pair) |
| | final SSIM ↑ | 0.8035 / 0.7878 / 0.7991 | **0.8165** / 0.7816 / 0.8006 | - |
| | final depth MAE ↓ | 0.2327 / 0.2399 / 0.2304 | **0.3352** / 0.3751 / 0.3623 | - |

Depth MAE rose for every method because Eq. 2 depth falls short of the surface
wherever the field is not opaque. WarpRF's AUSE (0.35) now sits in the paper's
Table 1 range (0.23-0.42).

The paper's direction holds on the hidden test, paired over the same 56 scenes:
- WarpRF beats random uncertainty by +0.240 AUSE [+0.204, +0.275] (sparse) and
  +0.349 [+0.318, +0.380] (dense).
- Its acquisitions beat random by **+0.83 dB** [+0.66, +1.00] (49/56 scenes).
- They beat farthest-camera selection by **+0.33 dB** [+0.12, +0.55] (39/56
  scenes). This is review item 1. The paper reports +0.3 to +0.9 dB over
  farthest with SVRaster.

On the 16-scene train split all differences have the same sign: +0.10 dB over
farthest, not significant.

`scoring/scores_test.json` and `scoring/scores_train.json` are the reference
run (WarpRF plus the evaluator's baselines) from the final job, and
`scoring/reference.json` holds the same test values. The paper's numbers are
unchanged under `_paper_reported`.

## Headroom check

- **No scored metric is at its bound.**
  - AUSE is 0.35 / 0.36 (bound 0); Spearman is 0.35 / 0.23 (bound 1).
  - WarpRF's acquisitions land at the 80th percentile of the 240 ordered pairs.
    The best pair is +0.94 dB [+0.77, +1.10] above it.
- **The headroom is real, not noise.** Acquiring (a, b) and (b, a) gives final
  PSNRs that correlate 0.91. Order noise is 0.22 dB against a 0.75 dB
  within-scene spread.
- **A variant above the reference beyond noise**: `warprf_edge`, on both depth
  experiments (above).
- **The experiment where the method falls short**: `depth_uncertainty_sparse`
  (3 views, large unobserved regions) is kept at weight 0.25.
- **Brute force cannot solve it.** The method never sees held-out views or
  unacquired candidate photographs, so it cannot compute the objective. Its own
  copy of the generator and backbone does not help without the hidden seeds.
- **Simple rules stay below the reference**: depth rules, camera-coverage
  planners, farthest and random.

## Weights

Unchanged: sparse depth uncertainty 0.25, dense depth uncertainty 0.25,
acquisition 0.5 (PSNR 1.0, SSIM 0.5, depth MAE 0.5 as the geometry constraint).

The paper's two halves (Table 1; Tables 2-7) still carry equal weight. The
acquisition experiment now reproduces both of the paper's downstream
comparisons, and it is where breaking the selection costs most.

## What I could not fix

- **No honest acquisition variant beats WarpRF beyond noise.** On the final
  build (test / train):
  - pair-aware lookahead: +0.01 [-0.12, +0.15] / +0.09 [-0.01, +0.22];
  - rank-sum with camera distance: -0.07 / +0.13;
  - Eq. 9 with source photographs: -0.00 / 0.00;
  - valid-pixel mean: -0.12 / +0.04;
  - rank-sum with Eq. 8: -0.20 / -0.10;
  - three-way ensemble: -0.26 / -0.18;
  - angular neighbourhood smoothing: -0.52 / -0.50;
  - occlusion-aware Eq. 9: -0.65 / -0.63.

  Even selecting by each candidate's *true* rendering error, a perfect version
  of the quantity WarpRF estimates, gains only +0.13 dB [-0.01, +0.28] (test) /
  +0.10 (train). WarpRF is close to the ceiling of max-uncertainty selection.

  The remaining headroom is real: the best one-step-then-best-second choice
  (greedy oracle) gains +0.72 dB [+0.55, +0.89]. Reaching it requires predicting
  which view improves *other* views, not which view is rendered worst. That is
  a research problem the interface allows (methods see all candidate renderings
  and the public train tables), but I did not find a method that solves it.

  So the acquisition experiment's headroom is shown by oracles, not by an honest
  variant. The test as a whole has demonstrated headroom through the depth
  experiments.
- **The train split is small.** With 16 scenes, `warprf_edge`'s train gains and
  WarpRF's train margin over farthest are not significant. The hidden test
  (56 scenes) is where the effects are resolved.
- **Scale.**
  - Procedural scenes at 48x48 and a tiny NeRF only (no 3DGS, SVRaster, LPIPS
    or meshes).
  - Two acquisitions instead of 8-16; depth MAE stands in for mesh metrics.
  - The paper's competing uncertainty methods (BayesRays, FisherRF, Manifold)
    need model internals and are not expressible through the rendering-only
    interface. The baselines are random and farthest-camera selection, as in
    the paper's SVRaster tables.
- **The time limit is wall-clock.** The prediction limit (1800 s per round) is
  hardware-dependent. The final audit ran 28 cases on the hidden test in 675 s
  on one CPU core.
- **Repository size.** The two rebuilds added about 320 MB of instance files to
  the git history (current tree: about 180 MB: 140 MB hidden test, 40 MB train).
