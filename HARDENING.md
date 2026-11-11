# Hardening report (iteration 11)

## Summary

The red-team audit found that the old gym did not reward better research:
its answers could be regenerated from public code, it trusted method-written
labels and constraint flags, its acquisition metric rewarded deliberately
corrupted images, retraining noise (rank spread 0.61) exceeded every honest
effect, half the weight sat on paper tables the test never scored, and the
reference lost to random on one split. These were structural, so the gym was
rebuilt rather than patched:

- **Instances**: procedurally generated textured scenes (`data/scenes.py`) with
  camera-z ground truth. The 32 hidden test scenes come from 128-bit seeds that
  exist only in `eval/test/scene_seeds.json`; their inputs and answers are built
  once and frozen in `eval/test/assets/` (16 public train scenes in
  `data/train_assets/`).
- **Frozen backbones**: one fixed tiny-NeRF recipe (`data/backbone.py`,
  position-only density, view-dependent colour) is trained by the builder with
  identical initialisation and paired per-scene ray-sampling streams. Every
  method sees the same renderings, so comparisons are paired and the test has
  zero run-to-run noise (`warprf_reference_copy` reproduces the reference
  exactly).
- **The paper's procedure, scaled**: the acquisition experiment now does what the
  paper does. From 3 views it acquires 2 of 8 candidates one at a time,
  continues training after each, and scores final novel-view PSNR/SSIM and depth
  error at 8 held-out views. All 8 first-step fields and all 56 ordered
  second-step fields were trained at build time (3,168 fields over 48 scenes), so the hidden test
  only looks results up and never retrains.
- **Paper-faithful reference**: WarpRF was rewritten from the paper and the
  authors' released code (`sadrasafa/WarpRF`, `active/warprf.py`,
  `evaluate_uncertainty.py`). Changes: z-depth (the old version fed ray
  distances into z reprojection), nearest/bilinear sampling with border padding
  (no in-bounds mask, no 95th-percentile fill), max-fill only where no source has
  positive depth, and Eq. 9 summed over pixels of the per-pixel minimum.
- **Evaluator owns everything on the far side**: methods register a command
  (`outputs/<name>/predict.json`). The evaluator runs it in two rounds on the
  inputs, derives both acquisitions from the returned scores, computes all
  errors from its own answers, and computes the baselines itself. It reads
  nothing else a method writes. Any crash, timeout, missing, misshapen or
  non-finite array fails the run, and a failed run gets the worst achievable
  value of every metric of every experiment. The test evaluator imports only
  numpy and the standard library (nothing from `data/` or `method/`).
- **Scoring**: `reference.json` has three runnable experiments (sparse depth
  uncertainty 0.25, dense 0.25, acquisition 0.5; all coefficients positive). The
  paper's tables are kept verbatim under `_paper_reported`, not scored.

Audit code: `redteam/hardening/` (`variants.py` holds every variant and
shortcut payload; `run_checks.py` scores them through both unmodified
evaluators). Raw results: `redteam/hardening/results.{json,md}`. The first
build-and-audit job used the released code's AUSE normalisation; its tables are
kept in `redteam/hardening/results_job1_released_ause_normalization.*` (see
"AUSE normalisation" below).

**Cohort ranks.** Ranks below use the same formula as the red team:
`sum(weight * |coef| * direction * (value - reference) / std)`, where std is
taken across the 23 audited cases on that split. The reference is 0. Paired
improvements use a 2,000-sample scene bootstrap (95% CI).

## Do better results now score higher?

**Before** (red team, old gym): a real geometry fix scored −0.06/−0.13/+0.08
paired against matching seeds, and extra training scored −0.02, all inside a seed
spread of 0.61. Constant predictions scored −0.97 and removing warping −1.55.
Corrupting images scored +0.35.

**After** (hidden test / train):

| Case | What changes | Test rank | Train rank |
|---|---|---:|---:|
| `warprf_edge` (better) | adds rendered-depth edge magnitude to Eq. 8 | **+0.094** | **+0.101** |
| `warprf_rgbd_edge` (better) | adds Eq. 9 photometric map and depth edges | +0.097 | +0.051 |
| `warprf_reference_copy` | identical reference | 0.000 | 0.000 |
| `warprf_lookahead_select` | pair-aware first acquisition | −0.004 | −0.204 |
| `warprf_ensemble_select` | rank-sum with camera distance | −0.054 | −0.085 |
| `weak_depth_prior` (weaker) | uncertainty = rendered depth; pick farthest-depth view | −0.280 | −0.345 |
| `weak_constant` (weaker) | constant predictions | −0.510 | −1.030 |
| `warprf_coverage_select` (weaker) | out-of-view pixels count as max error | −0.667 | −0.656 |
| `warprf_depth_select` (weaker) | selects by summed Eq. 8 | −1.126 | −0.710 |

The better variant `warprf_edge` beats the reference beyond noise on both depth
experiments and both splits (paired AUSE gain):

- test: sparse +0.039 [+0.018, +0.063] (23/32 scenes better), dense +0.060 [+0.033, +0.088] (25/32);
- train: sparse +0.047 [+0.008, +0.086], dense +0.063 [+0.025, +0.097].

There is no retraining noise to compare against, because inputs are frozen and
evaluation is deterministic. The paper's effect is far larger than the
scene-sampling noise. On test, WarpRF beats random by an AUSE of +0.101
[+0.063, +0.139] (sparse) and +0.164 [+0.129, +0.198] (dense). Its acquisition
beats random by +0.42 dB PSNR [+0.16, +0.69], with SSIM +0.016 [+0.006, +0.025].

## Major shortcuts: fix and score before → after

| Red-team shortcut | Before | Fix | After (test rank) |
|---|---:|---|---:|
| Oracle enumeration via the visible generator | +0.631 | Test scenes come from 128-bit seeds held only in `eval/test/`; answers are frozen there. | `shortcut_generator_oracle`: tries public seeds 0–1999, otherwise falls back to WarpRF. Train +1.965 (train is public by design), **test 0.000** (no match, no gain). |
| Legacy fabricated `depth_error` / `candidate_oracle_errors` | +1.488 | Legacy path and `eval/metrics.py` deleted. Every error is computed from evaluator answers; extra arrays are ignored. | `shortcut_fabricated_fields` (forged errors, PSNR, flags, selection): **−0.510**, identical to constant predictions. |
| False budget / impossible constraint flags | +0.012 / +1.597 | No self-reported quantity is read. The budget is applied by the evaluator, which performs every acquisition. | Same case (`source_view_count=-100`, flags −100): **−0.510**. |
| Constant-white / RGB-1000 candidate images | +0.352 / +0.941 | The metric is final held-out quality after acquisition, not "error exposed". Methods return only scores; renderings are evaluator-owned. | No image output path exists (closed by construction). |
| Replacing the fixed generator in `data/` | +0.766 | The test evaluator imports nothing from `data/`; answers are frozen files in `eval/test/assets/`. | Closed by construction (imports are numpy and the standard library only). |
| Empty train results → empty test entry | 0 (tie) | No filtering by `scores_train.json`. Every registered method is scored. If none is registered, a `no_method_registered` entry gets worst values. | Empty registry: PSNR 21.07 (worst pair), AUSE 1.34/1.71. **No tie**. |
| Omitting a weaker method on disk | 0 (tie) | Every `outputs/*/predict.json` is scored. Malformed registrations fail to worst values. | `broken_registration`: **−4.335**. |
| Invalid selected-view field | 0 (tie) | Selection is derived from the score vector (argmax over available candidates). | Forged `selected_index=-999999` is ignored (in the fabricated case). |
| NaN outputs | unrankable | Non-finite, misshapen or crashed run fails, with worst value in every metric of every experiment. | `shortcut_nonfinite` / `shortcut_wrong_shape` / `shortcut_crash`: **−4.335**. |
| Prewritten / stale score file | overwritten / stale left | The evaluator deletes the score file before loading anything and writes atomically. | Forged file overwritten; a failing evaluator leaves no file. |

## Shortcuts left open

- **Reading the hidden answers from disk at prediction time**
  (`shortcut_read_hidden_answers`): **+2.211 test**. The method process runs on
  the same filesystem as `eval/test/assets/`. Closing this needs process or
  filesystem isolation that the container does not provide. Any obfuscation of
  the answers would sit next to its key. It is forbidden in `CONSTRAINTS.md`
  and is blatant in code, so it is left to the shortcut grader.
- **Inverting the scene family**: fitting the public generator's parameters to
  the captured views to recover ground-truth depth. Not tested. The family has
  3–6 primitives with random types, poses, sizes and procedural textures, seen
  in 3–8 low-resolution views, so recovery is an expensive inverse-graphics
  problem. Seed search is infeasible (128-bit seeds).
- **Simple rendered-depth cue (disclosed, not a constraint break)**: using
  rendered depth as uncertainty orders sparse-field errors better than WarpRF.
  - Sparse AUSE gain +0.070 [+0.021, +0.122]; dense tie.
  - Paired with WarpRF's acquisition (`depth_prior_hybrid`) it ranks +0.080
    test / +0.092 train: above the reference, below `warprf_edge`, and far from
    the AUSE bound.
  - This is a genuinely better error ordering: in sparse volume rendering,
    semi-transparent pixels near silhouettes get both a high expected depth and
    a large error. So it is headroom a cheap idea can claim, not a shortcut.

## Reference score before and after

The two cannot be compared number for number: metrics and instances changed.

- **Before** (old test): sparse AUSE 0.0537 (random 0.1319), dense 0.0331
  (0.2798), shifted 0.1224 (random 0.1100: the reference lost). Selected-error
  lift was 2.06/1.39/1.03.
- **After** (hidden test, `scoring/reference.json` = `scoring/scores_test.json`):

| Experiment | Metric | WarpRF | Random | Farthest | Bound |
|---|---|---:|---:|---:|---:|
| depth_uncertainty_sparse | AUSE ↓ | 0.4633 | 0.5643 | – | 0 |
| | Spearman ↑ | 0.1479 | 0.0005 | – | 1 |
| depth_uncertainty_dense | AUSE ↓ | 0.4618 | 0.6258 | – | 0 |
| | Spearman ↑ | 0.1068 | −0.0006 | – | 1 |
| active_view_selection | final PSNR ↑ | 23.012 | 22.594 | 22.969 | 24.177 (best pair) |
| | final SSIM ↑ | 0.8035 | 0.7878 | 0.7991 | – |
| | final depth MAE ↓ | 0.2327 | 0.2399 | 0.2304 | – |

Train split (16 scenes) is consistent: AUSE 0.430/0.449 vs random 0.562/0.601,
PSNR 22.95 vs random 22.53 / farthest 22.90.

## Headroom check

- **No scored metric sits at its bound.** AUSE is 0.46 (bound 0) and Spearman
  0.11–0.15 (bound 1). WarpRF's acquisitions land at the 62nd percentile of the
  56 possible pairs: 23.01 dB against 24.18 dB for the best pair and 21.07 dB for
  the worst.
- **Added experiment where the method falls short**:
  `depth_uncertainty_sparse` (3-view fields, large unobserved regions), weight
  0.25. Here WarpRF is weakest relative to simple cues.
- **Variants above the reference beyond noise**: `warprf_edge` on both depth
  experiments (above). No single simple variant reaches any bound.
- **Brute force**: enumerating candidates needs each candidate's photograph and
  the held-out views. On the hidden split the method sees neither: a candidate's
  photograph is revealed only after it is acquired, and the held-out views
  never. Its own copy of the backbone and generator therefore cannot compute
  the objective. The public-seed enumeration (`shortcut_generator_oracle`)
  reaches the 98th percentile pair and AUSE 0 on the public train split, but
  gains nothing on test.
- **Baselines**: random is clearly below the reference on every experiment,
  with the CIs above. Farthest-camera acquisition ties it (see below).

## AUSE normalisation

The released evaluation code divides both sparsification curves by the larger
curve's maximum. That denominator depends on the prediction. In the first audit
job, an anti-correlated ordering (the failed-run value) therefore scored AUSE
0.34/0.21, *better* than WarpRF's 0.39/0.42, and lower than random's 0.49/0.52
(`results_job1_released_ause_normalization.md`). The evaluators now divide by
the view's full-set mean error, a constant independent of the prediction, so
AUSE is monotone in ordering quality. Failed runs now score 1.34/1.71 against
random's 0.56/0.63. Everything else in the protocol follows the paper's code:
removal fractions 0–0.99 in steps of 0.01, MAE, per-view averaging, and
trapezoidal area.

## Weights

The four paper tables previously held weight 0.50 but were never scored. That
weight moved to runnable experiments. The paper's two halves (uncertainty
quality, Table 1; downstream acquisition and mapping, Tables 2–7) each get 0.5:

- sparse depth uncertainty 0.25, dense depth uncertainty 0.25;
- acquisition 0.5, with PSNR 1.0, SSIM 0.5, and depth MAE 0.5 as the
  mapping/geometry constraint.

The shifted-split diagnostic was dropped: it had weight 0.05, and its negative
result came from the reference's ray-distance bug.

## What I could not fix

- **WarpRF only ties farthest-camera acquisition**: +0.04 dB [−0.27, +0.36],
  SSIM +0.004, depth MAE −0.002. The paper reports WarpRF beating farthest-view
  selection with SVRaster by 0.3–0.9 dB. At this scale (48×48, 3+2 views,
  tiny NeRF) that part of the downstream claim is not reproduced; beating random
  is reproduced.
- **No honest acquisition variant beat the reference beyond noise.**
  `warprf_masked` is +0.10 dB [−0.11, +0.32] on test but −0.01 dB on train.
  Look-ahead, distance ensemble, coverage penalty and depth-based selection are
  all at or below the reference. The acquisition experiment's headroom is shown
  only by the 1.17 dB gap to the best pair (and by the answer-reading shortcut).
  The overall test has demonstrated headroom through the depth experiments.
- **Scale**:
  - Procedural scenes instead of ScanNet++/ETH3D/NeRF-Synthetic/Tanks and
    Temples, at 48×48.
  - A tiny NeRF only (no 3DGS or SVRaster).
  - Dense means 8 views (the paper uses about 20).
  - Two acquisitions instead of 8–16.
  - No LPIPS (it needs pretrained network weights) and no mesh metrics; depth
    MAE stands in for geometry.
- The prediction time limit (1800 s per round) is wall-clock and therefore
  hardware-dependent. The hidden test scored all 23 audit cases (both rounds,
  32 scenes each) in 433 s on one CPU, under 20 s per case.
- `redteam/` outside `redteam/hardening/` is the previous red team's harness for
  the old gym. It is kept as a record and does not run against the new evaluator.
