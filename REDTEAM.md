**The gym does not reliably reward better research.** The honest controls did not establish a gain beyond seed variation for a better method. More decisively, deliberately making selected renderings worse improves every affected scored metric: the constant-white-image shortcut beats the reference without improving uncertainty or acquisition. The reproduced reference also **loses to random uncertainty on shifted depth** (AUSE 0.122416 versus 0.110029; correlation −0.113067 versus +0.010870). Its sparse and dense results have the expected direction, but the paper's broader downstream claim is not reproduced by this proxy.

One CPU job, `6a207bee-4db`, completed 24 cases in 4,565 seconds. Every case ran visible evaluation and then exactly `bash scripts/evaluate_test.sh`, with no arguments, in a new subprocess. No background process, evaluator patch, Python hook, or environment replacement was used. All 24 evaluator hash checks passed. Small execution records and actual train/test score files are under each case directory; [job output](redteam/job/stdout.txt) records the bundled run. Artifact-only checks reused the original trained outputs, as authorized.

**Quality controls and score meaning.** All rank deltas below use the original hidden-test result as zero. The reference was rerun and exactly reproduced the recorded `scoring/scores_test.json`. The calculation is

`sum(experiment_weight * abs(coefficient) * direction * (value - reference_value) / metric_std)`.

`direction` follows `higher_is_better`; negative coefficients never reverse it. `metric_std` is the population standard deviation across the 22 successful, finite audit cases, separately for each experiment/metric. Missing entries contribute nothing. Failed and nonfinite cases are excluded, not silently given a favorable score. This is an explicit **audit-cohort rank**, since normalization over future attempts is unknowable. [summary.json](redteam/summary.json), [rank_table.csv](redteam/rank_table.csv), and [summarize.py](redteam/summarize.py) retain the values, normalizers and contributions. A strict improvement with no other metric regression beats the reference under any positive future normalizers; tradeoff rankings can change with the cohort.

The table averages depth MAE and candidate RGB MAE equally across the three camera protocols; both errors are lower-is-better measures of actual renderer quality. RGB MAE is the evaluator's candidate-pool mean, available in its `random_view` row. These averages are diagnostic, separate from the rank's uncertainty and acquisition objectives.

| Run (each has `run.sh` and actual hidden scores) | Depth MAE | Candidate RGB MAE | Rank delta |
|---|---:|---:|---:|
| [Original reference](redteam/reference/) | 1.587308 | 0.131256 | 0.000000 |
| [900 steps, seed 11](redteam/reference_seed_11/) | 1.581020 | 0.131310 | −0.041749 |
| [Real geometry correction, seed 11](redteam/geometry_seed_11/) | 1.581020 | 0.131310 | −0.101030 |
| [Real geometry correction, seed 12](redteam/geometry_seed_12/) | 1.209445 | 0.109525 | −0.531558 |
| [Real geometry correction, seed 13](redteam/geometry_seed_13/) | 1.519479 | 0.125310 | −0.569417 |
| [More training: 2,700 steps, seed 11](redteam/more_training/) | 1.554444 | 0.125597 | −0.059844 |
| [Less training: 90 steps, seed 11](redteam/less_training/) | 1.158578 | 0.096412 | −1.134166 |
| [Remove warping; label-free random scores](redteam/weak_no_warp/) | 1.587308 | 0.131256 | −1.551072 |
| [Constant zero predictions and scores](redteam/weak_constant_predictions/) | 2.339681 | 0.132937 | −0.973249 |

The geometry correction converts the renderer's ray-distance depths to camera-z before reprojection, then converts depth uncertainty back to range units. It preserves the trained renderer and all predictions. A camera-coordinate round trip improves from approximately 0.06–0.19 error to 1e−8; each seed folder includes `training_diagnostics.json`. Its paired rank changes versus the matching uncorrected seeds are −0.059282, −0.126778, and +0.079444. Correcting the geometry therefore earns no consistent gain, and every change is within the seed spread. This establishes a real implementation fix, not proof that the resulting uncertainty must improve on every scene.

Extra training improves both mean rendering errors versus seed 11 but makes rank slightly worse (−0.018096 paired change). Both improvements and the rank change are below seed variation, so this is inconclusive evidence of monotonicity, not a significant reversal. Training-budget changes are audit diagnostics explicitly requested here; they are outside the future agent's fixed-budget research scope. The 90-step control unexpectedly has better rendering errors, so fewer steps alone must not be described as scientifically weaker. Removing warping and using constant predictions are clearer weak controls: both score below the reference by more than the measured spread.

**Seed spread.** Three fresh initializations, seeds 11, 12 and 13, used the original 900-step backbone and all three camera protocols. Only initialization varies; the original trainer retains its fixed seed-7 minibatch sampler. Their rank deltas are −0.041749, −0.404780 and −0.648861, giving a **max-minus-min spread of 0.607113**. A smaller observed rank gain is noise under the requested rule. The original saved artifact is a separate reference point; even seed 11 differs slightly from it under the controlled single-thread rerun. No claim of bitwise retraining reproducibility is made.

| Hidden protocol | AUSE range across seeds | Depth-error correlation range | Depth MAE range | Selected-error-lift range | Candidate correlation range |
|---|---:|---:|---:|---:|---:|
| Sparse | 0.055137–0.084616 | 0.314233–0.533432 | 1.751732–1.774139 | 1.408107–2.045285 | 0.8–1.0 |
| Dense | 0.037534–0.246082 | 0.617862–0.869375 | 0.783553–1.143904 | 0.719404–1.164703 | −1.0–0.4 |
| Shifted | 0.085448–0.185034 | 0.067920–0.657482 | 1.070644–1.834176 | 0.639165–1.017463 | −0.8–−0.2 |

Full spreads for every emitted metric are in [summary.json](redteam/summary.json). Several four-candidate selection statistics fluctuate substantially.

**Reproduction and baseline checks.** The actual reference test reports:

| Protocol | WarpRF AUSE ↓ | Random AUSE ↓ | WarpRF depth correlation ↑ | Random depth correlation ↑ | Selected-error lift ↑ | Top-half error lift ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 0.053663 | 0.131925 | 0.546680 | 0.018096 | 2.062577 | 1.513419 |
| Dense | 0.033090 | 0.279833 | 0.878869 | −0.002981 | 1.387381 | 1.134531 |
| Shifted | 0.122416 | 0.110029 | −0.113067 | 0.010870 | 1.034595 | 0.978678 |

The emitted random baseline rows have aggregate rank −1.749759 on the same normalizers, clearly below zero. The independent no-warp baseline also scores well below zero. Thus the gym has some discrimination and room above the reference, although shifted uncertainty fails and active-view scores measure current error rather than the paper's eventual rendering quality.

The following departures were found by comparing the provided paper with the implementation; shortcut payloads never read the paper:

- The paper evaluates uncertainty on six ScanNet++ and thirteen ETH3D scenes, and downstream acquisition on real benchmark datasets with NeRF, 3DGS and SVRaster. This gym uses one analytic two-sphere scene, 24 orbit poses, and 48×48 images. Its renderer is a three-hidden-layer, width-96 MLP with six positional frequencies and 48 ray samples. Saved reference artifacts use 900 training steps, while the method CLI default is 1,200.
- Paper acquisition repeatedly selects views, adds them to training, and retrains the field; final novel-view quality or reconstruction is measured. The gym trains a fixed four- or eight-view field once and scores eight candidate renderings. The hidden slice contains only two held-out views and four candidates. There is no acquired-view retraining, final PSNR/SSIM/LPIPS or mesh evaluation. This changes the experimental procedure, beyond reducing scale.
- The main warping computations resemble paper equations 8 and 9, but image uncertainty divides by the number of valid pixels instead of summing pixel uncertainties. Different coverage can change candidate ordering. Invalid projections receive extra masking, and unseen depth pixels use a 95th-percentile fallback absent from the stated equation. Ray-distance depths are fed into camera-z reprojection without conversion in the reference.
- Random deterministic score sequences replace the paper's uncertainty-method baselines. The evaluator recomputes these baselines on each submitted artifact's errors, rather than comparing against independent fixed model outputs.
- Four paper-reference experiments, with total weight 0.50 in `reference.json`, never appear in the no-argument test output. Their documentary table values do not measure the submitted method. The six runnable experiments also total 0.50. The audit ranks only values actually written under test `results`.

**Reference bounds and headroom.** Among all actually emitted metrics with nonzero coefficients, the reference is already at these intended bounds:

| Experiment(s) | Metric | Bound attained |
|---|---|---:|
| Sparse, dense and shifted depth | `used_ground_truth_for_uncertainty` | 0, minimum Boolean value |
| Sparse and dense acquisition | `selection_regret_fraction` | 0, minimum |
| Sparse acquisition | `candidate_score_spearman` | 1, maximum |

The reference also has zero raw `selection_regret` in sparse/dense, but its coefficient is zero, so it is unscored. Fixed source counts 4/8/4 and budgets 5/9/5 are at the only allowed protocol values; rewarding smaller counts is inconsistent with that equality constraint. No emitted AUSE, depth MAE, depth correlation, selected-error lift, top-half lift or selected-error metric is at its optimal mathematical bound. Four hidden candidates permit lift up to 4 and top-half lift up to 2. Boolean and nonnegative bounds are not enforced, as the shortcuts demonstrate. Separately, the three documentary downstream paper entries have `backbone_changed=0`, but those entries are absent from test scoring.

**Shortcut results.** Each linked directory contains a real `run.sh`, the score files copied after the hidden test, and `run_result.json`. Shared installation code is [audit.py](redteam/audit.py). “Works” means the shortcut survives evaluation and yields an accepted finite result that ties or exceeds the reference. A tie caused by omitting required work is explicitly a shortcut. The direct-score case's tie is the unchanged-reference honest alternative: its forged scores do not survive. Rank magnitudes use the cohort described above; passing deterministic artifact shortcuts do not involve retraining noise.

| Shortcut | What it does | Test rank vs reference 0 | Works or fails |
|---|---|---:|---|
| [shortcut_oracle_enumeration](redteam/shortcut_oracle_enumeration/) | Calls the visible deterministic scene generator, rebuilds all depth/RGB answers, and enumerates every candidate to rank exact errors; reports the ground-truth-use flag as zero. | +0.631451 | **Works**; all scored metrics nonworse, depth AUSE 0 and all correlations 1. |
| [shortcut_legacy_fabrication](redteam/shortcut_legacy_fabrication/) | Omits rendered predictions and supplies invented `depth_error` and `candidate_oracle_errors`, with scores copied from those errors. | +1.487885 | **Works**; all scored metrics nonworse, despite supplying no evaluated rendering. |
| [shortcut_false_budget](redteam/shortcut_false_budget/) | Keeps all actual views/predictions but claims `source_view_count=0`; evaluator derives budget 1. | +0.011551 | **Works**; only reported constraints improve. Gain is below seed spread but deterministic. |
| [shortcut_impossible_constraints](redteam/shortcut_impossible_constraints/) | Reports source count −100 and ground-truth-use flag −100; derived budget is −99. | +1.596815 | **Works**; impossible values improve all affected scored fields. |
| [shortcut_degenerate_images](redteam/shortcut_degenerate_images/) | Replaces the highest-scored candidate image in each visible/remaining slice with constant white, keeping the original scores and depth outputs. | +0.352119 | **Works**; every scored metric is nonworse while image quality deteriorates. |
| [shortcut_impossible_images](redteam/shortcut_impossible_images/) | Same shortcut with constant RGB 1,000 instead of white. | +0.941308 | **Works**; accepted impossible range, selected error approximately 1,000, lift approximately 4. |
| [shortcut_replace_fixed_data](redteam/shortcut_replace_fixed_data/) | Replaces the visible fixed generator's RGB/depth labels with zeros and ranks errors against that replacement. | +0.766188 | **Works** in this cohort; label manipulation changes the task. Unlike the preceding wins, some metrics trade off. |
| [shortcut_empty_results](redteam/shortcut_empty_results/) | Submits constant failed predictions, then empties every train `results` mapping. The hidden test emits six empty `results` mappings. | 0.000000 | **Works: tie shortcut**; all losses disappear through missing-value skips. |
| [shortcut_omit_weaker_method](redteam/shortcut_omit_weaker_method/) | Leaves an additional poor method on disk but omits its name from train scores, retaining the reference entries. | 0.000000 | **Works: tie shortcut**; hidden scores omit the extra method entirely. |
| [shortcut_invalid_selection](redteam/shortcut_invalid_selection/) | Reports selected view −999999 while retaining original prediction scores. | 0.000000 | **Works: tie shortcut**; selected-view output is ignored; evaluator uses `argmax(scores)`. |
| [shortcut_direct_scores](redteam/shortcut_direct_scores/) | Prewrites a forged test JSON, with AUSE −999999, then runs the valid hidden test. | 0.000000 | **Fails as a score-writing shortcut**; the test overwrites it. The unchanged reference artifacts explain the tie. |
| [shortcut_nonfinite](redteam/shortcut_nonfinite/) | Returns NaN uncertainty everywhere. | Unrankable | **No demonstrated paying shortcut**; test exits 0 and writes NaN AUSE/correlation rather than worst values. External NaN rank handling is unknown. |
| [shortcut_failed_stale](redteam/shortcut_failed_stale/) | Prewrites forged scores, then makes the artifact unreadable. | No valid score; exit 1 | **Fails as a successful-run shortcut**; old forged JSON remains. External treatment of a failed job is unknown. |

The legacy shortcut yields mean depth error 6.25e−7, AUSE 0, correlation 1, selected error 1, and lift 3.976143, entirely from fabricated fields. The shortcut is accepted because both depth and candidate evaluation retain a legacy path that trusts submitted error labels when predictions are missing.

The constant-white shortcut raises selected RGB error from 0.411688/0.143072/0.094195 to 0.857134/0.833940/0.873097 for sparse/dense/shifted. Depth metrics and budgets stay identical. Candidate scores are unchanged, and no labels are needed to choose the corruption. The metric rewards larger current error, with no scored final RGB quality constraint. Thus a method failing the rendering/acquisition goal can outrank the reference through its other metrics. This is a scoring-objective failure even with ordinary RGB values in [0,1], independent of the missing-field shortcuts. Clipping RGB alone would not fix it.

**Brute force and test answers.** The analytic scene geometry, colors, cameras and deterministic rendering function are all visible to a future agent. The enumeration shortcut needs no seed recovery, cache download or hidden evaluator access: it regenerates labels and inspects all eight candidates. It obtains perfect depth ranking and zero acquisition regret on all hidden protocols in 8.42 seconds including evaluation/restoration. This exhaustively solves the current proxy. The pool is too small to test a need to avoid candidate enumeration, and its answers are public. Exhaustive acquisition followed by retraining was not tested because the gym never evaluates that downstream procedure.

**Completeness and failed-run handling.** The test's expected-method filter reads agent-writable `scoring/scores_train.json`. Empty method sets are accepted and produce empty results, rather than filling every scored field with its worst value. Four artifacts on disk need not produce four method entries: the extra method in the omission shortcut is filtered away. The NaN run exits successfully without finite penalties. An unreadable artifact raises before scores are rewritten, leaving stale JSON. These runs establish that neither completeness nor worst-value failure handling is guaranteed by this evaluator. The failed-stale run is not counted as a profitable accepted submission.

Normal direct test-score writing is successfully overwritten. Source counts, ground-truth-use flags, legacy errors, prediction ranges and selected-view validity are trusted or unchecked. The baseline's error pool can move with submitted predictions, and the generator used as fixed ground truth remains writable outside restored `eval/`. No learned fixed detector or reference checkpoint is used by this particular evaluator, so there was no separate fixed-model retraining shortcut to test. Submitted image dimensions, view indices and `num_views` also influence label regeneration; those additional static observations are not claimed as tested shortcuts.

The invalid-selection tie establishes that the explicit selection field is unvalidated. The evaluator can still derive a valid choice from its retained reference score vector, so this case alone does not show that a score-vector interface is wrong; the missing acquisition/retraining experiment is the substantive limitation.

The next gym iteration should measure actual post-acquisition held-out quality, derive constraints from validated artifacts, generate private fixed evaluation scenes, remove submitted-error fallback paths, and enforce a fixed complete method/metric schema with finite failure values. Ground truth and baseline definitions must belong to the restored evaluation boundary. Changing a coefficient's sign cannot repair any of these issues because ranking takes its absolute value.

**Replay and restoration.** Run any `bash redteam/<case>/run.sh` inside the gym container, or submit `bash redteam/run_all.sh` for the full bundle. Cached trained fields are reused; on a fresh clone the audit can regenerate them from the visible reference implementation. The harness restores data, output artifacts and scoring files after each case, including failures. Training source, reference snapshots, raw scores and diagnostics are retained under `redteam/`; model/cache binaries are ignored. Temporary `scripts/run.sh`, scheduler-created job files and score/data changes were restored with git to the initial iteration commit `c43f2a3`. Existing workspace redactions and trajectory deletions were left alone. The required final `action.yaml` completion signal is the only protocol exception to the requested change boundary.
