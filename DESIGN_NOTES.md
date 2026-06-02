# Train/Test Evaluation Design Notes

## What I Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side for copy-paste errors. Train routes to `eval/train/evaluate.py` and defaults to `scoring/scores_train.json`; test routes to `eval/test/evaluate.py` and defaults to `scoring/scores_test.json`.
- Checked that neither slice hardcodes a single future method name. Both scripts default to the `outputs` directory, and the Python evaluators discover `warprf_outputs.npz` artifacts recursively.
- Checked for direct train/test cross-imports with grep. None were found.
- Checked for early `scoring/scores_train.json` or `scoring/scores_test.json` stubs. None were present.
- Checked metric names for paper-agreement patterns. None were found.

## Changes Made

- Added active-view triangulation metrics to both split evaluators:
  - `candidate_score_spearman`: rank correlation between submitted candidate scores and evaluator-computed candidate rendering errors.
  - `top_half_error_lift`: mean exposed error among the top half of scored candidates divided by the full candidate mean.
- These metrics make the active-view evaluation harder to game by tuning only the visible best-candidate choice. A real improvement should rank the candidate pool well on both slices; a shortcut that memorizes or exploits visible candidate positions should lose ranking quality on the held-out odd-position slice.
- Updated `scoring/reference.json` with definitions and descriptions for the new active-view metrics.
- Added one-sentence `description` fields to existing metric definitions so the scoring schema explains the scientific quantity each metric measures.

## Runtime Concerns For Next Iteration

- I did not run evaluators or generate score files, per the design-only instruction. The next runner should confirm that the added active-view metrics are finite for all candidate pools in the produced artifacts.
- The train/test split still uses even vs. odd held-out and candidate positions within each artifact. This is lightweight and matched across slices, but the candidate pools may be small enough that rank metrics are somewhat noisy.
- The split evaluators are artifact-based and do not import from `method/`, but they reconstruct analytic labels through `data.synthetic_radiance.render_dataset`. That is intentional for evaluator-owned labels, but future methods must keep writing `test_rendered_depths`, `test_indices`, `candidate_rendered_images`, `candidate_indices`, `candidate_scores`, and metadata fields consistently.
- A legacy top-level `eval/metrics.py` single-slice evaluator still exists and is used by the original `scripts/evaluate.sh`. I did not edit it because this iteration's scope was the train/test split files, split scripts, scoring reference, and this handoff. Future split validation should focus on `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`.
