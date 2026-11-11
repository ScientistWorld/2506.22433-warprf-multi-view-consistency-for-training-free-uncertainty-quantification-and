# Scientist Constraints

## Backbone

The radiance fields are trained by the benchmark with one fixed recipe and are
frozen. Methods receive their renderings and may not replace, retrain or
fine-tune them, and may not substitute renderings from another model. The
uncertainty must describe the provided renderings.

## Data

Use only the arrays the evaluator passes to the method for each instance, plus
general knowledge and the public development split (`data/train_assets/`) for
designing and tuning the method. Do not read evaluator files (`eval/`), score
files, or anything else on disk at prediction time, and do not add external
images, depth maps, meshes or pretrained scene priors.

## View Budget

The initial views, the candidate pool and the two-acquisition budget are fixed.
A method changes only how candidates are scored; the evaluator performs every
acquisition from those scores.

## Compute

Each prediction round over all scenes must finish within 1800 s; slower runs
fail. The hidden test never retrains anything on behalf of a method.

## Evaluation

Held-out views, ground-truth depth and unacquired candidate photographs are
used only by the evaluator. Report through the registration interface only;
anything else a method writes is ignored.

## Compliance Checklist

- [ ] The method is registered as `outputs/<name>/predict.json` and runs through `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` without errors.
- [ ] Predictions are computed only from the arrays passed for the current instance.
- [ ] No evaluator files, answers or score files are read at prediction time.
- [ ] No radiance field is retrained or replaced.
- [ ] Every round finishes within the time limit.
