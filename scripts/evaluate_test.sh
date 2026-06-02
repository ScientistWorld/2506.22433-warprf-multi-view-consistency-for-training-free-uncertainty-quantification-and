#!/bin/bash
# Evaluate the held-out test slice of saved method artifacts.
#
# Mirrors scripts/evaluate.sh's positional contract:
#   ./evaluate_test.sh [artifact_or_outputs_dir] [scores_path] [experiment_prefix] [merge_existing]
#
# With no artifact argument, all outputs/*/warprf_outputs.npz artifacts are
# enumerated. When scoring/scores_train.json exists, the test evaluator scores
# exactly the same experiment/method keys on the held-out slice.

set -e

cd /home/user

mkdir -p scoring
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

ARTIFACT_INPUT="${1:-outputs}"
SCORES="${2:-scoring/scores_test.json}"
EXPERIMENT_PREFIX="${3:-}"
MERGE="${4:-false}"
TRAIN_SCORES="${TRAIN_SCORES:-scoring/scores_train.json}"

CMD=(python eval/test/evaluate.py --artifact "$ARTIFACT_INPUT" --scores "$SCORES" --expected-scores "$TRAIN_SCORES")
if [ -n "$EXPERIMENT_PREFIX" ]; then
  CMD+=(--experiment-prefix "$EXPERIMENT_PREFIX")
fi
if [ "$MERGE" = "true" ]; then
  CMD+=(--merge-existing)
fi

"${CMD[@]}"
