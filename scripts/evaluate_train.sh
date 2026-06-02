#!/bin/bash
# Evaluate the visible train slice of saved method artifacts.
#
# Mirrors scripts/evaluate.sh's positional contract:
#   ./evaluate_train.sh [artifact_or_outputs_dir] [scores_path] [experiment_prefix] [merge_existing]
#
# With no artifact argument, all outputs/*/warprf_outputs.npz artifacts are
# enumerated and scored. The evaluator reads artifacts only; it does not train
# or import from method/.

set -e

cd /home/user

mkdir -p scoring
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

ARTIFACT_INPUT="${1:-outputs}"
SCORES="${2:-scoring/scores_train.json}"
EXPERIMENT_PREFIX="${3:-}"
MERGE="${4:-false}"

CMD=(python eval/train/evaluate.py --artifact "$ARTIFACT_INPUT" --scores "$SCORES")
if [ -n "$EXPERIMENT_PREFIX" ]; then
  CMD+=(--experiment-prefix "$EXPERIMENT_PREFIX")
fi
if [ "$MERGE" = "true" ]; then
  CMD+=(--merge-existing)
fi

"${CMD[@]}"
