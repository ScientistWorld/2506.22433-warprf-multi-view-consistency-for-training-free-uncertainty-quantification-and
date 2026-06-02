#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

python -m baseline.random_uncertainty

if [ -e outputs/warprf/warprf_outputs.npz ]; then
  bash scripts/evaluate.sh outputs/warprf/warprf_outputs.npz
else
  echo "No WarpRF artifact found. Run scripts/method.sh first; the evaluator computes the random baseline from the same artifact."
fi
