#!/bin/bash
# Baselines are owned by the evaluator, not by a method: every evaluation run
# recomputes them on the same instances as the methods it scores.
#   random_uncertainty - fixed pseudo-random pixel uncertainty
#   random_view        - exact expectation of uniformly random sequential acquisition
#   farthest_view      - acquire the candidate camera farthest from all current cameras
#
# Usage: bash scripts/baseline.sh   (prints the baseline rows of the train split)

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p tmp
python eval/train/evaluate.py --scores tmp/baseline_scores.json > /dev/null
python - <<'PY'
import json
s = json.load(open("tmp/baseline_scores.json"))
for exp, data in s["experiments"].items():
    for name in ("random_uncertainty", "random_view", "farthest_view"):
        if name in data["results"]:
            print(exp, name, data["results"][name])
PY
