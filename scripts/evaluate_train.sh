#!/bin/bash
# Score every registered method on the visible train instances (data/train_assets/).
#
# Usage: bash scripts/evaluate_train.sh        (no arguments)
#
# Same protocol, metrics and baselines as the hidden test; writes
# scoring/scores_train.json.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

rm -f scoring/scores_train.json
python eval/train/evaluate.py
