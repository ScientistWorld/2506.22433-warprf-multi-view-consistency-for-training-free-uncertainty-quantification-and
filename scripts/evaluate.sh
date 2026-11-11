#!/bin/bash
# Standard evaluation entry point: score every registered method
# (outputs/<name>/predict.json) on the visible train instances and write
# scoring/scores.json. Uses only code in eval/.
#
# Usage: bash scripts/evaluate.sh

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring
python eval/train/evaluate.py --scores scoring/scores.json
