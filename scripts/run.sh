#!/bin/bash
# Regenerate the scaled WarpRF reproduction and score visible/held-out splits.

set -euo pipefail

cd /home/user

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh /home/user/outputs scoring/scores_train.json
bash scripts/evaluate_test.sh /home/user/outputs scoring/scores_test.json
python validate.py --compare
