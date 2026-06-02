#!/bin/bash
# Score the visible train split and held-out test split from saved artifacts.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh /home/user/outputs scoring/scores_train.json
bash scripts/evaluate_test.sh /home/user/outputs scoring/scores_test.json
