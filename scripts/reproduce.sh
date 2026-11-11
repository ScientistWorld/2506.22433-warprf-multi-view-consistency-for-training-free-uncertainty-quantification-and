#!/bin/bash
# Reproduce the reference end to end: build the frozen instances if they are
# missing (GPU), register WarpRF, and score it with both evaluators.

set -euo pipefail

cd /home/user

bash scripts/build_assets.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
