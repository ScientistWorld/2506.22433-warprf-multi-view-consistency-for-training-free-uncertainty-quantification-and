#!/bin/bash
# Register the paper's method (WarpRF, method/predict.py) and score it on the
# visible train instances. WarpRF is training-free: registration only records the
# command the evaluators run on each instance.
#
# Usage: bash scripts/method.sh

set -euo pipefail

cd /home/user

mkdir -p outputs/warprf
cat > outputs/warprf/predict.json <<'JSON'
{"command": ["python", "-m", "method.predict"]}
JSON

bash scripts/evaluate.sh
