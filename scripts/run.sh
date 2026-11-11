#!/bin/bash
# Reproduce the reference end to end and re-run the hardening audit.
#   1. scripts/reproduce.sh: build the frozen instances if missing (GPU, ~1.5 h),
#      register WarpRF, score it on the train and hidden test splits.
#   2. redteam/hardening/run_checks.py: score variants and shortcut payloads
#      through both unmodified evaluators (CPU, ~10 min).

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user

bash scripts/reproduce.sh > /dev/null
python - <<'PY'
import json
for split in ("train", "test"):
    s = json.load(open(f"scoring/scores_{split}.json"))
    print(split, json.dumps(s["experiments"], indent=1), s["_methods"], flush=True)
PY

python redteam/hardening/run_checks.py
python validate.py || true
