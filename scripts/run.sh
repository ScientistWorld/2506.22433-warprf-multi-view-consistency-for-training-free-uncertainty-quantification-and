#!/bin/bash
# Reproduce the reference end to end and re-run the hardening audit.
#   0. smoke test: 2+2 public scenes at 1% of the training steps through the
#      builder, both evaluators and the audit (catches code errors in minutes).
#   1. scripts/reproduce.sh: build the frozen instances if missing (GPU, resumable),
#      register WarpRF, score it on the train and hidden test splits.
#   2. redteam/hardening/run_checks.py: score variants and shortcut payloads
#      through both unmodified evaluators (CPU).

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [ ! -f eval/test/assets/manifest.json ]; then
  rm -rf tmp/smoke
  python -m data.build_assets --seeds 900-901 --out tmp/smoke/train_assets --steps-scale 0.01 --group 64 > logs/smoke.log 2>&1
  python -m data.build_assets --seeds 902-903 --out tmp/smoke/test_assets --steps-scale 0.01 --group 64 >> logs/smoke.log 2>&1
  python redteam/hardening/run_checks.py --assets-train tmp/smoke/train_assets --assets-test tmp/smoke/test_assets \
    --expected 2 2 --work tmp/smoke/work --out tmp/smoke >> logs/smoke.log 2>&1 || { tail -n 60 logs/smoke.log; exit 1; }
  grep -E "^\| (warprf|shortcut_escape|weak_constant|broken)|Escape routes|isolation=" tmp/smoke/results.md || true
  echo "smoke test passed"
fi

bash scripts/reproduce.sh > logs/reproduce.log 2>&1 || { tail -n 60 logs/reproduce.log; exit 1; }
tail -n 20 logs/reproduce.log
python - <<'PY'
import json
for split in ("train", "test"):
    s = json.load(open(f"scoring/scores_{split}.json"))
    print(split, json.dumps(s["experiments"], indent=1), s["_methods"], s.get("_isolation"), flush=True)
PY

python redteam/hardening/run_checks.py
python validate.py || true
