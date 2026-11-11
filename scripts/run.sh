#!/bin/bash
# Hardening job: (1) smoke-test the full pipeline on tiny instances, (2) build the
# frozen benchmark instances on the GPU, (3) register and score the reference,
# (4) run the hardening audit (variants + shortcuts) through both evaluators.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user

python - <<'PY'
import numpy, torch
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB", flush=True)
PY

echo "=== smoke test ==="
rm -rf tmp/smoke
python -m data.build_assets --seeds 0-1 --out tmp/smoke/train_assets --steps-scale 0.01
python eval/test/build_assets.py --count 2 --out tmp/smoke/test_assets --steps-scale 0.01
python redteam/hardening/run_checks.py --assets-train tmp/smoke/train_assets --assets-test tmp/smoke/test_assets \
  --expected 2 2 --work tmp/smoke/work --out tmp/smoke

echo "=== full build ==="
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA is required for the full build')"
bash scripts/build_assets.sh
du -sh data/train_assets eval/test/assets

echo "=== reference ==="
bash scripts/method.sh > /dev/null
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null
python - <<'PY'
import json
for split in ("train", "test"):
    s = json.load(open(f"scoring/scores_{split}.json"))
    print(split, json.dumps(s["experiments"], indent=1), s["_methods"], flush=True)
PY

echo "=== hardening audit ==="
python redteam/hardening/run_checks.py

python validate.py || true
