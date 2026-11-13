#!/bin/bash
# Build the frozen benchmark instances (GPU job; ~2.5 h on two A100s, ~5 h on one).
#   data/train_assets/ - 16 public development scenes (seeds 0-15)
#   eval/test/assets/  - 40 hidden test scenes (seeds held only in eval/test/)
# Resumable: finished scenes are kept, so rerunning continues an interrupted
# build. FORCE=1 deletes both instance sets first.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

if [ "${FORCE:-0}" = "1" ]; then
  rm -rf data/train_assets eval/test/assets
fi
mkdir -p logs

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$NGPU" -ge 2 ]; then
  # GPU 0: the 16 train scenes, then test scenes 28-39; GPU 1: test scenes 0-27.
  ( python -m data.build_assets --seeds 0-15 --out data/train_assets --device cuda:0 \
    && python eval/test/build_assets.py --indices 28-39 --device cuda:0 ) > logs/build_gpu0.log 2>&1 &
  PID0=$!
  python eval/test/build_assets.py --indices 0-27 --device cuda:1 > logs/build_gpu1.log 2>&1 &
  PID1=$!
  STATUS=0
  wait "$PID0" || STATUS=1
  wait "$PID1" || STATUS=1
  tail -n 3 logs/build_gpu0.log logs/build_gpu1.log
  [ "$STATUS" = "0" ] || { echo "asset build failed"; exit 1; }
else
  python -m data.build_assets --seeds 0-15 --out data/train_assets
  python eval/test/build_assets.py
fi

# Manifests are written once every scene of a split exists.
python -m data.build_assets --seeds 0-15 --out data/train_assets --manifest-only
python eval/test/build_assets.py --manifest-only
