#!/bin/bash
# Build the frozen benchmark instances (GPU job, ~1 h on one GPU).
#   data/train_assets/ - 16 public development scenes (seeds 0-15)
#   eval/test/assets/  - 32 hidden test scenes (seeds held only in eval/test/)
# Existing instance sets are kept unless FORCE=1.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

if [ "${FORCE:-0}" = "1" ] || [ ! -f data/train_assets/manifest.json ]; then
  rm -rf data/train_assets
  python -m data.build_assets --seeds 0-15 --out data/train_assets
fi
if [ "${FORCE:-0}" = "1" ] || [ ! -f eval/test/assets/manifest.json ]; then
  rm -rf eval/test/assets
  python eval/test/build_assets.py
fi
