#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

OUT_DIR="${1:-outputs/warprf}"
PREFIX="${2:-tiny_nerf}"
SPLIT_OFFSET="${3:-0}"
MERGE_SCORES="${4:-false}"
STEPS="${WARPRF_STEPS:-1200}"

mkdir -p "$OUT_DIR"

python -m method.tiny_nerf_warprf \
  --output-dir "$OUT_DIR" \
  --steps "$STEPS" \
  --batch-size "${WARPRF_BATCH_SIZE:-1024}" \
  --width "${WARPRF_WIDTH:-48}" \
  --height "${WARPRF_HEIGHT:-48}" \
  --split-offset "$SPLIT_OFFSET"

bash scripts/evaluate.sh "$OUT_DIR/warprf_outputs.npz" scoring/scores.json "$PREFIX" "$MERGE_SCORES"
