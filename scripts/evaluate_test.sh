#!/bin/bash
# Score every registered method on the hidden test instances.
#
# Usage: bash scripts/evaluate_test.sh        (no arguments)
#
# Methods are registered as outputs/<name>/predict.json (see briefing/evaluation.md).
# The evaluator runs each method's command on the hidden inputs held only in
# eval/test/assets/, derives every acquisition itself, and writes all scores to
# scoring/scores_test.json. Any stale score file is deleted first.

set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

rm -f scoring/scores_test.json
python eval/test/evaluate.py
