#!/bin/bash
# Design study for the acquisition rebuild (one GPU, ~30 min); see
# redteam/hardening/design_study.py. Also probes namespace isolation in the
# job container.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python redteam/hardening/design_study.py
