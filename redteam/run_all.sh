#!/bin/bash
set -euo pipefail
cd /home/user
# No installations, network activity, GPU code or evaluator process hooks.
# Single-threaded CPU training matches the original CPU reproduction regime.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m redteam.audit --all
