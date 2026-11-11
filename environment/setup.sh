#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# The overlay may be recreated on container rebuilds — a venv inside
# it will be lost and all subsequent jobs will fail.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).

set -e

# CUDA build of torch: the frozen benchmark instances are built on a GPU
# (scripts/build_assets.sh). Evaluation and the reference method need only numpy.
if ! python -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  uv pip install --system --reinstall-package torch --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1"
fi
uv pip install --system numpy
