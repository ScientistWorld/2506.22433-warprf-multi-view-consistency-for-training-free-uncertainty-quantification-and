#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This benchmark needs no external data or pretrained weights: scenes are
# generated procedurally by data/scenes.py and the frozen benchmark instances
# are built by scripts/build_assets.sh (GPU) and tracked in git.

set -e

cd /home/user

mkdir -p data/downloads
echo "Nothing to download: the benchmark uses procedurally generated scenes."
