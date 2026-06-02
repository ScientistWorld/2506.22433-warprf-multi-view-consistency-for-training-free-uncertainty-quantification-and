#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This paper-specific reproduction uses no external downloads. If future
# variants add public datasets or models, fetch them into data/downloads/ with
# idempotent commands here.

set -e

cd /home/user

mkdir -p data/downloads
echo "WarpRF reproduction uses a tracked analytic synthetic scene and downloads no external data."
