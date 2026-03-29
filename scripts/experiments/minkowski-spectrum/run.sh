#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
A_MAX="${1:-50}"
N="${2:-40}"
echo "Compiling minkowski_spectrum (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o minkowski_spectrum scripts/experiments/minkowski-spectrum/minkowski_spectrum.cu -lm
echo "Done."
mkdir -p scripts/experiments/minkowski-spectrum/results
./minkowski_spectrum "$A_MAX" "$N" 2>&1 | tee scripts/experiments/minkowski-spectrum/results/run.log
