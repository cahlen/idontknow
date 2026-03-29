#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
MAX_DIGIT="${1:-10}"
N="${2:-40}"
echo "Compiling lyapunov_spectrum (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o lyapunov_spectrum scripts/experiments/lyapunov-spectrum/lyapunov_spectrum.cu -lm
echo "Done."
mkdir -p scripts/experiments/lyapunov-spectrum/results
./lyapunov_spectrum "$MAX_DIGIT" "$N" 2>&1 | tee "scripts/experiments/lyapunov-spectrum/results/run_n${MAX_DIGIT}.log"
