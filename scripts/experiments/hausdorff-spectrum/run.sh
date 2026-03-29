#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"

MAX_DIGIT="${1:-10}"
N="${2:-40}"

echo "Compiling hausdorff_spectrum (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o hausdorff_spectrum \
    scripts/experiments/hausdorff-spectrum/hausdorff_spectrum.cu -lm
echo "Done."

mkdir -p scripts/experiments/hausdorff-spectrum/results

echo ""
echo "=== Computing Hausdorff dimension spectrum for {1,...,$MAX_DIGIT} ==="
echo "=== Chebyshev order N=$N ==="
echo ""
./hausdorff_spectrum "$MAX_DIGIT" "$N" 2>&1 | tee "scripts/experiments/hausdorff-spectrum/results/run_n${MAX_DIGIT}.log"
