#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"

N_BILLIONS="${1:-1}"

echo "Compiling flint_hills (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o flint_hills \
    scripts/experiments/flint-hills/flint_hills.cu -lm
echo "Done."

mkdir -p scripts/experiments/flint-hills/results

echo ""
echo "=== Flint Hills Series: S_N to N = ${N_BILLIONS} billion ==="
echo ""
./flint_hills "$N_BILLIONS" 2>&1 | tee "scripts/experiments/flint-hills/results/run_${N_BILLIONS}B.log"
