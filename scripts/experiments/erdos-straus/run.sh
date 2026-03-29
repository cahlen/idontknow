#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
MAX_M="${1:-100}"
echo "Compiling erdos_straus (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o erdos_straus scripts/experiments/erdos-straus/erdos_straus.cu -lm
echo "Done."
mkdir -p scripts/experiments/erdos-straus/results
echo ""
echo "=== Erdos-Straus f(p) for primes up to ${MAX_M}M ==="
echo ""
./erdos_straus "$MAX_M" 2>&1 | tee "scripts/experiments/erdos-straus/results/run_${MAX_M}M.log"
