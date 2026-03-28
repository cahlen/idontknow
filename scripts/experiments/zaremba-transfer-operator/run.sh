#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"

echo "Compiling transfer operator..."
nvcc -O3 -arch=sm_100a -o transfer_op \
    scripts/experiments/zaremba-transfer-operator/transfer_operator.cu \
    -lcusolver -lcublas -lm
echo "Done."

mkdir -p logs/transfer-operator

echo ""
echo "=== Phase 1: Hausdorff dimension (N=200) ==="
./transfer_op 200 1 2>&1 | tee logs/transfer-operator/phase1.log

echo ""
echo "=== Phase 2: Congruence spectral gaps (N=20, m up to 30) ==="
./transfer_op 20 2 30 2>&1 | tee logs/transfer-operator/phase2.log
