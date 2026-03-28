#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o kronecker_compute scripts/experiments/kronecker-coefficients/kronecker_compute.cu
mkdir -p logs/kronecker

echo "=== Kronecker Coefficients for S_n ==="
echo "Phase 1: Full table for n=30 (validation)..."
./kronecker_compute 30 all 2>&1 | tee logs/kronecker/n30.log

echo "Phase 2: GCT-relevant triples for n=80..."
./kronecker_compute 80 gct 2>&1 | tee logs/kronecker/n80_gct.log

echo "Phase 3: Push to n=120..."
./kronecker_compute 120 gct 2>&1 | tee logs/kronecker/n120_gct.log
