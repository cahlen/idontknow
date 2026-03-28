#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o mersenne_check scripts/experiments/mersenne-sieve/mersenne_check.cu
mkdir -p logs/mersenne
echo "Sieving Mersenne candidates up to p=10,000,000 with factor limit 10^12..."
./mersenne_check 10000000 1000000000000 2>&1 | tee logs/mersenne/run.log
