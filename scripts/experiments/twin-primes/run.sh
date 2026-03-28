#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
echo "Compiling..."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o twin_prime_count scripts/experiments/twin-primes/twin_prime_count.cu
mkdir -p logs/twin-primes
echo "Counting twin primes up to 10^12..."
./twin_prime_count 1000000000000 2>&1 | tee logs/twin-primes/run.log
