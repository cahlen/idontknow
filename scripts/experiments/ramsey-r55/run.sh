#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o ramsey_search scripts/experiments/ramsey-r55/ramsey_search.cu -lcurand
mkdir -p logs/ramsey

echo "=== Phase 1: Verify known lower bound (n=43) ==="
./ramsey_search 43 100000 1000000 2>&1 | tee logs/ramsey/n43.log

echo ""
echo "=== Phase 2: Attack n=44 (would improve lower bound) ==="
./ramsey_search 44 1000000 10000000 2>&1 | tee logs/ramsey/n44.log

echo ""
echo "=== Phase 3: Long run on n=44 if Phase 2 failed ==="
./ramsey_search 44 10000000 100000000 2>&1 | tee logs/ramsey/n44_long.log
