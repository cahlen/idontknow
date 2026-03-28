#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
echo "Compiling..."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o goldbach_verify scripts/experiments/goldbach/goldbach_verify.cu
mkdir -p logs/goldbach

# 8 GPUs, 125B even numbers each = 1 trillion even numbers
for i in $(seq 0 7); do
    START=$((i * 250000000000 + 4))
    END=$(((i + 1) * 250000000000))
    CUDA_VISIBLE_DEVICES=$i ./goldbach_verify $START $END > logs/goldbach/gpu${i}.log 2>&1 &
    echo "GPU $i: even n=$START..$END (PID $!)"
done
echo "1 trillion even numbers across 8 GPUs. Monitor: for i in \$(seq 0 7); do tail -1 logs/goldbach/gpu\${i}.log; done"
