#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o class_number_rqf scripts/experiments/class-numbers/class_number_rqf.cu -lm
mkdir -p logs/class-numbers

# 8 GPUs, each handles a range of discriminants
# Target: d = 10^11 to 10^13 (extending beyond known frontier)
for i in $(seq 0 7); do
    START=$((100000000000 + i * 1162500000000))
    END=$((100000000000 + (i + 1) * 1162500000000))
    CUDA_VISIBLE_DEVICES=$i ./class_number_rqf $START $END > logs/class-numbers/gpu${i}.log 2>&1 &
    echo "GPU $i: d=$START..$END (PID $!)"
done
echo "Computing class numbers for d = 10^11 to 10^13 across 8 GPUs."
