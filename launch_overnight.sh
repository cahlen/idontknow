#!/bin/bash
# Overnight Zaremba density runs — one experiment per GPU.
# Ordered by expected completion time (fastest first).
# Each writes progress + checkpoints to logs/results dir.
set -e
cd /home/amsysistestdrive2026/idontknow
R="scripts/experiments/zaremba-density/results"
B="./zaremba_density_gpu"

echo "$(date): Starting overnight Zaremba density batch"
echo ""

# GPU 0: {1,2,8} @ 10^11 — fastest (large digits)
CUDA_VISIBLE_DEVICES=0 nohup stdbuf -oL $B 100000000000 1,2,8 > "$R/gpu_A128_1e11.log" 2>&1 &
echo "GPU 0: {1,2,8}@1e11 PID=$!"

# GPU 1: {1,3,5} @ 10^11 — fast
CUDA_VISIBLE_DEVICES=1 nohup stdbuf -oL $B 100000000000 1,3,5 > "$R/gpu_A135_1e11.log" 2>&1 &
echo "GPU 1: {1,3,5}@1e11 PID=$!"

# GPU 2: {1,2,6} @ 10^11 — medium
CUDA_VISIBLE_DEVICES=2 nohup stdbuf -oL $B 100000000000 1,2,6 > "$R/gpu_A126_1e11.log" 2>&1 &
echo "GPU 2: {1,2,6}@1e11 PID=$!"

# GPU 3: {1,2,5} @ 10^11 — medium
CUDA_VISIBLE_DEVICES=3 nohup stdbuf -oL $B 100000000000 1,2,5 > "$R/gpu_A125_1e11.log" 2>&1 &
echo "GPU 3: {1,2,5}@1e11 PID=$!"

# GPU 4: {1,2,4} @ 10^11 — slower
CUDA_VISIBLE_DEVICES=4 nohup stdbuf -oL $B 100000000000 1,2,4 > "$R/gpu_A124_1e11.log" 2>&1 &
echo "GPU 4: {1,2,4}@1e11 PID=$!"

# GPU 5: {1,2,3,4,5} @ 10^11 — Zaremba digits
CUDA_VISIBLE_DEVICES=5 nohup stdbuf -oL $B 100000000000 1,2,3,4,5 > "$R/gpu_A12345_1e11.log" 2>&1 &
echo "GPU 5: {1,2,3,4,5}@1e11 PID=$!"

# GPU 6: {1,2,3} @ 10^11 — slowest (digit-1 heavy), has checkpoint
CUDA_VISIBLE_DEVICES=6 nohup stdbuf -oL $B 100000000000 1,2,3 > "$R/gpu_A123_1e11.log" 2>&1 &
echo "GPU 6: {1,2,3}@1e11 PID=$!"

# GPU 7: {1,2,3} @ 10^12 — longest, has checkpoint
CUDA_VISIBLE_DEVICES=7 nohup stdbuf -oL $B 1000000000000 1,2,3 > "$R/gpu_A123_1e12.log" 2>&1 &
echo "GPU 7: {1,2,3}@1e12 PID=$!"

echo ""
echo "All 8 launched. Checkpoints every 5 min in $R/checkpoint_*.bin"
echo "Check progress: for f in $R/gpu_A*_1e1*.log; do echo \"\$f:\"; tail -1 \"\$f\"; done"
echo "$(date): Batch started"
