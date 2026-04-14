#!/bin/bash
set -e
cd /home/amsysistestdrive2026/idontknow
R="scripts/experiments/zaremba-density/results"

CUDA_VISIBLE_DEVICES=0 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,3 > "$R/gpu_A123_1e11.log" 2>&1 &
echo "GPU0: {1,2,3}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=1 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,4 > "$R/gpu_A124_1e11.log" 2>&1 &
echo "GPU1: {1,2,4}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=2 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,5 > "$R/gpu_A125_1e11.log" 2>&1 &
echo "GPU2: {1,2,5}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=3 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,6 > "$R/gpu_A126_1e11.log" 2>&1 &
echo "GPU3: {1,2,6}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=4 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,3,4,5 > "$R/gpu_A12345_1e11.log" 2>&1 &
echo "GPU4: {1,2,3,4,5}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=5 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,2,8 > "$R/gpu_A128_1e11.log" 2>&1 &
echo "GPU5: {1,2,8}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=6 nohup stdbuf -oL ./zaremba_density_gpu 100000000000 1,3,5 > "$R/gpu_A135_1e11.log" 2>&1 &
echo "GPU6: {1,3,5}@1e11 PID=$!"

CUDA_VISIBLE_DEVICES=7 nohup stdbuf -oL ./zaremba_density_gpu 1000000000000 1,2,3 > "$R/gpu_A123_1e12.log" 2>&1 &
echo "GPU7: {1,2,3}@1e12 PID=$!"

echo ""
echo "All 8 launched. Monitor: tail -f $R/gpu_A*_1e11.log"
