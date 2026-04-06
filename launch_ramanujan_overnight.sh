#!/bin/bash
# Overnight Ramanujan Machine chain on GPU 6
# Runs: deg4_range3 (30s) → deg4_range6 (3.8h) → deg3_range11 (2.2h) ≈ 6.5h total
#
# All output logged to files. Hits saved to CSV for HuggingFace upload.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
BIN="/home/amsysistestdrive2026/idontknow/scripts/experiments/ramanujan-machine/ramanujan_gpu"
RESULTS="/home/amsysistestdrive2026/idontknow/scripts/experiments/ramanujan-machine/results"
cd /home/amsysistestdrive2026/idontknow

echo "[$(date)] Starting Ramanujan overnight chain on GPU 6"

# Run 1: deg4 range3 (~30s, fills gap)
echo "[$(date)] === Run 1/3: deg4 range3 ==="
stdbuf -oL "$BIN" 4 3 200 0 2>&1 | tee "$RESULTS/run_deg4_range3.log"
echo "[$(date)] === Run 1/3 complete ==="

# Run 2: deg4 range6 (~3.8 hrs, extends deg4 coverage)
echo "[$(date)] === Run 2/3: deg4 range6 ==="
stdbuf -oL "$BIN" 4 6 200 0 2>&1 | tee "$RESULTS/run_deg4_range6.log"
echo "[$(date)] === Run 2/3 complete ==="

# Run 3: deg3 range11 (~2.2 hrs, extends deg3 coverage)
echo "[$(date)] === Run 3/3: deg3 range11 ==="
stdbuf -oL "$BIN" 3 11 200 0 2>&1 | tee "$RESULTS/run_deg3_range11.log"
echo "[$(date)] === Run 3/3 complete ==="

echo "[$(date)] All 3 runs complete. Results in $RESULTS/"
echo "Files:"
ls -lh "$RESULTS"/hits_deg4_range3.csv "$RESULTS"/hits_deg4_range6.csv "$RESULTS"/hits_deg3_range11.csv 2>/dev/null
