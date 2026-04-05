#!/bin/bash
# Launch a Zaremba density computation across all 8 GPUs, then merge results.
#
# Usage: ./run_multi_gpu.sh <max_d> <digits> [num_gpus]
# Example: ./run_multi_gpu.sh 100000000000 1,2,3 8
#
set -e
cd /home/amsysistestdrive2026/idontknow

MAX_D="$1"
DIGITS="$2"
NUM_GPUS="${3:-8}"
BINARY="./zaremba_density_gpu"
RESULTS="scripts/experiments/zaremba-density/results"
BITSET_PREFIX="$RESULTS/bitset_A${DIGITS}_${MAX_D}"

# Replace commas in prefix for filename safety
BITSET_PREFIX=$(echo "$BITSET_PREFIX" | tr ',' '_')

echo "========================================"
echo "Multi-GPU Zaremba Density"
echo "Range: 1 to $MAX_D"
echo "Digits: {$DIGITS}"
echo "GPUs: $NUM_GPUS"
echo "========================================"
echo ""

# Launch all shards in parallel
PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_OUT="${BITSET_PREFIX}.shard${gpu}.bin"
    LOG="$RESULTS/shard_${gpu}.log"
    echo "GPU $gpu: shard $gpu/$NUM_GPUS -> $SHARD_OUT"
    CUDA_VISIBLE_DEVICES=$gpu nohup stdbuf -oL \
        $BINARY $MAX_D $DIGITS --shard $gpu $NUM_GPUS --bitset-out "$SHARD_OUT" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS shards launched. Waiting..."
echo ""

# Wait for all shards, report as they finish
FAILED=0
for i in $(seq 0 $((NUM_GPUS - 1))); do
    pid=${PIDS[$i]}
    if wait $pid; then
        echo "  GPU $i (PID $pid): DONE"
    else
        echo "  GPU $i (PID $pid): FAILED (exit code $?)"
        FAILED=1
    fi
done

if [ "$FAILED" = "1" ]; then
    echo "ERROR: some shards failed. Check logs in $RESULTS/shard_*.log"
    exit 1
fi

echo ""
echo "All shards complete. Merging bitsets..."
echo ""

# Merge — runs on CPU, reads all shard files, ORs them, prints results
$BINARY --merge $MAX_D $DIGITS $NUM_GPUS "$BITSET_PREFIX"
