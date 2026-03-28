#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Collatz Conjecture — 8×B200 GPU verification
#
# Verifies that the Collatz sequence reaches 1 for all n in [1, 10^12].
# 8 GPUs × 125 billion each = 1 trillion values.
#
# Also records:
#   - Maximum stopping time (steps to reach 1) and which n achieves it
#   - Maximum trajectory height and which n achieves it
#
# Usage: ./scripts/experiments/collatz/run.sh
# ============================================================================

cd "$(dirname "$0")/../../.."

echo "=============================================="
echo "  Collatz Conjecture — 8×B200 GPU Verification"
echo "=============================================="
echo ""

# Compile
echo "Compiling CUDA kernel..."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o collatz_verify scripts/experiments/collatz/collatz_verify.cu
echo "Done."
echo ""

mkdir -p logs/collatz

# Launch: 8 GPUs, 125B each = 1 trillion total
TOTAL=1000000000000  # 10^12
PER_GPU=$((TOTAL / 8))

for i in $(seq 0 7); do
    START=$((i * PER_GPU + 1))
    END=$(((i + 1) * PER_GPU))
    CUDA_VISIBLE_DEVICES=$i ./collatz_verify $START $END > logs/collatz/gpu${i}.log 2>&1 &
    echo "GPU $i: n=$START..$END (PID $!)"
done

echo ""
echo "1 trillion values launched across 8 B200 GPUs."
echo "Logs: logs/collatz/gpu*.log"
echo ""
echo "Monitor: for i in \$(seq 0 7); do echo \"GPU \$i: \$(tail -1 logs/collatz/gpu\${i}.log)\"; done"
