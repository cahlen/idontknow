#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Riemann Zeta Zeros — 8×B200 GPU verification
#
# Verify that the first N non-trivial zeros of the Riemann zeta function
# lie on the critical line Re(s) = 1/2.
#
# Uses the Hardy Z-function with Riemann-Siegel formula, evaluated on a
# fine grid to detect sign changes. Zero count validated against the
# Riemann-von Mangoldt formula.
#
# Usage: ./scripts/experiments/riemann-zeta/run.sh [num_zeros] [grid_density]
#   num_zeros: how many zeros to verify (default: 10_000_000)
#   grid_density: grid points per unit t (default: 20, increase for precision)
# ============================================================================

cd "$(dirname "$0")/../../.."

NUM_ZEROS="${1:-10000000}"
GRID_DENSITY="${2:-20}"

echo "=============================================="
echo "  Riemann Zeta Zeros — 8×B200 GPU Verification"
echo "=============================================="
echo ""
echo "Target: $NUM_ZEROS zeros"
echo "Grid density: $GRID_DENSITY points per unit t"
echo ""

# Compile
echo "Compiling CUDA kernel..."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o riemann_zeros scripts/experiments/riemann-zeta/riemann_zeros.cu -lm
echo "Done."
echo ""

mkdir -p logs/riemann

# Run (the kernel internally distributes across GPUs)
./riemann_zeros "$NUM_ZEROS" "$GRID_DENSITY" 2>&1 | tee logs/riemann/run.log
