#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
echo "Compiling..."
export PATH="/usr/local/cuda/bin:$PATH"
nvcc -O3 -arch=sm_100a -o mandelbrot_deep scripts/experiments/mandelbrot-depth/mandelbrot_deep.cu
mkdir -p logs/mandelbrot output/mandelbrot

echo "=== Rendering 5 zoom levels ==="

# Full set overview
./mandelbrot_deep 8192 8192 -0.5 0.0 1.0 10000 output/mandelbrot/full_set.raw 2>&1 | tee logs/mandelbrot/full_set.log

# Seahorse valley
./mandelbrot_deep 8192 8192 -0.7463 0.1102 500.0 50000 output/mandelbrot/seahorse.raw 2>&1 | tee logs/mandelbrot/seahorse.log

# Deep zoom: Misiurewicz point
./mandelbrot_deep 8192 8192 -0.74364388703 0.13182590421 1e8 100000 output/mandelbrot/misiurewicz_1e8.raw 2>&1 | tee logs/mandelbrot/misiurewicz_1e8.log

# Ultra deep: 10^12 zoom
./mandelbrot_deep 8192 8192 -0.74364388703 0.13182590421 1e12 500000 output/mandelbrot/deep_1e12.raw 2>&1 | tee logs/mandelbrot/deep_1e12.log

# Extreme: 10^15 zoom (pushing double-double precision)
./mandelbrot_deep 8192 8192 -0.7436438870371587 0.1318259042145380 1e15 1000000 output/mandelbrot/extreme_1e15.raw 2>&1 | tee logs/mandelbrot/extreme_1e15.log

echo "Done. Raw files in output/mandelbrot/"
