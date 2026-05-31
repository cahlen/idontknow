#!/usr/bin/env bash
# CFD chaotic advection — standard map Lyapunov sweep (RTX 5090)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:${PATH:-}"

N_K="${1:-512}"
N_IC="${2:-4096}"
N_ITERS="${3:-20000}"
K_MAX="${4:-5.0}"

echo "Compiling standard_map_lyapunov (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o standard_map_lyapunov \
    scripts/experiments/cfd-chaotic-advection/standard_map_lyapunov.cu -lm
echo "Done."

mkdir -p scripts/experiments/cfd-chaotic-advection/results

LOG="scripts/experiments/cfd-chaotic-advection/results/run_k${N_K}_ic${N_IC}_iter${N_ITERS}.log"
echo ""
echo "=== Standard map Lyapunov sweep ==="
echo "=== K grid: ${N_K}, ICs/K: ${N_IC}, iters: ${N_ITERS}, K_max: ${K_MAX} ==="
echo ""
./standard_map_lyapunov "${N_K}" "${N_IC}" "${N_ITERS}" "${K_MAX}" 2>&1 | tee "${LOG}"
