#!/usr/bin/env bash
# CFD Phase 2 — 2D pseudospectral Navier-Stokes BKM diagnostic (RTX 5090)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:${PATH:-}"

N="${1:-512}"
NU="${2:-0.0001}"
STEPS="${3:-5000}"
DT="${4:-0.005}"
IC="${5:-random}"

echo "Compiling ns2d_bkm (sm_120 for RTX 5090)..."
nvcc -O3 -arch=sm_120 -o scripts/experiments/cfd-ns-bkm/ns2d_bkm \
    scripts/experiments/cfd-ns-bkm/ns2d_bkm.cu -lcufft -lm
echo "Done."

mkdir -p scripts/experiments/cfd-ns-bkm/results

LOG="scripts/experiments/cfd-ns-bkm/results/run_n${N}_nu${NU}_steps${STEPS}_${IC}.log"
echo ""
echo "=== 2D NS pseudospectral BKM ==="
echo "=== N=${N}, nu=${NU}, steps=${STEPS}, dt=${DT}, IC=${IC} ==="
echo ""
./scripts/experiments/cfd-ns-bkm/ns2d_bkm "${N}" "${NU}" "${STEPS}" "${DT}" "${IC}" \
    scripts/experiments/cfd-ns-bkm/results 2>&1 | tee "${LOG}"
