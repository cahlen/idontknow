#!/usr/bin/env bash
# CFD Phase 3 — 3D pseudospectral NS BKM (RTX 5090)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH="/usr/local/cuda/bin:${PATH:-}"

N="${1:-128}"
NU="${2:-0.001}"
STEPS="${3:-1000}"
DT="${4:-0.002}"
IC="${5:-random}"

echo "Compiling ns3d_bkm (sm_120)..."
nvcc -O3 -arch=sm_120 -o scripts/experiments/cfd-ns3d-bkm/ns3d_bkm \
    scripts/experiments/cfd-ns3d-bkm/ns3d_bkm.cu -lcufft -lm

mkdir -p scripts/experiments/cfd-ns3d-bkm/results
LOG="scripts/experiments/cfd-ns3d-bkm/results/run_n${N}_nu${NU}_steps${STEPS}_${IC}.log"
echo "=== 3D NS BKM: N=${N}, nu=${NU}, steps=${STEPS}, dt=${DT}, IC=${IC} ==="
./scripts/experiments/cfd-ns3d-bkm/ns3d_bkm "${N}" "${NU}" "${STEPS}" "${DT}" "${IC}" \
    scripts/experiments/cfd-ns3d-bkm/results 2>&1 | tee "${LOG}"
