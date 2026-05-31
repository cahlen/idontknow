#!/usr/bin/env bash
# Phase 5a: Kerr-type antiparallel vortex IC + viscosity sweep at 256³
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN="${ROOT}/scripts/experiments/cfd-ns3d-bkm/run.sh"

echo "=== Phase 5a smoke: 64³ Kerr, 200 steps ==="
"${RUN}" 64 0.01 200 0.002 kerr

echo "=== Phase 5a: Kerr ν sweep at 256³ ==="
# ν=1e-3 — moderate Re baseline (compare to random at same ν)
"${RUN}" 256 0.001 1000 0.001 kerr
"${RUN}" 256 0.001 1000 0.001 random

# ν=1e-4 — Phase 4 blowup-search Reynolds number
"${RUN}" 256 0.0001 2000 0.001 kerr

# ν=1e-5 — toward Euler limit (smaller dt)
"${RUN}" 256 0.00001 2000 0.0005 kerr

echo "=== Phase 5a complete ==="
