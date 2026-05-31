# CFD Phase 2: 2D Navier–Stokes BKM Diagnostic

Pseudospectral 2D incompressible Navier–Stokes on `[0,2π)²` with **Beale–Kato–Majda** vorticity tracking.

## What this is

- **Vorticity form** NS with RK4 + 2/3 dealiasing (cuFFT C2C)
- Tracks `max |ω|`, enstrophy, and cumulative `∫ ||ω||_∞ dt` (BKM diagnostic)
- **2D has global regularity** — no blowup is expected; this builds certifying infrastructure toward 3D BKM searches

## Hardware

RTX 5090 (`-arch=sm_120`), CUDA 13+

## Quick start

```bash
# smoke (Taylor–Green decay validation)
./scripts/experiments/cfd-ns-bkm/run.sh 256 0.001 2000 0.01 taylor-green

# standard (random vorticity, lower viscosity)
./scripts/experiments/cfd-ns-bkm/run.sh 512 0.0001 5000 0.005 random
```

## Plot

```bash
python3 scripts/experiments/cfd-ns-bkm/plot_bkm.py \
  scripts/experiments/cfd-ns-bkm/results/bkm_n512_nu1e-04_steps5000.csv \
  -o /path/to/bkm_diagnostic.svg
```

## Certificate

Exit code **2** on NaN/Inf (`CERTIFICATE_ERROR`).

## Dataset

Upload via `upload_hf.py` → [cahlen/cfd-ns-bkm](https://huggingface.co/datasets/cahlen/cfd-ns-bkm)
