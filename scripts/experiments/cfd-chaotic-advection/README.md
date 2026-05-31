# CFD Chaotic Advection — Standard Map Lyapunov Spectrum

First **bigcompute-style CFD experiment**: GPU verification of chaotic mixing in the
**Chirikov standard map**, the canonical phase-space model for **2D chaotic advection**
in periodically driven incompressible flows.

## Conjecture class

- **Integrability → chaos transition:** for small $K$, orbits are mostly regular; above
  $K_{\mathrm{crit}} \approx 0.971635406$, a chaotic sea dominates (Chirikov, 1979).
- **Mixing rate:** the largest Lyapunov exponent $\Lambda(K)$ quantifies how fast
  nearby fluid parcels diverge — directly tied to **stretching rates** in laminar
  advection–diffusion problems.

This is the fluid-dynamics analogue of our Hausdorff / transfer-operator work: same
ergodic-theory toolkit, different physical map.

## Method

For each $K \in [0, K_{\max}]$ on a uniform grid:

1. Sample `n_ic` random initial conditions $(\theta_0, p_0) \in \mathbb{T}^2$
2. Iterate the standard map $n_{\mathrm{iters}}$ steps
3. Track tangent-vector growth (Benettin) → estimate $\Lambda(K)$
4. Aggregate mean, std, min, max, fraction of ICs with $\Lambda > 0$

One CUDA thread per $(K, \mathrm{IC})$ pair. Sized for **single RTX 5090** (32 GB).

## Hardware

| Target | Parameters | ~Runtime |
|--------|------------|----------|
| Smoke test | `64 512 5000 2.0` | ~1 s |
| Standard (5090) | `512 4096 20000 5.0` | ~6 s |
| Deep certifying | `2048 8192 50000 5.0` | ~2 min |

## Plot

```bash
python3 scripts/experiments/cfd-chaotic-advection/plot_lyapunov.py \
  scripts/experiments/cfd-chaotic-advection/results/lyapunov_k2048_ic8192_iter50000.csv \
  -o lyapunov_spectrum.svg
```

## Reproduction

```bash
cd idontknow
./scripts/experiments/cfd-chaotic-advection/run.sh          # defaults
./scripts/experiments/cfd-chaotic-advection/run.sh 64 512 5000 2.0   # quick test
```

Compile manually:

```bash
nvcc -O3 -arch=sm_120 -o standard_map_lyapunov \
  scripts/experiments/cfd-chaotic-advection/standard_map_lyapunov.cu -lm
./standard_map_lyapunov 512 4096 20000 5.0
```

Adjust `-arch=sm_89` for RTX 4090, `sm_120` for RTX 5090.

## Outputs

- `results/lyapunov_k*_ic*_iter*.csv` — per-$K$ statistics
- `results/run_*.log` — certifying log (device, params, validation, timing)

Exit code **2** = numerical certificate failure (NaN/Inf).

## Related repos

- **cfd** (`~/dev/cfd`) — aerospace RANS demo; no custom CUDA yet
- **bigcompute.science** — publication layer for this experiment

## References

- Chirikov, B. V. (1979). *Phys. Rep.* — standard map chaos threshold
- Ottino, J. M. (1989). *The Kinematics of Mixing* — chaotic advection
- Aref, H. (1984). *J. Fluid Mech.* — chaotic advection in Stokes flow
