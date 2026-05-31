---
license: mit
tags:
  - fluid-dynamics
  - navier-stokes
  - cuda
  - pseudospectral
  - beale-kato-majda
---

# cfd-ns-bkm

2D pseudospectral Navier–Stokes BKM diagnostic sweeps from [bigcompute.science](https://bigcompute.science/experiments/cfd-ns-bkm/).

## Configs

| Config | Description |
|--------|-------------|
| `smoke_taylor_green` | N=256, ν=1e-3, Taylor–Green decay validation |
| `standard_random` | N=512, ν=1e-4, random vorticity blob |

## Columns

`step, time, max_vorticity, enstrophy, bkm_cumulative`

## Code

- Kernel: [idontknow/scripts/experiments/cfd-ns-bkm](https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-ns-bkm)
- CUDA kernels mirror: [bigcompute-cuda-kernels/cfd-ns-bkm](https://huggingface.co/cahlen/bigcompute-cuda-kernels/tree/main/cfd-ns-bkm)

**Note:** 2D NS is globally regular; BKM integral growth here is diagnostic infrastructure, not evidence of singularity.
