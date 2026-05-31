---
license: mit
tags:
  - fluid-dynamics
  - navier-stokes
  - 3d-dns
  - beale-kato-majda
---

# cfd-ns3d-bkm

3D pseudospectral Navier–Stokes BKM blowup-search runs from [bigcompute.science](https://bigcompute.science/experiments/cfd-ns3d-bkm/).

| Config | Description |
|--------|-------------|
| `smoke_taylor_green` | 64³, ν=0.01, Taylor–Green |
| `standard_random` | 128³, ν=1e-3, random vorticity |

Columns: `step, time, max_vorticity, enstrophy, bkm_cumulative`

**Note:** No blowup observed at tested resolution/Re. Infrastructure toward full BKM search.
