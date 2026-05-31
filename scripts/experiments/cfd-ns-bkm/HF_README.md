---
license: mit
task_categories:
  - tabular-regression
tags:
  - fluid-dynamics
  - navier-stokes
  - cuda
  - pseudospectral
  - beale-kato-majda
  - gpu-computation
  - computational-fluid-dynamics
  - bigcompute
pretty_name: "2D Navier–Stokes BKM Diagnostic (GPU Pseudospectral DNS)"
size_categories:
  - n<1K
configs:
  - config_name: smoke_taylor_green
    data_files: "data/smoke_taylor_green/*.csv"
    description: "N=256², ν=1e-3, Taylor–Green IC, 2000 steps — decay validation on RTX 5090"
  - config_name: standard_random
    data_files: "data/standard_random/*.csv"
    description: "N=512², ν=1e-4, random vorticity blob, 5000 steps — diagnostic sweep"
dataset_info:
  - config_name: smoke_taylor_green
    features:
      - name: step
        dtype: int64
      - name: time
        dtype: float64
      - name: max_vorticity
        dtype: float64
      - name: enstrophy
        dtype: float64
      - name: bkm_cumulative
        dtype: float64
    splits:
      - name: train
        num_examples: 97
  - config_name: standard_random
    features:
      - name: step
        dtype: int64
      - name: time
        dtype: float64
      - name: max_vorticity
        dtype: float64
      - name: enstrophy
        dtype: float64
      - name: bkm_cumulative
        dtype: float64
    splits:
      - name: train
        num_examples: 100
---

# 2D Navier–Stokes BKM Diagnostic

**Pseudospectral vorticity-form Navier–Stokes** on a periodic torus \\([0,2\pi)^2\\), with the **Beale–Kato–Majda (BKM)** diagnostic

$$
\int_0^T \lVert \omega(\cdot,t) \rVert_{L^\infty}\, dt
$$

tracked alongside enstrophy. Computed with a custom **CUDA + cuFFT** kernel on **NVIDIA RTX 5090** (Blackwell, sm_120).

> Part of the [bigcompute.science](https://bigcompute.science) CFD conjecture program — GPU infrastructure toward 3D BKM blowup searches. **2D incompressible flow is globally regular**; these runs are certifying diagnostics, not blowup evidence.

## Quick Start

```python
from datasets import load_dataset

tg = load_dataset("cahlen/cfd-ns-bkm", "smoke_taylor_green", split="train")
rand = load_dataset("cahlen/cfd-ns-bkm", "standard_random", split="train")
print(tg[-1])  # final Taylor–Green row
```

## What's In This Dataset

Each row is one logged time step from a pseudospectral DNS run:

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Time-step index |
| `time` | float | Physical time \\(t\\) |
| `max_vorticity` | float | \\(\lVert \omega \rVert_{L^\infty}\\) (max vorticity on the grid) |
| `enstrophy` | float | \\(\tfrac{1}{2}\int \lVert \omega \rVert^2 \, dx\\) |
| `bkm_cumulative` | float | Running BKM integral \\(\int_0^t \lVert \omega \rVert_{L^\infty}\, ds\\) |

Certifying logs are in `logs/`. Run metadata in `metadata.json`.

### Configurations

| Config | Grid | \\(\nu\\) | IC | Steps | \\(\Delta t\\) | Final max \\(\lVert \omega \rVert_{L^\infty}\\) | Final BKM | Throughput |
|--------|------|---|-----|-------|-----|-----------------|-----------|------------|
| `smoke_taylor_green` | \\(256^2\\) | \\(10^{-3}\\) | Taylor–Green | 2000 | 0.01 | 0.157 at \\(t=20\\) | 12.80 | ~1108 steps/s |
| `standard_random` | \\(512^2\\) | \\(10^{-4}\\) | Random blob | 5000 | 0.005 | 0.026 at \\(t=25\\) | 1.77 | ~532 steps/s |

Both runs: **zero NaN/Inf** (exit certificate).

### Method (summary)

Vorticity equation:

$$
\partial_t \omega + \mathbf{u}\cdot\nabla\omega = \nu \nabla^2 \omega
$$

- Streamfunction Poisson solve in Fourier space; **2/3 Orszag dealiasing**; **RK4**; **fp64**
- Random IC: Gaussian-envelope vorticity blob at \\((\pi,\pi)\\) with SplitMix64 amplitudes

## Key Results

- Taylor–Green: \\(\lVert \omega \rVert_{L^\infty}\\) decays **2.0 → 0.16** by \\(t=20\\); validates spectral accuracy
- Random IC at \\(\nu=10^{-4}\\): BKM integral **≈ 1.77** over \\(t=25\\); peak vorticity remains bounded
- Infrastructure validated for Phase 3 3D extension

## Reproduction

```bash
git clone https://github.com/cahlen/idontknow.git
cd idontknow
./scripts/experiments/cfd-ns-bkm/run.sh 256 0.001 2000 0.01 taylor-green
./scripts/experiments/cfd-ns-bkm/run.sh 512 0.0001 5000 0.005 random
python3 scripts/experiments/cfd-ns-bkm/upload_hf.py
```

CUDA kernel: [ns2d_bkm.cu](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/cfd-ns-bkm/ns2d_bkm.cu)

## Related

- CFD program hub: [cfd-chaotic-advection experiment](https://bigcompute.science/experiments/cfd-chaotic-advection/)
- Experiment page: [cfd-ns-bkm](https://bigcompute.science/experiments/cfd-ns-bkm/)
- Finding: [2D NS BKM diagnostic](https://bigcompute.science/findings/cfd-ns-bkm-diagnostic/) — **bronze** / ACCEPT w/ revision (3-model review; consensus = most conservative: 2× silver + 1× bronze)
- Phase 3 dataset: [cahlen/cfd-ns3d-bkm](https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm)
- Code: [idontknow/scripts/experiments/cfd-ns-bkm](https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-ns-bkm)

## Citation

```bibtex
@misc{humphreys2026cfdnsbkm,
  author = {Humphreys, Cahlen},
  title = {2D Navier–Stokes BKM Diagnostic (GPU Pseudospectral DNS)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/cahlen/cfd-ns-bkm}}
}
```

Human–AI collaborative research. Peer-reviewed finding on bigcompute.science. All code and data open for verification.
