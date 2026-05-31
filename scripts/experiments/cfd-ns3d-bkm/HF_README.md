---
license: mit
task_categories:
  - tabular-regression
tags:
  - fluid-dynamics
  - navier-stokes
  - 3d-dns
  - beale-kato-majda
  - cuda
  - pseudospectral
  - gpu-computation
  - computational-fluid-dynamics
  - bigcompute
pretty_name: "3D Navier–Stokes BKM Blowup Search (GPU Pseudospectral DNS)"
size_categories:
  - n<1K
configs:
  - config_name: smoke_taylor_green
    data_files: "data/smoke_taylor_green/*.csv"
    description: "64³, ν=0.01, Taylor–Green IC, 200 steps — smoke validation on RTX 5090"
  - config_name: standard_random
    data_files: "data/standard_random/*.csv"
    description: "128³, ν=1e-3, random vorticity IC, 1000 steps — first certifying 3D BKM sweep"
  - config_name: blowup_search
    data_files: "data/blowup_search/*.csv"
    description: "256³, ν=1e-4, random IC, 500 steps — Phase 4 higher-Re blowup monitor on RTX 5090"
  - config_name: blowup_search_long
    data_files: "data/blowup_search_long/*.csv"
    description: "256³, ν=1e-4, random IC, 2000 steps — extended Phase 4 BKM monitor (t=2)"
  - config_name: taylor_green_256
    data_files: "data/taylor_green_256/*.csv"
    description: "256³, ν=1e-3, Taylor–Green IC, 1000 steps — structured higher-Re benchmark"
  - config_name: blowup_search_5000
    data_files: "data/blowup_search_5000/*.csv"
    description: "256³, ν=1e-4, random IC, 5000 steps — extended Phase 4 BKM monitor (t=5)"
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
        num_examples: 68
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
        num_examples: 92
  - config_name: blowup_search
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
        num_examples: 84
  - config_name: blowup_search_long
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
  - config_name: taylor_green_256
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
        num_examples: 92
  - config_name: blowup_search_5000
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

# 3D Navier–Stokes BKM Blowup Search

**Three-dimensional vorticity-form incompressible Navier–Stokes** with **vortex stretching** \\((\boldsymbol{\omega}\cdot\nabla)\mathbf{u}\\), pseudospectral **cuFFT C2C** on a periodic cube, and BKM integral tracking. Custom **CUDA** kernel on **NVIDIA RTX 5090**.

$$
\partial_t \omega + (\mathbf{u}\cdot\nabla)\omega = (\boldsymbol{\omega}\cdot\nabla)\mathbf{u} + \nu \nabla^2 \omega
$$

$$
\hat{\mathbf{u}} = \frac{i(\mathbf{k}\times\hat{\boldsymbol{\omega}})}{\lVert \mathbf{k} \rVert^2}
$$

> Phase 3 of the [bigcompute.science](https://bigcompute.science) CFD program. Extends the [2D BKM diagnostic](https://huggingface.co/datasets/cahlen/cfd-ns-bkm). **No blowup observed** at tested resolution/Re — certifying infrastructure at moderate grid sizes.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("cahlen/cfd-ns3d-bkm", "standard_random", split="train")
row = ds[-1]
print(f"t={row['time']:.3f}, max|ω|={row['max_vorticity']:.4f}, BKM={row['bkm_cumulative']:.4f}")
```

## What's In This Dataset

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Time-step index |
| `time` | float | Physical time \\(t\\) |
| `max_vorticity` | float | \\(\lVert \omega \rVert_{L^\infty}\\) |
| `enstrophy` | float | \\(\tfrac{1}{2}\int \lVert \omega \rVert^2 \, dV\\) |
| `bkm_cumulative` | float | BKM integral \\(\int_0^t \lVert \omega \rVert_{L^\infty}\, ds\\) |

Certifying logs in `logs/`. Metadata in `metadata.json`.

### Configurations

| Config | Grid | \\(\nu\\) | IC | Steps | \\(\Delta t\\) | Final max \\(\lVert \omega \rVert_{L^\infty}\\) | Final BKM |
|--------|------|---|-----|-------|-----|-----------------|-----------|
| `smoke_taylor_green` | \\(64^3\\) | 0.01 | Taylor–Green | 200 | 0.002 | 4.17 at \\(t=0.4\\) | 1.63 |
| `standard_random` | \\(128^3\\) | \\(10^{-3}\\) | Random blob | 1000 | 0.002 | 0.614 at \\(t=2.0\\) | 1.24 |
| `blowup_search` | \\(256^3\\) | \\(10^{-4}\\) | Random blob | 500 | 0.001 | 0.878 at \\(t=0.5\\) | 0.44 |
| `blowup_search_long` | \\(256^3\\) | \\(10^{-4}\\) | Random blob | 2000 | 0.001 | 0.887 at \\(t=2.0\\) | 1.76 |
| `blowup_search_5000` | \\(256^3\\) | \\(10^{-4}\\) | Random blob | 5000 | 0.001 | 0.903 at \\(t=5.0\\) | 4.45 |
| `taylor_green_256` | \\(256^3\\) | \\(10^{-3}\\) | Taylor–Green | 1000 | 0.001 | 4.44 at \\(t=1.0\\) | 4.23 |

All runs: **zero NaN/Inf**.

### Method (summary)

- 3D vorticity form with explicit vortex-stretching term in physical space
- Velocity from Fourier relation \\(\hat{\mathbf{u}} = i(\mathbf{k}\times\hat{\boldsymbol{\omega}})/\lVert \mathbf{k} \rVert^2\\); **2/3 dealiasing**; **RK4**; **fp64**
- BKM integral accumulated each step for blowup-criterion monitoring

## Key Results

- First certifying **3D pseudospectral BKM** runs on RTX 5090
- Random IC at \\(128^3\\): BKM **≈ 1.24** by \\(t=2\\); vorticity remains bounded at tested Re
- **256³ blowup search** at \\(\nu=10^{-4}\\): BKM **≈ 0.44** by \\(t=0.5\\); extended to **≈ 1.76** by \\(t=2.0\\); **≈ 4.45** by \\(t=5.0\\) (5000 steps); **2.3 steps/s**; no blowup signal
- **256³ Taylor–Green** at \\(\nu=10^{-3}\\): BKM **≈ 4.23** by \\(t=1.0\\); max \\(\lVert \omega \rVert_{L^\infty} \approx 4.44\\)
- **512³** exceeds 32 GB VRAM on RTX 5090 (cuFFT allocation OOM); **256³** is the practical ceiling on this hardware
- No finite-time blowup signal at this resolution — consistent with viscous DNS at moderate Re

## Reproduction

```bash
git clone https://github.com/cahlen/idontknow.git
cd idontknow
./scripts/experiments/cfd-ns3d-bkm/run.sh 64 0.01 200 0.002 taylor-green
./scripts/experiments/cfd-ns3d-bkm/run.sh 128 0.001 1000 0.002 random
python3 scripts/experiments/cfd-ns3d-bkm/upload_hf.py
```

CUDA kernel: [ns3d_bkm.cu](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/cfd-ns3d-bkm/ns3d_bkm.cu)

## Related

- Phase 2 (2D): [cahlen/cfd-ns-bkm](https://huggingface.co/datasets/cahlen/cfd-ns-bkm)
- CFD program: [cfd-chaotic-advection](https://bigcompute.science/experiments/cfd-chaotic-advection/)
- Experiment page: [cfd-ns3d-bkm](https://bigcompute.science/experiments/cfd-ns3d-bkm/)
- Finding (3D): [cfd-ns3d-bkm-infrastructure](https://bigcompute.science/findings/cfd-ns3d-bkm-infrastructure/)
- Finding (2D): [cfd-ns-bkm-diagnostic](https://bigcompute.science/findings/cfd-ns-bkm-diagnostic/)
- Code: [idontknow/scripts/experiments/cfd-ns3d-bkm](https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-ns3d-bkm)

## Citation

```bibtex
@misc{humphreys2026cfdns3dbkm,
  author = {Humphreys, Cahlen},
  title = {3D Navier–Stokes BKM Blowup Search (GPU Pseudospectral DNS)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm}}
}
```

Human–AI collaborative research. All code and data open for verification.
