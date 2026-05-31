---
license: cc-by-4.0
task_categories:
  - tabular-regression
tags:
  - fluid-dynamics
  - chaotic-advection
  - dynamical-systems
  - lyapunov-exponent
  - chirikov-standard-map
  - gpu-computation
  - mathematics
  - computational-fluid-dynamics
  - bigcompute
pretty_name: "Chirikov Standard Map Lyapunov Spectrum (GPU-Computed)"
size_categories:
  - n<1K
configs:
  - config_name: deep_sweep
    data_files: "data/deep_sweep/*.csv"
    description: "2048 K × 8192 ICs × 50000 iterations, K ∈ [0, 5] — certifying run on RTX 5090"
  - config_name: standard_sweep
    data_files: "data/standard_sweep/*.csv"
    description: "512 K × 4096 ICs × 20000 iterations"
  - config_name: smoke_test
    data_files: "data/smoke_test/*.csv"
    description: "64 K × 512 ICs × 5000 iterations — quick validation"
dataset_info:
  - config_name: deep_sweep
    features:
      - name: k_index
        dtype: int64
      - name: K
        dtype: float64
      - name: mean_lyapunov
        dtype: float64
      - name: std_lyapunov
        dtype: float64
      - name: min_lyapunov
        dtype: float64
      - name: max_lyapunov
        dtype: float64
      - name: fraction_positive
        dtype: float64
    splits:
      - name: train
        num_examples: 2048
---

# Chirikov Standard Map Lyapunov Spectrum

**Largest Lyapunov exponent** Λ(K) for the Chirikov standard map on T², computed with a custom CUDA Benettin kernel on **NVIDIA RTX 5090** (sm_120).

> Part of the [bigcompute.science](https://bigcompute.science) CFD conjecture program — GPU-accelerated exploration of open questions in fluid mixing and dynamical systems.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("cahlen/cfd-chaotic-advection", "deep_sweep", split="train")
row = ds[1200]
print(f"K={row['K']:.4f}, mean Λ={row['mean_lyapunov']:.6f}")
```

## What's In This Dataset

Each row is one coupling parameter K on a uniform grid in [0, K_max]:

| Column | Type | Description |
|--------|------|-------------|
| `k_index` | int | Grid index |
| `K` | float | Standard map coupling parameter |
| `mean_lyapunov` | float | Mean largest Lyapunov exponent over ICs |
| `std_lyapunov` | float | Standard deviation across ICs |
| `min_lyapunov` | float | Minimum over ICs |
| `max_lyapunov` | float | Maximum over ICs |
| `fraction_positive` | float | Fraction of ICs with Λ > 0 |

### Configurations

| Config | n_k | n_ic | n_iters | K_max | Trajectories | Wall time |
|--------|-----|------|---------|-------|--------------|-----------|
| `deep_sweep` | 2048 | 8192 | 50000 | 5.0 | 16,777,216 | 116.6 s |
| `standard_sweep` | 512 | 4096 | 20000 | 5.0 | 2,097,152 | 5.9 s |
| `smoke_test` | 64 | 512 | 5000 | 2.0 | 32,768 | ~1 s |

Certifying logs are in `logs/`. Metadata in `metadata.json`.

## Key Results (deep_sweep)

- Λ(0) = 0 (integrable limit validated)
- At literature K_crit ≈ 0.971635406: mean Λ ≈ 0.0446, >99.9% ICs positive
- At K = 5: mean Λ ≈ 0.956
- Zero NaN/Inf across all trajectories

## Reproduction

```bash
git clone https://github.com/cahlen/idontknow.git
cd idontknow
./scripts/experiments/cfd-chaotic-advection/run.sh 2048 8192 50000 5.0
```

CUDA kernel: [cahlen/bigcompute-cuda-kernels](https://huggingface.co/cahlen/bigcompute-cuda-kernels) (`cfd-chaotic-advection/standard_map_lyapunov.cu`)

## Related

- Experiment: [bigcompute.science/experiments/cfd-chaotic-advection](https://bigcompute.science/experiments/cfd-chaotic-advection/)
- Finding: [Standard Map Chaos Onset](https://bigcompute.science/findings/cfd-standard-map-chaos-onset/)
- Aerospace CFD context: [enfuse/cfd-ai-poc](https://github.com/enfuse/cfd-ai-poc)

## Citation

```bibtex
@misc{humphreys2026cfdchaoticadvection,
  author = {Humphreys, Cahlen},
  title = {Chirikov Standard Map Lyapunov Spectrum (GPU-Computed)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/cahlen/cfd-chaotic-advection}}
}
```

Human–AI collaborative research. Not peer-reviewed. All code and data open for verification.
