---
license: cc-by-4.0
task_categories:
  - tabular-classification
tags:
  - mathematics
  - number-theory
  - continued-fractions
  - zaremba-conjecture
  - gpu-computation
  - hausdorff-dimension
size_categories:
  - 1K<n<10K
---

# Zaremba Density: GPU-Computed Exception Sets and Density Data

Complete GPU-computed density measurements for Zaremba's conjecture across 65 digit set / range combinations, computed on an 8x NVIDIA B200 cluster (1.4 TB VRAM).

## What is Zaremba's Conjecture?

For every positive integer d, there exists a coprime a/d whose continued fraction has all partial quotients bounded by 5. The **density** of a digit set A at range N is the fraction of integers d in [1, N] that have at least one coprime a/d with all partial quotients in A.

## Dataset Contents

### `zaremba_density_all_results.csv`
Summary of all 65 completed experiments: digit set, range, density, uncovered count, runtime.

### `density_sweep_1e6.csv` / `density_all_subsets_n10_1e6.csv`
Complete density sweep of all 1,023 non-empty subsets of {1,...,10} at 10^6.

### `logs/`
All 65 raw GPU log files with full output including uncovered denominator lists for closed exception sets.

## Key Results

### Closed Exception Sets (verified to 10^11)
| Digit Set | Exceptions | Verified To |
|-----------|-----------|-------------|
| {1,2,3} | 27 | 10^9 |
| {1,2,4} | 64 | 10^10 |
| {1,2,5} | 374 | 10^11 |
| {1,2,6} | 1,834 | 10^11 |
| {1,2,7} | 7,178 | 10^11 |

### Open (Growing) Exception Sets at 10^11
| Digit Set | Exceptions at 10^11 |
|-----------|-------------------|
| {1,2,8} | 23,590 |
| {1,2,9} | 77,109 |
| {1,2,10} | 228,514 |

### {1,2} Logarithmic Convergence
| Range | Density |
|-------|---------|
| 10^6 | 61.28% |
| 10^9 | 72.06% |
| 10^10 | 76.55% |
| 10^11 | 80.75% |
| 10^12 | 84.58% |

Fits: density ~ 31.5 + 4.47 * log10(N)

## Hardware

- 8x NVIDIA B200 (180 GB each, ~1.4 TB total VRAM)
- Intel Xeon w9-3595X
- Custom CUDA kernels with persistent threads, prefix-sorted work distribution

## Reproduction

All source code: https://github.com/cahlen/idontknow/tree/main/scripts/experiments/zaremba-density

```bash
nvcc -O3 -arch=sm_90 -o zaremba_density_gpu zaremba_density_gpu.cu -lm
./zaremba_density_gpu 100000000000 1,2,3
```

## Citation

```bibtex
@dataset{humphreys2026zaremba_density,
  author = {Humphreys, Cahlen},
  title = {Zaremba Density: GPU-Computed Exception Sets and Density Data},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/cahlen/zaremba-density}
}
```

## License

CC BY 4.0. This work was produced through human-AI collaboration (Cahlen Humphreys + Claude). All code and data open for verification.
