---
license: cc-by-4.0
task_categories:
  - tabular-classification
tags:
  - number-theory
  - class-numbers
  - real-quadratic-fields
  - cohen-lenstra
  - gpu-computation
  - mathematics
  - computational-number-theory
  - algebraic-number-theory
  - continued-fractions
pretty_name: "Class Numbers of Real Quadratic Fields (GPU-Computed)"
size_categories:
  - 1B<n<10B
configs:
  - config_name: 1e9_to_1e10
    data_files: "data/1e9_to_1e10/*.parquet"
    description: "All fundamental discriminants d in [10^9, 10^10)"
dataset_info:
  - config_name: 1e9_to_1e10
    features:
      - name: discriminant
        dtype: uint64
      - name: class_number
        dtype: int32
    splits:
      - name: train
        num_examples: 2735671820
---

# Class Numbers of Real Quadratic Fields

**2.74 billion** class numbers of real quadratic fields Q(√d), computed for every fundamental discriminant d in [10⁹, 10¹⁰) on an 8× NVIDIA B200 DGX cluster in 30 minutes.

This dataset does not exist anywhere else. The previous systematic frontier was d ≤ 10¹¹ (Jacobson, Ramachandran, Williams 2006), but their raw per-discriminant data was never published. This is the first openly available, per-discriminant class number table at this scale.

> Part of the [bigcompute.science](https://bigcompute.science) project — GPU-accelerated exploration of open conjectures in number theory and combinatorics.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("cahlen/class-numbers-real-quadratic", "1e9_to_1e10", split="train", streaming=True)

for row in ds.take(10):
    print(f"d = {row['discriminant']}, h(d) = {row['class_number']}")
```

## What's In This Dataset

Every row is a fundamental discriminant d and its class number h(d):

| Column | Type | Description |
|--------|------|-------------|
| `discriminant` | `uint64` | Fundamental discriminant d > 0 |
| `class_number` | `int32` | Class number h(d) of the real quadratic field Q(√d) |

A **fundamental discriminant** is either:
- d ≡ 1 (mod 4) and squarefree, or
- d = 4m where m ≡ 2 or 3 (mod 4) and m is squarefree

The **class number** h(d) measures the failure of unique factorization in the ring of integers of Q(√d). When h(d) = 1, the ring has unique factorization.

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| Range | d ∈ [10⁹, 10¹⁰) |
| Fundamental discriminants | 2,735,671,820 |
| Computation time | 30 minutes |
| Hardware | 8× NVIDIA B200 DGX (1.43 TB VRAM, NVLink 5) |
| Throughput | 1.53 million discriminants/sec |

### Class Number Distribution

| h | Count | Fraction |
|---|-------|----------|
| 1 | 456,984,420 | 16.70% |
| 2 | 606,415,562 | 22.17% |
| 3 | 73,409,125 | 2.68% |
| 4 | 540,733,202 | 19.77% |
| 5 | 22,715,143 | 0.83% |
| 6 | 96,852,027 | 3.54% |
| 7 | 10,849,013 | 0.40% |
| 8 | 298,291,861 | 10.90% |
| 9 | 9,027,194 | 0.33% |
| 10 | 30,106,984 | 1.10% |
| 12 | 85,877,392 | 3.14% |
| 16 | 123,589,441 | 4.52% |

### Cohen-Lenstra p-Divisibility

| Divisor | Observed | Cohen-Lenstra (asymptotic) |
|---------|----------|---------------------------|
| 3 divides h | 15.28% | ~43.99% |
| 5 divides h | 4.89% | ~23.84% |
| 7 divides h | 2.35% | ~16.33% |

## Key Finding: Non-Monotone Convergence

Cohen and Lenstra (1984) predict that h(d) = 1 occurs with probability ≈ 75.446% asymptotically. Our data shows the observed rate is **decreasing** at this scale:

| Range | h = 1 fraction |
|-------|---------------|
| d < 10⁴ | 42.1% |
| d ~ 10⁶ | 25.7% |
| d ∈ [10⁹, 10¹⁰) | 16.7% |
| Asymptotic prediction | 75.4% |

The rate must eventually reverse and increase toward 75.4%, but at d ~ 10¹⁰ it hasn't turned around yet. This is because genus theory (the 2-part of the class group, determined by the number of prime factors of d) dominates at moderate discriminants. The values h = 2, 4, 8, 16 alone account for 57% of all discriminants. The odd part of the class group — where Cohen-Lenstra actually applies — must eventually dominate, but convergence is extremely slow.

See the [full analysis](https://bigcompute.science/findings/class-number-convergence/) on bigcompute.science.

## Computation Method

For each fundamental discriminant d, we compute h(d) via the analytic class number formula:

```
h(d) = round( sqrt(d) * L(1, χ_d) / (2 * R(d)) )
```

### Step 1: GPU Squarefree Sieve

Each GPU thread checks its position for divisibility by p² for all primes p ≤ √d. Classifies fundamental discriminants and stream-compacts into a packed array. All on-device — no CPU bottleneck.

### Step 2: Regulator R(d)

The regulator R(d) = log(ε_d) is computed from the continued fraction expansion, entirely in log-space to avoid integer overflow at d > 10⁹:

- d ≡ 0 (mod 4): CF expansion of √(d/4), with first D = 1 detection for cycle completion
- d ≡ 1 (mod 4): CF expansion of (1 + √d)/2 with reduced-state cycle detection

### Step 3: L-Function via Euler Product

```
L(1, χ_d) = ∏(p ≤ 99991) (1 - χ_d(p)/p)⁻¹
```

9,592 primes stored in CUDA `__constant__` memory. Kronecker symbol χ_d(p) = (d/p) computed via modular exponentiation (Jacobi symbol algorithm).

### Step 4: Assembly

Round sqrt(d) * L / (2R) to nearest integer. Atomic histogram updates for aggregate statistics.

### Validation

- **Exact match** with PARI/GP `qfbclassno()` on 1,000 randomly sampled discriminants across the full range
- h = 1 rate of 42.13% for d < 10⁴ matches PARI exactly
- Cross-validated: regulator values match PARI `quadregulator()` to 12+ digits

## Hardware

| Component | Specification |
|-----------|---------------|
| Node | NVIDIA DGX B200 |
| GPUs | 8× NVIDIA B200 (183 GB VRAM each) |
| Total VRAM | 1.43 TB |
| Interconnect | NVLink 5 (NV18), full mesh |
| CPUs | 2× Intel Xeon Platinum 8570 (112 cores / 224 threads) |
| System RAM | 2 TB DDR5 |

## Reproduce It Yourself

```bash
git clone https://github.com/cahlen/idontknow
cd idontknow

# Compile (adjust -arch for your GPU: sm_100a for B200, sm_120a for RTX 5090)
nvcc -O3 -arch=sm_100a -o class_v2 \
    scripts/experiments/class-numbers/class_numbers_v2.cu -lpthread -lm

# Validate against PARI/GP (should give h=1 at 42.13%)
./class_v2 5 10000

# Full run: d = 10^9 to 10^10 (~30 min on 8x B200, longer on fewer GPUs)
./class_v2 1000000000 10000000000 | tee run.log

# Raw (d, h) binary files appear in data/class-numbers/raw_gpu*.bin
# Format: repeating (uint64 discriminant, int32 class_number) = 12 bytes per record
```

The kernel auto-detects available GPUs and distributes the range evenly.

## Planned Extensions

| Range | Est. Discriminants | Est. Time (8x B200) |
|-------|-------------------|---------------------|
| [10¹⁰, 10¹¹) | ~27B | ~65 hours (running now) |
| [10¹¹, 10¹²) | ~270B | ~27 days |
| [10¹², 10¹³) | ~2.7T | ~270 days |

The [10¹⁰, 10¹¹) computation is in progress as of 2026-03-30 and will be added to this dataset when complete.

## Related

- **Source code**: [github.com/cahlen/idontknow](https://github.com/cahlen/idontknow) — CUDA kernels, experiment infrastructure
- **Experiment page**: [bigcompute.science/experiments/class-numbers-real-quadratic](https://bigcompute.science/experiments/class-numbers-real-quadratic/)
- **Finding writeup**: [bigcompute.science/findings/class-number-convergence](https://bigcompute.science/findings/class-number-convergence/)
- **All experiments**: [bigcompute.science](https://bigcompute.science) — Zaremba's conjecture, Ramsey R(5,5), Hausdorff spectrum, and more
- **Agent-readable index**: [bigcompute.science/llms.txt](https://bigcompute.science/llms.txt)

## Citation

```bibtex
@dataset{humphreys2026classnumbers,
  title   = {Class Numbers of Real Quadratic Fields: GPU-Accelerated Computation to 10^10},
  author  = {Humphreys, Cahlen},
  year    = {2026},
  month   = mar,
  publisher = {Hugging Face},
  url     = {https://huggingface.co/datasets/cahlen/class-numbers-real-quadratic},
  note    = {2.74 billion fundamental discriminants, 8x NVIDIA B200}
}
```

## References

1. Cohen, H. and Lenstra, H.W. Jr. (1984). "Heuristics on class groups of number fields." *Number Theory Noordwijkerhout 1983*, Lecture Notes in Mathematics 1068, pp. 33-62.
2. Jacobson, M.J. Jr., Ramachandran, S., and Williams, H.C. (2006). "Numerical results on class groups of imaginary quadratic fields." *Mathematics of Computation*, 75(254), pp. 1003-1024.
3. Stevenhagen, P. (1993). "The number of real quadratic fields having units of negative norm." *Experimental Mathematics*, 2(2), pp. 121-136.
4. Watkins, M. (2004). "Class numbers of imaginary quadratic fields." *Mathematics of Computation*, 73(246), pp. 907-938.

## License

CC BY 4.0 — free to use with attribution.
