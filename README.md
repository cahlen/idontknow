# idontknow

GPU-accelerated computational mathematics — exploring open conjectures with custom CUDA kernels, interval arithmetic, and heavy compute on NVIDIA B200 + RTX 5090.

**This work is produced through human–AI collaboration.** CUDA kernels, mathematical arguments, documentation, and analysis are developed jointly by [Cahlen Humphreys](https://github.com/cahlen) and AI agents (Claude). No results have been independently peer-reviewed. All claims are grounded in computational evidence and reproducible code — not formal proof unless explicitly stated. Everything is open for independent verification.

Results published openly at [bigcompute.science](https://bigcompute.science). Raw data on [Hugging Face](https://huggingface.co/cahlen).

## Experiments

| Experiment | Method | Key Result | Status |
|---|---|---|---|
| **Zaremba's Conjecture** | GPU brute force (210B) + MOW spectral theory + arb interval arithmetic | Computer-assisted proof framework for all d ≥ 1. D₀ ≈ 3.4×10¹⁰. Not yet peer-reviewed. | [Paper](paper/zaremba-proof.pdf) |
| **Ramsey R(5,5)** | SA + exhaustive extension + 4-SAT (Glucose3) | 656/656 K₄₂ colorings UNSAT. Strongest computational evidence R(5,5) = 43. | Complete |
| **Class Numbers** | GPU sieve + CF regulator (log-space) + Euler product (9592 primes) | 2.74B discriminants for d ∈ [10⁹, 10¹⁰]. Cohen-Lenstra convergence is non-monotone. | In progress |
| **Hausdorff Spectrum** | Transfer operator + Chebyshev collocation on RTX 5090 | To our knowledge, first complete dim_H for all 2²⁰ - 1 subsets of {1,...,20} | Complete |
| **Lyapunov Spectrum** | Transfer operator eigenvalue computation | All 1,048,575 subsets | Complete |
| **Minkowski ?(x)** | Multifractal analysis | To our knowledge, first numerical singularity spectrum f(α) | Complete |
| **Flint Hills Series** | Quad-double CUDA arithmetic | Partial sums to 10¹⁰ | Complete |
| **LLM Theorem Proving** | Goedel-Prover + Kimina-Prover → Lean 4 | 19/20 formal proofs | Complete |
| **Kronecker Coefficients** | GPU-accelerated representation theory | To n=120 for geometric complexity theory | Planned |

## Structure

```
paper/                          # Zaremba proof paper (LaTeX + PDF)
scripts/experiments/
  zaremba-effective-bound/      # Brute force, spectral gaps, Dolgopyat, arb certification
  ramsey-r55/                   # SA kernels, extension search, 4-SAT, Exoo data
  class-numbers/                # GPU sieve, regulator, L-function, Cohen-Lenstra stats
  hausdorff-dimension-spectrum/ # Transfer operator eigenvalue computation
  lyapunov-exponent-spectrum/   # Lyapunov exponent computation
  minkowski-spectrum/           # Multifractal singularity spectrum
  flint-hills/                  # Quad-double partial sums
  mcts-proof-search/            # LLM + MCTS for theorem proving
lean4-proving/                  # Lean 4 formalizations + LLM proving loop
data/                           # Raw computation output (large files on HF)
logs/                           # Computation logs
docs/                           # Research notes
```

## Hardware

| Environment | GPUs | VRAM | Role |
|---|---|---|---|
| **B200 Cluster** | 8× NVIDIA B200 | 1.43 TB (NVLink 5) | Primary compute |
| **Local** | RTX 5090 | 32 GB | Development + smaller experiments |

## Key Technical Details

### Zaremba Computer-Assisted Proof Framework (not yet peer-reviewed)
- Brute force: 210B denominators verified, zero failures (6962s on 8×B200)
- Spectral gaps: MPFR 256-bit certified (σ_p ≥ 0.651 for 11 covering primes)
- Dolgopyat bound: ρ_η ≤ 0.771 via arb ball arithmetic (FLINT, 70 certified digits)
- All 8 constants interval-certified via arb/MPFR
- MOW theorem matching verified against actual paper (Crelle 2019)
- Transitivity argument via Dickson's classification (AI-assisted, not independently verified)

### Ramsey R(5,5)
- Fixed critical initialization bug (adj[i]=0 inside loop destroyed back-edges)
- Exhaustive: 2⁴² = 4.4T extensions of Exoo's K₄₂ → zero valid (130s on 8×B200)
- 4-SAT: all 656 K₄₂ colorings checked in 3 seconds (Glucose3)
- Direct K₄₃ SAT (903 vars, 1.9M clauses) remains open

### Class Numbers
- GPU sieve + CF regulator + Euler product — all on-device, 1.5M disc/sec
- Regulator: CF of √(d/4) for d≡0 mod 4, CF of (1+√d)/2 with reduced-state cycle detection for d≡1 mod 4
- Validated: exact match with PARI/GP on 1000 discriminants
- Finding: Cohen-Lenstra h=1 convergence is non-monotone (42% at d~10⁴ → 17% at d~10¹⁰)

## Data Preservation

All raw data from GPU computations is preserved:
- **Small data** (< 100MB): committed to `data/` in this repo
- **Large data** (> 100MB): uploaded to [Hugging Face](https://huggingface.co/cahlen) as datasets
- All computations logged with timestamps, parameters, and aggregate statistics

## Related

- **[bigcompute.science](https://bigcompute.science)** — Publishing platform for results
- **[bigcompute.science repo](https://github.com/cahlen/bigcompute.science)** — Website source (Astro + KaTeX)
- **[/llms.txt](https://bigcompute.science/llms.txt)** — Agent-consumable structured data
