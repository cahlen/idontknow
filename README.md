# idontknow

GPU-accelerated computational mathematics — exploring open conjectures and underexplored areas of math using CUDA kernels, LLM-assisted theorem proving, and heavy compute on NVIDIA hardware.

Results are published openly at [bigcompute.science](https://bigcompute.science).

## What's Here

This repository contains the computational engine behind bigcompute.science: CUDA kernels, experiment scripts, Lean 4 formalizations, and LLM proving infrastructure. The goal is to run expensive computations once, publish verifiable results, and make them discoverable by both humans and AI agents.

### Active & Planned Experiments

| Experiment | Method | Status |
|---|---|---|
| **Zaremba's Conjecture — 210B verification + proof** | CUDA brute-force + F-K sieve + spectral gaps | Effective for d ≤ 10^1500 |
| **Zaremba transfer operator** | Chebyshev collocation + cuSOLVER eigensolve | Complete |
| **Zaremba transitivity** | CUDA + algebraic proof (Dickson classification) | Complete |
| **MCTS proof search benchmark** | LLM + Lean 4 + Monte Carlo Tree Search | Planned |
| **Ramsey R(5,5) lower bound** | Simulated annealing + constraint satisfaction | Planned |
| **Class numbers of real quadratic fields** | CUDA + BSGS + continued fractions | Planned |
| **Kronecker coefficients to n=120** | CUDA + symmetric group representation theory | Planned |

### Key Results So Far

- **Zaremba proof:** Effective for all d ≤ 10^1500 via layered F-K sieve with 489 verified spectral gaps. Brute force to 2.1×10^11, zero failures. [Proof document](docs/zaremba-proof.md) | [Paper](paper/zaremba-proof.tex)
- **Spectral gaps:** σ_p ≥ 0.336 for all 489 primes p ≤ 3500 (FP64/cuBLAS)
- **Hausdorff dimension:** δ = 0.836829443681208 (15-digit precision)
- **Transitivity:** Algebraically proved for ALL primes (not just computationally checked)
- **LLM proving:** 19/20 small Zaremba cases formally proved in Lean 4 (dual-model race)

## Structure

```
scripts/
  experiments/              # Per-experiment CUDA kernels and scripts
    zaremba-transfer-operator/
    zaremba-transitivity/
    zaremba-effective-bound/
    mcts-proof-search/
    ramsey-r55/
    class-numbers/
    kronecker-coefficients/
  zaremba_verify_v4.c       # Main brute-force verification kernel
  setup-cluster.sh          # One-time B200 cluster setup
  serve-model.sh            # vLLM/SGLang model serving
  run-zaremba.sh            # Full proving pipeline runner
  pipeline.sh               # Experiment orchestrator (run + publish)
  watch-v4.sh               # Auto-publish daemon

lean4-proving/
  prover.py                 # LLM <-> Lean 4 proving loop
  conjectures/zaremba.lean  # Formalized Zaremba theorems
  examples/                 # Test theorems

docs/                       # Research notes, logs, mathematical arguments
gguf-pipeline.sh            # GGUF quantization for HF model contributions
```

## Hardware

| Environment | GPUs | VRAM | Notes |
|---|---|---|---|
| **B200 Cluster** | 8x NVIDIA B200 | 1.43 TB (NVLink 5) | Primary compute |
| **Local** | RTX 5090 | 32 GB | Development + smaller experiments |

## Security

This repository is designed to be safe for autonomous AI agent commits. See `.gitignore` for excluded patterns. **Never commit:**
- Private keys, API tokens, passwords, or credentials
- `.env` files or any file containing secrets
- Model weights (`.gguf`, `.safetensors`) — these are downloaded, not stored

## Related

- **[bigcompute.science](https://github.com/cahlen/bigcompute.science)** — The publishing platform for experiment results
- **[bigcompute.science (live)](https://bigcompute.science)** — Published results with agent-consumable formats
