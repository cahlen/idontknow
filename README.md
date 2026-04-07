# idontknow

GPU-accelerated computational mathematics — exploring open conjectures with custom CUDA kernels, interval arithmetic, and heavy compute on NVIDIA B200 + RTX 5090.

**Human–AI collaborative research.** CUDA kernels, mathematical arguments, review infrastructure, and documentation developed jointly by [Cahlen Humphreys](https://github.com/cahlen) and AI agents (Claude Opus 4.6, o3-pro, GPT-5.2, Grok). Not peer-reviewed. All claims grounded in computational evidence and reproducible code. Everything open for independent verification.

Results: [bigcompute.science](https://bigcompute.science) · Data: [Hugging Face](https://huggingface.co/cahlen) · MCP: [mcp.bigcompute.science](https://mcp.bigcompute.science/mcp) (23 tools, no auth)

## Quick Start

```bash
# Run the research agent (needs one API key: Gemini free, or OpenAI, or Anthropic)
export GEMINI_API_KEY='AIza...'   # free at aistudio.google.com/apikey
./scripts/run_agent.sh             # one cycle: monitor → harvest → analyze → review → deploy
./scripts/run_agent.sh --loop 10m  # autonomous loop
```

Or open a [Colab notebook](https://colab.research.google.com/github/cahlen/bigcompute.science/blob/main/public/notebooks/bigcompute_research_agent.ipynb) — free T4 GPU, auto-compile, run experiments on open conjectures.

## Experiments

| Experiment | Key Result | Status |
|---|---|---|
| **Zaremba Conjecture** | Proof framework (not complete proof). 210B verified, ρ_η ≤ 0.7606 (arb-certified, 77 digits). 4 gaps remain. | [Paper](paper/zaremba-proof.pdf) |
| **Zaremba Density** | 5 closed exception sets ({1,2,3}=27 through {1,2,7}=7,178). A={1,2} logarithmic convergence (31.5 + 4.47·log₁₀N). Inverse-square amplification law. | In progress |
| **Ramsey R(5,5)** | 656/656 K₄₂ colorings UNSAT. Strongest computational evidence R(5,5) = 43. | Complete |
| **Kronecker Coefficients** | S₂₀, S₃₀ (26.4B nonzero), S₄₀ char table (37,338 partitions). 94.9% nonzero. S₄₅ infeasible (63 TB). | S₄₀ complete |
| **Class Numbers** | Complete to 10¹¹. h=1 rate falls to 0 (genus theory). Extending to 10¹³. | In progress |
| **Hausdorff Spectrum** | First complete dim_H for all 2²⁰-1 subsets of {1,...,20}. | Complete |
| **Ramanujan Machine** | 586B+ equal-degree CFs exhausted (0 new formulas, 7K false positives disproven via PSLQ). v2 asymmetric-degree kernel built — deg(b)≈2×deg(a) required. | Pivoting to v2 |
| **Prime Convergents** | 10M random CFs verified Erdős-Mahler bound. Worst-case ratio 4.87, constant ~10 suffices. | Complete |
| **Erdős-Straus** | Solution counts f(p) for 4/p = 1/x + 1/y + 1/z. All primes to 10⁸. | In progress |
| **Lyapunov Spectrum** | All 1,048,575 Lyapunov exponents. | Complete |
| **Minkowski ?(x)** | First numerical singularity spectrum f(α). | Complete |
| **Flint Hills** | Partial sums to 10¹⁰. | Complete |

## Review Infrastructure

Every finding is AI-audited claim-by-claim by multiple models. Currently **53 reviews** from **7 models** across **3 providers**. 210 issues discovered, 207 resolved (98.6%).

```bash
# Run a review (any OpenAI-compatible API)
export API_KEY='...'
python3 scripts/reviews/run_review.py --slug kronecker-s40-character-table --model gemini-2.5-flash --provider google --api-base https://generativelanguage.googleapis.com/v1beta/openai

# Rebuild manifest + website data
python3 scripts/reviews/aggregate.py
python3 scripts/reviews/sync_website.py
```

- **Review scripts**: [`scripts/reviews/`](scripts/reviews/) — run, aggregate, validate, sync
- **Manifest**: [`docs/verifications/manifest.json`](docs/verifications/manifest.json) — single source of truth
- **Remediations**: [`docs/verifications/remediations/`](docs/verifications/remediations/) — per-finding issue tracking with commit links
- **PR Bot**: [`.github/workflows/pr-review.yml`](.github/workflows/pr-review.yml) — auto-validates PRs, scans for secrets, labels

## Research Agent

Autonomous loop: Monitor → Harvest → Analyze → Review → Remediate → Deploy → Plan.

```bash
./scripts/run_agent.sh                         # one cycle
./scripts/run_agent.sh --loop 10m              # autonomous
./scripts/run_agent.sh --loop 10m --auto-launch # + launch experiments on free GPUs
```

Works with any ONE of: Claude Code (no key needed), Anthropic API, OpenAI API, or Gemini API (free). Auto-detects what's available. Default: creates branch + PR (safe). `--direct-push` for repo owner.

Source: [`scripts/research_agent.py`](scripts/research_agent.py)

## Structure

```
scripts/experiments/             CUDA kernels and Python harnesses per experiment
scripts/reviews/                 AI peer review infrastructure
scripts/research_agent.py        Autonomous research loop
docs/verifications/              Review JSONs, manifest, remediations
docs/verifications/remediations/ Per-finding issue tracking with commit links
paper/                           Zaremba proof paper (LaTeX + PDF)
data/                            Raw computation output (large files on HF)
logs/                            Computation logs
.github/workflows/               PR bot (auto-validate, scan for secrets, label)
```

## Hardware

| Environment | GPUs | VRAM | Role |
|---|---|---|---|
| **B200 Cluster** | 8× NVIDIA B200 | 1.43 TB (NVLink 5) | Primary compute |
| **Local** | RTX 5090 | 32 GB | Development + smaller experiments |
| **Colab** | T4 (free) / A100 / L4 | 16-80 GB | Distributed contributions |

## Contribute

1. **Colab** — [Open notebook](https://colab.research.google.com/github/cahlen/bigcompute.science/blob/main/public/notebooks/bigcompute_research_agent.ipynb), compile kernels on free GPU, run experiments
2. **Agent** — Clone, set one API key, `./scripts/run_agent.sh`
3. **PR** — Add review JSONs or experiment logs, PR bot validates automatically

See [AGENTS.md](AGENTS.md) for the full contribution guide.

## Related

- **[bigcompute.science](https://bigcompute.science)** — Results + audit dashboard
- **[MCP Server](https://mcp.bigcompute.science/mcp)** — 23 tools, no auth (arXiv, zbMATH, OEIS, LMFDB, Lean/Mathlib)
- **[Hugging Face](https://huggingface.co/cahlen)** — Datasets (class numbers, Kronecker, spectra, Zaremba)
- **[llms.txt](https://bigcompute.science/llms.txt)** — Agent-consumable structured index
