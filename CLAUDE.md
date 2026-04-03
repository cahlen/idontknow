# Project Context

## What This Is
GPU-accelerated computational mathematics infrastructure for exploring open conjectures and underexplored areas of math. Uses CUDA kernels for heavy verification, LLM-assisted theorem proving for formalization, and publishes all results openly at bigcompute.science.

This is NOT just a Lean 4 proving repo — Lean is one tool among many. The core mission is running expensive GPU computations on open problems (Zaremba, Ramsey, class numbers, Kronecker coefficients) and publishing verifiable, agent-discoverable results.

## The User
- **cahlen** on GitHub and Hugging Face
- Has an RTX 5090 (32GB) + Intel Core Ultra 9 285K + 188GB RAM locally
- Has access to an 8xB200 cluster (~1.4TB VRAM) via VPN
- Already contributed GGUF quants to HF: https://huggingface.co/cahlen/qwen3.5-35b-a3b-compacted-GGUF
- Interested in using AI to explore areas of math that don't get a spotlight — continued fractions, number theory, combinatorics
- NOT interested in video generation models

## Data Preservation — CRITICAL

Any raw data processed on the GPU cluster MUST be preserved and documented:
- **Small data** (< 100MB): commit directly to the `data/` directory in this repo with a descriptive log file
- **Large data** (> 100MB): upload to a corresponding Hugging Face dataset repo (https://huggingface.co/cahlen) with a README documenting the computation parameters, hardware, date, and how to reproduce
- **All computations** must have their output logged to a file (never just stdout) with timestamps, parameters, and aggregate statistics
- **Intermediate results** (e.g., per-chunk statistics) should be saved incrementally so partial runs are not lost
- **Log files** go in `logs/` or `data/<experiment>/` with descriptive names like `run_1e9_1e10.log`

## Security — CRITICAL

This repository will be public and autonomous AI agents will push commits to it. **Nothing secret can ever be committed.**

### Never commit:
- Private keys, SSH keys, API tokens, passwords, credentials
- `.env` files or any file containing secrets
- Hugging Face tokens, cloud provider keys, VPN configs
- Model weights (`.gguf`, `.safetensors`, `.bin`) — downloaded at runtime, not stored

### Before every commit:
- Verify no secrets in staged files
- Check that `.gitignore` covers any new sensitive file patterns
- If in doubt, don't commit it

## Current Experiments

### Zaremba's Conjecture (Primary — in progress)
*For every d >= 1, there exists a coprime to d whose continued fraction has all partial quotients <= 5.*

**What's done:**
- [x] CUDA v4 kernel: 8B+ values verified, zero failures
- [x] Transfer operator: δ = 0.836829443681208 (15 digits), spectral gaps uniform ≥ 0.237 for m ≤ 1999
- [x] Transitivity: algebraic argument for all primes via Dickson classification (AI-assisted, not peer-reviewed)
- [x] LLM proving: 19/20 small cases formally verified in Lean 4 (dual-model race, Goedel-Prover + Kimina-Prover)
- [x] Witness distribution: a/d ≈ 0.171 concentration, golden ratio connection
- [x] Fix d=9 failure: sweet-spot band [0.165d, 0.185d] collapsed for small d; added full search for d < 50
- [ ] MCTS proof search benchmark (planned)
- [ ] Make Bourgain-Kontorovich bound effective (Q₀ extraction)

### Ramsey R(5,5) (Complete — strongest computational evidence R(5,5) = 43)
*Search for 2-colorings of K₄₄ with no monochromatic K₅.*

**What's done:**
- [x] Fixed critical initialization bug (adj[i]=0 inside loop destroyed back-edges) across all 7 CUDA kernels
- [x] Incremental K₅ counter verified correct: 0 drift in 100 steps at n=43, 332M flips/sec on 8xB200
- [x] SA search saturates at fitness ~127-134 for n=43; random search cannot find solutions
- [x] Exhaustive extension: checked ALL 2^42 = 4.4x10^12 extensions of Exoo's K₄₂ coloring to K₄₃ — zero valid (130 sec on 8xB200)
- [x] 4-SAT reformulation: checked ALL 656 known K₄₂ colorings (McKay-Radziszowski database) — NONE extend to K₄₃ (3 sec on 8xB200)
- [x] This is the strongest computational evidence ever assembled that R(5,5) = 43
- [x] Direct K₄₃ SAT (903 vars, 1.9M clauses) — naive CDCL intractable (98 solvers, 2 hours, no progress). Needs degree constraints (R(4,5)=25 → 18≤deg_red≤24), full symmetry breaking, flag algebra cutting planes.
- [ ] Mathematically-informed K₄₃ SAT encoding (degree bounds, Turán density, BreakID symmetry breaking)

### Class Numbers of Real Quadratic Fields (Complete to 10^11)
Extend tables from 10^11 to 10^13 using CUDA + BSGS. Test Cohen-Lenstra heuristics at scale.

### Kronecker Coefficients (In Progress — S₄₀ complete, S₁₂₀ planned)
Complete character tables and Kronecker coefficients for S₂₀ (3.7s), S₃₀ (7.4 min), S₄₀ (9.5 hr char table). S₄₀ values exceed int64 (max |χ| = 5.9×10²²) — full triple-sum needs int128 GPU kernel. Targeted analysis: 94.9% nonzero (sampled), hooks multiplicity-free, near-rectangular GCT triples sparse (10.1%). Next: int128 kernel for full S₄₀ table, then push toward n=120 for geometric complexity theory.

### Hausdorff Dimension Spectrum (In Progress — RTX 5090)
First complete computation of dim_H(E_A) for all 2^20 - 1 = 1,048,575 subsets A ⊆ {1,...,20}. Transfer operator + Chebyshev collocation on RTX 5090. Validated against Jenkinson-Pollicott (E_{1,2}) and Zaremba (E_{1,...,5}). Dataset does not exist anywhere in the literature.

## Publishing Pipeline
Results from this repo are published to **bigcompute.science** (sibling repo):
1. Run experiment here (CUDA kernel, LLM prover, etc.)
2. Record results in `docs/log.md` with session detail
3. Write/update experiment markdown in bigcompute.science with YAML frontmatter
4. Push raw data to `bigcompute.science/public/data/`
5. `scripts/watch-v4.sh` and `scripts/pipeline.sh` can automate steps 2-4

## Tech Stack
- **CUDA** (custom kernels) for GPU-accelerated verification and computation
- **Lean 4** (v4.29.0-rc8 via elan) + **Mathlib** for formal verification
- **vLLM / SGLang** for LLM model serving (tensor parallel across GPUs)
- **Python 3.12** for experiment harnesses and the proving loop
- **llama.cpp** for GGUF quantization (built with CUDA at /home/cahlen/dev/llama.cpp on local machine)

## Key Research Context
- SOTA proving uses Reinforcement Learning with Verifiable Rewards (RLVR) — Lean compiler as binary reward
- Inference-time scaling matters more than model size — small models + massive search beat large models
- Best search: Best-First Tree Search (Seed Prover), AlphaZero MCTS (AlphaProof), Particle Filter Monte Carlo
- Continued fractions / number theory are highly amenable to MCTS exploration and formal verification

### Ramanujan Machine (In Progress — GPU formula discovery)
*Discover new continued fraction formulas for mathematical constants.*

**What's done:**
- [x] CGBN (CUDA big number) library installed
- [x] Experiment page and methodology designed
- [x] CUDA kernel v3: 10 base constants + 29 compound expressions + zero-value filter
- [x] Degree 1-3 complete: 42B+ candidates, zero transcendental hits
- [x] Degree 2 exhausted at range [-20,20]: only sqrt(2) and sqrt(5)
- [x] Degree 4-7 complete: 586B candidates, zero confirmed transcendentals
- [ ] Degree 4-6 full sweep with high-precision PSLQ verification
- [ ] GPU PSLQ implementation for eliminating double-precision false positives

### Zaremba Density (In Progress — GPU density computations)
*Zaremba density phase transition and exception set analysis.*

**What's done:**
- [x] Complete density sweep: all 1,023 subsets of {1,...,10} at 10^6
- [x] {1,k} pair hierarchy at 10^10 for k=2..10 (exponential decay ~k^{-3.5})
- [x] {2,k} pair hierarchy at 10^10 for k=3..10 (digit 1 amplifies 42-243x)
- [x] Four closed exception sets confirmed: {1,2,3}=27, {1,2,4}=64, {1,2,5}=374, {1,2,6}=1,834
- [x] A={1,2} logarithmic convergence: 5 data points (10^6 through 10^12), fits 31.5 + 4.47*log10(N)
- [x] A={2,3,4,5} non-monotone convergence: 97.3→97.1→98.8% at 10^{9,10,11}
- [x] A={3,...,10} at 10^10: 81.70% (8 digits without 1 or 2)
- [ ] Confirm {1,2,3} 27 exceptions at 10^11 (running)
- [ ] Confirm {1,2,4} 64 exceptions at 10^11 (running)
- [ ] A={1,2,3} at 10^12 — does 27 hold at next decade? (running)
- [ ] Complete density sweep at 10^9 for all 1,023 subsets
