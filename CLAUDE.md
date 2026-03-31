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
- [ ] Fix d=9 failure (witness search issue)
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
- [ ] Direct K₄₃ SAT (903 vars, 1.9M clauses) remains open — this IS the open problem

### Class Numbers of Real Quadratic Fields (Planned)
Extend tables from 10^11 to 10^13 using CUDA + BSGS. Test Cohen-Lenstra heuristics at scale.

### Kronecker Coefficients (Planned)
GPU-accelerated computation to n=120 for geometric complexity theory.

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
