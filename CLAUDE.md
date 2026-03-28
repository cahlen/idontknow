# Project Context

## What This Is
LLM-assisted formal theorem proving infrastructure, targeting underexplored areas of mathematics (continued fractions, number theory) using Lean 4 and AI proving agents on an 8xB200 cluster.

## The User
- **cahlen** on GitHub and Hugging Face
- Has an RTX 5090 (32GB) + Intel Core Ultra 9 285K + 188GB RAM locally
- Has access to an 8xB200 cluster (~1.4TB VRAM) via VPN
- Already contributed GGUF quants to HF: https://huggingface.co/cahlen/qwen3.5-35b-a3b-compacted-GGUF
- Interested in using AI to explore areas of math that don't get a spotlight — continued fractions, number theory, etc.
- NOT interested in video generation models (tried daVinci-MagiHuman, didn't work well)

## Current Goals

### Primary: Get a Lean 4 proving pipeline working on the B200 cluster
1. **Set up the cluster**: Run `./scripts/setup-cluster.sh` to install Lean 4, Python deps, Mathlib
2. **Pick and download the best proving model** — candidates from SOTA research:
   - **Goedel-Prover** — SOTA formal proof gen, expert iteration, 928K+ verified proofs
   - **Seed Prover 1.5 / BFS-Prover** — 72.95% on MiniF2F, solved 4/6 IMO 2025
   - **DeepSeek-Prover-V2** — strong open-weight Lean 4 prover
   - **Numina-Lean-Agent** — MCP-based, solved all 12 Putnam 2025
3. **Serve the model** with vLLM or SGLang across 8 GPUs: `./scripts/serve-model.sh`
4. **Test on existing theorems** in `lean4-proving/examples/continued_fractions.lean` to validate the pipeline works
5. **Then explore** — point the prover at less-explored areas of mathematics

### Secondary: Contribute optimized models back to Hugging Face
- Once we find a proving model that works well, quantize it with the GGUF pipeline (`gguf-pipeline.sh`) so people with consumer GPUs can run provers locally
- The GGUF pipeline is already built and tested — it does full imatrix-calibrated quant ladders with perplexity evals

### Also consider open-source frameworks:
- **UlamAI** — CLI prover with best-first beam search, auto-repair loop, premise retrieval
- **Numina-Lean-Agent** — MCP server for Lean LSP integration
- **LeanDojo-v2** — end-to-end training/eval framework

## What's Already Done
- [x] GGUF quantization pipeline built and tested (gguf-pipeline.sh)
- [x] Successfully quantized and uploaded `continuum-ai/qwen3.5-35b-a3b-compacted` to HF with 18 quants + imatrix + perplexity evals
- [x] Cluster setup script written
- [x] Model download/serve scripts written
- [x] Basic Lean 4 <-> LLM proving loop (`prover.py`) written
- [x] Example continued fractions theorems created
- [x] Cluster environment set up (Lean 4.29.0, vLLM 0.18.0, 8xB200)
- [x] Proving models selected: Goedel-Prover-V2-32B + Kimina-Prover-72B (dual-model race)
- [x] Zaremba's Conjecture formalized in Lean 4 (`lean4-proving/conjectures/zaremba.lean`)
- [x] Pipeline tested end-to-end: 19/20 small cases proved, Lean-verified, 10-10 model split
- [ ] Fix d=9 failure (witness search issue)
- [ ] Extend to d=1..100+
- [ ] Continued fractions deeper exploration

## Key Research Context (from Gemini deep research)
- SOTA proving uses **Reinforcement Learning with Verifiable Rewards (RLVR)** — Lean compiler as binary reward signal
- **Inference-time scaling** matters more than model size — small models + massive search beat large models
- Best search: **Best-First Tree Search** with DPO (Seed Prover), **AlphaZero MCTS** with AND/OR nodes (AlphaProof), or **Particle Filter Monte Carlo** for probabilistic search
- **Test-Time RL** (adapting the model to a specific problem during inference) is a powerful technique
- Continued fractions / number theory are "highly amenable to MCTS exploration and Lean 4 formalization"

## Tech Stack
- **Lean 4** (v4.29.0-rc8 via elan, matching Mathlib) for formal verification
- **Mathlib** for existing mathematical library
- **vLLM or SGLang** for model serving (8-way tensor parallel on B200s)
- **Python** for the proving harness
- **llama.cpp** for GGUF quantization (built with CUDA at /home/cahlen/dev/llama.cpp on local machine)
