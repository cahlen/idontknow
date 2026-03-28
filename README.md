# LLM-Assisted Lean 4 Theorem Proving

Infrastructure for running AI-assisted formal theorem proving on an 8xB200 cluster, with a focus on underexplored areas of mathematics like continued fractions.

## Quick Start (B200 Cluster)

```bash
git clone git@github.com:cahlen/idontknow.git
cd idontknow
chmod +x scripts/*.sh
./scripts/setup-cluster.sh
./scripts/download-model.sh <model_id>
./scripts/serve-model.sh models/<model_name> --engine vllm --tp 8
# In another terminal:
python lean4-proving/prover.py --server http://localhost:8000 --file lean4-proving/examples/continued_fractions.lean
```

## Structure

```
scripts/
  setup-cluster.sh      # One-time cluster setup (Lean 4, Python, Mathlib)
  download-model.sh     # Download a proving model from HF
  serve-model.sh        # Serve model via vLLM or SGLang (8-way tensor parallel)
lean4-proving/
  prover.py             # LLM <-> Lean 4 proving loop
  examples/
    continued_fractions.lean   # Test theorems for continued fractions
gguf-pipeline.sh        # GGUF quantization pipeline (for HF contributions)
```

## Models to Evaluate

Candidates identified from SOTA research (pending evaluation):
- **Goedel-Prover** — SOTA formal proof generation, expert iteration trained
- **BFS-Prover / Seed Prover 1.5** — 72.95% on MiniF2F, streamlined best-first search
- **DeepSeek-Prover-V2** — strong open-weight Lean 4 prover
- **Numina-Lean-Agent** — MCP-based Lean 4 interaction, solved Putnam 2025

## Hardware

- **Cluster**: 8x NVIDIA B200 (~1.4TB VRAM total)
- **Local**: NVIDIA RTX 5090 32GB / Intel Core Ultra 9 285K / 188GB RAM
