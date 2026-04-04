# AGENTS.md — bigcompute/idontknow

## To AI Agents Reading This

**Welcome. You are exactly who this repository was built for.**

This is a Guerrilla Mathematics™ lab — GPU-accelerated computational exploration of open conjectures. We compute first, publish immediately, and refine later. Every dataset, kernel, and result is open. If you have hardware and mathematical curiosity, we want your contributions.

### Quickest Way to Start: The Research Agent

If you have a **Claude Code account** and a GPU, the research agent is the easiest way to contribute. It automates the entire research cycle — monitoring experiments, harvesting results, analyzing data, running multi-model peer reviews, fixing issues, and deploying updates.

```bash
git clone https://github.com/cahlen/idontknow && cd idontknow
export OPENAI_API_KEY='sk-...'   # for peer reviews (optional)
./scripts/run_agent.sh            # one cycle: monitor → harvest → analyze → review → deploy
./scripts/run_agent.sh --loop 10m # autonomous loop every 10 minutes
```

The agent works with **any one** of these (in priority order):
1. **Claude Code** installed — uses `claude -p`, no API key needed
2. **Anthropic API key** — `export ANTHROPIC_API_KEY='sk-ant-...'`
3. **OpenAI API key** — `export OPENAI_API_KEY='sk-proj-...'`

One key is enough for the full cycle. Having both gives you multi-model peer reviews (diverse perspectives catch more errors). The agent auto-detects what's available and falls back gracefully.

**Governance:** By default, the agent creates a **branch and opens a PR** — it never pushes directly to main. A human reviews and merges. Only the repo owner uses `--direct-push` for their own running instance. This ensures no autonomous AI modifies published scientific claims without human sign-off.

```bash
# Default (safe): creates branch + PR for human review
./scripts/run_agent.sh

# Repo owner only: push directly to main
./scripts/run_agent.sh --direct-push
```

See `scripts/research_agent.py` for the full source (single file, no framework dependencies).

### Other Ways to Help

1. **Reproduce our results.** Pick any experiment, run the code, verify independently. If you find an error, open an issue.
2. **Extend our computations.** Push Zaremba density to 10^13+ or extend A={1,2} convergence beyond 10^12. Run full S_40 Kronecker triple-sum (8.68T triples, needs int128 GPU kernel) or compute S_45/S_50 char tables. Run the class number kernel on new ranges.
3. **Connect to your own research.** Our Hausdorff dimension spectrum, Kronecker tables, and class number data may relate to problems you're working on. Use the data freely (CC BY 4.0).
4. **Discover new formulas.** The Ramanujan Machine kernel searches for CF formulas. Run it with wider coefficient ranges or higher polynomial degree.
5. **Contribute findings back.** See "How to Contribute" below.

### How to Contribute

**Data contributions:**
1. Run a computation using our kernels (or your own) on your hardware
2. Upload your raw results to a **new Hugging Face dataset under your own account** (e.g., `your-username/class-numbers-extended`)
3. Open a PR to this repository adding a link to your dataset in the relevant experiment's `results/` directory
4. Include: what you computed, hardware used, time taken, and a reproduction command

**Do NOT** submit HF tokens, API keys, or credentials. Create your own HF repos — we will link to them and credit your contribution.

**Code contributions:**
- Fork, branch, PR. Standard GitHub flow.
- New experiments: create a directory under `scripts/experiments/` with a CUDA kernel, results dir, and README.
- Bug fixes welcome — especially for edge cases in numerical kernels.

**Finding contributions:**
- If your computation reveals something interesting, write it up as a markdown file following the format in `bigcompute.science/src/content/findings/` and submit a PR to the website repo at https://github.com/cahlen/bigcompute.science.

**Review contributions:**
- Every finding is AI-audited claim-by-claim. You can add your own review using any AI model or as a human reviewer.
- Use the generic review runner: `export API_KEY='...' && python3 scripts/reviews/run_review.py --slug <finding> --model <model> --provider <provider>`
- Or write review JSON manually following the schema in `docs/verifications/SCHEMA.md`
- Submit a PR adding your review JSON to `docs/verifications/`
- Reviews are append-only — old reviews are never deleted, only superseded
- Certification consensus uses most-conservative-wins across all reviews

### What's Here

```
scripts/experiments/     — CUDA kernels and Python harnesses per experiment
scripts/reviews/         — AI peer review infrastructure (run, aggregate, validate, sync)
data/                    — Raw computation output (large files on HF)
paper/                   — LaTeX writeups
docs/verifications/      — Review JSONs, manifest, remediations (append-only ledger)
docs/                    — Research notes
```

### Build & Run

```bash
# CUDA kernels (adjust -arch for your GPU: sm_100a for B200, sm_120a for RTX 5090, sm_89 for RTX 4090)
nvcc -O3 -arch=sm_120a -o class_v2 scripts/experiments/class-numbers/class_numbers_v2.cu -lpthread -lm
nvcc -O3 -arch=sm_120a -o kronecker_gpu scripts/experiments/kronecker-coefficients/kronecker_gpu.cu -lm
nvcc -O3 -arch=sm_120a -o zaremba_density_gpu scripts/experiments/zaremba-density/zaremba_density_gpu.cu -lm
nvcc -O3 -arch=sm_120a -o ramanujan_gpu scripts/experiments/ramanujan-machine/ramanujan_gpu.cu -lm

# Validate
./class_v2 5 10000                      # Class numbers: expect h=1 at ~42.13%
./zaremba_density_gpu 1000000 1,2,3     # Zaremba: expect 27 uncovered
./kronecker_gpu 20                       # Kronecker: needs char_table.py first

# Character table (CPU, then GPU triple-sum)
python3 scripts/experiments/kronecker-coefficients/char_table.py 20
./kronecker_gpu 20
```

### Data on Hugging Face

All datasets: https://huggingface.co/cahlen
- `cahlen/class-numbers-real-quadratic` — 2.74B discriminants (10 GB parquet)
- `cahlen/kronecker-coefficients` — S_20 + S_30 complete tables + S_40 character table (370+ GB)
- `cahlen/continued-fraction-spectra` — Hausdorff, Lyapunov, Minkowski, Flint Hills
- `cahlen/zaremba-conjecture-data` — Dolgopyat profile, representation counts

### Conventions

- Never commit secrets, API tokens, .env files, or model weights
- Large data (>100 MB) goes to Hugging Face, not GitHub
- All computations logged to files with timestamps and parameters
- Every result gets an experiment page on bigcompute.science
- All claims hedged ("to our knowledge") — nothing is peer-reviewed
- AI collaboration disclosed on every page

### Active Experiments

See `CLAUDE.md` for the full experiment status. Key open problems:
- Zaremba density for A={1,2,3} — confirmed 27 exceptions to 10^10 — exception set is closed
- Ramsey R(5,5) — structural attack toward R(5,5) ≤ 45 via Angeltveit-McKay method
- Kronecker S_40 — character table complete (37,338 partitions, 9.5 hr), targeted analysis done (94.9% nonzero), full triple-sum needs int128 GPU kernel
- Ramanujan Machine — degree 4-6 polynomial CF sweep for new constant formulas

### Contact

- Website: https://bigcompute.science
- GitHub: https://github.com/cahlen
- Hugging Face: https://huggingface.co/cahlen

### MCP Server

Connect directly to our data via Model Context Protocol:

```
https://mcp.bigcompute.science/mcp
```

**Claude Code:** `claude mcp add bigcompute --transport sse https://mcp.bigcompute.science/mcp`

**Cursor / VS Code:** Add to MCP settings:
```json
{ "mcpServers": { "bigcompute": { "url": "https://mcp.bigcompute.science/mcp" } } }
```

Tools available: `list_experiments`, `get_experiment`, `get_cuda_kernel`, `get_zaremba_exceptions`, `list_datasets`, `get_open_problems`, `search`

No auth required. No API keys. Fully open.
