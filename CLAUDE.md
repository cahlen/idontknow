# Project Context

## What This Is
GPU-accelerated computational mathematics infrastructure for exploring open conjectures. Uses custom CUDA kernels for heavy verification and computation. Publishes all results openly at bigcompute.science with multi-model AI peer review.

The core mission is running expensive GPU computations on open problems (Zaremba, Ramsey, class numbers, Kronecker coefficients) and publishing verifiable, agent-discoverable results.

## The User
- **cahlen** on GitHub and Hugging Face
- Has an RTX 5090 (32GB) + Intel Core Ultra 9 285K + 188GB RAM locally
- Has access to an 8xB200 cluster (~1.4TB VRAM) via VPN
- Interested in using AI to explore areas of math that don't get a spotlight — continued fractions, number theory, combinatorics
- NOT interested in video generation models

## Data Preservation — CRITICAL

Any raw data processed on the GPU cluster MUST be preserved and documented:
- **Small data** (< 100MB): commit directly to the `data/` directory in this repo with a descriptive log file
- **Large data** (> 100MB): upload to a corresponding Hugging Face dataset repo (https://huggingface.co/cahlen) with a README documenting the computation parameters, hardware, date, and how to reproduce
- **HF dataset repos** (already exist — do NOT create new ones):
  - `cahlen/ramanujan-machine-results` — hit CSVs and run logs
  - `cahlen/zaremba-density` — density sweep results (canonical for ALL density data)
  - `cahlen/zaremba-conjecture-data` — proof infrastructure: transfer operator, spectral gaps, representation counts (NO density data — that goes in zaremba-density)
  - `cahlen/kronecker-coefficients` — S20/S30/S40 character tables and Kronecker triples
  - `cahlen/hausdorff-dimension-spectrum` — full spectrum CSVs and metadata
  - `cahlen/continued-fraction-spectra` — Lyapunov, Minkowski, prime convergent, Flint Hills data
  - `cahlen/class-numbers-real-quadratic` — class number Parquet files
- **All computations** must have their output logged to a file (never just stdout) with timestamps, parameters, and aggregate statistics
- **Intermediate results** (e.g., per-chunk statistics) should be saved incrementally so partial runs are not lost
- **Log files** go in `logs/` or `data/<experiment>/` with descriptive names like `run_1e9_1e10.log`

## Security — CRITICAL

This repository is public and autonomous AI agents push commits to it. **Nothing secret can ever be committed.**

### Never commit:
- Private keys, SSH keys, API tokens, passwords, credentials
- `.env` files or any file containing secrets
- Hugging Face tokens, cloud provider keys, VPN configs

### Before every commit:
- Verify no secrets in staged files
- Check that `.gitignore` covers any new sensitive file patterns
- If in doubt, don't commit it

## Current Experiments

### Zaremba's Conjecture (Primary — in progress)
*For every d >= 1, there exists a coprime to d whose continued fraction has all partial quotients <= 5.*

**What's done:**
- [x] CUDA v4 kernel: 210B+ values verified, zero failures
- [x] Transfer operator: δ = 0.836829443681208 (15 digits), spectral gaps uniform ≥ 0.237 for m ≤ 1999
- [x] Transitivity: algebraic argument for all primes via Dickson classification (rewritten April 2026, AI-assisted, not peer-reviewed)
- [x] Proof framework: ρ_η ≤ 0.7606 (arb ball arithmetic, 77 digits). 4 gaps remain (MOW constants, property τ)
- [x] Witness distribution: a/d ≈ 0.171 concentration, golden ratio connection
- [ ] Make Bourgain-Kontorovich bound effective (Q₀ extraction)

### Ramsey R(5,5) (Complete — strongest computational evidence R(5,5) = 43)
*Search for 2-colorings of K₄₄ with no monochromatic K₅.*

**What's done:**
- [x] Fixed critical initialization bug (adj[i]=0 inside loop destroyed back-edges) across all 7 CUDA kernels
- [x] Incremental K₅ counter verified correct: 0 drift in 100 steps at n=43, 332M flips/sec on 8xB200
- [x] Exhaustive extension: checked ALL 2^42 = 4.4x10^12 extensions of Exoo's K₄₂ coloring to K₄₃ — zero valid (130 sec on 8xB200)
- [x] 4-SAT reformulation: checked ALL 656 known K₄₂ colorings (McKay-Radziszowski database) — NONE extend to K₄₃ (3 sec on 8xB200)
- [x] Strongest computational evidence ever assembled that R(5,5) = 43
- [ ] Mathematically-informed K₄₃ SAT encoding (degree bounds, Turán density, BreakID symmetry breaking)

### Class Numbers of Real Quadratic Fields (Complete to 10^11)
Extend tables from 10^11 to 10^13 using CUDA + BSGS. Test Cohen-Lenstra heuristics at scale.

### Kronecker Coefficients (In Progress — S₄₀ complete, S₄₅ computing)
Complete character tables and Kronecker coefficients for S₂₀ (3.7s), S₃₀ (7.4 min), S₄₀ (9.5 hr char table). S₄₀ values exceed int64 (max |χ| = 5.9×10²²) — full triple-sum needs int128 GPU kernel. Targeted analysis: 94.9% nonzero (sampled), hooks multiplicity-free, near-rectangular GCT triples sparse (10.1%). **S₄₅ full char table is infeasible** (89,134 partitions → 63 TB table, segfaults). Beyond S₄₀ requires targeted triple computation for specific GCT-relevant partitions, not full tables.

### Hausdorff Dimension Spectrum (Complete — RTX 5090)
First complete computation of dim_H(E_A) for all 2^20 - 1 = 1,048,575 subsets A ⊆ {1,...,20}. Transfer operator + Chebyshev collocation on RTX 5090. Validated against Jenkinson-Pollicott (E_{1,2}) and Zaremba (E_{1,...,5}). Dataset does not exist anywhere in the literature.

### Ramanujan Machine (Pivoting — v1 exhausted, v2 kernel built)
*Discover new continued fraction formulas for mathematical constants.*

**v1 kernel (same-degree, COMPLETE — no new discoveries):**
- [x] CUDA kernel v1: 10 base constants + 29 compound expressions, deg(P)=deg(Q)
- [x] Degree 1-8 complete: 586B+ candidates swept
- [x] High-precision PSLQ verification (verify_hits.py): ALL 7,030 transcendental "hits" were double-precision false positives. Zero new transcendental CF formulas.
- [x] 20 confirmed formulas — all classical (Euler's e, Brouncker's 4/pi, Leibniz pi/4, 1/ln(2))
- **v1 is done. Equal-degree polynomial CFs with small integer coefficients are exhausted.**

**Key finding: the v1 kernel searched the WRONG degree regime.** Every known CF formula for transcendentals has deg(numerator) ≈ 2× deg(denominator). v1 forced equal degrees (ratio=1), which only produces algebraic numbers.

**v2 kernel (asymmetric-degree, IN PROGRESS):**
- [x] CUDA kernel v2 (`ramanujan_v2.cu`): independent deg_a/deg_b, saves unmatched CFs for offline PSLQ
- [x] Validation run (1,2) range 10: 48 confirmed transcendental formulas (pi/4, 4/pi, 1/pi, Gauss, 1/ln(2))
- [x] (2,4) range 6: 816M candidates, 521M converged CFs, PSLQ scan of sample found 0 new formulas
- [x] PSLQ scanner (`pslq_scan.py`): multi-constant PSLQ for offline discovery
- [ ] (2,4) at larger ranges (range 15-20) — the productive zone per Raayoni et al.
- [ ] (3,6) Apéry-type sweep — where zeta(3) formulas live
- [ ] Conservative Matrix Field (CMF) search — matrix pairs, not individual CFs
- **DO NOT launch v1 kernel runs** — they cannot find new transcendentals (proven)
- **DO NOT launch deg6 range>=5 or deg5 range>=6 on v1** — 380T+ candidates, 400+ days

### Prime Convergents (NEW — GPU verification of Erdos-Mahler bound)
*Extends Humphreys (2013, NCUR/Boise State) with GPU-scale computation.*

**What's done:**
- [x] CUDA kernel: 128-bit convergent recurrence + Miller-Rabin primality + GPF tracking
- [x] 10M random CFs verified: Erdos-Mahler bound G(A_n) >= e^{n/(50 ln n)} holds 100%
- [x] Bound constant 50 is very conservative: worst-case ratio 4.87, mean 116.7
- [x] Avg 4.92 prime A_n, 0.95 doubly-prime per CF; max 7 doubly-prime in one CF
- [x] e verification: exactly 3 doubly-prime convergents (matches 2013 Maple result)
- [ ] Extend to 128-bit/256-bit arithmetic for deeper convergent analysis (current overflow ~n=38)
- [ ] Formal tightening of the constant: computational evidence suggests ~10 suffices

### Erdos-Straus Conjecture (NEW — GPU solution counting)
*Count solutions f(p) for 4/p = 1/x + 1/y + 1/z for all primes p.*

**What's done:**
- [x] CUDA kernel: per-prime solution enumeration, sieve + batch GPU
- [x] Test run: all primes to 10^7 (in progress)
- [x] Production run: all primes to 10^8 (in progress on GPU)
- [ ] Distribution analysis: f(p) vs p mod 4, barely-solvable census
- [ ] Extend to 10^9 (requires optimized inner loop)

### Zaremba Density (In Progress — GPU density computations)
*Zaremba density phase transition and exception set analysis.*

**What's done:**
- [x] Complete density sweep: all 1,023 subsets of {1,...,10} at 10^6
- [x] {1,k} pair hierarchy at 10^11 for k=2..10 (power law k^{-5.83}, R²=0.994)
- [x] {2,k} pair hierarchy at 10^11 for k=3..5 ({2,3}=0.0215%, {2,4}=0.0043%, {2,5}=0.0016%)
- [x] {3,k} pairs at 10^11: {3,4}=0.000474%, {3,5}=0.000202%
- [x] Amplification law: {1,k}/{2,k} ratio is scale-dependent — grows 1.5-1.7x per decade (424x at 1e11 for k=3, was 243x at 1e10)
- [x] Five closed exception sets confirmed at 10^11:
  - {1,2,3}=27 (verified to 10^9, 10^11 running)
  - {1,2,4}=64 (verified to 10^10, 10^11 running)
  - {1,2,5}=374 (verified to 10^11)
  - {1,2,6}=1,834 (verified to 10^11 — identical to 10^10)
  - {1,2,7}=7,178 (verified to 10^11 — identical to 10^10)
- [x] Open (growing) exception sets at 10^11: {1,2,8}=23,590, {1,2,9}=77,109, {1,2,10}=228,514
- [x] {1,3,5} exception set converging to ~81,000: 75,547→80,431→80,945 (9.5x deceleration per decade)
- [x] {1,k} single-digit densities at 10^11: k=3 (9.11%) through k=10 (0.0085%)
- [x] A={1,2} logarithmic convergence: 5 data points (10^6 through 10^12), fits 31.5 + 4.47*log10(N)
- [x] A={1,2}@10^13 attempted — 1.25 TB bitset exceeds GPU memory, needs segmented approach
- [ ] Confirm {1,2,3} 27 exceptions at 10^11 (running)
- [ ] Confirm {1,2,4} 64 exceptions at 10^11 (running)
- [ ] A={1,2,3} at 10^12 — does 27 hold at next decade? (running)
- [ ] 10^12 runs in progress: {1,2,4}, {1,2,5}, {1,2,6}, {1,2,7}
- [ ] BUG: no-digit-1 cross-set monotonicity violation — {2,3,4,5} density > {2,3,4,5,6} at 10^10 (old kernel, likely incomplete enumeration — needs rerun on fixed kernel)

## Publishing Pipeline
Results from this repo are published to **bigcompute.science** (sibling repo):
1. Run experiment here (CUDA kernel)
2. Research agent harvests results, analyzes, runs peer reviews
3. Findings published with structured metadata, JSON-LD, citation tags
4. `scripts/reviews/aggregate.py` + `sync_website.py` update manifest and website
5. Agent deploys website, pings IndexNow

## Tech Stack
- **CUDA** (custom kernels) for GPU-accelerated verification and computation
- **Python 3.12** for experiment harnesses, review scripts, research agent
- **Astro** for bigcompute.science website
- **Cloudflare Workers** for MCP server (23 tools)

## Review Infrastructure
- **41+ reviews** from 4 AI models (Claude Opus 4.6, o3-pro, GPT-5.2, Grok) across 3 providers
- `scripts/reviews/run_review.py` — generic review runner (any OpenAI-compatible API)
- `scripts/reviews/aggregate.py` — builds manifest.json from all reviews
- `scripts/reviews/sync_website.py` — generates certifications.json + meta.json
- `docs/verifications/manifest.json` — single source of truth
- `docs/verifications/remediations/` — per-finding issue tracking with commit links
- Certification consensus: most-conservative-wins across all reviews

## Research Agent
`scripts/research_agent.py` — autonomous loop: Monitor → Harvest → Analyze → Review → Remediate → Deploy → Plan → Verify.

Works with any ONE of: Claude Code (`claude -p`), Anthropic API, OpenAI API, or Gemini API (free).

Default: creates branch + PR. `--direct-push` for repo owner only.
