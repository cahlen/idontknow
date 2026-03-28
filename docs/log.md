# Project Log: LLM-Assisted Theorem Proving on B200 Cluster

## 2026-03-28 — Session 1: First Blood

### Hardware Profile

Connected to a DGX B200 node:
- **8x NVIDIA B200** — 183GB VRAM each, 1.43TB total
- **2x Intel Xeon Platinum 8570** — 112 cores / 224 threads
- **2TB system RAM**
- **NVLink 5 (NV18)** full mesh — every GPU directly connected at 956 GB/s bidirectional
- **~28TB raw NVMe** storage

### Target: Zaremba's Conjecture (1972)

> For every positive integer d, there exists a with gcd(a, d) = 1 whose continued fraction expansion has all partial quotients ≤ 5.

**Why this conjecture:**
- Cleanly statable in Lean 4
- Has known partial results (Bourgain-Kontorovich 2014: true for density-1 set of integers)
- Small cases are computationally verifiable — perfect for pipeline validation
- Nobody has formalized it in Lean 4 before
- The frontier is tractable for MCTS exploration

### Cluster Setup

1. Installed Lean 4.29.0 via elan (had to fix dead URL at elan.lean-lang.org → GitHub raw)
2. Set up Python venv with PyTorch 2.10.0 (CUDA), vLLM 0.18.0
3. Downloaded Mathlib4 cache (8220 files)
4. Aligned Lean toolchain to v4.29.0-rc8 to match Mathlib

**Gotcha:** The original setup script used `leanprover-lean4-v4.16.0` as the toolchain name, but elan wants the format `leanprover/lean4:v4.29.0-rc8`. Also the old `elan.lean-lang.org` domain is dead — had to switch to `raw.githubusercontent.com/leanprover/elan/master/elan-init.sh`.

### Zaremba Formalization

Created `lean4-proving/conjectures/zaremba.lean` with:
- **Layer 1:** Computable definitions — `cfQuotients` (Euclidean algorithm), `allBoundedBy`, `HasZarembaWitness`
- **Layer 2:** 20 sorry-based theorems for d=1..20, each with A=5
- **Layer 3:** The full conjecture statement (sorry — it's open!)
- **Layer 4:** Bourgain-Kontorovich placeholder

**Key design decision:** All definitions are computable, so proofs can be discharged with `native_decide`. The LLM just needs to find the witness `a` and emit:
```
exact ⟨a, by decide, by decide, by native_decide, by native_decide⟩
```

Manually verified this proof pattern works for d=1, d=7, d=20. Clean compile.

### Witness Table (computed with Python, verified in Lean)

| d | witness a | CF(a/d) | max quotient |
|---|-----------|---------|-------------|
| 1 | 1 | [1] | 1 |
| 2 | 1 | [0, 2] | 2 |
| 3 | 1 | [0, 3] | 3 |
| 4 | 1 | [0, 4] | 4 |
| 5 | 1 | [0, 5] | 5 |
| 6 | 5 | [0, 1, 5] | 5 |
| 7 | 2 | [0, 3, 2] | 3 |
| 8 | 3 | [0, 2, 1, 2] | 2 |
| 9 | 2 | [0, 4, 2] | 4 |
| 10 | 3 | [0, 3, 3] | 3 |
| 11 | 2 | [0, 5, 2] | 5 |
| 12 | 5 | [0, 2, 2, 2] | 2 |
| 13 | 3 | [0, 4, 3] | 4 |
| 14 | 3 | [0, 4, 1, 2] | 4 |
| 15 | 4 | [0, 3, 1, 3] | 3 |
| 16 | 3 | [0, 5, 3] | 5 |
| 17 | 3 | [0, 5, 1, 2] | 5 |
| 18 | 5 | [0, 3, 1, 1, 2] | 3 |
| 19 | 4 | [0, 4, 1, 3] | 4 |
| 20 | 9 | [0, 2, 4, 2] | 4 |

### Model Selection: The Race

Instead of picking one model, we decided to race two SOTA provers:

| Model | HF ID | Size | MiniF2F | GPUs | Port |
|-------|--------|------|---------|------|------|
| Goedel-Prover-V2-32B | `Goedel-LM/Goedel-Prover-V2-32B` | 32B (Qwen3) | 88-90% (Pass@32) | 0-3 | 8000 |
| Kimina-Prover-72B | `AI-MO/Kimina-Prover-72B` | 72B (Qwen2.5) | 84% (Pass@32), #1 PutnamBench | 4-7 | 8001 |

**Why race instead of just the best model?**
- We have 1.43TB VRAM — using only 64GB for one model wastes 95% of the hardware
- The models have different strengths: Goedel excels on MiniF2F, Kimina leads PutnamBench
- Racing gives us comparative data for the blog
- The prover queries both servers in parallel, first valid proof wins
- Each model gets its own scoreboard: we'll see which one proves more theorems

### Prover Harness Updates

Adapted `prover.py` to support:
- **Multi-server racing** — `--server URL` can be specified multiple times
- **Zaremba-specialized prompts** — `--zaremba` flag generates witness-focused prompts
- **Scoreboard** — tracks which model proved what
- **Chat completions API** — some models prefer chat over raw completions
- **Parallel querying** — ThreadPoolExecutor queries all servers simultaneously

### Pipeline Debugging: The Real Story

Getting the pipeline working took three iterations. Good blog material — this is what "real" AI-assisted proving looks like.

**Bug 1: sorry acceptance.** The Lean compiler exits 0 even when proofs contain `sorry` (it's a warning, not an error). Our first run reported 23/23 proved — the model was generating markdown + code with sorry still embedded, and we were accepting it. Fix: reject suggestions that contain the string "sorry".

**Bug 2: Output parsing.** The model wraps proofs in full theorem statements with markdown headers. When we substitute the entire response for `sorry`, we get garbage like `theorem zaremba_d1 : HasZarembaWitness 1 5 := by exact ⟨1...⟩ ### Proof of zaremba_d2...` injected into the middle of the file. Fix: extract just the proof body after `:= by`, truncate at the next theorem/section marker, and bracket-match `exact ⟨...⟩` expressions.

**Bug 3: `decide` vs `native_decide`.** The models always generate `by decide` for all four proof obligations. But `decide` uses the kernel evaluator which times out on CF computation. `native_decide` compiles to native code and runs instantly. Fix: post-process to replace the last two `by decide` with `by native_decide`.

**Lesson:** The model knows the proof *pattern* perfectly. Every suggestion was structurally correct: `exact ⟨N, by decide, by decide, by decide, by decide⟩`. The failures were all infrastructure — parsing, verification, and tactic selection. This is fixable engineering, not a fundamental AI limitation.

### Race Results

**Final: 20/23 sorries proved (19/20 small cases + 1 utility)**

| Model | Proofs | Role |
|-------|--------|------|
| Goedel-Prover-V2-32B | 10 | Won trivial cases (a=1 works), faster response |
| Kimina-Prover-72B | 10 | Won witness search — found a=5,9,4 for harder cases |

**Theorem-by-theorem results:**

| d | Winner | Witness | Expected | Notes |
|---|--------|---------|----------|-------|
| 1 | Goedel | a=1 | a=1 | |
| 2 | Goedel | a=1 | a=1 | |
| 3 | Goedel | a=1 | a=1 | |
| 4 | Goedel | a=1 | a=1 | |
| 5 | Goedel | a=1 | a=1 | |
| 6 | Kimina | a=5 | a=5 | Goedel kept trying a=1 |
| 7 | Kimina | a=2 | a=2 | |
| 8 | Kimina | a=3 | a=3 | |
| 9 | **FAIL** | — | a=2 | Both fixated on a=1,3 |
| 10 | Kimina | a=3 | a=3 | |
| 11 | Kimina | a=2 | a=2 | |
| 12 | Goedel | a=5 | a=5 | |
| 13 | Kimina | a=3 | a=3 | |
| 14 | Goedel | a=3 | a=3 | |
| 15 | Kimina | a=4 | a=4 | |
| 16 | Goedel | a=3 | a=3 | |
| 17 | Kimina | a=3 | a=3 | |
| 18 | Goedel | a=5 | a=5 | |
| 19 | Kimina | a=4 | a=4 | |
| 20 | Kimina | a=9 | a=9 | Hardest witness |

The 3 unproved sorries:
- **d=9**: Both models couldn't find a=2 in 5 attempts x 16 samples = 80 tries
- **zaremba_conjecture**: The open conjecture — expected failure
- **bourgain_kontorovich**: Density result — expected failure

**All 20 proved results verified by the Lean 4 compiler.** Zero errors.

### Key Insights

1. **The models nail proof structure, struggle with search.** Every attempt had the right pattern. Failures were always wrong witness values.

2. **Multi-model racing works.** Goedel solo: 13/23. Race: 20/23. Kimina found witnesses Goedel couldn't.

3. **Witness search is the bottleneck, not proof generation.** This validates the SOTA research: inference-time scaling (more search) matters more than model size.

4. **Next step is obvious: add a search layer.** The model doesn't need to find the witness — we can enumerate candidates programmatically and just ask the model to format the proof. Or better: use best-first tree search / MCTS with the model as a tactic generator.

### Status

- [x] Cluster setup complete (Lean 4.29.0, vLLM 0.18.0, 8xB200)
- [x] Zaremba formalization complete and compiling
- [x] Prover harness adapted for multi-model racing
- [x] Models downloaded (Goedel-32B 62GB, Kimina-72B 136GB)
- [x] Both models served on split GPU sets (GPUs 0-3, GPUs 4-7)
- [x] Race completed: 19/20 small cases proved, Lean-verified
- [ ] Fix d=9 (add witness enumeration or more search)
- [ ] Extend to d=1..100
- [ ] Continued fractions exploration started

### Attempting Level 3: Prove the Full Conjecture

#### LLM Proof Strategy Attempts

Asked both Goedel-32B and Kimina-72B to propose novel proof strategies with full mathematical context (Bourgain-Kontorovich, Huang, semigroup formulation, Rickards-Stange).

**Goedel-32B:** Spiraled into manual CF computation for 4000 tokens. Tried constructive approaches (find a for each d), correctly identified they don't generalize, but couldn't produce anything novel.

**Kimina-72B:** Extended chain-of-thought reasoning. Explored semigroup, inductive, and Euclidean algorithm approaches. Correctly concluded that approach (b) — effective B-K + computational verification — is most viable. Produced a valid but trivial Lean 4 proof (coprime a/d exists in (0,1), without the CF bound condition).

**Verdict:** Neither model can invent new mathematics. They correctly identify the landscape and known approaches, but cannot produce the novel insight needed to close the gap between density-1 and the full conjecture.

#### Computational Verification: CPU Phase

Verified d=1..1,000,000 on 112 CPU cores: **0 failures** in 751.7s.

#### Witness Distribution Analysis (d=1..100,000)

This is original mathematical data:

- **99.7%** of smallest witnesses have CF starting with `[0, 5, 1, ...]`
- **99.9%** hit max partial quotient of exactly 5 — the conjecture is tight
- **Mean a/d = 0.1712**, concentrated in [0.1708, 0.1745] (99% interval)
- This is close to 1/(5 + 1/(1+...)) — witnesses cluster near a specific CF prefix
- CF length peaks at 12-13 digits (logarithmic in d)
- Only d=1..5 can use the trivial witness a=1

**Key insight:** The witness distribution is remarkably concentrated. Almost every d has its smallest witness at a ≈ 0.171*d, with CF starting [0, 5, 1, ...]. This tight clustering suggests deep structural regularity.

#### CUDA-Accelerated Verification: GPU Phase

Wrote a CUDA kernel (`scripts/zaremba_verify.cu`) to parallelize verification across the B200 GPUs.

**Performance:**
- CPU (112 cores, Python): 1,330 d/sec for d~1M → **751s for 1M**
- GPU (1x B200, CUDA): 534,000 d/sec for d~1M → **1.9s for 1M**
- **Speedup: ~400x per GPU, ~3200x total cluster vs CPU**

**Optimization:** Analysis showed 99.9% of witnesses are in [d/7, d/3], so the kernel searches that range first before falling back to full search. Also fixed uint32 overflow for d > 4.3B.

**Current run:** 8 GPUs × 1B each = **8 billion values** being verified in parallel. All 8 B200s at 100% utilization. ETA ~2 hours.

If this completes with 0 failures: **Zaremba's Conjecture verified for all d ≤ 8,000,000,000 with A=5.**

This would be (to our knowledge) the largest computational verification of Zaremba's Conjecture ever performed.

### Brute-Force Scaling Wall (7 hours in)

The brute-force kernel hit a scaling wall. Rate drops dramatically with d:

| GPU | Range | Progress | Rate | Projected ETA |
|-----|-------|----------|------|---------------|
| 0 | 1-1B | 26% | 10,239 d/sec | ~20 hours |
| 1 | 1B-2B | 3% | 1,223 d/sec | ~9 days |
| 2 | 2B-3B | 1% | 603 d/sec | ~19 days |
| 3 | 3B-4B | 1% | 397 d/sec | ~29 days |
| 4-7 | 4B-8B | <1% | Even slower | Weeks+ |

**Zero failures across all ranges tested.** The conjecture holds everywhere we've looked.

**Root cause:** For each d, the kernel scans a ≈ d/7 to d/3 until it finds a coprime a with bounded CF quotients. For d ≈ 3B, that's scanning ~430M candidates, each requiring a gcd + CF computation. The search is O(d) per value of d.

**The fix:** Our witness distribution analysis showed a ≈ 0.170*d for 99% of cases. Instead of scanning [d/7, d/3], we should:
1. Start at a = floor(0.170 * d) and search outward in a narrow band
2. Skip non-coprime candidates using a small-prime sieve
3. Use the CF prefix [0, 5, 1, ...] as an early-exit filter

This should reduce the search from O(d) to O(d^epsilon) — potentially 1000× faster for large d.
