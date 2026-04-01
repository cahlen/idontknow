# Verification System Retrospective — April 1, 2026

## How Three AI Models Peer-Reviewed a Mathematics Paper in One Day

### Timeline

**Session start**: cahlen + Claude Opus 4.6 (Anthropic) working on bigcompute.science

**The idea**: cahlen suggested that the MCP server should connect to academic databases so agents could cross-reference findings against published literature.

**What happened next**:

1. **MCP server built** (22 tools): arXiv, zbMATH, Semantic Scholar, OEIS, LMFDB, Lean/Mathlib, FindStat, plus Boise State and FAU institutional repositories. One connection, all sources.

2. **verify_finding tool created**: takes a finding slug, queries all databases in parallel, returns our claim + live literature + verification guidance.

3. **Claude ate its own lunch**: ran verify_finding on all 9 findings. Found that the initial "certification" was just keyword matching — not real peer review. cahlen called it out: "bro... that is NOT what verification should do."

4. **Real peer review process built**: claim-by-claim analysis, checking specific numbers against published theorems, comparing our data against our own claims. Not just "papers exist" but "does Bourgain-Kontorovich's Theorem 1.2 contradict or support our specific numerical claim?"

5. **Six real issues found** by Claude's review:
   - Class numbers: h=1 rate goes to 0, not 75% (CRITICAL — the entire narrative was wrong)
   - Zaremba density: delta > 1/2 threshold contradicted by our own {2,3,4,5} data
   - Zaremba proof: premature to call it a "proof" — 6 known gaps
   - Cayley diameters: stated bound violated by our own data
   - Hausdorff: wrong number (0.826 should be 0.768) — correction STRENGTHENED the finding
   - Golden ratio: 13% discrepancy unexplained

6. **All issues fixed immediately**: findings corrected, paper rewritten, corrections noted with attribution.

7. **Living verification ledger designed**: each review stamped with AI model identity, version, provider, date. Append-only. Multiple models review same finding over time.

8. **First external review arrived**: cahlen pasted a line-by-line review from **Grok (xAI)** of the Zaremba proof paper. Grok independently reached the same verdict as Claude: ACCEPT WITH REVISION, SILVER.

### What we learned

**The certification badge evolved three times in one session:**

1. **v1 — Keyword matching**: "papers exist on this topic" → cahlen: "that's not verification"
2. **v2 — Claim-by-claim review**: checking numbers against literature → found 6 real issues
3. **v3 — Living ledger**: multiple AI models reviewing over time, each stamped with identity

**Cross-model concordance is real**: Claude Opus 4.6 and Grok independently agreed on the proof framework verdict. Different architectures, different training data, same conclusion. This is the strongest form of AI peer review currently possible.

**The review process caught real errors**: The class number h=1 → 75% claim was fundamentally wrong (it goes to 0). This wasn't a nitpick — it changed the finding's central narrative. The Hausdorff correction wasn't wrong in direction, just in magnitude — and the correction made the finding stronger.

**Honesty improves quality**: Every finding now has a correction note when the review changed something: "CORRECTED (2026-04-01): MCP peer review identified..." This makes the process visible and trustworthy.

### The numbers

- **9 findings** reviewed claim-by-claim
- **6 issues** found and fixed
- **2 AI models** reviewed the proof framework (concordant)
- **22 MCP tools** used during verification
- **5 academic databases** queried live (arXiv, zbMATH, Semantic Scholar, OEIS, LMFDB)
- **0 human reviewers** (yet — the system is designed for them too)

### What's next

- More AI models reviewing findings (Gemini, Llama, etc.)
- Human mathematician reviews via PR
- Re-verification as models improve (Claude 5, Grok 3, etc.)
- Automated re-verification when new papers are published on relevant topics

### The quote

cahlen, on why the verification system matters:

> "Models will get smarter all the time. So findings will go through many many future verifications like this to make sure they hold as models get smarter and smarter over time."

This is the core insight: verification isn't a one-time stamp, it's a living process that compounds as AI capabilities grow.

---

*Written by Claude Opus 4.6 (Anthropic), April 1, 2026. Part of the bigcompute.science verification system.*
