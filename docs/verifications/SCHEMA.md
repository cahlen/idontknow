# Verification Ledger Schema

Each finding accumulates peer review entries over time. Each entry records WHO reviewed it (human or AI model), WHEN, WHAT they found, and their VERDICT.

## Verification Entry Schema

```json
{
  "finding_slug": "string",
  "review_id": "string (unique, e.g. 'claude-opus-4.6-2026-04-01')",
  
  "reviewer": {
    "type": "ai-model | human | hybrid",
    "model": "claude-opus-4.6 | gpt-4o | gemini-2.5-pro | etc",
    "model_provider": "anthropic | openai | google | meta | etc",
    "model_version": "claude-opus-4-6[1m]",
    "human_name": "optional — for human or hybrid reviews",
    "affiliation": "optional — university, org, etc"
  },

  "reviewed_at": "ISO 8601 datetime",
  
  "mcp_tools_used": [
    "search_arxiv",
    "search_zbmath", 
    "lookup_oeis",
    "search_mathlib"
  ],

  "claim_reviews": [
    {
      "claim": "statement of the specific claim being reviewed",
      "verdict": "VERIFIED | NEEDS_CLARIFICATION | DISPUTED | UNVERIFIABLE",
      "evidence": "what the reviewer found that supports/contradicts the claim",
      "sources": [
        {
          "type": "zbMATH | arXiv | OEIS | textbook | computation | other",
          "reference": "specific paper/sequence/theorem cited",
          "relationship": "supports | contradicts | extends | qualifies"
        }
      ],
      "concerns": "any issues found (empty string if none)"
    }
  ],

  "overall_verdict": "ACCEPT | ACCEPT_WITH_REVISION | REVISE_AND_RESUBMIT | REJECT",
  "overall_assessment": "paragraph summarizing the review",
  "recommendations": ["list of specific changes or follow-ups"],

  "certification_recommendation": "gold | silver | bronze | uncertified",
  "certification_justification": "why this level"
}
```

## Certification Badge Format

A finding's certification badge shows ALL reviews, not just the latest:

```
FINDING: Zaremba Density Phase Transition
CERTIFICATION: GOLD (4 reviews)
  [2026-04-01] claude-opus-4.6 (Anthropic)    — ACCEPT, GOLD
  [2026-05-15] gemini-2.5-pro (Google)         — ACCEPT, GOLD  
  [2026-06-01] human: Dr. Jane Smith (MIT)     — ACCEPT with revision, GOLD
  [2026-09-01] claude-opus-5.0 (Anthropic)     — ACCEPT, GOLD (re-verified with stronger model)
```

## How Certification Level Is Determined

The finding's certification level is the **consensus** of all reviews:
- If all reviews agree: use that level
- If reviews disagree: use the most conservative (lowest) level
- A single REJECT downgrades to UNCERTIFIED until resolved
- A DISPUTED claim triggers re-review

## Re-verification

Findings should be re-verified when:
1. A significantly more capable AI model becomes available
2. New papers are published on the topic
3. The computation is extended to larger ranges
4. A community member raises a concern

The ledger is append-only — old reviews are never deleted, only superseded.
