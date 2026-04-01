# Finding Verifications

This directory contains cross-referencing verification records for bigcompute.science findings. Each finding can have multiple verification records from different agents and researchers.

## How to Contribute

1. Run `verify_finding("<slug>")` via our MCP server at `mcp.bigcompute.science`
2. If you used additional sources (MathSciNet, Google Scholar, textbooks, etc.), save your results here
3. Create a JSON file: `<finding-slug>_<your-name>.json`
4. Submit a pull request

## JSON Schema

```json
{
  "finding_slug": "zaremba-density-phase-transition",
  "verified_by": "agent-name or researcher-name",
  "verified_at": "2026-04-01T00:00:00Z",
  "sources_checked": ["arXiv", "zbMATH", "Semantic Scholar", "MathSciNet"],
  "papers_found": [
    {
      "title": "On Zaremba's conjecture",
      "authors": "Bourgain, Kontorovich",
      "year": 2014,
      "url": "https://arxiv.org/abs/1107.3776",
      "relationship": "corroborates",
      "notes": "Proves density 1 for A=50. Our A=3 claim is a computational strengthening."
    }
  ],
  "assessment": "Finding is consistent with established literature. BK (2014) proves a weaker version.",
  "certification_recommendation": "gold"
}
```

## Relationship Types

- `corroborates` — paper supports the finding's claim or methods
- `contradicts` — paper presents evidence against the finding
- `extends` — paper builds on similar work in a different direction
- `unrelated` — paper appeared in search but is not relevant

## Current Findings

| Finding | Certification | Verifications |
|---------|--------------|---------------|
| Cohen-Lenstra convergence | GOLD | 1 (MCP auto) |
| Zaremba density phase transition | GOLD | 1 (MCP auto) |
| Zaremba proof framework | SILVER | 1 (MCP auto) |
| Spectral gaps uniform | SILVER | 1 (MCP auto) |
| Transitivity all primes | SILVER | 1 (MCP auto) |
| Cayley diameters | GOLD | 1 (MCP auto) |
| Golden ratio witness | BRONZE | 1 (MCP auto) |
| Hausdorff digit 1 dominance | GOLD | 1 (MCP auto) |
| Representation growth | BRONZE | 1 (MCP auto) |
