# Review Infrastructure

Scripts for running AI peer reviews on bigcompute.science findings, aggregating results, and syncing with the website.

## Quick Start

```bash
# Run a review (API key from environment only — NEVER in files)
export API_KEY='sk-...'
python3 scripts/reviews/run_review.py --slug zaremba-density-phase-transition --model o3-pro --provider openai

# Review all findings with a model
python3 scripts/reviews/run_review.py --all --model gpt-4.1 --provider openai --skip-existing

# Validate all review JSONs
python3 scripts/reviews/validate.py --all

# Rebuild manifest from all reviews
python3 scripts/reviews/aggregate.py

# Generate website data
python3 scripts/reviews/sync_website.py --manifest docs/verifications/manifest.json --output ../bigcompute.science/src/data/certifications.json
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_review.py` | Run an AI peer review via any OpenAI-compatible API |
| `aggregate.py` | Build manifest.json from all review JSONs |
| `validate.py` | Validate review JSONs against schema |
| `sync_website.py` | Generate certifications.json for the website |

## API Key Safety

**API keys are NEVER saved to files, committed to git, or passed as CLI arguments.**

The only way to provide an API key is via the `API_KEY` environment variable:
```bash
export API_KEY='sk-...'
```

The scripts will refuse to run without this variable set. If you accidentally commit a key, rotate it immediately.

## Supported Providers

Any OpenAI-compatible API works. Examples:

```bash
# OpenAI
--model o3-pro --provider openai --api-base https://api.openai.com/v1

# Google Gemini (via OpenAI compatibility layer)
--model gemini-2.5-pro --provider google --api-base https://generativelanguage.googleapis.com/v1beta/openai

# Local model via vLLM
--model meta-llama/Llama-3.3-70B --provider local --api-base http://localhost:8000/v1

# Anthropic via proxy
--model claude-opus-4-6 --provider anthropic --api-base https://anthropic-proxy.example.com/v1
```

## Review File Naming

New reviews follow: `{slug}_review_{provider}-{model}_{YYYY-MM-DD}.json`

Old files (pre-convention) are preserved as-is. The aggregation script keys on the `finding_slug` field inside the JSON, not the filename.

## Certification Consensus

When multiple reviews exist for a finding, the **most conservative** certification level wins:
- gold > silver > bronze > uncertified
- Any REJECT verdict → uncertified
- Most conservative level across all reviews = consensus

## Remediation Tracking

Issues found in reviews are tracked in `docs/verifications/remediations/{slug}/`:
- `README.md` — summary table of all issues
- `{issue-id}.json` — individual issue with status, fix commit, lineage
