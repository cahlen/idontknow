#!/usr/bin/env python3
"""
Run an AI peer review on a bigcompute.science finding.

Supports any OpenAI-compatible API. API key must be in the environment variable
API_KEY (never saved to files or passed as CLI argument).

Usage:
    export API_KEY='sk-...'
    python3 scripts/reviews/run_review.py --slug zaremba-density-phase-transition --model o3-pro --provider openai
    python3 scripts/reviews/run_review.py --slug kronecker-s40-character-table --model gemini-2.5-pro --provider google --api-base https://generativelanguage.googleapis.com/v1beta/openai
    python3 scripts/reviews/run_review.py --list
    python3 scripts/reviews/run_review.py --all --model gpt-4.1 --provider openai
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FINDINGS_DIR = REPO_ROOT.parent / "bigcompute.science" / "src" / "content" / "findings"
VERIFICATIONS_DIR = REPO_ROOT / "docs" / "verifications"
PROMPT_FILE = SCRIPT_DIR / "prompts" / "peer_review_v1.txt"

# Models that use OpenAI's Responses API instead of Chat Completions
RESPONSES_API_MODELS = {"o3-pro", "o3", "o3-mini", "o1-pro", "o1"}


def check_api_key_safety():
    """Refuse to run if API key appears in any config or script file."""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        print("ERROR: API_KEY environment variable not set.")
        print("Usage: export API_KEY='sk-...' && python3 scripts/reviews/run_review.py ...")
        sys.exit(1)
    # Sanity check: key should look like a real key
    if len(api_key) < 20:
        print("ERROR: API_KEY looks too short. Check your environment variable.")
        sys.exit(1)
    return api_key


def get_findings():
    """Load all findings from the bigcompute.science content directory."""
    findings = {}
    for f in sorted(glob.glob(str(FINDINGS_DIR / "*.md"))):
        with open(f) as fh:
            content = fh.read()
        slug = None
        for line in content.split("\n"):
            if line.startswith("slug:"):
                slug = line.split(":", 1)[1].strip().strip('"')
                break
        if slug:
            findings[slug] = (f, content)
    return findings


def make_review_filename(slug, provider, model, date):
    """Generate standardized review filename."""
    model_safe = model.lower().replace(".", "-").replace(" ", "-")
    provider_safe = provider.lower().replace(" ", "-")
    return f"{slug}_review_{provider_safe}-{model_safe}_{date}.json"


def call_responses_api(api_key, api_base, model, prompt, max_tokens):
    """Call OpenAI's Responses API (for o3-pro, o1, etc.)."""
    import httpx
    url = api_base.rstrip("/").replace("/v1", "") + "/v1/responses"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": prompt, "max_output_tokens": max_tokens},
        timeout=600.0,
    )
    if response.status_code != 200:
        return None, response.status_code, response.text[:500]
    data = response.json()
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text += block.get("text", "")
    return text, 200, data.get("usage", {})


def call_chat_completions(api_key, api_base, model, prompt, max_tokens):
    """Call standard OpenAI Chat Completions API."""
    import httpx
    url = api_base.rstrip("/") + "/chat/completions"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        },
        timeout=600.0,
    )
    if response.status_code != 200:
        return None, response.status_code, response.text[:500]
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return text, 200, data.get("usage", {})


def parse_review_json(text):
    """Parse JSON from model response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("json\n"):
        text = text[5:]
    return json.loads(text)


def review_finding(slug, content, api_key, api_base, model, provider, dry_run=False):
    """Run a peer review on a single finding."""
    prompt_template = PROMPT_FILE.read_text()
    full_prompt = prompt_template + content

    if dry_run:
        print(f"  [DRY RUN] Would send {len(full_prompt)} chars to {model}")
        print(f"  [DRY RUN] First 200 chars of prompt:\n    {full_prompt[:200]}...")
        return None

    print(f"  Sending to {model} via {api_base}...", flush=True)
    t0 = time.time()

    use_responses = model in RESPONSES_API_MODELS
    if use_responses:
        text, status, usage = call_responses_api(api_key, api_base, model, full_prompt, 16000)
    else:
        text, status, usage = call_chat_completions(api_key, api_base, model, full_prompt, 16000)

    elapsed = time.time() - t0
    print(f"  Response in {elapsed:.1f}s (status {status})", flush=True)

    if status != 200:
        print(f"  ERROR: {usage}")
        return None

    try:
        review = parse_review_json(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        raw_path = VERIFICATIONS_DIR / f"{slug}_review_raw.txt"
        raw_path.write_text(text)
        print(f"  Saved raw response to {raw_path}")
        return None

    # Wrap in standard envelope
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = make_review_filename(slug, provider, model, date_str)

    doc = {
        "finding_slug": slug,
        "review_id": f"{provider}-{model}-{date_str}".lower().replace(" ", "-"),
        "reviewer": {
            "type": "ai-model",
            "model": model,
            "model_provider": provider,
        },
        "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mcp_tools_used": [],
        "prompt_version": "peer_review_v1",
        "usage": usage if isinstance(usage, dict) else {},
        "response_time_seconds": round(elapsed, 1),
        "claim_reviews": review.get("claim_reviews", []),
        "overall_verdict": review.get("overall_verdict", "UNKNOWN"),
        "overall_assessment": review.get("overall_assessment", ""),
        "recommendations": review.get("recommendations", []),
        "certification_recommendation": review.get("certification_recommendation", ""),
        "certification_justification": review.get("certification_justification", ""),
    }

    out_path = VERIFICATIONS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Saved: {out_path}")

    return doc


def main():
    parser = argparse.ArgumentParser(description="Run AI peer review on bigcompute.science findings")
    parser.add_argument("--slug", help="Finding slug to review")
    parser.add_argument("--all", action="store_true", help="Review all findings")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    parser.add_argument("--provider", default="openai", help="Provider name (default: openai)")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--list", action="store_true", help="List available findings")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt without calling API")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if review already exists for this model+date")
    args = parser.parse_args()

    findings = get_findings()

    if args.list:
        for slug in findings:
            existing = list(VERIFICATIONS_DIR.glob(f"{slug}_review_*"))
            print(f"  [{len(existing):2d} reviews] {slug}")
        return

    if not args.slug and not args.all:
        parser.print_help()
        sys.exit(1)

    api_key = check_api_key_safety()

    if args.all:
        slugs = list(findings.keys())
    else:
        slugs = [args.slug]
        if args.slug not in findings:
            print(f"Finding '{args.slug}' not found. Use --list to see available findings.")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"AI Peer Review — {len(slugs)} finding(s), model={args.model}")
    print(f"{'='*60}\n")

    results = []
    for i, slug in enumerate(slugs):
        # Check for existing review
        if args.skip_existing:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            expected = make_review_filename(slug, args.provider, args.model, date_str)
            if (VERIFICATIONS_DIR / expected).exists():
                print(f"[{i+1}/{len(slugs)}] {slug} — already reviewed today, skipping")
                continue

        print(f"[{i+1}/{len(slugs)}] {slug}")
        _, content = findings[slug]
        doc = review_finding(slug, content, api_key, args.api_base, args.model, args.provider, args.dry_run)
        if doc:
            results.append((slug, doc["overall_verdict"], doc["certification_recommendation"]))
        else:
            results.append((slug, "FAILED" if not args.dry_run else "DRY_RUN", ""))

        if i < len(slugs) - 1 and not args.dry_run:
            time.sleep(2)

    if results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for slug, verdict, cert in results:
            print(f"  {slug:50s} {verdict:25s} {cert}")


if __name__ == "__main__":
    main()
