#!/usr/bin/env python3
"""
Peer review all bigcompute.science findings using OpenAI o3-pro.

Reads each finding markdown, sends to o3-pro with a structured mathematical
peer review prompt, and saves the review as JSON in docs/verifications/.

Usage:
    export OPENAI_API_KEY='...'
    python3 scripts/o3pro_peer_review.py [slug]         # review one finding
    python3 scripts/o3pro_peer_review.py                # review all findings
    python3 scripts/o3pro_peer_review.py --list         # list findings
"""

import os
import sys
import json
import time
import glob
from datetime import datetime, timezone

FINDINGS_DIR = "/home/amsysistestdrive2026/bigcompute.science/src/content/findings"
VERIFICATIONS_DIR = "/home/amsysistestdrive2026/idontknow/docs/verifications"
MODEL = "o3-pro"

REVIEW_PROMPT = """You are a mathematical peer reviewer. You have been asked to review a computational mathematics finding from bigcompute.science — an open research project that uses GPU clusters to explore open conjectures.

Your job is to review the finding CLAIM BY CLAIM and assess:
1. Is the mathematics correct?
2. Are the computational claims plausible given the methods described?
3. Are there any errors, unsupported claims, or misleading statements?
4. How does this relate to published literature?

Be rigorous but fair. These are computational results, not formal proofs. The authors are honest about limitations.

Respond with a JSON object (and nothing else — no markdown fences, no commentary) in exactly this format:

{
  "claim_reviews": [
    {
      "claim": "specific claim extracted from the finding",
      "verdict": "VERIFIED | NEEDS_CLARIFICATION | DISPUTED | UNVERIFIABLE",
      "evidence": "your reasoning and any literature references",
      "concerns": "specific concerns, or empty string if none"
    }
  ],
  "overall_verdict": "ACCEPT | ACCEPT_WITH_REVISION | REVISE_AND_RESUBMIT | REJECT",
  "overall_assessment": "2-4 sentence summary of the review",
  "recommendations": ["list of specific recommendations"],
  "certification_recommendation": "gold | silver | bronze",
  "certification_justification": "1-2 sentence justification for the certification level"
}

Certification levels:
- gold: 3+ published papers corroborate the methods, validated against known benchmarks
- silver: 1+ published paper corroborates, methods grounded in established literature
- bronze: novel observation, no direct literature precedent

Here is the finding to review:

---
"""


def get_findings():
    """Get all finding markdown files."""
    files = sorted(glob.glob(os.path.join(FINDINGS_DIR, "*.md")))
    findings = []
    for f in files:
        with open(f) as fh:
            content = fh.read()
        # Extract slug from frontmatter
        slug = None
        for line in content.split("\n"):
            if line.startswith("slug:"):
                slug = line.split(":", 1)[1].strip().strip('"')
                break
        if slug:
            findings.append((slug, f, content))
    return findings


def review_finding(slug, content, api_key):
    """Send finding to o3-pro for review."""
    import httpx

    print(f"  Sending to {MODEL}...", flush=True)
    t0 = time.time()

    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "input": REVIEW_PROMPT + content,
            "max_output_tokens": 16000,
        },
        timeout=600.0,  # o3-pro can take a while (reasoning)
    )

    elapsed = time.time() - t0
    print(f"  Response in {elapsed:.1f}s", flush=True)

    if response.status_code != 200:
        print(f"  ERROR {response.status_code}: {response.text[:500]}")
        return None

    data = response.json()
    # Extract text from responses API output
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text += content_block.get("text", "")
    usage = data.get("usage", {})

    print(f"  Tokens: {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out", flush=True)

    # Parse JSON from response (strip markdown fences if present)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        review = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw response:\n{text[:1000]}")
        # Save raw response for manual inspection
        raw_path = os.path.join(VERIFICATIONS_DIR, f"{slug}_o3pro-raw.txt")
        with open(raw_path, "w") as f:
            f.write(text)
        print(f"  Saved raw response to {raw_path}")
        return None

    return review, usage, elapsed


def save_review(slug, review, usage, elapsed):
    """Save review as JSON in the verification format."""
    doc = {
        "finding_slug": slug,
        "review_id": f"o3-pro-openai-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "reviewer": {
            "type": "ai-model",
            "model": "o3-pro",
            "model_provider": "OpenAI",
            "model_version": "o3-pro-2025-06-10",
        },
        "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mcp_tools_used": ["none — direct markdown review"],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "response_time_seconds": round(elapsed, 1),
        },
        "claim_reviews": review.get("claim_reviews", []),
        "overall_verdict": review.get("overall_verdict", "UNKNOWN"),
        "overall_assessment": review.get("overall_assessment", ""),
        "recommendations": review.get("recommendations", []),
        "certification_recommendation": review.get("certification_recommendation", ""),
        "certification_justification": review.get("certification_justification", ""),
    }

    path = os.path.join(VERIFICATIONS_DIR, f"{slug}_o3pro-review.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Saved: {path}")
    return path


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    findings = get_findings()

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for slug, path, _ in findings:
            existing = os.path.exists(os.path.join(VERIFICATIONS_DIR, f"{slug}_o3pro-review.json"))
            print(f"  {'[done]' if existing else '[    ]'} {slug}")
        return

    # Filter to specific slug if provided
    if len(sys.argv) > 1 and sys.argv[1] != "--list":
        target = sys.argv[1]
        findings = [(s, p, c) for s, p, c in findings if s == target]
        if not findings:
            print(f"Finding '{target}' not found")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"o3-pro Peer Review — {len(findings)} findings")
    print(f"{'='*60}\n")

    results = []
    for i, (slug, path, content) in enumerate(findings):
        # Skip if already reviewed
        review_path = os.path.join(VERIFICATIONS_DIR, f"{slug}_o3pro-review.json")
        if os.path.exists(review_path):
            print(f"[{i+1}/{len(findings)}] {slug} — already reviewed, skipping")
            continue

        print(f"[{i+1}/{len(findings)}] {slug}")
        result = review_finding(slug, content, api_key)
        if result:
            review, usage, elapsed = result
            save_review(slug, review, usage, elapsed)
            verdict = review.get("overall_verdict", "?")
            cert = review.get("certification_recommendation", "?")
            print(f"  Verdict: {verdict}, Cert: {cert}")
            results.append((slug, verdict, cert))
        else:
            print(f"  FAILED — skipping")
            results.append((slug, "FAILED", ""))

        # Brief pause between requests
        if i < len(findings) - 1:
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for slug, verdict, cert in results:
        print(f"  {slug:50s} {verdict:25s} {cert}")
    print(f"\nDone. {len([r for r in results if r[1] != 'FAILED'])}/{len(results)} successful.")


if __name__ == "__main__":
    main()
