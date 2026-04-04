#!/usr/bin/env python3
"""
Aggregate all review JSONs into a single manifest.json.

Reads all *.json files in docs/verifications/, extracts finding_slug and
review data, computes consensus certification levels, and writes manifest.json.

Usage:
    python3 scripts/reviews/aggregate.py
    python3 scripts/reviews/aggregate.py --verifications-dir docs/verifications --findings-dir ../bigcompute.science/src/content/findings
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VERIFICATIONS = REPO_ROOT / "docs" / "verifications"
DEFAULT_FINDINGS = REPO_ROOT.parent / "bigcompute.science" / "src" / "content" / "findings"

CERT_ORDER = {"uncertified": 0, "bronze": 1, "silver": 2, "gold": 3}
VERDICT_SEVERITY = {
    "REJECT": 0, "REVISE_AND_RESUBMIT": 1, "ACCEPT_WITH_REVISION": 2, "ACCEPT": 3
}


def load_finding_titles(findings_dir):
    """Extract slug -> title mapping from finding markdowns, or from GitHub."""
    titles = {}

    # Try local directory
    findings_path = Path(findings_dir)
    if findings_path.exists():
        for f in glob.glob(str(findings_path / "*.md")):
            slug = title = None
            with open(f) as fh:
                for line in fh:
                    if line.startswith("slug:"):
                        slug = line.split(":", 1)[1].strip().strip('"')
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip().strip('"')
                    if slug and title:
                        break
            if slug:
                titles[slug] = title or slug

    # Fallback: extract titles from review JSONs (always available)
    if not titles:
        verifications = Path(findings_dir).parent.parent / "docs" / "verifications" if findings_path.exists() else Path(__file__).parent.parent.parent / "docs" / "verifications"
        for f in glob.glob(str(verifications / "*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                slug = data.get("finding_slug", "")
                if slug and slug not in titles:
                    titles[slug] = slug.replace("-", " ").title()
            except:
                pass

    return titles


def load_reviews(verifications_dir):
    """Load all review JSONs, grouping by finding_slug."""
    reviews_by_slug = defaultdict(list)
    responses_by_slug = defaultdict(list)
    skipped = []

    for f in sorted(glob.glob(str(Path(verifications_dir) / "*.json"))):
        basename = os.path.basename(f)
        if basename == "manifest.json":
            continue

        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped.append((basename, "invalid JSON"))
            continue

        slug = data.get("finding_slug")
        if not slug:
            skipped.append((basename, "no finding_slug"))
            continue

        # Distinguish reviews from revision responses
        entry_type = data.get("entry_type", "")
        if entry_type == "revision-response" or "revision" in basename:
            responses_by_slug[slug].append({"file": basename, "data": data})
        elif "claim_reviews" in data or "overall_verdict" in data:
            reviews_by_slug[slug].append({"file": basename, "data": data})
        else:
            # Old mcp-auto format — treat as review
            reviews_by_slug[slug].append({"file": basename, "data": data})

    if skipped:
        print(f"  Skipped {len(skipped)} files: {[s[0] for s in skipped]}")

    return reviews_by_slug, responses_by_slug


def load_remediations(verifications_dir):
    """Load remediation records from remediations/ subdirectories."""
    remediations_by_slug = defaultdict(list)
    rem_dir = Path(verifications_dir) / "remediations"
    if not rem_dir.exists():
        return remediations_by_slug

    for slug_dir in sorted(rem_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        for f in sorted(slug_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                remediations_by_slug[slug].append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    return remediations_by_slug


def compute_consensus(reviews):
    """Compute consensus certification level (most conservative wins)."""
    if not reviews:
        return "uncertified"

    levels = []
    for r in reviews:
        data = r["data"]
        cert = data.get("certification_recommendation", "").lower()
        if cert in CERT_ORDER:
            levels.append(cert)

    if not levels:
        return "uncertified"

    # Any REJECT → uncertified
    for r in reviews:
        verdict = r["data"].get("overall_verdict", "")
        if verdict == "REJECT":
            return "uncertified"

    # Most conservative wins
    return min(levels, key=lambda x: CERT_ORDER.get(x, 0))


def summarize_review(review_data):
    """Create a compact summary of a review."""
    data = review_data["data"]
    reviewer = data.get("reviewer", {})

    claims = data.get("claim_reviews", [])
    claims_summary = {
        "total": len(claims),
        "verified": sum(1 for c in claims if c.get("verdict") == "VERIFIED"),
        "disputed": sum(1 for c in claims if c.get("verdict") == "DISPUTED"),
        "needs_clarification": sum(1 for c in claims if c.get("verdict") == "NEEDS_CLARIFICATION"),
        "unverifiable": sum(1 for c in claims if c.get("verdict") == "UNVERIFIABLE"),
    }

    # Determine the actual verdict (not the assessment text)
    verdict_raw = data.get("overall_verdict", "")
    valid_verdicts = {"ACCEPT", "ACCEPT_WITH_REVISION", "REVISE_AND_RESUBMIT", "REJECT"}
    overall_verdict = verdict_raw if verdict_raw in valid_verdicts else "ACCEPT_WITH_REVISION"

    return {
        "review_id": data.get("review_id", "unknown"),
        "file": review_data["file"],
        "reviewer": {
            "type": reviewer.get("type", "unknown"),
            "model": reviewer.get("model", reviewer.get("verified_by", "unknown")),
            "provider": reviewer.get("model_provider", "unknown"),
        },
        "reviewed_at": data.get("reviewed_at", data.get("verified_at", "")),
        "overall_verdict": overall_verdict,
        "certification_recommendation": data.get("certification_recommendation", ""),
        "claims": claims_summary,
        "key_finding": (
            data.get("overall_assessment", "")[:200]
            if data.get("overall_assessment")
            else data.get("assessment", "")[:200]
        ),
    }


def build_manifest(verifications_dir, findings_dir):
    """Build the complete manifest."""
    print("Loading finding titles...", flush=True)
    titles = load_finding_titles(findings_dir)

    print(f"Loading reviews from {verifications_dir}...", flush=True)
    reviews_by_slug, responses_by_slug = load_reviews(verifications_dir)

    print("Loading remediations...", flush=True)
    remediations_by_slug = load_remediations(verifications_dir)

    # Collect all slugs
    all_slugs = set(titles.keys()) | set(reviews_by_slug.keys())

    # Aggregate stats
    total_reviews = sum(len(v) for v in reviews_by_slug.values())
    total_remediations = sum(len(v) for v in remediations_by_slug.values())
    resolved = sum(
        1 for slugremds in remediations_by_slug.values()
        for r in slugremds if r.get("status") == "resolved"
    )

    unique_models = set()
    unique_providers = set()
    for slug_reviews in reviews_by_slug.values():
        for r in slug_reviews:
            reviewer = r["data"].get("reviewer", {})
            model = reviewer.get("model", reviewer.get("verified_by", ""))
            provider = reviewer.get("model_provider", "")
            if model:
                unique_models.add(model)
            if provider:
                unique_providers.add(provider)

    # Build per-finding entries
    findings = {}
    for slug in sorted(all_slugs):
        reviews = reviews_by_slug.get(slug, [])
        consensus = compute_consensus(reviews)
        remediations = remediations_by_slug.get(slug, [])

        issues_found = len(remediations)
        issues_resolved = sum(1 for r in remediations if r.get("status") == "resolved")

        findings[slug] = {
            "title": titles.get(slug, slug),
            "certification": {
                "level": consensus,
                "review_count": len(reviews),
                "consensus_method": "most_conservative",
            },
            "reviews": [summarize_review(r) for r in reviews],
            "remediations": [
                {
                    "issue_id": r.get("issue_id", "unknown"),
                    "severity": r.get("severity", "unknown"),
                    "description": r.get("description", ""),
                    "status": r.get("status", "unknown"),
                    "resolved_commit": r.get("commits", [""])[0] if r.get("commits") else "",
                }
                for r in remediations
            ],
            "issues_found": issues_found,
            "issues_resolved": issues_resolved,
            "last_reviewed": max(
                (r["data"].get("reviewed_at", r["data"].get("verified_at", ""))
                 for r in reviews),
                default=""
            ),
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/reviews/aggregate.py",
        "schema_version": 3,
        "stats": {
            "total_findings": len(findings),
            "total_reviews": total_reviews,
            "total_remediations": total_remediations,
            "remediations_resolved": resolved,
            "unique_models": len(unique_models),
            "unique_providers": len(unique_providers),
            "models": sorted(unique_models),
            "providers": sorted(unique_providers),
        },
        "findings": findings,
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Aggregate review JSONs into manifest.json")
    parser.add_argument("--verifications-dir", default=str(DEFAULT_VERIFICATIONS))
    parser.add_argument("--findings-dir", default=str(DEFAULT_FINDINGS))
    parser.add_argument("--output", default=str(DEFAULT_VERIFICATIONS / "manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest(args.verifications_dir, args.findings_dir)

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    stats = manifest["stats"]
    print(f"\nManifest written: {args.output}")
    print(f"  {stats['total_findings']} findings, {stats['total_reviews']} reviews")
    print(f"  {stats['unique_models']} models from {stats['unique_providers']} providers")
    print(f"  {stats['total_remediations']} remediations ({stats['remediations_resolved']} resolved)")


if __name__ == "__main__":
    main()
