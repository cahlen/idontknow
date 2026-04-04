#!/usr/bin/env python3
"""
Generate website data files from the review manifest.

Reads manifest.json and produces:
1. certifications.json — structured data for verification.astro
2. Updates to llms.txt findings section (optional)

Usage:
    python3 scripts/reviews/sync_website.py --manifest docs/verifications/manifest.json --output ../bigcompute.science/src/data/certifications.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "verifications" / "manifest.json"
DEFAULT_OUTPUT = REPO_ROOT.parent / "bigcompute.science" / "src" / "data" / "certifications.json"


def build_certifications(manifest):
    """Transform manifest into website-ready certifications data."""
    findings = manifest.get("findings", {})

    # Build certification entries sorted by level (gold first) then by title
    cert_order = {"gold": 0, "silver": 1, "bronze": 2, "uncertified": 3}
    entries = []

    for slug, finding in findings.items():
        cert = finding.get("certification", {})
        reviews = finding.get("reviews", [])

        # Build models summary (exclude "unknown" from old auto-reviews)
        models = set()
        providers = set()
        for r in reviews:
            rev = r.get("reviewer", {})
            model = rev.get("model", "")
            provider = rev.get("provider", "")
            if model and model != "unknown":
                short = model.replace("Claude Opus 4.6", "Claude").replace("claude-opus-4-6[1m]", "Claude")
                models.add(short)
            if provider and provider != "unknown":
                providers.add(provider)

        # Build verdict summary
        review_count = cert.get("review_count", len(reviews))
        valid_verdicts = {"REJECT", "REVISE_AND_RESUBMIT", "ACCEPT_WITH_REVISION", "ACCEPT"}
        verdicts = [r.get("overall_verdict", "") for r in reviews if r.get("overall_verdict", "") in valid_verdicts]
        worst = min(verdicts, key=lambda v: {"REJECT": 0, "REVISE_AND_RESUBMIT": 1, "ACCEPT_WITH_REVISION": 2, "ACCEPT": 3}.get(v, 2), default="ACCEPT_WITH_REVISION")

        if worst == "REVISE_AND_RESUBMIT":
            verdict_text = f"Revise & resubmit ({review_count} reviews)"
        elif worst == "ACCEPT_WITH_REVISION":
            verdict_text = f"Accept w/ revision ({review_count} reviews)"
        elif worst == "ACCEPT":
            verdict_text = f"Accept ({review_count} reviews)"
        else:
            verdict_text = f"{worst} ({review_count} reviews)"

        models_text = " + ".join(sorted(models))

        entries.append({
            "slug": slug,
            "title": finding.get("title", slug),
            "level": cert.get("level", "uncertified"),
            "review_count": review_count,
            "verdict_summary": verdict_text,
            "models_summary": models_text,
            "issues_found": finding.get("issues_found", 0),
            "issues_resolved": finding.get("issues_resolved", 0),
            "last_reviewed": finding.get("last_reviewed", ""),
            "reviews": [
                {
                    "date": r.get("reviewed_at", "")[:10],
                    "model": r.get("reviewer", {}).get("model", ""),
                    "provider": r.get("reviewer", {}).get("provider", ""),
                    "verdict": r.get("overall_verdict", ""),
                    "level": r.get("certification_recommendation", ""),
                }
                for r in reviews
                if r.get("reviewer", {}).get("model", "unknown") != "unknown"
            ],
            "remediations": [
                {
                    "description": r.get("description", ""),
                    "severity": r.get("severity", ""),
                    "status": r.get("status", ""),
                    "commit": r.get("resolved_commit", ""),
                }
                for r in finding.get("remediations", [])
            ],
        })

    # Sort: gold first, then silver, bronze, uncertified; within each, alphabetical
    entries.sort(key=lambda e: (cert_order.get(e["level"], 3), e["title"]))

    # Compute aggregate stats
    total_issues = sum(e["issues_found"] for e in entries)
    resolved_issues = sum(e["issues_resolved"] for e in entries)

    return {
        "generated_at": manifest.get("generated_at", ""),
        "stats": {
            "findings_audited": len(entries),
            "total_reviews": manifest.get("stats", {}).get("total_reviews", 0),
            "issues_discovered": total_issues,
            "issues_fixed": resolved_issues,
            "unique_models": manifest.get("stats", {}).get("unique_models", 0),
            "unique_providers": manifest.get("stats", {}).get("unique_providers", 0),
        },
        "certifications": entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate website certification data from manifest")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output certifications.json path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("Run aggregate.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    certs = build_certifications(manifest)

    with open(output_path, "w") as f:
        json.dump(certs, f, indent=2)

    print(f"Certifications written: {output_path}")
    print(f"  {certs['stats']['findings_audited']} findings")
    print(f"  {certs['stats']['total_reviews']} reviews")
    print(f"  {certs['stats']['issues_discovered']} issues ({certs['stats']['issues_fixed']} fixed)")

    # Also generate meta.json (machine-readable site index)
    meta_path = output_path.parent.parent.parent / "public" / "meta.json"
    if meta_path.parent.exists():
        stats = certs["stats"]
        models = manifest.get("stats", {}).get("models", [])
        providers = manifest.get("stats", {}).get("providers", [])
        finding_urls = {e["slug"]: f"https://bigcompute.science/findings/{e['slug']}/" for e in certs["certifications"]}

        meta = {
            "name": "bigcompute.science",
            "description": "Open computational mathematics on GPUs. AI-audited, not peer-reviewed.",
            "url": "https://bigcompute.science",
            "generated_at": manifest.get("generated_at", ""),
            "author": {"name": "Cahlen Humphreys", "github": "https://github.com/cahlen", "huggingface": "https://huggingface.co/cahlen"},
            "license": "CC-BY-4.0",
            "stats": {
                "findings": stats["findings_audited"],
                "reviews": stats["total_reviews"],
                "issues": stats["issues_discovered"],
                "issues_fixed": stats["issues_fixed"],
                "models": models,
                "providers": providers,
            },
            "findings": finding_urls,
            "structured_data": {
                "json_ld": "Every finding page emits ScholarlyArticle + Dataset JSON-LD",
                "meta_tags": "Highwire Press (citation_*) + Dublin Core (DC.*)",
                "sitemap": "https://bigcompute.science/sitemap-index.xml",
                "rss": "https://bigcompute.science/rss.xml",
            },
            "key_pages": {
                "findings": "https://bigcompute.science/findings/",
                "verification": "https://bigcompute.science/verification/",
                "cite": "https://bigcompute.science/cite/",
                "about": "https://bigcompute.science/about/",
                "interactive": "https://bigcompute.science/interactive/",
            },
            "machine_endpoints": {
                "llms_txt": "https://bigcompute.science/llms.txt",
                "llms_full": "https://bigcompute.science/llms-full.txt",
                "mcp_server": "https://mcp.bigcompute.science/mcp",
                "meta_json": "https://bigcompute.science/meta.json",
            },
            "repositories": {
                "experiments": "https://github.com/cahlen/idontknow",
                "website": "https://github.com/cahlen/bigcompute.science",
                "manifest": "https://github.com/cahlen/idontknow/blob/main/docs/verifications/manifest.json",
                "research_agent": "https://github.com/cahlen/idontknow/blob/main/scripts/research_agent.py",
            },
            "datasets": {
                "zaremba": "https://huggingface.co/datasets/cahlen/zaremba-conjecture-data",
                "kronecker": "https://huggingface.co/datasets/cahlen/kronecker-coefficients",
                "spectra": "https://huggingface.co/datasets/cahlen/continued-fraction-spectra",
            },
            "colab_notebooks": {
                "research_agent": "https://colab.research.google.com/github/cahlen/bigcompute.science/blob/main/public/notebooks/bigcompute_research_agent.ipynb",
                "mcp_explorer": "https://colab.research.google.com/github/cahlen/bigcompute.science/blob/main/public/notebooks/bigcompute_mcp_explorer.ipynb",
            },
            "ai_disclosure": f"Human-AI collaborative. {len(models)} models, {len(providers)} providers. Not peer-reviewed.",
            "recommended_citation_verbs": {
                "safe": ["computed", "archived", "benchmarked", "observed", "verified", "released"],
                "with_caveats": ["suggests", "appears", "is consistent with", "within the tested range"],
                "do_not_use": ["proved", "established", "confirmed"],
            },
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Meta written: {meta_path}")


if __name__ == "__main__":
    main()
