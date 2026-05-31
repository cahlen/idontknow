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
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "verifications" / "manifest.json"


def _site_repo() -> Path:
    for candidate in (
        REPO_ROOT.parent / "research" / "bigcompute.science",
        REPO_ROOT.parent / "bigcompute.science",
    ):
        if (candidate / "src" / "data").exists():
            return candidate
    return REPO_ROOT.parent / "bigcompute.science"


DEFAULT_OUTPUT = _site_repo() / "src" / "data" / "certifications.json"


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


def _read_frontmatter_field(text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*\"(.+?)\"", text, re.M)
    if m:
        return m.group(1)
    m = re.search(rf"^{field}:\s*(\S+)", text, re.M)
    return m.group(1) if m else ""


def _truncate(s: str, n: int = 110) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def generate_llms_txt(site_root: Path, certs: dict) -> None:
    """Regenerate public/llms.txt from certifications and experiment frontmatter."""
    llms_path = site_root / "public" / "llms.txt"
    if not llms_path.parent.exists():
        return

    exp_dir = site_root / "src" / "content" / "experiments"
    findings_dir = site_root / "src" / "content" / "findings"
    summaries: dict[str, str] = {}
    for md in findings_dir.glob("*.md"):
        text = md.read_text()
        slug = _read_frontmatter_field(text, "slug")
        summary = _read_frontmatter_field(text, "summary")
        if slug and summary:
            summaries[slug] = summary

    experiments = []
    for md in exp_dir.glob("*.md"):
        text = md.read_text()
        slug = _read_frontmatter_field(text, "slug")
        if not slug:
            continue
        experiments.append({
            "slug": slug,
            "title": _read_frontmatter_field(text, "title"),
            "status": _read_frontmatter_field(text, "status") or "unknown",
            "summary": _read_frontmatter_field(text, "summary"),
            "date": _read_frontmatter_field(text, "date"),
        })
    experiments.sort(key=lambda e: e["date"], reverse=True)

    stats = certs["stats"]
    models = stats.get("unique_models", 0)
    providers = stats.get("unique_providers", 0)

    exp_lines = [
        f"- [{e['title']}](/experiments/{e['slug']}/): {_truncate(e['summary'])} ({e['status']})"
        for e in experiments
    ]

    finding_lines = []
    for entry in certs["certifications"]:
        slug = entry["slug"]
        if slug.startswith("_"):
            continue
        summary = _truncate(summaries.get(slug, entry["title"]))
        level = entry.get("level", "uncertified").upper()
        finding_lines.append(
            f"- [{entry['title']}](/findings/{slug}/): {summary} [{level}]"
        )

    body = f"""# bigcompute.science

> Open computational mathematics. GPU clusters, CUDA kernels, open data. Human-AI collaborative. AI-audited, not peer-reviewed.

All work produced through human-AI collaboration (Cahlen Humphreys + Claude). AI-audited against published literature (arXiv, zbMATH, OEIS). Not independently peer-reviewed. CC BY 4.0.

## Experiments ({len(experiments)} total)

{chr(10).join(exp_lines)}

## Findings ({stats['findings_audited']} total, all AI-audited)

{chr(10).join(finding_lines)}

## Datasets (Hugging Face)

- [Zaremba Density](https://huggingface.co/datasets/cahlen/zaremba-density): 65 GPU experiments, exception sets, density measurements to 10^12
- [Zaremba Data](https://huggingface.co/datasets/cahlen/zaremba-conjecture-data): Dolgopyat profile, transfer operator data, proof framework
- [Hausdorff Spectrum](https://huggingface.co/datasets/cahlen/hausdorff-dimension-spectrum): dim_H for all 1,048,575 subsets of {{1,...,20}} — first complete computation
- [Ramanujan Machine](https://huggingface.co/datasets/cahlen/ramanujan-machine-results): 586B candidate evaluations through deg 7
- [Kronecker Coefficients](https://huggingface.co/datasets/cahlen/kronecker-coefficients): S_20, S_30, S_40 character tables + Kronecker triples
- [Class Numbers](https://huggingface.co/datasets/cahlen/class-numbers-real-quadratic): Real quadratic fields to 10^11
- [Continued Fraction Spectra](https://huggingface.co/datasets/cahlen/continued-fraction-spectra): Hausdorff, Lyapunov, Minkowski, Flint Hills
- [CFD Chaotic Advection](https://huggingface.co/datasets/cahlen/cfd-chaotic-advection): Standard map Lyapunov sweeps (16.8M trajectories)
- [CFD 2D NS BKM](https://huggingface.co/datasets/cahlen/cfd-ns-bkm): 2D pseudospectral BKM diagnostic CSVs
- [CFD 3D NS BKM](https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm): 3D BKM blowup-monitor CSVs (incl. Phase 5 Kerr IC sweep)

## Tools

- [MCP Server](https://mcp.bigcompute.science/mcp): 23 tools, no auth. arXiv, zbMATH, OEIS, LMFDB, Lean/Mathlib + experiment data.
- [Colab Notebook](https://colab.research.google.com/github/cahlen/bigcompute.science/blob/main/public/notebooks/bigcompute_mcp_explorer.ipynb): Pre-loaded MCP client + HF datasets.
- [Experiment code](https://github.com/cahlen/idontknow): CUDA kernels, Python harnesses
- [Website source](https://github.com/cahlen/bigcompute.science): Astro + KaTeX
- [Agent guide](https://github.com/cahlen/idontknow/blob/main/AGENTS.md): How to contribute
- [Audit Ledger](/verification/): AI audit process + all reviews
- [Citation](https://github.com/cahlen/idontknow/blob/main/CITATION.cff): CITATION.cff

## Review Infrastructure

All findings are AI-audited claim-by-claim. Current: {stats['total_reviews']} reviews from {models} models across {providers} providers. {stats['issues_discovered']} issues discovered, {stats['issues_fixed']} fixed. Certification uses most-conservative-wins consensus.

- [Review scripts](https://github.com/cahlen/idontknow/tree/main/scripts/reviews): `run_review.py` (any OpenAI-compatible API), `aggregate.py`, `validate.py`, `sync_website.py`
- [Review schema](https://github.com/cahlen/idontknow/blob/main/docs/verifications/SCHEMA.md): JSON format for reviews
- [Manifest](https://github.com/cahlen/idontknow/blob/main/docs/verifications/manifest.json): Generated aggregate of all reviews
- [Remediations](https://github.com/cahlen/idontknow/tree/main/docs/verifications/remediations): Issue tracking with full lineage (review → issue → fix → commit)
- To contribute: write review JSON per schema, submit PR to [cahlen/idontknow](https://github.com/cahlen/idontknow)

## Machine-Readable Metadata

- [/meta.json](/meta.json): Complete machine-readable index of all structured data, pages, repos, datasets, notebooks, and citation guidance
- [/cite/](/cite/): Citation page with BibTeX/APA for every finding
- Every finding page has: ScholarlyArticle JSON-LD, Dataset JSON-LD (when applicable), Highwire Press meta tags (citation_title, citation_author, citation_date), Dublin Core metadata
- [Sitemap](/sitemap-index.xml) · [RSS](/rss.xml) · [robots.txt](/robots.txt)
"""
    llms_path.write_text(body)
    print(f"llms.txt written: {llms_path}")


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
                "zaremba_density": "https://huggingface.co/datasets/cahlen/zaremba-density",
                "kronecker": "https://huggingface.co/datasets/cahlen/kronecker-coefficients",
                "spectra": "https://huggingface.co/datasets/cahlen/continued-fraction-spectra",
                "cfd_chaotic_advection": "https://huggingface.co/datasets/cahlen/cfd-chaotic-advection",
                "cfd_ns_bkm": "https://huggingface.co/datasets/cahlen/cfd-ns-bkm",
                "cfd_ns3d_bkm": "https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm",
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

    site_root = output_path.parent.parent.parent
    generate_llms_txt(site_root, certs)

    changelog_script = site_root / "scripts" / "generate_changelog.py"
    if changelog_script.exists():
        result = subprocess.run(
            [sys.executable, str(changelog_script)],
            cwd=str(site_root),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Changelog update skipped: {result.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
