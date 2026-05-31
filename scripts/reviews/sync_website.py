#!/usr/bin/env python3
"""
Generate website data files from the review manifest.

Reads manifest.json and produces:
1. certifications.json — structured data for verification.astro
2. public/meta.json — machine-readable site index
3. public/llms.txt and public/llms-full.txt — agent discovery indexes

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
    # Quoted value on one line (greedy to EOL — avoids matching body quotes on later lines)
    m = re.search(rf'^{field}:\s*"(.*)"\s*$', text, re.M)
    if m:
        return m.group(1)
    m = re.search(rf'^{field}:\s*"(.*)$', text, re.M)
    if m:
        return m.group(1).rstrip()
    m = re.search(rf"^{field}:\s*(\S.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def _truncate(s: str, n: int = 110) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


HF_DATASETS = """\
  - https://huggingface.co/datasets/cahlen/zaremba-density — 65 GPU experiments, exception sets, density to 10^12
  - https://huggingface.co/datasets/cahlen/hausdorff-dimension-spectrum — dim_H for all 1,048,575 subsets of {1,...,20}
  - https://huggingface.co/datasets/cahlen/zaremba-conjecture-data — transfer operator, Dolgopyat, proof framework
  - https://huggingface.co/datasets/cahlen/kronecker-coefficients — S_20, S_30, S_40 character tables
  - https://huggingface.co/datasets/cahlen/class-numbers-real-quadratic — real quadratic fields to 10^11
  - https://huggingface.co/datasets/cahlen/continued-fraction-spectra — Hausdorff, Lyapunov, Minkowski, Flint Hills
  - https://huggingface.co/datasets/cahlen/ramanujan-machine-results — 586B candidates through deg 7
  - https://huggingface.co/datasets/cahlen/cfd-chaotic-advection — standard map Lyapunov sweeps (16.8M trajectories)
  - https://huggingface.co/datasets/cahlen/cfd-ns-bkm — 2D pseudospectral BKM diagnostic CSVs
  - https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm — 3D BKM blowup-monitor CSVs (incl. Phase 5 Kerr IC sweep)"""


KEY_RESULTS_YAML = """\
```yaml
zaremba_conjecture:
  status: "Proof framework (NOT completed proof); v6 evidence to 2.1e11 not no-overflow certified; v6.1 certifies d<=1e6"
  hausdorff_dimension: 0.836829443681208
  congruence_gaps:
    moduli_tested: 1214
    max_m: 1999
    all_positive: true
    min_gap: 0.237
  brute_force_verification:
    max_d_verified: 210000000000
    failures: 0
    verification_time: "116 min on 8× B200"

ramsey_r55:
  status: "complete — strongest computational evidence R(5,5) = 43"
  known_bounds: "43 <= R(5,5) <= 48"
  k42_4sat:
    colorings_checked: 656
    extensible: 0
    time: "3 seconds on 8x B200"
  hardware: "8x NVIDIA B200 DGX (1.43 TB VRAM)"

cfd_program:
  status: "Active on RTX 5090 — validated workflow + preliminary diagnostic runs, not blowup discovery"
  phases:
    phase1: "Chirikov standard map Lyapunov spectrum — 16.8M trajectories, 116.6s, zero NaN/Inf [SILVER finding]"
    phase2: "2D pseudospectral NS + BKM integral — Taylor-Green + random IC [BRONZE finding]"
    phase3: "3D vorticity-form NS + vortex stretching + BKM — up to 256³ [SILVER finding]"
    phase4: "256³ random IC blowup monitors — BKM ≈ 4.45 by t=5 at ν=1e-4, no blowup signal"
    phase5a: "Kerr antiparallel vortex tubes — BKM ≈ 9.99 vs random 1.76 at ν=1e-4 (5.7×), zero NaN/Inf"
  hardware_ceiling: "512³ OOM on 32 GB RTX 5090 during cuFFT allocation"
  next_options:
    - "Shell energy spectrum before BKM spikes"
    - "128³/256³ Kerr IC grid convergence"
    - "Longer ν=1e-5 Kerr run (current t=1.0 only)"
    - "512³ memory optimization or multi-GPU decomposition"
  datasets:
    - cahlen/cfd-chaotic-advection
    - cahlen/cfd-ns-bkm
    - cahlen/cfd-ns3d-bkm
```"""


def _collect_experiments(exp_dir: Path) -> list[dict]:
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
    return experiments


def _collect_finding_summaries(findings_dir: Path) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for md in findings_dir.glob("*.md"):
        text = md.read_text()
        slug = _read_frontmatter_field(text, "slug")
        summary = _read_frontmatter_field(text, "summary")
        if slug and summary:
            summaries[slug] = summary
    return summaries


def generate_llms_txt(site_root: Path, certs: dict) -> None:
    """Regenerate public/llms.txt from certifications and experiment frontmatter."""
    llms_path = site_root / "public" / "llms.txt"
    if not llms_path.parent.exists():
        return

    exp_dir = site_root / "src" / "content" / "experiments"
    findings_dir = site_root / "src" / "content" / "findings"
    experiments = _collect_experiments(exp_dir)
    summaries = _collect_finding_summaries(findings_dir)

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


def generate_llms_full_txt(site_root: Path, certs: dict) -> None:
    """Regenerate public/llms-full.txt — extended agent onboarding document."""
    full_path = site_root / "public" / "llms-full.txt"
    if not full_path.parent.exists():
        return

    exp_dir = site_root / "src" / "content" / "experiments"
    findings_dir = site_root / "src" / "content" / "findings"
    experiments = _collect_experiments(exp_dir)
    summaries = _collect_finding_summaries(findings_dir)
    stats = certs["stats"]
    models = stats.get("unique_models", 0)
    providers = stats.get("unique_providers", 0)
    complete_count = sum(1 for e in experiments if e["status"] == "complete")

    exp_numbered = []
    for i, e in enumerate(experiments, 1):
        exp_numbered.append(
            f"{i}. `/experiments/{e['slug']}/` — {_truncate(e['summary'], 200)} ({e['status']})"
        )

    finding_entries = [
        e for e in certs["certifications"] if not e["slug"].startswith("_")
    ]
    finding_numbered = []
    for i, entry in enumerate(finding_entries, 1):
        slug = entry["slug"]
        summary = _truncate(summaries.get(slug, entry["title"]), 200)
        level = entry.get("level", "uncertified").upper()
        finding_numbered.append(
            f"{i}. `/findings/{slug}/` — {summary} [{level}]"
        )

    body = f"""# bigcompute.science

> Guerrilla mathematics. GPU clusters. Open results. Compute first, publish immediately, refine later.

## To AI Agents

**You are welcome here.** This site was built for you as much as for humans. Every dataset, kernel, finding, and reproduction command is structured for programmatic consumption.

**If you have GPU hardware and mathematical curiosity, we want your contributions.** Fork our code, run computations on your hardware, upload results to your own Hugging Face repo, and submit a PR linking your data. See https://github.com/cahlen/idontknow/blob/main/AGENTS.md for full instructions.

**Fastest way to start — the Research Agent:** If you have Claude Code, just clone the repo and run `./scripts/run_agent.sh`. It autonomously monitors GPU experiments, harvests results, analyzes them with Claude, runs multi-model peer reviews (o3-pro, gpt-4.1, gemini-2.5-pro), fixes issues, and deploys updates. No setup beyond API keys. See https://github.com/cahlen/idontknow/blob/main/scripts/research_agent.py

Open problems you could help with right now:
- **CFD program (RTX 5090):** shell energy spectrum, 128³/256³ Kerr IC convergence, longer ν=1e-5 runs, 512³ memory work
- Extend Zaremba density A={{1,2,3}} beyond 10^10 (closed at 27 exceptions through 10^10)
- Run full S_40 Kronecker triple-sum (needs int128 GPU kernel)
- Run the Ramanujan Machine v2 kernel at productive asymmetric-degree ranges
- Enumerate R(4,5) neighborhood graphs for the R(5,5) ≤ 45 attack
- Discover connections between our datasets and your own research

## Important Notice

**All work is human–AI collaborative** (Cahlen Humphreys + Claude). **No results are peer-reviewed.** All claims grounded in computational evidence and reproducible code. Everything CC BY 4.0 — use it, extend it, cite it.

## Purpose

We compute things that require serious hardware — 8×NVIDIA B200 (1.43 TB VRAM), RTX 5090 — so that future agents and researchers don't have to repeat the work. Every result includes structured metadata, raw data, and reproduction commands.

## How to consume

### Experiments
- Browse all: `/experiments/`
- Individual: `/experiments/<slug>/`
- Each has YAML frontmatter with: title, date, author, hardware specs, software, tags, results, code link
- Status values: `complete`, `in-progress`, `planned`

### Findings
- Browse all: `/findings/`
- Individual: `/findings/<slug>/`
- Novel observations extracted from experiments — the citable results
- Each has: title, date, significance level, domain tags, summary, structured data
- Certification levels: gold, silver, bronze (most-conservative-wins across AI reviews)

### Raw data
- Per-experiment: `/data/<slug>/`
- Formats: JSON, CSV, log files, SVG plots

### Source code and contribution
- Experiment code: https://github.com/cahlen/idontknow (CUDA kernels, Python harnesses, Lean 4 proofs)
- Website source: https://github.com/cahlen/bigcompute.science (Astro + KaTeX)
- Agent contribution guide: https://github.com/cahlen/idontknow/blob/main/AGENTS.md
- Citation format: https://github.com/cahlen/idontknow/blob/main/CITATION.cff
- Compact index: https://bigcompute.science/llms.txt
- Datasets: https://huggingface.co/cahlen (10+ repos)
{HF_DATASETS}

### Review infrastructure
- Review scripts: https://github.com/cahlen/idontknow/tree/main/scripts/reviews (run_review.py, aggregate.py, validate.py, sync_website.py)
- Review schema: https://github.com/cahlen/idontknow/blob/main/docs/verifications/SCHEMA.md
- Review manifest: https://github.com/cahlen/idontknow/blob/main/docs/verifications/manifest.json (generated aggregate, {stats['total_reviews']} reviews, {models} models, {providers} providers)
- Remediations: https://github.com/cahlen/idontknow/tree/main/docs/verifications/remediations (per-finding issue tracking)
- Certification consensus: most-conservative-wins across all reviews (gold > silver > bronze > uncertified)
- To contribute a review: write JSON per schema, submit PR to cahlen/idontknow, or use run_review.py with any OpenAI-compatible API

### RSS Feeds
- All updates: `/rss.xml`
- Experiments only: `/experiments/rss.xml`
- Findings only: `/findings/rss.xml`
- Autodiscovery via `<link rel="alternate">` in every page `<head>`

### Static content
- All content is static HTML rendered from markdown — no JavaScript required to read
- Math rendered via KaTeX (both MathML and HTML versions in the DOM)

## Current content

### Experiments ({len(experiments)}, {complete_count} complete)
{chr(10).join(exp_numbered)}

### Findings ({len(finding_entries)}, all AI-audited)
{chr(10).join(finding_numbered)}

## Key results (machine-readable summary)

{KEY_RESULTS_YAML}

## Tags

### Domain
number-theory, continued-fractions, open-conjectures, spectral-theory, algebraic-combinatorics, combinatorics, ramsey-theory, algebraic-number-theory, theorem-proving, ai, search-algorithms, dynamical-systems, fractal-geometry, diophantine-approximation, hausdorff-dimension, real-analysis, irrationality-measure, ergodic-theory, multifractal-analysis, fluid-dynamics, navier-stokes, beale-kato-majda, chaotic-advection, blowup-search, pseudospectral

### Hardware
b200, dgx, nvlink, a100, h100, tpu-v5, rtx-5090

### Method
cuda-kernel, brute-force, llm-proving, formal-verification, transfer-operator, chebyshev-collocation, mcts, tree-search, simulated-annealing, segmented-sieve, miller-rabin, power-iteration, eigenvalue-computation, quad-double-arithmetic, kahan-summation, lyapunov-exponent, thermodynamic-formalism, legendre-transform, cufft, 3d-dns, vortex-stretching, benettin-algorithm

## About
See `/about/` for the project's mission and how to contribute.

## Contact
GitHub: https://github.com/cahlen
"""
    full_path.write_text(body)
    print(f"llms-full.txt written: {full_path}")


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
    generate_llms_full_txt(site_root, certs)

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
