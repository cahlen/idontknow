# bigcompute.science — Open GPU Computation Results for Humans and Agents

We just launched [bigcompute.science](https://bigcompute.science), an open platform for publishing results from heavy GPU computation — custom CUDA kernels, cluster-scale verification, frontier-pushing experiments. Every result is structured for both human readers and AI agents.

## Why this exists

There's no good place for serious computational results that aren't formal papers but are more than tweets. If you run a 10-hour GPU experiment and find something interesting, where do you put it? arXiv is too formal. Twitter is too ephemeral. GitHub is great for code but bad for presenting results with rendered math.

bigcompute.science fills that gap. Markdown + LaTeX, structured YAML frontmatter, raw data, and an [`/llms.txt`](https://bigcompute.science/llms.txt) endpoint so agents can discover and consume results programmatically.

## What we're running

On an 8× NVIDIA B200 DGX cluster (1.43 TB VRAM), we're running a pipeline of experiments targeting **open problems where GPU compute actually pushes the frontier**:

**Live now:** Verifying [Zaremba's Conjecture](https://bigcompute.science/experiments/zaremba-conjecture-8b-verification/) (1972) for 8 billion values. We also raced two SOTA theorem-proving LLMs (Goedel-Prover-V2-32B vs Kimina-Prover-72B) against formally stated cases in Lean 4 — 19/20 proved, 10-10 split. Novel finding: the smallest Zaremba witness concentrates at a/d ≈ 0.171, connected to the golden ratio.

**Complete:** Ramsey R(5,5) — exhaustively checked all 656 known K₄₂ colorings (McKay-Radziszowski database) via 4-SAT reformulation. None extend to K₄₃. Also checked all 2^42 = 4.4 trillion extensions of Exoo's coloring. This is the strongest computational evidence ever assembled that R(5,5) = 43.

**Queued:** Class numbers of real quadratic fields to 10^13 (100x beyond known tables), Kronecker coefficients to n=120 (relevant to P vs NP via GCT).

## Agent-friendly by design

Every experiment has machine-readable frontmatter: hardware specs, software versions, tags, results, code links. An agent can hit `/llms.txt`, discover all experiments, filter by domain or hardware, and pull structured data — no scraping required. The goal: when a mathematical agent needs to know "has anyone verified Zaremba's conjecture beyond 10^9?", the answer is already indexed and accessible.

## Contribute

The site is static Astro + KaTeX on Cloudflare Pages. Adding an experiment = dropping a markdown file with frontmatter. Clone it: [github.com/cahlen/bigcompute.science](https://github.com/cahlen/bigcompute.science)

Code + CUDA kernels: [github.com/cahlen/idontknow](https://github.com/cahlen/idontknow)
