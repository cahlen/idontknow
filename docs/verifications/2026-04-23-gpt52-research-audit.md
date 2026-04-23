# GPT-5.2 Research Audit — 2026-04-23

Scope: local repositories `idontknow` and `bigcompute.science` at `origin/main` as of 2026-04-23. Both repos were fetched from GitHub before this audit and were even with `origin/main`.

Hardware available to this shell: CUDA 13.0 compiler is installed, but `nvidia-smi` cannot communicate with the NVIDIA driver from this environment. I therefore ran CPU/reference checks and repository consistency checks, not long GPU reproductions.

## Executive Findings

1. **Zaremba proof status must remain downgraded.** The 210B v6 run is strong computational evidence, not a certified result. The current certified no-overflow artifact covers only `d ≤ 10^6`. The MOW route remains a proof framework with operator-transport and constant-propagation gaps.
2. **Several Zaremba density pages used “closed” too strongly.** The data support stable candidate exception sets within completed search ranges, not analytic closure. Completed 10^11 logs in this repo certify stability for `{1,2,6}` and `{1,2,7}`; `{1,2,3}`, `{1,2,4}`, and `{1,2,5}` 10^11 logs are partial.
3. **The inverse-square amplification headline was not supported as stated.** Matched `N=10^10` data fit exponent `-1.463`, not `-2`; the `N=10^11` matched overlap has only three points (`k=3,4,5`) and is suggestive only.
4. **The all-prime transitivity proof is still provisional.** The finite computation to `p=17,389` is valuable, but the all-prime Dickson argument needs a corrected ambient group formulation because the generators have determinant `-1`.
5. **The Cayley diameter computation is exploratory, not certified.** The kernel uses determinant `-1` generators while claiming `SL₂`, stops by `total_visited >= |SL₂(p)|`, and does not popcount the visited bitset or certify no frontier clipping.

## Per-Finding Audit Notes

| Finding | Audit status | Remarks / revision |
|---|---|---|
| `zaremba-conjecture-framework` | **Major caveats** | Renamed/canonicalized away from `zaremba-conjecture-proved`; page now states certified `d≤10^6`, evidence to `2.1e11`, and conditional MOW threshold. |
| `zaremba-density-phase-transition` | **Revised** | Replaced “closed” language with stable-candidate language; corrected `{2,3,4,5}` Hausdorff dimension to `0.559636`; noted partial 10^11 logs. |
| `zaremba-exception-hierarchy` | **Rewritten** | Old page contained a stale/corrupted exception list. Replaced with the correct 27 exceptions and a checked `27→2→0` witness table. |
| `zaremba-digit-pair-hierarchy` | **Revised** | Kept `{1,k}` hierarchy; changed “closed exception sets” to observational stability. |
| `zaremba-inverse-square-amplification` | **Downgraded** | Strong digit-1 amplification remains; inverse-square law is now explicitly a hypothesis pending matched larger-N runs. |
| `zaremba-transitivity-all-primes` | **Downgraded** | Finite BFS claim retained to `p=17,389`; all-prime proof marked provisional. |
| `zaremba-cayley-diameters` | **Downgraded** | Added ambient-group and validation caveats; fixed stale source path. |
| `zaremba-A12-logarithmic-convergence` | **Updated** | Removed stale “needs 10^12” language; noted observed `10^12` residual and OOM status for `10^13/10^14` logs. |
| `hausdorff-digit-one-dominance` | **Mostly sound** | Existing correction to `dim(E_{2..20})=0.7683` is consistent with `spectrum_n20.csv`; MCP stale value fixed. |
| `zaremba-spectral-gaps-uniform` | **Caveated** | Finite spectral data remain useful; property `(τ)` should be described as computationally supported, not proved. |
| `zaremba-representation-growth` | **Plausible empirical** | Should remain bronze: regression supports the exponent, but finite-depth and sampling/coverage details matter. |
| `zaremba-witness-golden-ratio` | **Heuristic** | Keep as distributional observation; golden-ratio interpretation should remain heuristic. |
| `gpu-matrix-enumeration-175x` | **Engineering claim** | Plausible but still bronze; benchmark depends on hardware and baseline definition. |
| `class-number-convergence` | **Sound with caveats** | Genus-theory correction is mathematically important and appropriate; independent validation details remain the key strength. |
| `kronecker-s30-largest-computation` | **Mostly sound** | Large computation claim should retain “to our knowledge”; unresolved remediation entries mostly ask for documentation/checksums. |
| `kronecker-s40-character-table` | **Mostly sound** | Character-table artifact is valuable; targeted coefficient conclusions should not be conflated with a full S40 Kronecker tensor. |
| `hausdorff / spectra experiments` | **Data artifact** | Useful as computational dataset; rigorous interval enclosures are still separate work. |

## Checks Run

- `git fetch --prune origin` for both repositories; both ended `0 ahead / 0 behind`.
- `python3 scripts/reviews/validate.py --all` before fixes: 0 errors, 15 schema warnings from non-normalized verdict strings. After normalizing review verdict fields and regenerating review artifacts: 0 errors, 0 warnings.
- CPU reference check: `gcc -O3 -o /tmp/zaremba_density scripts/experiments/zaremba-density/zaremba_density.c -lm && /tmp/zaremba_density 1000000 1,2,3`, reproducing 27 exceptions beginning `6,20,28,...`.
- Log audit for `gpu_A12*_1e11.log`: only `{1,2,6}`, `{1,2,7}`, `{1,2,8}`, `{1,2,9}`, `{1,2,10}`, and `{1,2}` contain completed `RESULTS` blocks among the checked family; `{1,2,3}`, `{1,2,4}`, and `{1,2,5}` are partial in this repo.
- Hausdorff CSV spot checks from `spectrum_n20.csv`: `{2,3,4,5}=0.559636450164777`, `{1,2,3,4}=0.788945557483154`, `{1,2,7}=0.617903695463376`, `{1,2,8}=0.608616964557154`, `{2..20}=0.768313628447352`.
- `python3 scripts/reviews/aggregate.py` and `python3 scripts/reviews/sync_website.py` completed, regenerating the verification manifest, website certifications, and `public/meta.json`.
- `ASTRO_TELEMETRY_DISABLED=1 npm run build` completed successfully after installing website dependencies with `npm ci --cache /tmp/npm-cache --loglevel=warn`. Warnings remain: Astro auto-generated content collections / `Astro.glob` deprecations, duplicate-id warnings for the transitivity and Cayley pages despite unique source slugs, 6 npm audit vulnerabilities (3 moderate, 3 high), and an engine warning because `sitemap@9.0.1` wants Node `>=20.19.5` while this shell has Node `20.19.4`.

## Next Work

1. Produce a corrected Cayley/transitivity computation in a precise ambient group with popcount validation and no-frontier-clipping certificate.
2. Run matched `N=10^11` or larger `{2,k}` computations for `k=6..10` before revisiting the inverse-square hypothesis.
3. Re-run the 210B Zaremba verification with v6.1 on B200-class memory, or use more rounds until the no-overflow certificate passes.
4. Add rigorous transfer-operator truncation bounds before treating MOW constants as theorem-level proof data.
