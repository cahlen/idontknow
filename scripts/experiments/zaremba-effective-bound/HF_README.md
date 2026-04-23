---
license: cc-by-4.0
tags:
  - number-theory
  - continued-fractions
  - zaremba-conjecture
  - transfer-operator
  - spectral-theory
  - gpu-computation
  - mathematics
pretty_name: "Zaremba's Conjecture — Computational Proof Framework Data"
configs:
  - config_name: dolgopyat-profile
    data_files:
      - split: train
        path: dolgopyat-profile/dolgopyat_profile_exact.csv
    default: true
  - config_name: representation-counts
    data_files:
      - split: train
        path: representation-counts/representation_counts_1M.csv
---

# Zaremba's Conjecture — Computational Proof Framework Data

Transfer operator spectral data, Dolgopyat contraction profile, spectral gap computations, and representation counts supporting the **computational proof framework** (not a completed proof) for Zaremba's Conjecture. Produced through human-AI collaboration (Cahlen Humphreys + Claude). **Not independently peer-reviewed. AI-audited against published literature.**

> Part of the [bigcompute.science](https://bigcompute.science) project.

**Note:** Density sweep data (exception sets, density measurements, digit pair hierarchies) lives in [cahlen/zaremba-density](https://huggingface.co/datasets/cahlen/zaremba-density).

---

## ⚠️ Verification Status (updated 2026-04-22)

This dataset accompanies a paper that is explicitly framed as a **proof framework**, not a completed proof. Several pieces have distinct verification levels. A reader should read the breakdown below before citing any result.

### Brute-force verification (d ≤ 2.1 × 10¹¹)

- The published run log `run_210B.log` reports `Uncovered: 0` in `6962.2 s` on 8× NVIDIA B200 (CUDA 13.0, driver 580.126.09), with 256 rounds × 8 GPUs and 119,210 seeds per chunk.
- **Software-audit caveat.** The original `matrix_enum_multipass.cu` (v6) kernel counts every expansion via `atomicAdd(out_count, 1ULL)` but writes only if `pos < max_out`, then clips the next frontier to `min(h_out, BUF_SLOTS)`. This means that if the true frontier ever exceeded the 2 × 10⁹ `BUF_SLOTS`, matrices were silently dropped. The original kernel emits **no machine-checkable no-overflow certificate**, so `Uncovered: 0` is conditional on "no overflow ever occurred" — which the original run did not prove.
- **Current status: strong computational evidence, not certified.**
- **Path to certification:** a hardened replacement, [`matrix_enum_multipass_v6_1.cu`](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu), adds (1) a hard overflow abort at every `expand_mark_compact_safe` call, (2) a per-round peak-frontier log, and (3) a final "NO-OVERFLOW CERTIFICATE" block. A v6.1 re-run of the 210B configuration whose tail reports `All peaks < BUF_SLOTS: YES` and `No-overflow abort fired: NO` upgrades the claim to certified. See [`paper/CERTIFICATE.md`](https://github.com/cahlen/idontknow/blob/main/paper/CERTIFICATE.md) for the exact procedure.
- **Self-audit in progress.** Local probes on a single RTX 5090 (32 GB, `BUF_SLOTS = 4 × 10⁸`) with the 210B chunk size (119,210 seeds per chunk, 2048 rounds) are collected under `idontknow/logs/v6_1_suite/`. Preliminary data at `max_d = 10⁸` shows per-chunk peak frontier ≈ 1.91 × 10⁹ — within 5% of the B200 `BUF_SLOTS`. Larger `max_d` probes are running.

### Spectral gap computation (congruence transfer operator)

- 11 covering primes certified at 256-bit MPFR precision (77 decimal digits), all gaps `σ_p ≥ 0.650`.
- **Caveat.** "Certification" applies to the **finite Galerkin matrix** (`N = 40` Chebyshev collocation), not to the infinite-dimensional transfer operator. A rigorous a-posteriori truncation error bound (Keller–Liverani type) has not been established.

### Dolgopyat contraction profile

- `ρ_η = sup_{t ≥ 1} ρ(t) ≤ 0.771` (arb-certified on `[1, 1000]` using FLINT ball arithmetic, 70 certified digits).
- **Caveat.** Same as above — certification is for the `N = 80` discretization, not the full operator.

### Layer 4 / property (τ)

- Currently invoked non-effectively. **No explicit constant.** This is one of the four gaps listed on the main finding page.

### MOW theorem matching

- Not yet verified theorem-by-theorem against Magee–Oh–Winter (Crelle 2019) and Calderón–Magee (JEMS 2025). **Pending independent verification.**

**In short:** the bounded computational result is the strongest piece, but even it is currently "strong computational evidence" until the v6.1 re-run lands. The analytic ingredients (spectral data, Dolgopyat profile) are interval-certified only for the finite discretizations.

---

## Datasets

### 1. Dolgopyat Transfer Operator Profile (`dolgopyat-profile/`)

Spectral radius profile `ρ(t)` computed via arb ball arithmetic (FLINT, 256-bit precision).

| File | Records | Description |
|------|---------|-------------|
| `dolgopyat_profile_exact.csv` | 20,001 | `ρ(t)` for `t ∈ [1, ~21]`, step 0.001 |

Supremum `ρ_η ≤ 0.771` establishes the Dolgopyat contraction for the MOW framework (on the `N = 80` Chebyshev discretization).

### 2. Spectral Gap Computations (`logs/`)

Transfer operator spectral gap logs for various matrix sizes `N` and moduli `m`.

| Files | Description |
|-------|-------------|
| `gaps_N{15..40}_m{34,638,1469}.log` | Spectral gaps at matrix sizes 15–40 for selected moduli |

These verify that spectral gaps of the finite Galerkin matrices remain uniform (`≥ 0.237`) across moduli — evidence consistent with (but not a proof of) property (τ) at this scale.

### 3. Representation Counts (`representation-counts/`)

`R(d)` = number of coprime fractions `a/d` with all CF partial quotients `≤ 5`, for `d = 1` to `1,000,000`.

| File | Records | Description |
|------|---------|-------------|
| `representation_counts_1M.csv` | 1,000,001 | `R(d)` for `d ∈ [0, 10⁶]` |

Growth: `R(d) ~ c₁ · d^(2δ - 1)` where `δ = 0.836829`. Observed exponent 0.654 (least-squares fit); theoretical 0.674. The slight undercount is expected from finite-depth effects.

## Related Datasets

- **[cahlen/zaremba-density](https://huggingface.co/datasets/cahlen/zaremba-density)** — GPU-computed density sweeps, exception sets, digit pair hierarchies (65+ experiments, `10⁶` through `10¹⁴`)
- **[cahlen/hausdorff-dimension-spectrum](https://huggingface.co/datasets/cahlen/hausdorff-dimension-spectrum)** — `dim_H(E_A)` for all `2²⁰ − 1` subsets

## Canonical code paths

- Brute-force kernel (v6, original 210B run): [`scripts/experiments/zaremba-effective-bound/matrix_enum_multipass.cu`](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/zaremba-effective-bound/matrix_enum_multipass.cu)
- Hardened kernel (v6.1, with no-overflow certificate): [`scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu`](https://github.com/cahlen/idontknow/blob/main/scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu)
- Verification manifest (SHA256 checksums, environment): [`paper/verification-manifest.txt`](https://github.com/cahlen/idontknow/blob/main/paper/verification-manifest.txt)
- Certification procedure: [`paper/CERTIFICATE.md`](https://github.com/cahlen/idontknow/blob/main/paper/CERTIFICATE.md)

## Hardware

- 8× NVIDIA B200 (DGX, 1.43 TB VRAM, NVLink 5) — 210B headline run
- RTX 5090 (32 GB) — v6.1 local self-audit probes

## Source

- **Paper**: [Proof framework (PDF)](https://github.com/cahlen/idontknow/blob/main/paper/zaremba-proof.pdf)
- **Code**: [github.com/cahlen/idontknow](https://github.com/cahlen/idontknow)
- **Findings page**: [Proof framework (bigcompute.science)](https://bigcompute.science/findings/zaremba-conjecture-proved/)
- **Experiment page**: [210B verification](https://bigcompute.science/experiments/zaremba-conjecture-verification/)
- **MCP server**: `mcp.bigcompute.science` (22 tools, no auth)

## Citation

```bibtex
@misc{humphreys2026zaremba,
  author = {Humphreys, Cahlen and Claude (Anthropic)},
  title = {Zaremba's Conjecture: Computational Proof Framework Data},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/cahlen/zaremba-conjecture-data}
}
```

Human-AI collaborative work. AI-audited against published literature. Not independently peer-reviewed. CC BY 4.0.
