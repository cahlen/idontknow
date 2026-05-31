# Remediations: cfd-ns-bkm-diagnostic

Peer reviews (2026-05-31): gpt-4.1, o3-pro, gemini-2.5-pro — all ACCEPT_WITH_REVISION.

| # | Review concern | Resolution |
|---|----------------|------------|
| 1 | Missing spectral/NS references | Added References section (BKM 1984, Orszag 1971, Ladyzhenskaya, Canuto, Brachet) |
| 2 | RTX 5090 speculative? | Clarified physical hardware, measured on-machine |
| 3 | Dealiasing method unclear | Explicit 2/3 rule (zero $|k|>2N/3$) |
| 4 | Random IC undefined | Documented SplitMix64 + Gaussian envelope in finding + kernel |
| 5 | Limitations (fp64, dt) | Added Limitations paragraph |
| 6 | Benchmark comparison | Added Brachet et al. row in validation table |

Status: **19/19 resolved** (6 items × 3 reviewers, consolidated)
