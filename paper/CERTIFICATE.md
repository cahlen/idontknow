# Zaremba 210B Verification: Computational Certificate

**Status as of 2026-04-22:** strong computational evidence, *not* yet a certified
computational result.

This document describes what was run, what artifacts exist, what can currently
be machine-checked, and what must still be done to upgrade the 210B headline
claim from "strong computational evidence" to "certified computational result".

---

## 1. Canonical artifact

| Field | Value |
|-------|-------|
| Kernel source | `scripts/experiments/zaremba-effective-bound/matrix_enum_multipass.cu` (v6 original) |
| Hardened replacement | `scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu` (v6.1) |
| Run log | `scripts/experiments/zaremba-effective-bound/run_210B.log` |
| Verification manifest | `paper/verification-manifest.txt` |
| Paper | `paper/zaremba-proof.tex` (PDF built separately) |
| Experiment page | `bigcompute.science/src/content/experiments/2026-03-28-zaremba-conjecture-8b-verification.md` |
| Main finding page | `bigcompute.science/src/content/findings/2026-03-29-zaremba-conjecture-proved.md` |
| Hugging Face dataset | `cahlen/zaremba-conjecture-data` |

**One canonical kernel path:** `scripts/experiments/zaremba-effective-bound/`. Any
document referring to `scripts/experiments/zaremba-conjecture-verification/` or
`scripts/zaremba_verify_v4.cu` for the 210B claim is stale and should be fixed.

---

## 2. Exact invocation (original 210B headline run)

```bash
nvcc -O3 -arch=sm_100a -o matrix_v6 \
    scripts/experiments/zaremba-effective-bound/matrix_enum_multipass.cu -lpthread
./matrix_v6 210000000000
```

| Parameter | Value |
|-----------|-------|
| max_d | 2.1 × 10¹¹ |
| Phase A depth | 12 |
| Phase A depth-12 live seeds | 244,140,625 (before pruning); log reports 2.44 × 10⁸ live after pruning |
| Phase B depth | 62 |
| num_rounds | 256 |
| Seeds per chunk per GPU | 119,210 |
| GPUs | 8 |
| Hardware | NVIDIA DGX B200, 8 × B200 (183 GB each), 2 × Xeon Platinum 8570 |
| CUDA | 13.0 |
| Driver | 580.126.09 |
| Wall time | 6,962.2 s (116 min) |
| Uncovered | 0 |

---

## 3. What is currently machine-checkable

- **Run log tail** (`run_210B.log`):
  - `Time: 6962.2s`
  - `Uncovered: 0` on the range `[1, 210000000000]`
  - Per-GPU progress lines `Round k/256: GPU g: 119210 matrices`
- **Manifest** (`paper/verification-manifest.txt`):
  - SHA256 of `matrix_enum_multipass.cu`
  - SHA256 of `run_210B.log`
  - SHA256 of Dolgopyat profile CSV
  - CUDA 13.0, driver 580.126.09
- **Reproducibility** on equivalent hardware:
  - Anyone with ≥1.5 TB of GPU memory (e.g. DGX B200, H200, H100 cluster) can
    re-run the exact command above and compare the tail of the log to the
    manifest.

## 4. What is NOT yet machine-checkable (the open gap)

The original v6 kernel contains the following lines in
`expand_mark_compact_safe`:

```c
unsigned long long pos = atomicAdd(out_count, 1ULL);
if (pos < max_out) {
    /* write matrix */
}
/* else: drop silently */
```

and, after each Phase B level,

```c
num = h_out < BUF_SLOTS ? h_out : BUF_SLOTS;
```

This means that if the true frontier of live matrices at any intermediate
level ever exceeded `BUF_SLOTS` (2 × 10⁹ in the B200 run), those matrices
were silently dropped, the next level continued from a clipped frontier,
and the final "Uncovered: 0" could in principle miss denominators whose
CF paths were on the dropped branches. The kernel emits no assertion,
no certificate, and no per-level peak-frontier report in the published log.

**Consequence.** "Uncovered: 0" is conditional on "no overflow ever
occurred". The original kernel does not prove that, and the published
manifest does not record a no-overflow event log.

---

## 5. The hardened replacement (v6.1)

`matrix_enum_multipass_v6_1.cu` is a drop-in replacement with three changes:

1. **Hard overflow abort.** Every call to `expand_mark_compact_safe` passes
   an `overflow_count` device-side counter; if the kernel atomically
   reports any overflow, the host prints a fatal diagnostic and exits
   with status `2`. This replaces silent clipping with a loud failure.
2. **Per-round peak-frontier logging.** Each Phase B round prints the
   maximum `out_count` it observed; the host also tracks `phase_a_peak`
   and `phase_b_peak` across the full run.
3. **Final no-overflow certificate block.** At the end of a successful run,
   the binary prints:

   ```
   --- NO-OVERFLOW CERTIFICATE ---
   BUF_SLOTS                 : 2000000000
   Phase A peak frontier     : <N>  (<f> of BUF_SLOTS)
   Phase A overflow events   : 0
   Phase B peak frontier     : <M>  (<g> of BUF_SLOTS)
   Phase B overflow events   : 0
   All peaks < BUF_SLOTS     : YES
   No-overflow abort fired   : NO
   ```

   A run whose tail reports `All peaks < BUF_SLOTS: YES` and
   `No-overflow abort fired: NO` is a certified computational artifact.
   Any other tail means the claim is not certified.

4. **`ZAREMBA_PROBE=1` diagnostic mode.** When set, the kernel disables the
   hard abort (it clips and continues, as v6 did), but still tracks peak
   frontiers and prints the certificate block with
   `No-overflow abort fired: YES` and a comment that the uncovered count
   is **INVALID** in probe mode. This is for measuring true peak
   frontiers on smaller GPUs to decide whether the original 210B
   chunk size was safe.

5. **`ZAREMBA_ROUNDS=<n>` override.** Allows matching a specific chunk size
   for cross-hardware probing (e.g. using 2048 rounds on a single RTX 5090
   to mimic the 119,210 seeds/chunk of the 8× B200 run).

---

## 6. Self-audit on local hardware (RTX 5090, 32 GB)

A suite was run in `logs/v6_1_suite/` with `BUF_SLOTS = 4 × 10⁸`
(compiled via `-DBUF_SLOTS=400000000ULL`). This is 1/5 of the B200
run's 2 × 10⁹ buffer, which means our local abort threshold is
**deliberately more sensitive** than the B200 run's.

**Probe mode** (single RTX 5090, 119,210 seeds per chunk, matching the
210B configuration):

| max_d | num_rounds | Max per-chunk peak frontier observed | vs BUF_SLOTS = 2 × 10⁹ |
|-------|-----------|--------------------------------------|------------------------|
| 10⁸ | 2048 | **1.91 × 10⁹** (full 2048/2048 rounds, 1407 s wall time) | under by 4.5% |
| 10⁹ | 2048 | **≥ 2.00 × 10⁹** (partial — already crossed BUF_SLOTS at round 23/2048) | at or above buffer wall |
| 10¹⁰ | 2048 | *pending* (probe 3) | expected higher still |

Full logs: `idontknow/logs/v6_1_suite/v6_1_PROBE_d*.log`.

**Interpretation (2026-04-22).** The monotonic scaling of per-chunk
peak frontier in `max_d` is now measured empirically, not just
predicted. At `max_d = 10⁸` we are 4.5% under BUF_SLOTS; at
`max_d = 10⁹` we are already at or above it; the B200 headline run
was at `max_d = 2.1 × 10¹¹`, a factor of 210 higher. **The original
210B run almost certainly clipped Phase B frontiers silently.** This
does not prove the 210B claim is wrong — clipped matrices could have
had their denominators marked via other unclipped CF paths — but it
does mean the original kernel did not produce a machine-checkable
computational certificate, and the claim must be treated as strong
computational evidence rather than certified until a v6.1 re-run is
performed on equivalent hardware.

**Measured growth, not predicted.** The earlier version of this
document listed the `max_d = 10⁹` peak as "expected to be ≥ 1.9 × 10⁹"
based on extrapolation. That prediction has now been confirmed
empirically: probe 2 (max_d = 10⁹) crossed 2.00 × 10⁹ at round 23 of
2048, matching the B200 `BUF_SLOTS` exactly. The `max_d = 10¹⁰` probe
and the 10¹¹-scale extrapolation both point to peaks substantially
above 2 × 10⁹, i.e. the 210B headline run was almost certainly above
its buffer wall. The definitive check remains a v6.1 re-run at
`max_d = 2.1 × 10¹¹` on hardware with ≥ 1.5 TB of GPU memory.

**Earlier, less informative probes** at `num_rounds = 1` (not the 210B
chunk size): peak frontier > 25 × 10⁹, i.e. an order of magnitude beyond
`BUF_SLOTS`. This means any earlier row in the experiment page's "scaling
table" that used `num_rounds = 1` (the `d ≤ 10⁹` row at 21.8 s, and
corresponding intermediate rows) was almost certainly clipping silently.
**Only the 210B headline run, with its 256-round chunking, is in the
candidate-safe regime, and even it is within 5 % of the buffer wall at
`max_d = 10⁸` — far from comfortable at `max_d = 2.1 × 10¹¹`.**

---

## 7. Path to certified status

To upgrade the 210B claim from "strong computational evidence" to
"certified computational result", the following are sufficient:

1. Compile `matrix_enum_multipass_v6_1.cu` on 8× B200 (or equivalent
   ≥ 1.5 TB aggregate GPU memory) with default `BUF_SLOTS = 2 × 10⁹`.
2. Run `./matrix_v6_1 210000000000` with `ZAREMBA_ROUNDS=256` (the
   original configuration).
3. Verify the tail contains
   - `Uncovered: 0`
   - `All peaks < BUF_SLOTS: YES`
   - `No-overflow abort fired: NO`
4. Add SHA256 of the new log and the v6.1 source to the manifest, and
   swap the experiment page's headline caveat from "strong computational
   evidence" to "certified" at the same commit.

If step 3 instead produces `No-overflow abort fired: YES`, the correct
response is to increase `num_rounds` (e.g. to 512 or 1024) until the
chunk size is safe, re-run, and record the new chunk size as the
canonical configuration.

## 8. Path to larger certified bounds on local hardware

For a single RTX 5090 (32 GB), the largest `max_d` at which v6.1
completes with `All peaks < BUF_SLOTS: YES` at `BUF_SLOTS = 4 × 10⁸`
defines the certified local bound. This is a real computational
artifact: it is smaller than 2.1 × 10¹¹, but unlike the headline run,
it comes with a published no-overflow certificate. The ongoing suite in
`logs/v6_1_suite/` will report this bound directly.
