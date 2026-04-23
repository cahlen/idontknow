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
210B configuration, `ZAREMBA_PROBE=1` so overflow continues rather
than aborting):

| max_d | num_rounds | `h_out` peak observed | `overflow_count` total | Wall time | Interpretation |
|-------|-----------|-----------------------|------------------------|-----------|----------------|
| 10⁸ | 2048 | 1.91 × 10⁹ | 0 | 1,407 s | **True unclipped peak**; no overflow at local BUF_SLOTS=4e8; 95.5% of B200 BUF_SLOTS=2e9 |
| 10⁹ | 2048 | 2.00 × 10⁹ (saturated) | 17.5 × 10¹² | 6,987 s | True peak > 4e8; saturation artifact at 5× BUF_SLOTS |
| 10¹⁰ | 2048 | 4.29 × 10⁹ (saturated higher) | 25.2 × 10¹² | 7,491 s | Higher saturation level; confirms peak grows with max_d |

Full probe logs: `idontknow/logs/v6_1_suite/v6_1_PROBE_d*.log`.

### Interpreting the probe data correctly (2026-04-22)

The observed `h_out` peak at `max_d = 10⁸` (1.91 × 10⁹) is the **true
unclipped frontier**: no overflow was recorded, so the atomic counter
was not saturating. This is the directly-measurable upper bound on the
Phase B working-set size at that `max_d`, and it sits at 95.5 % of
the B200's 2 × 10⁹ `BUF_SLOTS`.

For `max_d ≥ 10⁹`, the probe's observed `h_out` peak does **not**
measure the true unclipped frontier. Because the input buffer to each
`expand_mark_compact_safe` call is itself clipped to `BUF_SLOTS = 4 ×
10⁸` by the previous level, each thread produces at most 5 children
(BOUND = 5), giving a structural upper bound `h_out ≤ 5 × BUF_SLOTS =
2 × 10⁹` that has nothing to do with the true CF-tree frontier. What
*is* informative at those max_d values is:

- **Overflow happens.** The probe recorded 17.5 × 10¹² overflow events
  at `max_d = 10⁹` (summed over all rounds and depths). These are
  events in which the atomic counter attempted to write past
  `BUF_SLOTS`. This confirms that at our 4 × 10⁸ buffer, the true
  per-level frontier exceeded 4 × 10⁸ repeatedly across the 2048
  rounds.
- **What this says about the B200 run.** The B200 had `BUF_SLOTS =
  2 × 10⁹`, five times our local buffer. We cannot directly observe
  from the 4 × 10⁸ probe whether the true peak at `max_d = 2.1 × 10¹¹`
  exceeds 2 × 10⁹. What we *can* say is:
    - At `max_d = 10⁸` the measured peak was 95.5 % of 2 × 10⁹.
    - Per-chunk peak is non-decreasing in `max_d` (more matrices
      survive the `q ≤ max_d` filter at every level), so at `max_d =
      2.1 × 10¹¹` the true peak is at least 1.91 × 10⁹ and almost
      certainly strictly larger.
    - Whether the excess is small enough (say, ≤ 4.5 %, keeping it
      under B200's 2 × 10⁹) or large (multiplying past 2 × 10⁹) is
      not determined by our local probes. The honest answer is
      "likely over the B200 buffer wall, but we cannot quantify by how
      much without a B200 v6.1 re-run".

### What this does and does not say about `Uncovered = 0`

Clipping **does not** directly invalidate the 210B `Uncovered = 0`
claim. In `expand_mark_compact_safe`, the bitset mark

```c
atomicOr(&bitset[n10 / 32], 1u << (n10 % 32));
atomicAdd(marks, 1);
```

fires *before* the `pos < max_out` check. Every child matrix with
`n10 ≤ max_d` marks its denominator, regardless of whether the
matrix itself survives into the next level's frontier. What a clipped
matrix loses is its *descendants* — the denominators its subtree
would have marked at greater depth. Because the 244M Phase A seeds
produce massively redundant CF coverage (most integers `d ≤ 2.1 ×
10¹¹` have many `A = 5` representations), a clipped subtree's
denominators are typically still marked by other, unclipped CF paths.

**So the accurate characterization is:**

- It is very likely that the v6 B200 run clipped a meaningful
  portion of its per-chunk Phase B frontier at `max_d = 2.1 × 10¹¹`.
- This does **not** mean `Uncovered = 0` is wrong. It is entirely
  consistent that `Uncovered = 0` is in fact correct even with
  significant clipping, because seed coverage is redundant.
- But the v6 kernel does not *prove* this. It emits no
  machine-checkable certificate. The only way to turn the 210B
  headline claim into a certified computational artifact is a v6.1
  re-run on equivalent hardware, whose tail must read
  `All peaks < BUF_SLOTS: YES` and `No-overflow abort fired: NO`.

This is a *software-audit* gap, not a mathematical one. The headline
claim is best described as **strong computational evidence, pending
certification by a v6.1 re-run**.

**Earlier, less informative probes** at `num_rounds = 1` (not the 210B
chunk size): peak frontier > 25 × 10⁹ — more than an order of magnitude
beyond the B200 buffer. This means any earlier row in the experiment
page's "scaling table" that used `num_rounds = 1` (the `d ≤ 10⁹` row
at 21.8 s, and corresponding intermediate rows) was clipping heavily.
Only the 210B headline run, with its 256-round chunking, is in the
candidate-safe regime.

### Local certified runs (CERTIFY mode, hard abort on overflow)

These runs use the same binary but without `ZAREMBA_PROBE`, so any
overflow immediately triggers a fatal abort with exit status 2. A
successful CERTIFY run produces a machine-checkable no-overflow
certificate in its tail.

| max_d | num_rounds | Chunk seeds | Outcome | Phase B peak | Overflow |
|-------|-----------|-------------|---------|--------------|----------|
| 10⁶ | 2048 | 105,631 | **CERTIFIED** (Uncovered=0, 226.4 s) | 262,804,169 (65.7 % of BUF_SLOTS) | 0 |
| 10⁷ | 2048 | 119,051 | hard abort at round 1 | Phase B round 1 h_out = 1.14 × 10⁹ | 7.42 × 10⁸ |
| 10⁸ | 16,384 | 14,902 | hard abort at round 1 | Phase B round 1 h_out = 1.01 × 10⁹ | 6.12 × 10⁸ |
| 10⁹ | 16,384 | 14,902 | hard abort at round 1 | Phase B round 1 h_out = 1.16 × 10⁹ | 7.61 × 10⁸ |

Logs: `idontknow/logs/v6_1_suite/v6_1_CERTIFY_*_v2.log`.

**First certified result (frozen).** The `max_d = 10⁶` run produced:

```
--- NO-OVERFLOW CERTIFICATE ---
BUF_SLOTS                 : 400000000
Phase A peak frontier     : 216330790  (0.5408 of BUF_SLOTS)
Phase A overflow events   : 0
Phase B peak frontier     : 262804169  (0.6570 of BUF_SLOTS)
Phase B overflow events   : 0
All peaks < BUF_SLOTS     : YES
No-overflow abort fired   : NO

RESULT: ALL d in [1, 1000000] are Zaremba denominators (A=5).
        Run is buffer-safe (no frontier ever reached BUF_SLOTS).
```

This is the first machine-checkable computational artifact produced
by the project: a v6.1 run whose tail proves every expansion landed
in-buffer and whose bitset count proves every integer in [1, 10⁶] is
covered. A full log is at
`idontknow/logs/v6_1_suite/v6_1_CERTIFY_d1000000_r2048_v2.log`.

**What this does and does not establish.** It establishes that the
v6.1 kernel is *auditable* (it emits a real certificate) and that the
`max_d = 10⁶` sub-range of the 210B claim is now certified, not
merely computationally-evident. It does **not** certify the full 210B
range — that requires a re-run on hardware where `BUF_SLOTS ≥ 2 × 10⁹`
(nominally the original 8× B200 configuration, since the local RTX
5090 is five times too small). See section 7.

**Failures on larger max_d are themselves informative.** The hard
aborts at `max_d ≥ 10⁷` at 119,051 seeds per chunk prove that even
for chunk sizes smaller than the 210B configuration, Phase B peak
frontier exceeds the local 4 × 10⁸ buffer on the very first round.
At `max_d = 10⁸` with `num_rounds = 16,384` (chunk size reduced to
14,902 seeds — 1/8 of the 210B per-chunk seeds), Phase B peak still
reaches ~10⁹. This strongly suggests that on the B200's 2 × 10⁹
buffer, the 210B configuration (119,210 seeds per chunk,
`max_d = 2.1 × 10¹¹`) was operating close to or past the buffer
wall — consistent with the probe-mode observations.

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
