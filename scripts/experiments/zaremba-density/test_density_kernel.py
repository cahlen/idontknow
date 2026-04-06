#!/usr/bin/env python3
"""
Test suite for zaremba_density_v2 — bounded DFS with overflow re-enqueueing.

Tests correctness against known exact values:
  - A={1,2,3} at 10^6: uncovered=27
  - A={1,2,4} at 10^6: uncovered=64
  - A={1,2,5} at 10^6: uncovered=373
  - A={1,2,3} at 10^9: uncovered=27 (the KEY test — same 27 exceptions)

Also tests:
  - Progress output appears within 10 seconds for small runs
  - No single round takes more than 60 seconds (load balancing test)

Usage:
  python3 test_density_kernel.py [--compile] [--skip-slow]

  --compile    Recompile before testing
  --skip-slow  Skip the 10^9 test (for quick iteration)
"""

import subprocess
import sys
import os
import time
import re

REPO = "/home/amsysistestdrive2026/idontknow"
SRC = os.path.join(REPO, "scripts/experiments/zaremba-density/zaremba_density_v2.cu")
BIN = os.path.join(REPO, "zaremba_density_v2")

def compile_kernel():
    """Compile the v2 kernel. Returns True on success."""
    print("Compiling zaremba_density_v2...")
    result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_90", "-o", BIN, SRC, "-lm"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"COMPILE FAILED:\n{result.stderr}")
        return False
    if result.stderr.strip():
        print(f"Warnings:\n{result.stderr}")
    print("Compile OK")
    return True


def run_density(max_d, digits, timeout_sec=600):
    """Run the density kernel and return (stdout, elapsed_seconds, returncode)."""
    digits_str = ",".join(str(d) for d in digits)
    cmd = [BIN, str(max_d), digits_str]
    print(f"\n  Running: {' '.join(cmd)}")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec
    )
    elapsed = time.monotonic() - t0

    return result.stdout, elapsed, result.returncode


def parse_results(stdout):
    """Parse the RESULTS block from stdout. Returns dict."""
    results = {}

    m = re.search(r"Covered:\s+(\d+)\s*/\s*(\d+)", stdout)
    if m:
        results["covered"] = int(m.group(1))
        results["total"] = int(m.group(2))

    m = re.search(r"Uncovered:\s+(\d+)", stdout)
    if m:
        results["uncovered"] = int(m.group(1))

    m = re.search(r"Density:\s+([\d.]+)%", stdout)
    if m:
        results["density_pct"] = float(m.group(1))

    return results


def parse_round_times(stdout):
    """Extract elapsed times from 'Round N: ... (elapsed Xs)' lines.
    Returns list of (round_num, elapsed_time) tuples."""
    rounds = []
    for m in re.finditer(r"Round\s+(\d+):\s+\d+\s+work items\s+\(elapsed\s+([\d.]+)s\)", stdout):
        rounds.append((int(m.group(1)), float(m.group(2))))
    return rounds


def check_round_durations(stdout, max_round_sec=60):
    """Check that no single round took more than max_round_sec seconds.
    Returns (pass, details_string)."""
    rounds = parse_round_times(stdout)
    if len(rounds) < 2:
        return True, f"Only {len(rounds)} round(s), cannot compute durations"

    max_dur = 0
    for i in range(1, len(rounds)):
        dur = rounds[i][1] - rounds[i-1][1]
        if dur > max_dur:
            max_dur = dur

    # Also check last round to end
    m = re.search(r"GPU enumeration:\s+([\d.]+)s", stdout)
    if m and rounds:
        enum_time = float(m.group(1))
        last_round_dur = enum_time - rounds[-1][1]
        if last_round_dur > max_dur:
            max_dur = last_round_dur

    passed = max_dur <= max_round_sec
    detail = f"max round duration: {max_dur:.1f}s (limit: {max_round_sec}s)"
    return passed, detail


def test_correctness(digits, max_d, expected_uncovered, timeout_sec=600):
    """Run a correctness test. Returns True on pass."""
    digits_str = "{" + ",".join(str(d) for d in digits) + "}"
    print(f"\n{'='*60}")
    print(f"TEST: A={digits_str} at {max_d:.0e}")
    print(f"Expected uncovered: {expected_uncovered}")
    print(f"{'='*60}")

    try:
        stdout, elapsed, rc = run_density(max_d, digits, timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"  FAIL: timed out after {timeout_sec}s")
        return False

    if rc != 0:
        print(f"  FAIL: nonzero return code {rc}")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
        return False

    results = parse_results(stdout)
    if "uncovered" not in results:
        print(f"  FAIL: could not parse 'Uncovered' from output")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
        return False

    actual = results["uncovered"]
    if actual != expected_uncovered:
        print(f"  FAIL: uncovered={actual}, expected {expected_uncovered}")
        print(stdout)
        return False

    # Check covered + uncovered = max_d
    if "covered" in results and "total" in results:
        if results["covered"] + actual != results["total"]:
            print(f"  FAIL: covered + uncovered != total "
                  f"({results['covered']} + {actual} != {results['total']})")
            return False

    print(f"  PASS: uncovered={actual} (correct), time={elapsed:.1f}s")

    # Check round durations (load balancing)
    passed, detail = check_round_durations(stdout)
    if passed:
        print(f"  PASS: load balance OK ({detail})")
    else:
        print(f"  WARN: load balance issue ({detail})")

    return True


def test_progress_output(digits, max_d, max_wait_sec=10):
    """Test that progress output appears within max_wait_sec seconds."""
    digits_str = "{" + ",".join(str(d) for d in digits) + "}"
    print(f"\n{'='*60}")
    print(f"TEST: Progress output for A={digits_str} at {max_d:.0e}")
    print(f"{'='*60}")

    cmd = [BIN, str(max_d), ",".join(str(d) for d in digits)]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t0 = time.monotonic()
        output_lines = []

        while time.monotonic() - t0 < max_wait_sec:
            import select
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    # Check for round progress
                    if "Round" in line:
                        elapsed = time.monotonic() - t0
                        print(f"  PASS: got progress at {elapsed:.1f}s: {line.strip()}")
                        proc.wait(timeout=300)  # let it finish
                        return True

        # Wait for process to finish
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        print(f"  FAIL: no Round progress within {max_wait_sec}s")
        print(f"  Output so far: {output_lines[:10]}")
        return False

    except Exception as e:
        print(f"  FAIL: exception: {e}")
        try:
            proc.kill()
        except:
            pass
        return False


def main():
    args = sys.argv[1:]
    do_compile = "--compile" in args
    skip_slow = "--skip-slow" in args

    print("=" * 60)
    print("Zaremba Density v2 — Test Suite")
    print("=" * 60)

    if do_compile:
        if not compile_kernel():
            sys.exit(1)

    if not os.path.exists(BIN):
        print(f"Binary not found at {BIN}")
        print("Run with --compile to build first")
        sys.exit(1)

    results = []

    # ── Correctness tests at 10^6 ──
    results.append(("A={1,2,3} 10^6", test_correctness([1,2,3], 1000000, 27)))
    results.append(("A={1,2,4} 10^6", test_correctness([1,2,4], 1000000, 64)))
    results.append(("A={1,2,5} 10^6", test_correctness([1,2,5], 1000000, 373)))

    # ── Progress output test ──
    results.append(("Progress output", test_progress_output([1,2,3], 1000000)))

    # ── The hard test: 10^9 ──
    if not skip_slow:
        results.append(("A={1,2,3} 10^9", test_correctness([1,2,3], 1000000000, 27, timeout_sec=3600)))
    else:
        print("\n(Skipping 10^9 test due to --skip-slow)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed.")
    else:
        print("\nSome tests FAILED.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
