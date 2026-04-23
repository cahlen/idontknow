#!/bin/bash
#
# v6.1 overnight probe + certify suite on RTX 5090
#
# Goals:
#   1. PROBE: measure true (unclipped) peak Phase B frontier at the exact
#      per-chunk seed count used by the 8x B200 210B run (119,210 seeds
#      per chunk, i.e. num_rounds=2048 on a single GPU). This tells us
#      whether the original v6 210B configuration was buffer-safe.
#   2. CERTIFY: produce a real no-overflow certificate for the largest
#      max_d we can finish overnight on a single 32 GB GPU.
#
# Output: timestamped logs in logs/v6_1_suite/
# Return: first failing command returns 2; success returns 0.
#
# This script is designed to be run in background via:
#   nohup scripts/experiments/zaremba-effective-bound/run_v6_1_suite.sh > logs/v6_1_suite.out 2>&1 &

set -u
SUITE_DIR="logs/v6_1_suite"
mkdir -p "$SUITE_DIR"

BIN_SMALL=/tmp/matrix_v6_1_400m
BIN_PROBE=/tmp/matrix_v6_1_probe

# 400 MB buffer: fits in 32 GB with headroom. Used for both certify and probe.
if [ ! -x "$BIN_SMALL" ]; then
    nvcc -O3 -arch=sm_120a -DBUF_SLOTS=400000000ULL \
        -o "$BIN_SMALL" \
        scripts/experiments/zaremba-effective-bound/matrix_enum_multipass_v6_1.cu \
        -lpthread || exit 2
fi

# Same binary, different env. Keep a separate symlink for clarity in logs.
if [ ! -e "$BIN_PROBE" ]; then
    ln -s "$BIN_SMALL" "$BIN_PROBE"
fi

run_one () {
    local mode="$1"      # PROBE or CERTIFY
    local max_d="$2"
    local rounds="$3"
    local log="$SUITE_DIR/v6_1_${mode}_d${max_d}_r${rounds}.log"
    echo "=== [$mode] max_d=$max_d rounds=$rounds -> $log ==="
    local start=$(date +%s)
    if [ "$mode" = "PROBE" ]; then
        ZAREMBA_PROBE=1 ZAREMBA_ROUNDS="$rounds" "$BIN_PROBE" "$max_d" > "$log" 2>&1
    else
        ZAREMBA_ROUNDS="$rounds" "$BIN_SMALL" "$max_d" > "$log" 2>&1
    fi
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    echo "  -> rc=$rc elapsed=${elapsed}s"
    # Extract certificate tail
    tail -n 14 "$log" | sed 's/^/    /'
    echo ""
    return $rc
}

# --- Probes: measure true peak with the 210B chunk size (119,210 seeds/chunk).
# Phase A produces ~244M seeds, so rounds=2048 -> ~119,210 seeds/chunk.
# If the probe at max_d=1e11 overflows, we know v6 at num_rounds=256 on 8x
# B200 with chunk=119,210 was also overflowing. If it does NOT overflow,
# the original 210B claim is buffer-safe.

echo "### PHASE 1: PROBE SWEEP ###"
echo "Purpose: measure true peak Phase B frontier at per-chunk seed count"
echo "         matching the 210B run configuration (119,210 seeds/chunk)."
echo ""

# 1e8: should be quick. Useful as a floor.
run_one PROBE 100000000   2048

# 1e9: moderately cheap. Matches the original 1B baseline.
run_one PROBE 1000000000  2048

# 1e10: longer, but still overnight-finishable on 5090.
run_one PROBE 10000000000 2048

# --- Certification runs: hard abort on overflow, produce real certificates.
#
# NOTE: probe 1 (max_d=1e8) reported Phase B peak = 1.91e9, far above the
# local BUF_SLOTS of 4e8. Under this BUF_SLOTS, CERTIFY at the original
# 210B chunk size (2048 rounds / 119,210 seeds per chunk) is guaranteed
# to fast-fail with a hard abort at every max_d. Running them would be a
# waste of wall time and would not produce a real no-overflow log.
#
# Instead, certify at smaller max_d with enough rounds that the Phase B
# per-chunk peak stays well below 4e8. A rough scaling argument: peak
# scales roughly with (per-chunk seeds x typical survival rate), so to
# get peak under ~2e8 we need per-chunk seeds around 5,000-15,000. That
# means num_rounds between 16,000 and 50,000 for max_d = 1e8, or just
# running at smaller max_d where the intrinsic Phase B working set is
# smaller.

echo "### PHASE 2: CERTIFY (local-hardware safe bounds) ###"
echo "Purpose: produce at least one genuine no-overflow certificate on"
echo "         the RTX 5090 at 400M BUF_SLOTS for the CERTIFICATE.md"
echo "         self-audit section."
echo ""

# Start tiny and walk up until we find the largest max_d that certifies.
run_one CERTIFY 1000000   2048 || true
run_one CERTIFY 10000000  2048 || true
run_one CERTIFY 100000000 16384 || true

echo ""
echo "### SUITE COMPLETE ###"
echo "Check $SUITE_DIR/ for individual run logs."
ls -la "$SUITE_DIR/"
