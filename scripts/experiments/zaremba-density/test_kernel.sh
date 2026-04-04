#!/bin/bash
# Test the zaremba_density_gpu kernel with known results.
# Run after compiling: nvcc -O3 -arch=sm_100a -o zaremba_density_gpu zaremba_density_gpu.cu -lm
#
# Known results (from verified runs on B200):
#   A={1,2,3} at 10^4:  uncovered=27, density=99.73%
#   A={1,2,3,4,5} at 10^5: uncovered=0, density=100%
#   A={1,2} at 10^4: uncovered=4855, density=51.45%

set -euo pipefail

BINARY="${1:-./zaremba_density_gpu}"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    local args="$2"
    local expected_uncovered="$3"
    local expected_density="$4"

    echo -n "  TEST: $name ... "

    output=$($BINARY $args 2>&1)

    uncovered=$(echo "$output" | grep "Uncovered:" | head -1 | awk '{print $2}')
    density=$(echo "$output" | grep "Density:" | head -1 | awk '{print $2}' | tr -d '%')

    # Check batch progress output exists
    has_batches=$(echo "$output" | grep -c "batch" || true)

    if [ "$uncovered" = "$expected_uncovered" ]; then
        echo "PASS (uncovered=$uncovered, density=$density, batches=$has_batches)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (expected uncovered=$expected_uncovered, got uncovered=$uncovered)"
        FAIL=$((FAIL + 1))
    fi
}

check_checkpoint() {
    local name="$1"
    local args="$2"
    local ckpt_pattern="$3"

    echo -n "  TEST: $name checkpoint ... "

    $BINARY $args > /dev/null 2>&1

    # Checkpoint should be cleaned up after successful completion
    if ls $ckpt_pattern 1>/dev/null 2>&1; then
        echo "FAIL (checkpoint not cleaned up)"
        FAIL=$((FAIL + 1))
    else
        echo "PASS (checkpoint cleaned up after completion)"
        PASS=$((PASS + 1))
    fi
}

echo "=== Zaremba Density Kernel Tests ==="
echo ""

# Correctness tests
echo "Correctness:"
run_test "A={1,2,3} at 10^4 (expect 27 uncovered)" "10000 1,2,3" "27" "99.73"
run_test "A={1,2,3,4,5} at 10^5 (expect 0 uncovered)" "100000 1,2,3,4,5" "0" "100.00"
run_test "A={1,2} at 10^4 (expect 4855 uncovered)" "10000 1,2" "4855" "51.45"
run_test "A={1,2,3,4} at 10^4 (expect 2 uncovered)" "10000 1,2,3,4" "2" "99.98"

echo ""

# Checkpoint tests
echo "Checkpoint cleanup:"
check_checkpoint "A={1,2,3} at 10^3" "1000 1,2,3" "scripts/experiments/zaremba-density/results/checkpoint_A1_2_3_1000.bin"

echo ""

# Progress output test
echo -n "  TEST: Progress output ... "
output=$(./zaremba_density_gpu 10000 1,2,3 2>&1)
if echo "$output" | grep -q "batch"; then
    echo "PASS (batch progress lines present)"
    PASS=$((PASS + 1))
else
    echo "FAIL (no batch progress output)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -gt 0 ]; then
    exit 1
fi
