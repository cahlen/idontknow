#!/bin/bash
# Test the kronecker_gpu kernel with known results.
# Requires: char_table_n5.bin, z_inv_n5.bin, partitions_n5.txt in results/
#
# Known: S_5 has 7 partitions, Kronecker table is fully known.

set -euo pipefail

BINARY="${1:-./kronecker_gpu}"
PASS=0
FAIL=0

echo "=== Kronecker GPU Kernel Tests ==="
echo ""

# Test 1: S_5 correctness
echo -n "  TEST: S_5 (7 partitions) ... "
if [ ! -f scripts/experiments/kronecker-coefficients/results/char_table_n5.bin ]; then
    echo "SKIP (char_table_n5.bin not found — run char_table.py 5 first)"
else
    output=$($BINARY 5 2>&1)

    # S_5 should have 7 partitions
    partitions=$(echo "$output" | grep "partitions" | head -1 | grep -o '[0-9]* partitions' | grep -o '[0-9]*')

    # Should have nonzero count
    nonzero=$(echo "$output" | grep "nonzero=" | tail -1 | grep -o 'nonzero=[0-9]*' | grep -o '[0-9]*')

    if [ "$partitions" = "7" ] && [ -n "$nonzero" ] && [ "$nonzero" -gt "0" ]; then
        echo "PASS (P=$partitions, nonzero=$nonzero)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (P=$partitions, nonzero=$nonzero)"
        FAIL=$((FAIL + 1))
    fi
fi

# Test 2: Progress output present
echo -n "  TEST: Progress output ... "
if [ ! -f scripts/experiments/kronecker-coefficients/results/char_table_n5.bin ]; then
    echo "SKIP"
else
    output=$($BINARY 5 2>&1)
    if echo "$output" | grep -q "j="; then
        echo "PASS (slab progress present)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no slab progress)"
        FAIL=$((FAIL + 1))
    fi
fi

# Test 3: RESULTS section present
echo -n "  TEST: RESULTS section ... "
if [ ! -f scripts/experiments/kronecker-coefficients/results/char_table_n5.bin ]; then
    echo "SKIP"
else
    output=$($BINARY 5 2>&1)
    if echo "$output" | grep -q "RESULTS"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no RESULTS section)"
        FAIL=$((FAIL + 1))
    fi
fi

# Test 4: Checkpoint cleaned up
echo -n "  TEST: Checkpoint cleanup ... "
if [ ! -f scripts/experiments/kronecker-coefficients/results/char_table_n5.bin ]; then
    echo "SKIP"
else
    $BINARY 5 > /dev/null 2>&1
    if [ -f scripts/experiments/kronecker-coefficients/results/checkpoint_n5.txt ]; then
        echo "FAIL (checkpoint not cleaned up)"
        FAIL=$((FAIL + 1))
    else
        echo "PASS"
        PASS=$((PASS + 1))
    fi
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -gt 0 ]; then exit 1; fi
