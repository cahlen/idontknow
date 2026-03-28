#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run the Zaremba proving pipeline — dual-model race
#
# Serves Goedel-Prover-V2-32B on GPUs 0-3 and Kimina-Prover-72B on GPUs 4-7,
# then races them against 20 Zaremba's Conjecture sorry-targets.
#
# Usage: ./scripts/run-zaremba.sh [--attempts N] [--single goedel|kimina]
# ============================================================================

ATTEMPTS=5
MODE="race"  # race, goedel, kimina
GOEDEL_MODEL="models/Goedel-Prover-V2-32B"
KIMINA_MODEL="models/Kimina-Prover-72B"
LEAN_FILE="lean4-proving/conjectures/zaremba.lean"
OUTPUT_FILE="lean4-proving/conjectures/zaremba_proved.lean"
GOEDEL_PORT=8000
KIMINA_PORT=8001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --attempts) ATTEMPTS="$2"; shift 2 ;;
        --single) MODE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PATH="$HOME/.elan/bin:$PATH"
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=============================================="
echo "  Zaremba's Conjecture — LLM Proving Race"
echo "=============================================="
echo ""
echo "Mode: $MODE"
echo "Max attempts per sorry: $ATTEMPTS"
echo ""

cleanup() {
    echo ""
    echo "Shutting down servers..."
    [ -f /tmp/goedel-vllm.pid ] && kill "$(cat /tmp/goedel-vllm.pid)" 2>/dev/null && rm /tmp/goedel-vllm.pid
    [ -f /tmp/kimina-vllm.pid ] && kill "$(cat /tmp/kimina-vllm.pid)" 2>/dev/null && rm /tmp/kimina-vllm.pid
    echo "Done."
}
trap cleanup EXIT

wait_for_server() {
    local url="$1"
    local name="$2"
    local max_wait=300  # 5 minutes
    local elapsed=0
    while ! curl -sf "$url/v1/models" > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: $name server did not start within ${max_wait}s"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  Waiting for $name... (${elapsed}s)"
    done
    echo "  $name ready!"
}

# ── Launch servers ──
SERVERS=""

if [ "$MODE" = "race" ] || [ "$MODE" = "goedel" ]; then
    echo "[1/3] Starting Goedel-Prover-V2-32B on GPUs 0-3 (port $GOEDEL_PORT)..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
        --model "$GOEDEL_MODEL" \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$GOEDEL_PORT" \
        --max-model-len 32768 \
        --dtype auto > logs/goedel-vllm.log 2>&1 &
    echo $! > /tmp/goedel-vllm.pid
    SERVERS="--server http://localhost:$GOEDEL_PORT"
fi

if [ "$MODE" = "race" ] || [ "$MODE" = "kimina" ]; then
    echo "[1/3] Starting Kimina-Prover-72B on GPUs 4-7 (port $KIMINA_PORT)..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
        --model "$KIMINA_MODEL" \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$KIMINA_PORT" \
        --max-model-len 32768 \
        --dtype auto > logs/kimina-vllm.log 2>&1 &
    echo $! > /tmp/kimina-vllm.pid
    SERVERS="$SERVERS --server http://localhost:$KIMINA_PORT"
fi

echo ""
echo "Waiting for servers to load models..."

if [ "$MODE" = "race" ] || [ "$MODE" = "goedel" ]; then
    wait_for_server "http://localhost:$GOEDEL_PORT" "Goedel-32B"
fi
if [ "$MODE" = "race" ] || [ "$MODE" = "kimina" ]; then
    wait_for_server "http://localhost:$KIMINA_PORT" "Kimina-72B"
fi

echo ""

# ── Verify prerequisites ──
echo "[2/3] Checking prerequisites..."
lean --version
echo ""

# ── Run the prover ──
echo "[3/3] Running prover..."
echo ""
python3 lean4-proving/prover.py \
    $SERVERS \
    --file "$LEAN_FILE" \
    --output "$OUTPUT_FILE" \
    --max-attempts "$ATTEMPTS" \
    --zaremba

echo ""
echo "=============================================="

# ── Verify output ──
echo ""
echo "Verifying output with Lean compiler..."
SORRY_COUNT=$(lean "$OUTPUT_FILE" 2>&1 | grep -c "sorry" || true)
ERROR_COUNT=$(lean "$OUTPUT_FILE" 2>&1 | grep -c "error" || true)

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "WARNING: Output has $ERROR_COUNT Lean error(s)"
    lean "$OUTPUT_FILE" 2>&1 | grep "error"
elif [ "$SORRY_COUNT" -gt 0 ]; then
    echo "Output compiles with $SORRY_COUNT sorry warning(s) remaining"
else
    echo "Output compiles clean — all sorries resolved!"
fi

echo ""
echo "Output saved to: $OUTPUT_FILE"
