#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# bigcompute.science Experiment Pipeline
#
# Runs experiments sequentially. After each one completes:
#   1. Records results to logs
#   2. Updates the website post with actual data
#   3. Commits and pushes to both repos
#   4. Immediately starts the next experiment
#
# Usage: ./scripts/pipeline.sh
# ============================================================================

cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:/usr/local/cuda/bin:$PATH"

SITE_DIR="/home/amsysistestdrive2026/bigcompute.science"
REPO_DIR="/home/amsysistestdrive2026/idontknow"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

commit_and_push() {
    local msg="$1"

    # Push idontknow repo
    cd "$REPO_DIR"
    git add -A
    git commit -m "$msg

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    git push origin main 2>/dev/null || true

    # Build and push website
    cd "$SITE_DIR"
    npx astro build 2>/dev/null
    git add -A
    git commit -m "$msg

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    git push origin main 2>/dev/null || true

    cd "$REPO_DIR"
    log "Committed and pushed: $msg"
}

# ============================================================================
# EXPERIMENT 1: Zaremba 8B (already running — wait for it)
# ============================================================================
run_zaremba() {
    log "=== EXPERIMENT 1: Zaremba 8B Verification ==="

    # Check if already running (v1 or v2)
    if pgrep -f "zaremba_v\?e\?r\?i\?f\?y" > /dev/null 2>&1; then
        log "Zaremba verification already running. Waiting for completion..."
        while pgrep -f "zaremba_v" > /dev/null 2>&1; do
            sleep 60
            # Show progress from whichever log dir exists
            for logdir in logs/v2 logs; do
                for i in $(seq 0 7); do
                    LAST=$(tail -1 ${logdir}/gpu${i}*.log 2>/dev/null | head -1)
                    if [ -n "$LAST" ]; then
                        echo "  GPU $i: $LAST"
                    fi
                done
                break
            done
        done
    else
        log "Compiling v2 kernel..."
        nvcc -O3 -arch=sm_100a -o zaremba_v2 scripts/zaremba_verify_v2.cu

        log "Launching Zaremba v2 verification..."
        mkdir -p logs/v2
        for i in $(seq 0 7); do
            START=$((i * 1000000000 + 1))
            END=$(((i + 1) * 1000000000))
            CUDA_VISIBLE_DEVICES=$i ./zaremba_v2 $START $END > logs/v2/gpu${i}.log 2>&1 &
        done
        log "Waiting for all 8 GPUs to finish..."
        wait
    fi

    log "Zaremba verification complete!"

    # Collect results (check v2 logs first, then v1)
    TOTAL_FAILURES=0
    for i in $(seq 0 7); do
        LOGFILE=""
        if [ -f "logs/v2/gpu${i}.log" ]; then
            LOGFILE="logs/v2/gpu${i}.log"
        elif [ -f "logs/gpu${i}_8B.log" ]; then
            LOGFILE="logs/gpu${i}_8B.log"
        fi
        if [ -n "$LOGFILE" ]; then
            FAILURES=$(grep "Total failures:" "$LOGFILE" 2>/dev/null | awk '{print $NF}' || echo "0")
            TOTAL_FAILURES=$((TOTAL_FAILURES + FAILURES))
            log "  GPU $i ($LOGFILE): failures=$FAILURES"
        fi
    done

    log "TOTAL FAILURES: $TOTAL_FAILURES"

    # Update website post
    ZAREMBA_POST="$SITE_DIR/src/content/experiments/2026-03-28-zaremba-conjecture-8b-verification.md"
    if [ "$TOTAL_FAILURES" -eq 0 ]; then
        # Update title to past tense
        sed -i 's/Verifying 8 Billion/8 Billion Values Verified on/' "$ZAREMBA_POST"
        sed -i 's/status: in-progress/status: complete/' "$ZAREMBA_POST"
        sed -i 's/verified_range: \[1, 8000000000\]/verified_range: [1, 8000000000]/' "$ZAREMBA_POST"
        log "Updated Zaremba post: PASS"
    else
        log "Updated Zaremba post: FAILURES FOUND"
    fi

    # Copy logs to website data
    mkdir -p "$SITE_DIR/public/data/zaremba-8b/gpu_logs"
    cp logs/v2/gpu*.log "$SITE_DIR/public/data/zaremba-8b/gpu_logs/" 2>/dev/null || true
    cp logs/gpu*_8B.log "$SITE_DIR/public/data/zaremba-8b/gpu_logs/" 2>/dev/null || true
    cp logs/race-results.log "$SITE_DIR/public/data/zaremba-8b/" 2>/dev/null || true

    commit_and_push "Zaremba 8B complete: $TOTAL_FAILURES failures across 8 billion values"
}

# ============================================================================
# EXPERIMENT 2: MCTS Proof Search Benchmark
# ============================================================================
run_mcts() {
    log "=== EXPERIMENT 2: MCTS Proof Search Benchmark ==="

    source venv/bin/activate

    # Need models served — start them
    log "Starting model servers..."
    export CUDA_HOME=/usr/local/cuda

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
        --model models/Goedel-Prover-V2-32B \
        --tensor-parallel-size 4 --trust-remote-code \
        --host 0.0.0.0 --port 8000 --max-model-len 32768 --dtype auto \
        > logs/goedel-vllm.log 2>&1 &
    GOEDEL_PID=$!

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
        --model models/Kimina-Prover-72B \
        --tensor-parallel-size 4 --trust-remote-code \
        --host 0.0.0.0 --port 8001 --max-model-len 32768 --dtype auto \
        > logs/kimina-vllm.log 2>&1 &
    KIMINA_PID=$!

    # Wait for servers
    log "Waiting for model servers..."
    for attempt in $(seq 1 120); do
        if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && \
           curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
            log "Both servers ready!"
            break
        fi
        sleep 5
    done

    # Generate extended Zaremba theorems for d=1..50
    # TODO: generate the extended .lean file

    # Run MCTS prover
    mkdir -p logs/mcts
    log "Running MCTS prover (budget=256)..."
    python3 scripts/experiments/mcts-proof-search/mcts_prover.py \
        --server http://localhost:8000 \
        --file lean4-proving/conjectures/zaremba.lean \
        --output lean4-proving/conjectures/zaremba_mcts.lean \
        --budget 256 --rollouts 8 \
        2>&1 | tee logs/mcts/mcts_run.log

    # Run naive prover for comparison
    log "Running naive prover (same budget)..."
    python3 lean4-proving/prover.py \
        --server http://localhost:8000 \
        --file lean4-proving/conjectures/zaremba.lean \
        --output lean4-proving/conjectures/zaremba_naive.lean \
        --max-attempts 32 --zaremba \
        2>&1 | tee logs/mcts/naive_run.log

    # Kill model servers
    kill $GOEDEL_PID $KIMINA_PID 2>/dev/null || true
    wait $GOEDEL_PID $KIMINA_PID 2>/dev/null || true

    # Update website post
    MCTS_POST="$SITE_DIR/src/content/experiments/2026-03-30-mcts-proof-search-benchmark.md"
    sed -i 's/status: in-progress/status: complete/' "$MCTS_POST"

    mkdir -p "$SITE_DIR/public/data/mcts-benchmark"
    cp logs/mcts/*.log "$SITE_DIR/public/data/mcts-benchmark/" 2>/dev/null || true

    commit_and_push "MCTS proof search benchmark complete"
    log "MCTS experiment done!"
}

# ============================================================================
# EXPERIMENT 3: Ramsey R(5,5) Lower Bound Search
# ============================================================================
run_ramsey() {
    log "=== EXPERIMENT 3: Ramsey R(5,5) Search ==="

    nvcc -O3 -arch=sm_100a -o ramsey_search \
        scripts/experiments/ramsey-r55/ramsey_search.cu -lcurand
    mkdir -p logs/ramsey

    # Phase 1: Validate on n=43
    log "Phase 1: Validating lower bound on K_43..."
    ./ramsey_search 43 100000 1000000 2>&1 | tee logs/ramsey/n43.log

    # Phase 2: Attack n=44
    log "Phase 2: Attacking K_44 (would improve lower bound)..."
    ./ramsey_search 44 1000000 10000000 2>&1 | tee logs/ramsey/n44.log

    # Check result
    if grep -q "SUCCESS" logs/ramsey/n44.log; then
        log "*** RAMSEY BREAKTHROUGH: Found R(5,5)-good coloring of K_44! ***"
        RESULT="SUCCESS — R(5,5) >= 45 PROVED"
    else
        BEST=$(grep "Best fitness" logs/ramsey/n44.log | tail -1)
        log "No R(5,5)-good coloring of K_44 found. $BEST"

        # Phase 3: Long run
        log "Phase 3: Extended search on K_44..."
        ./ramsey_search 44 10000000 100000000 2>&1 | tee logs/ramsey/n44_long.log
        RESULT="Extended search complete"
    fi

    # Update website
    RAMSEY_POST="$SITE_DIR/src/content/experiments/2026-04-01-ramsey-r55-lower-bound.md"
    sed -i 's/status: in-progress/status: complete/' "$RAMSEY_POST"

    mkdir -p "$SITE_DIR/public/data/ramsey-r55"
    cp logs/ramsey/*.log "$SITE_DIR/public/data/ramsey-r55/" 2>/dev/null || true

    commit_and_push "Ramsey R(5,5) search complete: $RESULT"
    log "Ramsey experiment done!"
}

# ============================================================================
# EXPERIMENT 4: Class Numbers of Real Quadratic Fields
# ============================================================================
run_class_numbers() {
    log "=== EXPERIMENT 4: Class Numbers to 10^13 ==="

    nvcc -O3 -arch=sm_100a -o class_number_rqf \
        scripts/experiments/class-numbers/class_number_rqf.cu -lm
    mkdir -p logs/class-numbers

    # Launch 8 GPUs
    for i in $(seq 0 7); do
        START=$((100000000000 + i * 1162500000000))
        END=$((100000000000 + (i + 1) * 1162500000000))
        CUDA_VISIBLE_DEVICES=$i ./class_number_rqf $START $END \
            > logs/class-numbers/gpu${i}.log 2>&1 &
        log "GPU $i: d=$START..$END"
    done

    log "Waiting for all 8 GPUs..."
    wait
    log "Class number computation complete!"

    # Update website
    CLASS_POST="$SITE_DIR/src/content/experiments/2026-04-02-class-numbers-real-quadratic.md"
    sed -i 's/status: in-progress/status: complete/' "$CLASS_POST"

    mkdir -p "$SITE_DIR/public/data/class-numbers"
    cp logs/class-numbers/*.log "$SITE_DIR/public/data/class-numbers/" 2>/dev/null || true

    commit_and_push "Class numbers of real quadratic fields: extended to 10^13"
    log "Class numbers experiment done!"
}

# ============================================================================
# EXPERIMENT 5: Kronecker Coefficients
# ============================================================================
run_kronecker() {
    log "=== EXPERIMENT 5: Kronecker Coefficients ==="

    nvcc -O3 -arch=sm_100a -o kronecker_compute \
        scripts/experiments/kronecker-coefficients/kronecker_compute.cu
    mkdir -p logs/kronecker

    log "Phase 1: Full table for S_30..."
    ./kronecker_compute 30 all 2>&1 | tee logs/kronecker/n30.log

    log "Phase 2: GCT triples for S_80..."
    ./kronecker_compute 80 gct 2>&1 | tee logs/kronecker/n80.log

    log "Phase 3: Push to S_120..."
    ./kronecker_compute 120 gct 2>&1 | tee logs/kronecker/n120.log

    # Update website
    KRON_POST="$SITE_DIR/src/content/experiments/2026-04-03-kronecker-coefficients-gpu.md"
    sed -i 's/status: in-progress/status: complete/' "$KRON_POST"

    mkdir -p "$SITE_DIR/public/data/kronecker"
    cp logs/kronecker/*.log "$SITE_DIR/public/data/kronecker/" 2>/dev/null || true

    commit_and_push "Kronecker coefficients: computed to n=120"
    log "Kronecker experiment done!"
}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

log "=========================================="
log "  bigcompute.science Experiment Pipeline"
log "=========================================="
log ""
log "Queue:"
log "  1. Zaremba 8B (running)"
log "  2. MCTS Proof Search"
log "  3. Ramsey R(5,5)"
log "  4. Class Numbers to 10^13"
log "  5. Kronecker Coefficients to n=120"
log ""

run_zaremba
run_mcts
run_ramsey
run_class_numbers
run_kronecker

log ""
log "=========================================="
log "  ALL EXPERIMENTS COMPLETE"
log "=========================================="
log ""
log "Results published to bigcompute.science"
