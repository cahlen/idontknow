#!/bin/bash
# Portfolio SAT solver for Ramsey R(5,5) K43
# Runs multiple solver configurations in parallel on idle CPUs
# Kills all others when one finishes (SAT or UNSAT)
#
# Usage: ./run_sat_portfolio.sh [cnf_file] [num_jobs]

set -e

CNF="${1:-/tmp/ramsey_k43_v2.cnf}"
NJOBS="${2:-32}"
LOGDIR="logs/ramsey-k43-sat"
mkdir -p "$LOGDIR"

echo "========================================"
echo "Ramsey R(5,5) K43 SAT Portfolio"
echo "CNF: $CNF"
echo "Jobs: $NJOBS"
echo "Log dir: $LOGDIR"
echo "Started: $(date -Iseconds)"
echo "========================================"

# Verify CNF exists
if [ ! -f "$CNF" ]; then
    echo "ERROR: CNF file not found: $CNF"
    exit 1
fi

head -4 "$CNF"
echo ""

# Array of PIDs
PIDS=()
CONFIGS=()

launch() {
    local solver="$1"
    local args="$2"
    local tag="$3"
    local logfile="$LOGDIR/${tag}.log"

    echo "Launching: $tag"
    echo "  cmd: $solver $args $CNF"

    $solver $args "$CNF" > "$logfile" 2>&1 &
    PIDS+=($!)
    CONFIGS+=("$tag")
}

# Kissat configurations with different random seeds and strategies
for seed in $(seq 1 $((NJOBS / 2))); do
    launch kissat "--seed=$seed" "kissat-seed${seed}"
done

# CaDiCaL configurations with different random seeds
for seed in $(seq 1 $((NJOBS / 2))); do
    launch cadical "--seed $seed" "cadical-seed${seed}"
done

echo ""
echo "Launched ${#PIDS[@]} solver instances"
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitoring... (Ctrl+C to stop all)"

# Monitor: wait for any to finish
while true; do
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        config=${CONFIGS[$i]}

        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished
            wait "$pid"
            exit_code=$?

            logfile="$LOGDIR/${config}.log"
            echo ""
            echo "========================================"
            echo "SOLVER FINISHED: $config (PID $pid)"
            echo "Exit code: $exit_code"
            echo "Time: $(date -Iseconds)"

            if [ $exit_code -eq 10 ]; then
                echo "RESULT: *** SAT *** — R(5,5) > 43 !!!"
                echo "THIS WOULD BE A MAJOR MATHEMATICAL RESULT"
                echo "Solution in: $logfile"
            elif [ $exit_code -eq 20 ]; then
                echo "RESULT: UNSAT — No valid 2-coloring of K43 exists"
                echo "This proves R(5,5) ≤ 43 (given R(5,5) ≥ 43, this means R(5,5) = 43)"
                echo "THIS WOULD RESOLVE THE OPEN PROBLEM"
            else
                echo "RESULT: UNKNOWN (timeout/error)"
                echo "Last 5 lines:"
                tail -5 "$logfile"
            fi

            echo "========================================"

            # Kill all other solvers
            echo "Killing remaining solvers..."
            for j in "${!PIDS[@]}"; do
                if [ "$j" != "$i" ]; then
                    kill "${PIDS[$j]}" 2>/dev/null || true
                fi
            done

            # Save summary
            echo "Summary saved to $LOGDIR/result.txt"
            {
                echo "Ramsey R(5,5) K43 SAT Result"
                echo "Date: $(date -Iseconds)"
                echo "Solver: $config"
                echo "Exit code: $exit_code"
                if [ $exit_code -eq 10 ]; then echo "RESULT: SAT"
                elif [ $exit_code -eq 20 ]; then echo "RESULT: UNSAT"
                else echo "RESULT: UNKNOWN"; fi
                echo "CNF: $CNF"
                echo "Log: $logfile"
            } > "$LOGDIR/result.txt"

            exit $exit_code
        fi
    done
    sleep 10
done
