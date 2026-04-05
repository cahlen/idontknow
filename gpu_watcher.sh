#!/bin/bash
# GPU Watcher: monitors for completed experiments, launches next in queue.
# Run with: nohup bash gpu_watcher.sh > logs/gpu_watcher.log 2>&1 &
set -e
cd /home/amsysistestdrive2026/idontknow
R="scripts/experiments/zaremba-density/results"
B="./zaremba_density_gpu"
LOG="logs/gpu_watcher.log"
mkdir -p logs

# ── Experiment queue: each line is "max_d digits description" ──
# These run AFTER the current overnight batch finishes.
# Ordered by scientific priority.
QUEUE=(
    # Extend pair hierarchies to 10^11
    "100000000000 1,2,7 A127_1e11"
    "100000000000 1,2,9 A129_1e11"
    "100000000000 1,2,10 A1210_1e11"
    # Larger digit sets at 10^11
    "100000000000 1,2,3,4 A1234_1e11"
    "100000000000 1,2,3,4,5,6 A123456_1e11"
    # No-digit-1 sets at 10^11
    "100000000000 2,3,4,5 A2345_1e11"
    "100000000000 2,3 A23_1e11"
    # Extend {1,2} logarithmic convergence to 10^13
    "10000000000000 1,2 A12_1e13"
    # Pairs without digit 1 at 10^11
    "100000000000 2,3,4,5,6,7,8,9,10 A2345678910_1e11"
    # {1,3} pair at 10^11
    "100000000000 1,3 A13_1e11"
    "100000000000 1,4 A14_1e11"
    "100000000000 1,5 A15_1e11"
    # Full Zaremba digits at 10^12
    "1000000000000 1,2,3,4,5 A12345_1e12"
)

QUEUE_IDX=0

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Track which GPUs have active experiments (by PID file)
declare -A GPU_PID

# Initialize: find currently running experiments
for gpu in 0 1 2 3 4 5 6 7; do
    pid=$(ps aux | grep "zaremba_density_gpu" | grep "CUDA_VISIBLE_DEVICES=$gpu " | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        GPU_PID[$gpu]=$pid
        log "GPU $gpu: already running PID $pid"
    fi
done

# Also check by nvidia-smi memory usage
for gpu in 0 1 2 3 4 5 6 7; do
    if [ -z "${GPU_PID[$gpu]}" ]; then
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        if [ "$mem" -gt 100 ] 2>/dev/null; then
            log "GPU $gpu: has ${mem} MiB used but no tracked PID — assuming busy"
            GPU_PID[$gpu]="unknown"
        fi
    fi
done

log "Starting GPU watcher. ${#QUEUE[@]} experiments in queue."
log "Current GPU state:"
for gpu in 0 1 2 3 4 5 6 7; do
    if [ -n "${GPU_PID[$gpu]}" ]; then
        log "  GPU $gpu: BUSY (PID ${GPU_PID[$gpu]})"
    else
        log "  GPU $gpu: FREE"
    fi
done

while true; do
    sleep 60

    # Check each GPU
    for gpu in 0 1 2 3 4 5 6 7; do
        pid="${GPU_PID[$gpu]}"

        if [ -n "$pid" ] && [ "$pid" != "unknown" ]; then
            # Check if PID is still running
            if ! kill -0 "$pid" 2>/dev/null; then
                log "GPU $gpu: PID $pid finished"
                unset GPU_PID[$gpu]
            fi
        elif [ "$pid" = "unknown" ]; then
            # Check memory usage
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
            if [ "$mem" -lt 100 ] 2>/dev/null; then
                log "GPU $gpu: freed up (memory ${mem} MiB)"
                unset GPU_PID[$gpu]
            fi
        fi

        # If GPU is free and we have work, launch next experiment
        if [ -z "${GPU_PID[$gpu]}" ] && [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; then
            exp="${QUEUE[$QUEUE_IDX]}"
            read -r max_d digits name <<< "$exp"
            logfile="$R/gpu_${name}.log"

            # Skip if this experiment already has results
            if grep -q "^Covered:" "$logfile" 2>/dev/null; then
                log "GPU $gpu: SKIPPING $name (already complete)"
                QUEUE_IDX=$((QUEUE_IDX + 1))
                continue
            fi

            log "GPU $gpu: LAUNCHING $name ($B $max_d $digits)"
            CUDA_VISIBLE_DEVICES=$gpu nohup stdbuf -oL $B $max_d $digits > "$logfile" 2>&1 &
            new_pid=$!
            GPU_PID[$gpu]=$new_pid
            log "GPU $gpu: started PID $new_pid -> $logfile"
            QUEUE_IDX=$((QUEUE_IDX + 1))
        fi
    done

    # Check if everything is done
    all_done=true
    for gpu in 0 1 2 3 4 5 6 7; do
        if [ -n "${GPU_PID[$gpu]}" ]; then
            all_done=false
            break
        fi
    done
    if $all_done && [ $QUEUE_IDX -ge ${#QUEUE[@]} ]; then
        log "All experiments complete. Queue exhausted. Exiting."
        break
    fi

    # Periodic status (every 10 min)
    min=$(( $(date +%s) / 600 ))
    if [ $(( min % 1 )) -eq 0 ]; then
        busy=0
        free=0
        for gpu in 0 1 2 3 4 5 6 7; do
            if [ -n "${GPU_PID[$gpu]}" ]; then
                busy=$((busy + 1))
            else
                free=$((free + 1))
            fi
        done
        remaining=$((${#QUEUE[@]} - QUEUE_IDX))
        log "Status: ${busy}/8 GPUs busy, ${free} free, ${remaining} queued"
    fi
done
