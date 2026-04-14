#!/bin/bash
# GPU Watcher v2: monitors for completed experiments, launches next in queue.
# Run with: nohup bash gpu_watcher.sh > logs/gpu_watcher.log 2>&1 &
set -e
cd /home/amsysistestdrive2026/idontknow
R="scripts/experiments/zaremba-density/results"
B="./zaremba_density_gpu"
mkdir -p logs

# ── Experiment queue ──
# Each line: "max_d digits logname"
# Ordered by expected speed (fast first to cycle GPUs)
QUEUE=(
    # Pairs without digit 1 at 1e11
    "100000000000 1,6 A16_1e11"
    "100000000000 1,7 A17_1e11"
    "100000000000 1,8 A18_1e11"
    "100000000000 1,9 A19_1e11"
    "100000000000 1,10 A110_1e11"
    "100000000000 2,3 A23_1e11_v2"
    "100000000000 2,4 A24_1e11"
    "100000000000 2,5 A25_1e11"
    "100000000000 3,4 A34_1e11"
    "100000000000 3,5 A35_1e11"
    # Larger digit sets at 1e11
    "100000000000 1,2,3,4,5,6,7 A1234567_1e11"
    # Push to 1e12 for confirmed closed sets
    "1000000000000 1,2,4 A124_1e12"
    "1000000000000 1,2,5 A125_1e12"
    "1000000000000 1,2,6 A126_1e12"
    "1000000000000 1,2,7 A127_1e12"
    # Extend {1,2} convergence
    "100000000000000 1,2 A12_1e14"
)

QUEUE_IDX=0

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1"; }

declare -A GPU_PID

# Initialize from running processes
for gpu in 0 1 2 3 4 5 6 7; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
    if [ "$mem" -gt 100 ] 2>/dev/null; then
        GPU_PID[$gpu]="unknown"
        log "GPU $gpu: busy (${mem} MiB)"
    else
        log "GPU $gpu: free"
    fi
done

log "Watcher v2 started. ${#QUEUE[@]} experiments queued."

while true; do
    sleep 60

    for gpu in 0 1 2 3 4 5 6 7; do
        pid="${GPU_PID[$gpu]}"

        # Check if GPU freed up
        if [ -n "$pid" ]; then
            freed=false
            if [ "$pid" = "unknown" ]; then
                mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
                [ "$mem" -lt 100 ] 2>/dev/null && freed=true
            else
                kill -0 "$pid" 2>/dev/null || freed=true
            fi
            if $freed; then
                log "GPU $gpu: freed up"
                unset GPU_PID[$gpu]
            fi
        fi

        # Launch next if free and queue has work
        if [ -z "${GPU_PID[$gpu]}" ] && [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; then
            exp="${QUEUE[$QUEUE_IDX]}"
            read -r max_d digits logname <<< "$exp"
            logfile="$R/gpu_${logname}.log"

            # Skip if already complete
            if grep -q "^Covered:" "$logfile" 2>/dev/null; then
                log "GPU $gpu: SKIP $logname (done)"
                QUEUE_IDX=$((QUEUE_IDX + 1))
                continue
            fi

            log "GPU $gpu: LAUNCH $logname ($B $max_d $digits)"
            CUDA_VISIBLE_DEVICES=$gpu nohup stdbuf -oL $B $max_d $digits > "$logfile" 2>&1 &
            GPU_PID[$gpu]=$!
            log "GPU $gpu: PID=${GPU_PID[$gpu]}"
            QUEUE_IDX=$((QUEUE_IDX + 1))
        fi
    done

    # Status every 10 min
    busy=0; free=0
    for gpu in 0 1 2 3 4 5 6 7; do
        [ -n "${GPU_PID[$gpu]}" ] && busy=$((busy+1)) || free=$((free+1))
    done
    remaining=$((${#QUEUE[@]} - QUEUE_IDX))
    log "Status: ${busy}/8 busy, ${free} free, ${remaining} queued"

    # Exit when everything is done
    [ $busy -eq 0 ] && [ $remaining -eq 0 ] && { log "All done."; break; }
done
