#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Serve a model on the B200 cluster using vLLM or SGLang
# Usage: ./scripts/serve-model.sh <model_path_or_id> [--engine vllm|sglang] [--tp 8]
# ============================================================================

MODEL="${1:?Usage: $0 <model_path_or_id> [--engine vllm|sglang] [--tp 8]}"
ENGINE="vllm"
TP=8  # tensor parallel across 8 GPUs
PORT=8000

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Serving $MODEL with $ENGINE (tp=$TP) on port $PORT"

if [ "$ENGINE" = "vllm" ]; then
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len 32768 \
        --dtype auto
elif [ "$ENGINE" = "sglang" ]; then
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp "$TP" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$PORT" \
        --mem-fraction-static 0.85
else
    echo "Unknown engine: $ENGINE (use vllm or sglang)"
    exit 1
fi
