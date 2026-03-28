#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Download a proving model to the cluster
# Usage: ./scripts/download-model.sh <model_id> [--quantized]
# ============================================================================

MODEL_ID="${1:?Usage: $0 <model_id> [--quantized]}"
MODELS_DIR="models"
MODEL_NAME=$(echo "$MODEL_ID" | sed 's|.*/||')
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

mkdir -p "$MODELS_DIR"

if [ -d "$MODEL_PATH" ] && [ -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo "Model already downloaded: $MODEL_PATH"
    exit 0
fi

echo "Downloading $MODEL_ID -> $MODEL_PATH"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_PATH')
print('Download complete.')
"

echo "Model ready at: $MODEL_PATH"
du -sh "$MODEL_PATH"
