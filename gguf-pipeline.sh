#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# GGUF Quantization Pipeline
# Converts HF models → GGUF → imatrix → full quant ladder → uploads to HF
# ============================================================================

LLAMA_CPP="/home/cahlen/dev/llama.cpp"
QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
IMATRIX="$LLAMA_CPP/build/bin/llama-imatrix"
CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
WORK_DIR="/home/cahlen/dev/gguf-workbench"
IMATRIX_DATASET="$WORK_DIR/calibration-data.txt"
HF_USER="cahlen"

# Quant types: full ladder from high to low quality
# Standard quants (don't need imatrix)
STANDARD_QUANTS="Q8_0 Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_L Q3_K_M Q3_K_S Q2_K"
# Importance-matrix quants (need imatrix for good quality)
IMATRIX_QUANTS="IQ4_XS IQ3_M IQ3_S IQ3_XXS IQ2_M IQ2_S IQ2_XXS IQ1_M"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  quantize <model_id> [--org <hf_org>] [--quants <types>] [--skip-imatrix] [--skip-upload]
      Download a model, convert to GGUF, quantize, and upload.
      Example: $(basename "$0") quantize meta-llama/Llama-3.1-8B-Instruct

  setup
      One-time setup: create work directory, download calibration data.

  list-quants
      Show available quantization types.

  clean [model_name]
      Remove working files for a model (keeps uploads).

Options:
  --org <org>         HF org/user to upload under (default: $HF_USER)
  --quants <types>    Comma-separated quant types (default: full ladder)
  --skip-imatrix      Skip importance matrix generation
  --skip-upload       Don't upload to HF, just produce local files
  --bf16              Use BF16 as base type instead of F16 (for BF16 source models)
  --help              Show this help
EOF
}

setup() {
    log "Setting up GGUF workbench..."
    mkdir -p "$WORK_DIR"

    if [ ! -f "$IMATRIX_DATASET" ]; then
        log "Downloading calibration dataset for importance matrix..."
        # Use a standard calibration dataset (wikitext)
        python3 -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
with open('$IMATRIX_DATASET', 'w') as f:
    for row in ds:
        text = row['text'].strip()
        if len(text) > 100:
            f.write(text + '\n')
print('Calibration data written.')
"
        log "Calibration dataset ready: $IMATRIX_DATASET"
    else
        log "Calibration dataset already exists."
    fi

    log "Setup complete. Work directory: $WORK_DIR"
}

list_quants() {
    echo "Standard quants (no imatrix needed):"
    echo "  $STANDARD_QUANTS"
    echo ""
    echo "Importance-matrix quants (imatrix recommended):"
    echo "  $IMATRIX_QUANTS"
}

quantize_model() {
    local model_id="$1"
    local hf_org="$HF_USER"
    local custom_quants=""
    local skip_imatrix=false
    local skip_upload=false
    local base_type="f16"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --org) hf_org="$2"; shift 2 ;;
            --quants) custom_quants="$2"; shift 2 ;;
            --skip-imatrix) skip_imatrix=true; shift ;;
            --skip-upload) skip_upload=true; shift ;;
            --bf16) base_type="bf16"; shift ;;
            *) err "Unknown option: $1"; exit 1 ;;
        esac
    done

    # Derive names
    local model_name
    model_name=$(echo "$model_id" | sed 's|.*/||')
    local model_dir="$WORK_DIR/$model_name"
    local gguf_dir="$model_dir/gguf"
    local base_gguf="$gguf_dir/${model_name}-${base_type^^}.gguf"
    local imatrix_file="$gguf_dir/imatrix.dat"
    local repo_name="${model_name}-GGUF"

    mkdir -p "$model_dir" "$gguf_dir"

    # ── Step 1: Download model ──
    if [ ! -d "$model_dir/hf" ] || [ -z "$(ls -A "$model_dir/hf" 2>/dev/null)" ]; then
        log "Downloading $model_id from Hugging Face..."
        huggingface-cli download "$model_id" --local-dir "$model_dir/hf"
    else
        log "Model already downloaded: $model_dir/hf"
    fi

    # ── Step 2: Convert to GGUF ──
    if [ ! -f "$base_gguf" ]; then
        log "Converting to GGUF ($base_type)..."
        python3 "$CONVERT" "$model_dir/hf" \
            --outfile "$base_gguf" \
            --outtype "$base_type"
        log "Base GGUF created: $base_gguf"
    else
        log "Base GGUF already exists: $base_gguf"
    fi

    # ── Step 3: Generate importance matrix ──
    if [ "$skip_imatrix" = false ] && [ ! -f "$imatrix_file" ]; then
        log "Generating importance matrix (this takes a while)..."
        "$IMATRIX" \
            -m "$base_gguf" \
            -f "$IMATRIX_DATASET" \
            -o "$imatrix_file" \
            --chunks 200 \
            -ngl 999
        log "Importance matrix saved: $imatrix_file"
    elif [ -f "$imatrix_file" ]; then
        log "Importance matrix already exists."
    fi

    # ── Step 4: Quantize ──
    local quants_to_run
    if [ -n "$custom_quants" ]; then
        quants_to_run="${custom_quants//,/ }"
    else
        quants_to_run="$STANDARD_QUANTS"
        if [ "$skip_imatrix" = false ]; then
            quants_to_run="$quants_to_run $IMATRIX_QUANTS"
        fi
    fi

    log "Quantizing: $quants_to_run"
    local produced_files=()

    for qtype in $quants_to_run; do
        local out_file="$gguf_dir/${model_name}-${qtype}.gguf"
        if [ -f "$out_file" ]; then
            log "  $qtype already exists, skipping."
            produced_files+=("$out_file")
            continue
        fi

        log "  Quantizing $qtype..."
        local imatrix_flag=""
        # Use imatrix for IQ* quants and low-bit K-quants
        if [[ "$qtype" == IQ* ]] || [[ "$qtype" == "Q2_K" ]] || [[ "$qtype" == Q3_K_* ]]; then
            if [ -f "$imatrix_file" ]; then
                imatrix_flag="--imatrix $imatrix_file"
            fi
        fi

        if "$QUANTIZE" $imatrix_flag "$base_gguf" "$out_file" "$qtype" 2>&1 | tail -3; then
            produced_files+=("$out_file")
            local size
            size=$(du -h "$out_file" | cut -f1)
            log "  ✓ $qtype → $size"
        else
            warn "  ✗ $qtype failed, skipping."
        fi
    done

    # ── Step 5: Generate model card ──
    local card_file="$gguf_dir/README.md"
    generate_model_card "$model_id" "$model_name" "$repo_name" "$card_file" "$gguf_dir"

    # ── Step 6: Upload to HF ──
    if [ "$skip_upload" = false ]; then
        local full_repo="${hf_org}/${repo_name}"
        log "Uploading to $full_repo..."

        # Create repo if it doesn't exist
        huggingface-cli repo create "$repo_name" --type model -y 2>/dev/null || true

        # Upload all GGUF files and the README
        huggingface-cli upload "$full_repo" "$gguf_dir/" . \
            --include "*.gguf" "README.md" \
            --commit-message "Add GGUF quantizations of $model_id"

        log "Upload complete! https://huggingface.co/$full_repo"
    else
        log "Skipping upload (--skip-upload). Files are in: $gguf_dir/"
    fi

    # Summary
    echo ""
    log "═══════════════════════════════════════════"
    log "  Quantization complete: $model_name"
    log "  Files: ${#produced_files[@]} quants produced"
    log "  Location: $gguf_dir/"
    if [ "$skip_upload" = false ]; then
        log "  HF Repo: https://huggingface.co/${hf_org}/${repo_name}"
    fi
    log "═══════════════════════════════════════════"
}

generate_model_card() {
    local source_model="$1"
    local model_name="$2"
    local repo_name="$3"
    local card_file="$4"
    local gguf_dir="$5"

    # Build file table
    local file_table=""
    file_table+="| Filename | Quant Type | Size |\n"
    file_table+="|----------|-----------|------|\n"
    for f in "$gguf_dir"/*.gguf; do
        [ -f "$f" ] || continue
        local fname size qtype
        fname=$(basename "$f")
        size=$(du -h "$f" | cut -f1)
        qtype=$(echo "$fname" | sed "s/${model_name}-//" | sed 's/\.gguf//')
        file_table+="| $fname | $qtype | $size |\n"
    done

    cat > "$card_file" <<EOF
---
base_model: $source_model
tags:
  - gguf
  - quantized
  - llama-cpp
license: other
---

# $repo_name

GGUF quantizations of [\`$source_model\`](https://huggingface.co/$source_model).

Quantized using [llama.cpp](https://github.com/ggml-org/llama.cpp) with importance matrix calibration for low-bit quants.

## Available Quantizations

$(echo -e "$file_table")

## How to Use

### With llama.cpp
\`\`\`bash
llama-cli -m ${model_name}-Q4_K_M.gguf -p "Hello" -ngl 999
\`\`\`

### With Ollama
\`\`\`bash
ollama run hf.co/$HF_USER/$repo_name:Q4_K_M
\`\`\`

### With LM Studio
Download any GGUF file above and load it in LM Studio.

## Quantization Details

- **Source model**: $source_model
- **Quantized by**: [$HF_USER](https://huggingface.co/$HF_USER)
- **Calibration**: Importance matrix generated from WikiText-2 (200 chunks)
- **Tool**: llama.cpp ($(cd "$LLAMA_CPP" && git rev-parse --short HEAD 2>/dev/null || echo "latest"))
- **Hardware**: NVIDIA RTX 5090 32GB / Intel Core Ultra 9 285K / 188GB RAM

## Choosing a Quant

| Use Case | Recommended |
|----------|------------|
| Maximum quality | Q8_0 |
| Good balance | Q5_K_M or Q4_K_M |
| Low VRAM (8GB) | Q3_K_M or IQ3_M |
| Extreme compression | IQ2_M or IQ2_S |
EOF

    log "Model card generated: $card_file"
}

clean_model() {
    local model_name="$1"
    local model_dir="$WORK_DIR/$model_name"
    if [ -d "$model_dir" ]; then
        log "Cleaning $model_dir..."
        rm -ri "$model_dir"
    else
        err "No working directory found for: $model_name"
    fi
}

# ── Main ──
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

command="$1"
shift

case "$command" in
    setup)
        setup
        ;;
    quantize)
        if [ $# -lt 1 ]; then
            err "Missing model ID. Example: $(basename "$0") quantize meta-llama/Llama-3.1-8B-Instruct"
            exit 1
        fi
        quantize_model "$@"
        ;;
    list-quants)
        list_quants
        ;;
    clean)
        if [ $# -lt 1 ]; then
            err "Missing model name."
            exit 1
        fi
        clean_model "$1"
        ;;
    --help|-h|help)
        usage
        ;;
    *)
        err "Unknown command: $command"
        usage
        exit 1
        ;;
esac
