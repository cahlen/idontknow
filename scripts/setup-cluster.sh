#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# B200 Cluster Setup — Lean 4 + LLM-assisted Theorem Proving
# Run this on the 8xB200 cluster after cloning the repo
# ============================================================================

echo "=== Setting up Lean 4 + LLM Proving Environment ==="

# ── System deps ──
echo "[1/6] Checking system dependencies..."
which git curl python3 pip nvcc || {
    echo "Installing system deps..."
    sudo apt-get update && sudo apt-get install -y git curl python3 python3-pip build-essential
}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Lean 4 (elan + lake) ──
echo "[2/6] Installing Lean 4 via elan..."
if ! command -v lean &>/dev/null; then
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    export PATH="$HOME/.elan/bin:$PATH"
    echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
else
    export PATH="$HOME/.elan/bin:$PATH"
    echo "  Lean already installed: $(lean --version)"
fi

# Use the toolchain that matches Mathlib
MATHLIB_TOOLCHAIN="leanprover/lean4:v4.29.0-rc8"
echo "  Setting default toolchain to $MATHLIB_TOOLCHAIN (matches Mathlib)..."
elan default "$MATHLIB_TOOLCHAIN"

# ── Python env ──
echo "[3/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers \
    accelerate \
    vllm \
    huggingface-hub \
    datasets \
    sglang \
    openai \
    requests \
    pyyaml

# ── Mathlib (optional, for testing against existing library) ──
echo "[4/6] Setting up Mathlib cache..."
if [ ! -d "lean4-proving/mathlib4" ]; then
    cd lean4-proving
    git clone --depth 1 https://github.com/leanprover-community/mathlib4.git
    cd mathlib4
    lake exe cache get || echo "  (Mathlib cache download failed, will build from source if needed)"
    cd ../..
else
    echo "  Mathlib already cloned."
fi

# ── Model download placeholder ──
echo "[5/6] Model download..."
echo "  Model will be configured after research phase."
echo "  Run: ./scripts/download-model.sh <model_id> to pull a specific model."

# ── Verify setup ──
echo "[6/6] Verifying setup..."
lean --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""
echo "=== Setup complete ==="
