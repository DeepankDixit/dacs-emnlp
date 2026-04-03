#!/bin/bash
# DACS Project — Fresh Lambda A10G Setup
# =======================================
# Run this once on a new Lambda instance after cloning the repo.
# Expected total time: ~8 minutes
#
# Usage:
#   git clone <your-repo> dacs-emnlp && cd dacs-emnlp
#   python3 -m venv venv && source venv/bin/activate
#   bash code/setup_lambda.sh

set -e  # exit on first error

echo "======================================================"
echo " DACS Lambda Setup"
echo "======================================================"
echo ""

# ── Step 1: PyTorch (cu124) ──────────────────────────────────────────────────
# MUST be installed before modelopt — modelopt 0.42.0 requires torch>=2.6
echo "[1/5] Installing PyTorch 2.6.0+cu124..."
pip install torch==2.6.0+cu124 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 -q
echo "      torch OK"

# ── Step 2: Quantization libraries ──────────────────────────────────────────
echo "[2/5] Installing quantization libraries..."
pip install nvidia-modelopt==0.42.0 -q
pip install autoawq -q
echo "      modelopt + autoawq OK"

# ── Step 3: HuggingFace + eval stack ────────────────────────────────────────
echo "[3/5] Installing HuggingFace stack + lm-eval..."
pip install \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    peft>=0.12.0 \
    datasets>=2.20.0 \
    tokenizers>=0.19.0 \
    sentencepiece>=0.2.0 \
    safetensors>=0.4.3 \
    lm-eval>=0.4.3 \
    numpy>=1.26.0 \
    tqdm>=4.66.0 \
    einops>=0.7.0 \
    packaging>=24.0 \
    scipy>=1.13.0 -q
echo "      HuggingFace stack OK"

# ── Step 4: Clone PurpleLlama (datasets kept for reference) ─────────────────
echo "[4/5] Cloning PurpleLlama (for MITRE dataset files)..."
if [ ! -d "PurpleLlama" ]; then
    git clone --depth 1 https://github.com/meta-llama/PurpleLlama.git PurpleLlama
    echo "      PurpleLlama cloned"
else
    echo "      PurpleLlama already present, skipping"
fi

# ── Step 5: Sanity checks ────────────────────────────────────────────────────
echo "[5/5] Running sanity checks..."

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(f"      torch {torch.__version__} — CUDA {torch.version.cuda} — GPU: {torch.cuda.get_device_name(0)}")

import modelopt.torch.quantization as mtq
print(f"      modelopt OK — INT8_SMOOTHQUANT_CFG found: {hasattr(mtq, 'INT8_SMOOTHQUANT_CFG')}")

from awq import AutoAWQForCausalLM
print(f"      autoawq OK")

import transformers
print(f"      transformers {transformers.__version__}")

# Confirm lm-eval has wmdp_cyber task
from lm_eval.tasks import TaskManager
tm = TaskManager()
assert "wmdp_cyber" in tm.all_tasks, "wmdp_cyber task not found in lm-eval!"
print(f"      lm-eval OK — wmdp_cyber task registered")

import sys
sys.path.insert(0, "code")
from unified_eval import evaluate_model
print(f"      unified_eval.py importable OK")
EOF

echo ""
echo "======================================================"
echo " Setup COMPLETE — all dependencies verified"
echo " Next: run quantization scripts or evaluate_all.py"
echo "======================================================"
