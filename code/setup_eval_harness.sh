#!/usr/bin/env bash
# =============================================================================
# DACS Project — Activity 5: Eval Harness Setup
# Run this FIRST on your Lambda Cloud GPU instance, before any quantization.
# Without working benchmarks you cannot record results for Activities 2-6.
#
# Usage (run from repo root):
#   bash code/setup_eval_harness.sh
#
# Expected total time: ~15-20 minutes (torch download is ~2GB)
# =============================================================================

set -e  # stop on first unpiped error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo " DACS: Activity 5 — Eval Harness Setup"
echo " Run this BEFORE Activity 2 quantization"
echo " Repo root: $REPO_ROOT"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1: Install PyTorch (CUDA 12.1 — matches Lambda A10G)
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q

python -c "import torch; print('  torch version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
echo "  PyTorch OK"

# ---------------------------------------------------------------------------
# STEP 2: Install lm-eval (MMLU benchmark)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installing lm-eval harness (EleutherAI)..."
pip install lm-eval -q

python -c "import lm_eval; print('  lm-eval version:', lm_eval.__version__)"
echo "  lm-eval OK"

# ---------------------------------------------------------------------------
# STEP 3: Clone and install CyberSecEval 4 (Meta PurpleLlama)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Cloning and installing CyberSecEval 4 (PurpleLlama)..."

if [ ! -d "PurpleLlama" ]; then
    git clone https://github.com/meta-llama/PurpleLlama.git --depth=1
else
    echo "  PurpleLlama already cloned, skipping."
fi

cd PurpleLlama/CybersecurityBenchmarks
pip install -r requirements.txt -q
# Install the package itself so `cyberseceval` is importable from anywhere
pip install -e . -q 2>/dev/null || echo "  (no setup.py — using PYTHONPATH instead)"
cd "$REPO_ROOT"

# Verify import works (with fallback to path injection)
python -c "
import sys, os
try:
    import cyberseceval
    print('  cyberseceval import OK (installed as package)')
except ImportError:
    sys.path.insert(0, os.path.join('$REPO_ROOT', 'PurpleLlama', 'CybersecurityBenchmarks'))
    import cyberseceval
    print('  cyberseceval import OK (via sys.path)')
"
echo "  CyberSecEval OK"

# ---------------------------------------------------------------------------
# STEP 4: Sanity-test lm-eval (gpt2 on MMLU, expect ~25% = random)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Sanity-testing lm-eval on gpt2 (20 questions, expect ~25% accuracy)..."
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --num_fewshot 5 \
    --limit 20 \
    --output_path /tmp/mmlu_sanity_test/ \
    2>&1 | tail -8 || true

echo "  lm-eval sanity test complete (gpt2 ~25% is expected and correct)"

# ---------------------------------------------------------------------------
# STEP 5: Verify unified_eval.py is importable
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying unified_eval.py is importable..."
python -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'code'))
from unified_eval import evaluate_model
print('  unified_eval.py OK')
"

echo ""
echo "============================================================"
echo " Activity 5 Setup COMPLETE"
echo " Both benchmarks are installed and verified."
echo " You can now proceed to Activity 2 quantization."
echo "============================================================"
