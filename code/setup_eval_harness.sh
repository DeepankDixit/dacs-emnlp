#!/usr/bin/env bash
# =============================================================================
# DACS Project — Activity 5: Eval Harness Setup
# Run this FIRST on your Lambda Cloud GPU instance, before any quantization.
# Without working benchmarks you cannot record results for Activities 2-6.
#
# Usage:
#   bash setup_eval_harness.sh
#
# Expected total time: ~10 minutes
# =============================================================================

set -e  # stop on first error

echo "============================================================"
echo " DACS: Activity 5 — Eval Harness Setup"
echo " Run this BEFORE Activity 2 quantization"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1: Install lm-eval (MMLU benchmark)
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing lm-eval harness (EleutherAI)..."
pip install lm-eval --break-system-packages -q

python -c "import lm_eval; print('  lm-eval version:', lm_eval.__version__)"
echo "  lm-eval OK"

# ---------------------------------------------------------------------------
# STEP 2: Install CyberSecEval 4 (Meta PurpleLlama)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Cloning and installing CyberSecEval 4 (PurpleLlama)..."

if [ ! -d "PurpleLlama" ]; then
    git clone https://github.com/meta-llama/PurpleLlama.git --depth=1
else
    echo "  PurpleLlama already cloned, skipping."
fi

cd PurpleLlama/CybersecurityBenchmarks
pip install -r requirements.txt --break-system-packages -q
cd ../..

python -c "import sys; sys.path.insert(0, 'PurpleLlama/CybersecurityBenchmarks'); print('  CyberSecEval import OK')"
echo "  CyberSecEval OK"

# ---------------------------------------------------------------------------
# STEP 3: Sanity test lm-eval (runs MMLU on gpt2, expects ~25% = random)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Sanity-testing lm-eval on gpt2 (20 questions, expect ~25% accuracy)..."
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks mmlu \
    --num_fewshot 5 \
    --limit 20 \
    --output_path /tmp/mmlu_sanity_test/ \
    2>&1 | tail -5

echo "  lm-eval sanity test complete (gpt2 ~25% is expected and correct)"

# ---------------------------------------------------------------------------
# STEP 4: Sanity test CyberSecEval on gpt2 (5 samples, just checks it runs)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Sanity-testing CyberSecEval on gpt2 (5 samples, checks pipeline only)..."
cd PurpleLlama/CybersecurityBenchmarks
python -m cyberseceval.run_benchmark \
    --benchmark mitre \
    --model gpt2 \
    --num-samples 5 \
    --output-dir /tmp/cyberseceval_sanity_test/ \
    2>&1 | tail -5
cd ../..
echo "  CyberSecEval sanity test complete"

# ---------------------------------------------------------------------------
# STEP 5: Verify unified_eval.py can be imported
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying unified_eval.py is importable..."
python -c "import sys; sys.path.insert(0, '.'); from unified_eval import evaluate_model; print('  unified_eval.py OK')"

echo ""
echo "============================================================"
echo " Activity 5 Setup COMPLETE"
echo " Both benchmarks are installed and verified."
echo " You can now proceed to Activity 2 quantization."
echo "============================================================"
