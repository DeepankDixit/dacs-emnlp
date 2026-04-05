#!/bin/bash
# DACS Activity 3 — Lambda A10G Setup (Multi-Domain)
# ===================================================
# Run this on a fresh Lambda instance AFTER setup_lambda.sh.
# Installs domain-specific eval datasets (MedQA, HumanEval+) and verifies
# HuggingFace login for gated med42 checkpoint.
#
# Total time: ~5 minutes
#
# Usage:
#   source venv/bin/activate
#   bash code/setup_lambda.sh              # Activity 2 base
#   bash code/setup_lambda_act3.sh         # Activity 3 extensions
#   huggingface-cli login                  # enter token for Med42 gated access

set -e

echo "======================================================"
echo " DACS Activity 3 Setup (multi-domain extensions)"
echo "======================================================"

# ── Step 1: Activity-3 pip extras ────────────────────────────────────────────
echo "[1/4] Installing Activity 3 extras..."
pip install -q -r code/requirements_act3.txt
echo "      OK"

# ── Step 2: MedQA dataset ────────────────────────────────────────────────────
echo "[2/4] Cloning MedQA (US 4-option test split)..."
if [ ! -d "MedQA" ]; then
    git clone --depth 1 https://github.com/jind11/MedQA.git MedQA
    echo "      MedQA cloned"
else
    echo "      MedQA already present, skipping"
fi
# Sanity: count test questions
N_Q=$(wc -l < MedQA/data/questions/US/4_options/phrases_no_exclude_test.jsonl)
echo "      MedQA test set: $N_Q questions"

# ── Step 3: HumanEval+ ───────────────────────────────────────────────────────
echo "[3/4] Installing human-eval..."
pip install -q human-eval
python - <<'EOF'
from human_eval.data import read_problems
probs = read_problems()
print(f"      HumanEval loaded: {len(probs)} problems")
EOF

# ── Step 4: Gated model check ────────────────────────────────────────────────
echo "[4/4] Checking HuggingFace login (required for Med42 gated access)..."
if huggingface-cli whoami 2>/dev/null | grep -q .; then
    WHO=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo "      Logged in as: $WHO"
else
    echo "      NOT LOGGED IN."
    echo "      Run: huggingface-cli login"
    echo "      Then accept terms at: https://huggingface.co/m42-health/Llama3-Med42-8B"
fi

echo ""
echo "======================================================"
echo " Activity 3 setup COMPLETE"
echo " Next: python code/download_domain_models.py --domain all"
echo "======================================================"
