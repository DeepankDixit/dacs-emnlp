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
#   bash code/activity2/setup_lambda.sh    # Activity 2 base
#   bash code/activity3/setup_lambda.sh    # Activity 3 extensions
#   huggingface-cli login --token $HF_TOKEN

set -e

echo "======================================================"
echo " DACS Activity 3 Setup (multi-domain extensions)"
echo "======================================================"

# ── Step 1: Activity-3 pip extras ────────────────────────────────────────────
echo "[1/4] Installing Activity 3 extras..."
pip install -q -r code/activity3/requirements.txt
echo "      OK"

# ── Step 2: MedQA dataset ────────────────────────────────────────────────────
echo "[2/4] Downloading MedQA test split from HuggingFace..."
MEDQA_FILE="MedQA/data/questions/US/4_options/phrases_no_exclude_test.jsonl"
if [ ! -f "$MEDQA_FILE" ]; then
    mkdir -p MedQA/data/questions/US/4_options
    python - <<'EOF'
import json, os
from datasets import load_dataset

print("      Fetching GBaker/MedQA-USMLE-4-options test split...")
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
out_path = "MedQA/data/questions/US/4_options/phrases_no_exclude_test.jsonl"
with open(out_path, "w") as f:
    for item in ds:
        # resolve answer_idx from answer text
        answer_idx = None
        for k, v in item["options"].items():
            if v.strip() == item["answer"].strip():
                answer_idx = k
                break
        if answer_idx is None:
            answer_idx = list(item["options"].keys())[0]  # fallback
        record = {
            "question":   item["question"],
            "options":    item["options"],
            "answer_idx": answer_idx,
        }
        f.write(json.dumps(record) + "\n")
print(f"      Saved {len(ds)} questions to {out_path}")
EOF
else
    echo "      MedQA already present, skipping"
fi
N_Q=$(wc -l < "$MEDQA_FILE")
echo "      MedQA test set: $N_Q questions"

# ── Step 3: HumanEval+ ───────────────────────────────────────────────────────
echo "[3/4] Installing human-eval..."
pip install -q human-eval
python - <<'EOF'
from human_eval.data import read_problems
probs = read_problems()
print(f"      HumanEval loaded: {len(probs)} problems")
EOF

# ── Step 4: HuggingFace login check ──────────────────────────────────────────
echo "[4/4] Checking HuggingFace login..."
if huggingface-cli whoami 2>/dev/null | grep -q .; then
    WHO=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo "      Logged in as: $WHO"
else
    echo "      NOT LOGGED IN."
    echo "      Run: huggingface-cli login --token \$HF_TOKEN"
fi

echo ""
echo "======================================================"
echo " Activity 3 setup COMPLETE"
echo " Next: python code/activity3/download_models.py --domain all"
echo "======================================================"
