"""
DACS Project — Activity 2C: Prepare DACS Calibration Data (C3)
===============================================================
This is the core contribution of the paper.
Sample 512 assistant-turn texts from the original SFT training corpus
(LoraForge Activity 0 data) to use as calibration data for quantization.

Why assistant turns only (not user/system prompts)?
  During fine-tuning, the cross-entropy loss was computed on ASSISTANT tokens only.
  Those are the tokens whose gradients shaped the weight updates.
  The weights now activate in specific ways when PRODUCING domain-style text.
  Calibrating on assistant text exposes exactly those activation patterns.

Why 512 samples (not 128 like AWQ default)?
  More samples → better coverage of the domain activation distribution.
  512 is the SmoothQuant default and also works well for AWQ.
  The DACS paper uses 512 for all three quantization formats.

Input:    LoraForge SFT training corpus — either a local JSONL or auto-downloaded
          from HuggingFace (Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset)
Output:   ./outputs/dacs_calib_512.jsonl
Runtime:  ~1-2 minutes (CPU; add ~2 min first time for HF download)
"""

import os
import sys
import json
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to the SFT training corpus from LoraForge Activity 0.
# If this file does not exist, the script will auto-download from HuggingFace
# and save it here for future runs.
SFT_CORPUS_PATH = os.environ.get(
    "SFT_CORPUS_PATH",
    "./outputs/cybersec_sft_train.jsonl",   # auto-saved here on first run
)

# HuggingFace dataset used in LoraForge Activity 0 (fallback if no local file)
HF_DATASET_PRIMARY  = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
HF_INSTRUCTION_COL  = "user"
HF_RESPONSE_COL     = "assistant"

OUTPUT_JSONL = "./outputs/dacs_calib_512.jsonl"
N_SAMPLES    = 512
SEED         = 42   # MUST be fixed for reproducibility (reported in paper)
# ---------------------------------------------------------------------------

DOMAIN_TERMS = [
    "CVE", "vulnerability", "exploit", "buffer", "injection",
    "MITRE", "ATT&CK", "attack", "malware", "threat", "security",
    "shellcode", "payload", "reverse shell", "privilege escalation",
]

def extract_assistant_text(example: dict) -> str | None:
    """
    Extract assistant-turn text from a training example.
    Handles three common SFT corpus formats:
      1. ChatML: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
      2. Instruction: {"instruction": "...", "output": "..."}
      3. Plain text: {"text": "..."}
    """
    # Format 1: ChatML (most common for instruction-tuned models)
    if "messages" in example:
        for msg in reversed(example["messages"]):
            if msg.get("role") == "assistant":
                return msg.get("content", "").strip()

    # Format 2: Instruction/output format
    if "output" in example and len(example["output"].strip()) > 20:
        return example["output"].strip()

    # Format 3: Plain text (fallback)
    if "text" in example and len(example["text"].strip()) > 20:
        return example["text"].strip()

    return None


def main():
    random.seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" Activity 2C — Prepare DACS Calibration Data (C3)")
    print("=" * 60)
    print(f"  SFT corpus: {SFT_CORPUS_PATH}")
    print(f"  Output:     {OUTPUT_JSONL}")
    print(f"  N samples:  {N_SAMPLES} (seed={SEED})")
    print()

    # Step 1: Load corpus — from local file or HuggingFace
    if not os.path.exists(SFT_CORPUS_PATH):
        print(f"  Local corpus not found at: {SFT_CORPUS_PATH}")
        print(f"  Auto-downloading from HuggingFace: {HF_DATASET_PRIMARY}")
        print(f"  (This happens once — saved to {SFT_CORPUS_PATH} for future runs)")
        print()
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' library not installed.")
            print("  Fix: pip install datasets")
            sys.exit(1)

        hf_ds = load_dataset(HF_DATASET_PRIMARY, split="train", trust_remote_code=True)
        print(f"  Downloaded {len(hf_ds)} examples from HuggingFace")

        # Save as JSONL in ChatML format so extract_assistant_text() can parse it
        print(f"  Saving to {SFT_CORPUS_PATH}...")
        with open(SFT_CORPUS_PATH, "w") as f:
            for row in hf_ds:
                record = {
                    "messages": [
                        {"role": "user",      "content": row[HF_INSTRUCTION_COL]},
                        {"role": "assistant", "content": row[HF_RESPONSE_COL]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(hf_ds)} examples")
        print()

    print(f"[1/4] Loading SFT corpus from {SFT_CORPUS_PATH}...")
    with open(SFT_CORPUS_PATH) as f:
        sft_corpus = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(sft_corpus)} training examples")

    if len(sft_corpus) < N_SAMPLES:
        print(f"  WARNING: Corpus has {len(sft_corpus)} examples but we need {N_SAMPLES}.")
        print(f"  Will use all {len(sft_corpus)} examples.")
        N_SAMPLES_ACTUAL = len(sft_corpus)
    else:
        N_SAMPLES_ACTUAL = N_SAMPLES

    # Step 2: Random sample N_SAMPLES examples (fixed seed for reproducibility)
    print(f"\n[2/4] Randomly sampling {N_SAMPLES_ACTUAL} examples (seed={SEED})...")
    sampled = random.sample(sft_corpus, N_SAMPLES_ACTUAL)

    # Step 3: Extract assistant-turn text only
    print(f"\n[3/4] Extracting assistant-turn text...")
    calib_texts = []
    skipped = 0
    for example in sampled:
        text = extract_assistant_text(example)
        if text:
            calib_texts.append(text)
        else:
            skipped += 1

    print(f"  Extracted {len(calib_texts)} texts  (skipped {skipped} with no assistant turn)")

    # Verify domain coverage
    all_text = " ".join(calib_texts).lower()
    found = {t: all_text.count(t.lower()) for t in DOMAIN_TERMS}
    print(f"\n  Domain term coverage:")
    for term, count in sorted(found.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 5, 30)
        print(f"    {term:25s}: {count:4d}  {bar}")

    no_domain = [t for t, c in found.items() if c == 0]
    if no_domain:
        print(f"\n  WARNING: These domain terms not found: {no_domain}")
        print(f"  This may indicate the corpus format was not parsed correctly.")

    # Step 4: Save
    print(f"\n[4/4] Saving {len(calib_texts)} DACS calibration texts to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w") as f:
        for text in calib_texts:
            f.write(json.dumps({"text": text}) + "\n")

    print()
    print("=" * 60)
    print(" DACS calibration data (C3) COMPLETE")
    print(f"  Output:    {OUTPUT_JSONL}")
    print(f"  Samples:   {len(calib_texts)}")
    print(f"  Seed:      {SEED}  ← record this in your paper for reproducibility")
    print()
    print("  Next: Run activity2c_awq_c3.py, activity2c_sq_c3.py, activity2c_fp8_c3.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
