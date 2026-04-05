"""
DACS Activity 3 — Build Code Domain C3 Calibration Corpus
==========================================================
Sample 512 Python code+explanation texts from a public code-instruction
dataset aligned with the CodeLlama-7B-Instruct fine-tune distribution.

Source: sahil2801/CodeAlpaca-20k (instruction/output, Python-heavy) with
        fallback to HuggingFaceH4/CodeAlpaca_20K.

Why this source?
  CodeLlama-Instruct was tuned on code-instruction pairs similar to CodeAlpaca.
  Sampling CodeAlpaca outputs approximates the assistant-turn distribution
  the model was adapted on.

Output:  ./outputs/code_c3_calib_512.jsonl
Runtime: ~1-2 minutes (download + sample)
"""

import os
import json
import random

OUTPUT_JSONL = "./outputs/code_c3_calib_512.jsonl"
N_SAMPLES    = 512
SEED         = 42

DOMAIN_TERMS = [
    "def ", "return", "import", "class ", "print", "for ", "while ",
    "if ", "else", "lambda", "list", "dict", "str", "int", "float",
]


def main():
    random.seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" Activity 3 — Code C3 Corpus")
    print("=" * 60)

    from datasets import load_dataset

    texts = []
    for ds_id in ["sahil2801/CodeAlpaca-20k", "HuggingFaceH4/CodeAlpaca_20K"]:
        try:
            print(f"[1/2] Trying {ds_id}...")
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            for row in ds:
                t = (row.get("output") or row.get("response") or "").strip()
                if len(t) > 60:
                    texts.append(t)
            print(f"  Got {len(texts)} texts from {ds_id}")
            break
        except Exception as e:
            print(f"  {ds_id} failed ({e})")

    print(f"\n[2/2] Sampling {N_SAMPLES} (seed={SEED})...")
    if len(texts) < N_SAMPLES:
        print(f"  WARNING: only {len(texts)} texts available; using all")
        sampled = texts
    else:
        sampled = random.sample(texts, N_SAMPLES)

    combined = " ".join(sampled)
    print("\n  Domain term coverage:")
    for term in DOMAIN_TERMS:
        print(f"    {term!r:12s}: {combined.count(term):4d}")

    with open(OUTPUT_JSONL, "w") as f:
        for t in sampled:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"\n DONE: {OUTPUT_JSONL}  |  {len(sampled)} samples  |  seed={SEED}")


if __name__ == "__main__":
    main()
