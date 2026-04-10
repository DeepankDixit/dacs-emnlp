"""
DACS Activity 3 — Build Medical Domain C3 Calibration Corpus
=============================================================
Sample 512 medical assistant-turn texts from a public medical instruction
dataset aligned with the Llama3-Med42-8B fine-tune distribution.

Source: qiaojin/PubMedQA (pqa_labeled long answers) +
        GBaker/MedQA-USMLE-4-options train split (board question explanations).
        Both are standard Parquet-format datasets, no loading scripts required.

Why these sources?
  Med42-8B's fine-tuning corpus mixes clinical vignettes + medical-board
  explanations. PubMedQA long_answers ≈ reference explanations; MedQA
  train split ≈ board-question walkthroughs. Together they approximate
  the distribution Med42 was tuned on.

Output:  ./outputs/med_c3_calib_512.jsonl
Runtime: ~2-3 minutes (download + sample)
"""

import os
import json
import random

OUTPUT_JSONL = "./outputs/med_c3_calib_512.jsonl"
N_SAMPLES    = 512
SEED         = 42

DOMAIN_TERMS = [
    "patient", "diagnosis", "treatment", "symptom", "syndrome",
    "clinical", "therapy", "pathophysiology", "differential",
    "dosage", "contraindication", "prognosis", "etiology",
]


def main():
    random.seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" Activity 3 — Medical C3 Corpus")
    print("=" * 60)

    from datasets import load_dataset

    texts = []

    # Source 1: PubMedQA long-form answers (qiaojin — standard Parquet, no loading script)
    print("[1/3] Loading qiaojin/PubMedQA (pqa_labeled)...")
    try:
        ds1 = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for row in ds1:
            t = (row.get("long_answer") or "").strip()
            if len(t) > 100:
                texts.append(t)
        print(f"  Got {len(texts)} texts from qiaojin/PubMedQA")
    except Exception as e:
        print(f"  qiaojin/PubMedQA unavailable ({e}) — skipping")

    # Source 2: MedQA USMLE train split (board question + explanation pairs)
    print("[2/3] Loading GBaker/MedQA-USMLE-4-options (train split)...")
    try:
        ds2 = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        for row in ds2:
            # Compose: question + all answer options as domain-rich medical text
            opts = " ".join(f"{k}: {v}" for k, v in row["options"].items())
            t = f"{row['question']} {opts}".strip()
            if len(t) > 100:
                texts.append(t)
        print(f"  Accumulated {len(texts)} texts total")
    except Exception as e:
        print(f"  MedQA train unavailable ({e}) — continuing with what we have")

    # Random sample 512 with fixed seed
    print(f"\n[3/3] Sampling {N_SAMPLES} (seed={SEED})...")
    if len(texts) < N_SAMPLES:
        print(f"  WARNING: only {len(texts)} texts available; using all")
        sampled = texts
    else:
        sampled = random.sample(texts, N_SAMPLES)

    # Domain-term coverage sanity check
    combined = " ".join(sampled).lower()
    print("\n  Domain term coverage:")
    for term in DOMAIN_TERMS:
        print(f"    {term:20s}: {combined.count(term):4d}")

    with open(OUTPUT_JSONL, "w") as f:
        for t in sampled:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"\n DONE: {OUTPUT_JSONL}  |  {len(sampled)} samples  |  seed={SEED}")


if __name__ == "__main__":
    main()
