"""
DACS Activity 3 — Build Medical Domain C3 Calibration Corpus
=============================================================
Sample 512 medical assistant-turn texts from a public medical instruction
dataset aligned with the Llama3-Med42-8B fine-tune distribution.

Source: bigbio/pubmed_qa (long_answers) + medalpaca/medical_meadow_medqa
        (instruction/output pairs). Both are permissive-license, publicly
        available on HuggingFace.

Why these sources?
  Med42-8B's fine-tuning corpus mixes clinical vignettes + medical-board
  explanations. PubMedQA long_answers ≈ reference explanations; medical
  meadow MedQA ≈ board-question walkthroughs. Together they approximate
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

    # Source 1: PubMedQA long-form answers
    print("[1/3] Loading pubmed_qa (pqa_labeled)...")
    ds1 = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_source",
                       split="train", trust_remote_code=True)
    for row in ds1:
        t = (row.get("LONG_ANSWER") or "").strip()
        if len(t) > 100:
            texts.append(t)
    print(f"  Got {len(texts)} texts from pubmed_qa")

    # Source 2: Medical Meadow MedQA explanations
    print("[2/3] Loading medalpaca/medical_meadow_medqa...")
    try:
        ds2 = load_dataset("medalpaca/medical_meadow_medqa",
                           split="train", trust_remote_code=True)
        for row in ds2:
            t = (row.get("output") or "").strip()
            if len(t) > 100:
                texts.append(t)
        print(f"  Accumulated {len(texts)} texts total")
    except Exception as e:
        print(f"  medalpaca dataset unavailable ({e}) — using pubmed_qa only")

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
