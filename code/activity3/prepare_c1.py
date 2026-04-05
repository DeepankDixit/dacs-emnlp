"""
DACS Activity 3 — Build Generic C1 Calibration Corpus
======================================================
512 WikiText-2 passages, used identically for both domains.

This is the "generic" baseline calibration used by every paper (AWQ original,
SmoothQuant original, ModelOpt FP8 default). Activity 3 keeps C1 identical
across med/code to isolate the domain effect in C3.

Output:  ./outputs/c1_generic_calib_512.jsonl
Runtime: ~30 seconds
"""

import os
import json
import random

OUTPUT_JSONL = "./outputs/c1_generic_calib_512.jsonl"
N_SAMPLES    = 512
SEED         = 42
MIN_CHARS    = 100


def main():
    random.seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" Activity 3 — C1 Generic Calibration (WikiText-2)")
    print("=" * 60)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > MIN_CHARS]
    print(f"  Loaded {len(texts)} passages with >{MIN_CHARS} chars")

    sampled = random.sample(texts, min(N_SAMPLES, len(texts)))

    with open(OUTPUT_JSONL, "w") as f:
        for t in sampled:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"\n DONE: {OUTPUT_JSONL}  |  {len(sampled)} samples  |  seed={SEED}")


if __name__ == "__main__":
    main()
