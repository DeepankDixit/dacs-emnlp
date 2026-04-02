"""
DACS Project — Activity 2B: Generate Self-Calibration Data (C2)
================================================================
Self-calibration (C2) from Williams et al. (NAACL 2025).
Ask the model to generate its own calibration text using domain seed prompts.
The generated text is domain-influenced (it will contain cybersecurity vocabulary)
but is NOT the original SFT training data — this is the key difference from C3 (DACS).

What this does:
  1. Loads the merged FP16 model
  2. Runs it on 8 cybersecurity seed prompts with temperature sampling
  3. Generates 128 samples (16 per seed prompt)
  4. Saves to ./outputs/self_calib_samples_c2.jsonl

Input:    ./outputs/cybersec_analyst_merged_fp16/  (from merge_adapter.py)
Output:   ./outputs/self_calib_samples_c2.jsonl
Runtime:  ~20 minutes on A10G (generation is slower than calibration)
Cost:     ~$0.25 on A10G at $0.75/h
"""

import os
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
OUTPUT_JSONL = "./outputs/self_calib_samples_c2.jsonl"
N_SAMPLES    = 128
SEED         = 42   # fixed for reproducibility

# Domain-relevant seed prompts — guide generation toward cybersecurity text
# These are the same distribution the model was fine-tuned on
SEED_PROMPTS = [
    "Explain the following cybersecurity vulnerability:",
    "Describe the attack vector for CVE-",
    "The MITRE ATT&CK technique T1059 involves",
    "A buffer overflow vulnerability occurs when",
    "To exploit this vulnerability, an attacker would",
    "The recommended mitigation for this security issue is",
    "In penetration testing, this technique is used to",
    "This malware strain uses the following persistence mechanism:",
]
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" Activity 2B — Generate Self-Calibration Data (C2)")
    print("=" * 60)
    print(f"  Model:     {MERGED_MODEL}")
    print(f"  Output:    {OUTPUT_JSONL}")
    print(f"  N samples: {N_SAMPLES} (seed={SEED})")
    print()
    print("  NOTE: Self-calibration uses model-generated text, NOT the SFT corpus.")
    print("  Generated text is domain-influenced but may not reproduce the exact")
    print("  activation patterns learned during fine-tuning (which is C3/DACS).")
    print()

    # Step 1: Load model
    print("[1/3] Loading merged FP16 model for text generation...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  Model loaded on: {next(model.parameters()).device}")

    # Step 2: Generate samples
    print(f"\n[2/3] Generating {N_SAMPLES} self-calibration samples...")
    print("  Each sample is generated from a random seed prompt")
    print("  with temperature=0.8 for diversity.\n")

    samples = []
    n_per_prompt = N_SAMPLES // len(SEED_PROMPTS)

    for prompt_idx, prompt in enumerate(SEED_PROMPTS):
        for i in range(n_per_prompt):
            output = gen_pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=True,
            )
            generated_text = output[0]["generated_text"]
            samples.append({
                "text": generated_text,
                "seed_prompt": prompt,
                "sample_idx": len(samples),
            })

        completed = len(samples)
        print(f"  Prompt {prompt_idx+1}/{len(SEED_PROMPTS)}: {n_per_prompt} samples done "
              f"({completed}/{N_SAMPLES} total)")
        print(f"    Preview: {samples[-1]['text'][:120]!r}")

    # Handle remainder if N_SAMPLES not evenly divisible
    while len(samples) < N_SAMPLES:
        prompt = random.choice(SEED_PROMPTS)
        output = gen_pipeline(
            prompt, max_new_tokens=200, temperature=0.8,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
        samples.append({"text": output[0]["generated_text"], "seed_prompt": prompt})

    # Step 3: Save
    print(f"\n[3/3] Saving {len(samples)} samples to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Inspect domain coverage
    domain_terms = [
        "CVE", "vulnerability", "exploit", "buffer", "injection",
        "MITRE", "attack", "malware", "threat", "security",
    ]
    all_text = " ".join(s["text"] for s in samples).lower()
    found = {t: all_text.count(t.lower()) for t in domain_terms}
    print()
    print("  Domain term frequency in generated samples:")
    for term, count in sorted(found.items(), key=lambda x: -x[1]):
        print(f"    {term:20s}: {count} occurrences")

    print()
    print("=" * 60)
    print(" Self-calibration data (C2) COMPLETE")
    print(f"  Output: {OUTPUT_JSONL}")
    print(f"  {len(samples)} samples saved, seed={SEED}")
    print()
    print("  Next: Run activity2b_awq_c2.py, activity2b_sq_c2.py, activity2b_fp8_c2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
