"""
DACS Activity 3 — Generate C2 Self-Calibration Corpus (per domain)
===================================================================
Sample 512 model-generated continuations from the FP16 checkpoint, using
domain-specific seed prompts. Mirrors Williams et al. (NAACL 2025) self-cal
protocol used in Activity 2.

Usage:
    python generate_self_calib_act3.py --domain med
    python generate_self_calib_act3.py --domain code

Input:   ./outputs/{domain}_fp16/
Output:  ./outputs/{domain}_c2_selfgen_512.jsonl
Runtime: ~15-20 minutes on A10G per domain
"""

import os
import json
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DOMAINS

N_SAMPLES     = 512
MAX_NEW_TOKENS = 128
SEED          = 42


def main(domain: str):
    random.seed(SEED)
    torch.manual_seed(SEED)
    cfg       = DOMAINS[domain]
    fp16_path = cfg["local_fp16"]
    out_path  = cfg["c2_selfgen"]
    seeds     = cfg["seed_prompts"]

    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(f" Activity 3 — C2 Self-Cal Generation  |  domain={domain}")
    print("=" * 60)

    print(f"[1/3] Loading FP16 model {fp16_path}...")
    tok   = AutoTokenizer.from_pretrained(fp16_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        fp16_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"\n[2/3] Generating {N_SAMPLES} continuations...")
    texts = []
    for i in range(N_SAMPLES):
        prompt = random.choice(seeds)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, top_p=0.9, temperature=0.8,
                pad_token_id=tok.eos_token_id,
            )
        generated = tok.decode(out[0], skip_special_tokens=True)
        texts.append(generated.strip())
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{N_SAMPLES}")

    print(f"\n[3/3] Saving to {out_path}...")
    with open(out_path, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

    print(f"\n DONE: {out_path}  |  {len(texts)} samples  |  seed={SEED}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["med", "code"])
    args = p.parse_args()
    main(args.domain)
