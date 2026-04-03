"""
DACS Project — Activity 2B: FP8, Self-Calibration (C2)
=======================================================
Identical to activity2a_fp8_c1.py except calibration data is self-generated.
Run generate_self_calib.py BEFORE this script.

Input:    ./outputs/cybersec_analyst_merged_fp16/ + ./outputs/self_calib_samples_c2.jsonl
Output:   ./outputs/cyber_fp8_c2/  (~8 GB)
Runtime:  ~30 minutes on A10G
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
CALIB_JSONL  = "./outputs/self_calib_samples_c2.jsonl"
OUTPUT_PATH  = "./outputs/cyber_fp8_c2/"
MAX_LENGTH   = 512
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2B — FP8  |  Calibration: C2 (Self-generated)")
    print("=" * 60)

    if not os.path.exists(CALIB_JSONL):
        print(f"ERROR: {CALIB_JSONL} not found. Run generate_self_calib.py first.")
        raise FileNotFoundError(CALIB_JSONL)

    print(f"[1/4] Loading self-calibration data...")
    with open(CALIB_JSONL) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    print(f"  Loaded {len(calib_texts)} self-generated samples")

    print(f"\n[2/4] Loading merged FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"\n[3/4] Tokenizing calibration data...")
    inputs = tokenizer(
        calib_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\n[4/4] Running FP8 quantization (C2 self-generated calibration data)...")
    try:
        import modelopt.torch.quantization as mtq
        def forward_loop(m):
            with torch.no_grad():
                m(**inputs)
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
    except ImportError:
        print("ERROR: Install nvidia-modelopt[torch]")
        sys.exit(1)

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    size_gb = sum(os.path.getsize(os.path.join(OUTPUT_PATH, f))
                  for f in os.listdir(OUTPUT_PATH)
                  if os.path.isfile(os.path.join(OUTPUT_PATH, f))) / 1e9

    print()
    print("=" * 60)
    print(f" Activity 2B FP8 (C2) COMPLETE  |  {OUTPUT_PATH}  |  {size_gb:.1f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
