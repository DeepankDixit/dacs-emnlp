"""
DACS Project — Activity 2C: FP8, DACS Calibration (C3)
=======================================================
Run prepare_dacs_calib.py BEFORE this script.

Input:    ./outputs/cybersec_analyst_merged_fp16/ + ./outputs/dacs_calib_512.jsonl
Output:   ./outputs/cyber_fp8_c3/  (~8 GB)
Runtime:  ~30 minutes on A10G
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
CALIB_JSONL  = "./outputs/dacs_calib_512.jsonl"
OUTPUT_PATH  = "./outputs/cyber_fp8_c3/"
MAX_LENGTH   = 512
CALIB_BATCH  = 4   # mini-batch size — avoids OOM from batch×seq_len² activations
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2C — FP8  |  Calibration: C3 (DACS — PROPOSED)")
    print("=" * 60)

    if not os.path.exists(CALIB_JSONL):
        print(f"ERROR: {CALIB_JSONL} not found. Run prepare_dacs_calib.py first.")
        raise FileNotFoundError(CALIB_JSONL)

    print(f"[1/4] Loading DACS calibration data...")
    with open(CALIB_JSONL) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    print(f"  Loaded {len(calib_texts)} DACS samples")

    print(f"\n[2/4] Loading merged FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"\n[3/4] Tokenizing DACS calibration data...")
    inputs = tokenizer(
        calib_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\n[4/4] Running FP8 quantization (C3 DACS calibration data)...")
    try:
        import modelopt.torch.quantization as mtq
        def forward_loop(m):
            n = inputs["input_ids"].shape[0]
            with torch.no_grad():
                for start in range(0, n, CALIB_BATCH):
                    batch = {k: v[start:start+CALIB_BATCH] for k, v in inputs.items()}
                    m(**batch)
        mtq.quantize(model, config=mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
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
    print(f" Activity 2C FP8 (C3/DACS) COMPLETE  |  {OUTPUT_PATH}  |  {size_gb:.1f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
