"""
DACS Project — Activity 2B: INT4 AWQ, Self-Calibration (C2)
============================================================
Condition 2 (C2) — Self-calibration baseline from Williams et al. (NAACL 2025).
Identical to 2A AWQ except calib_data comes from generate_self_calib.py output
instead of WikiText-2. The only line that changes is where we load calibration data.

Run generate_self_calib.py BEFORE this script.

Input:    ./outputs/cybersec_analyst_merged_fp16/ + ./outputs/self_calib_samples_c2.jsonl
Output:   ./outputs/cyber_int4_awq_c2/  (~4 GB)
Runtime:  ~45 minutes on A10G
"""

import os
import json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
MERGED_MODEL    = "./outputs/cybersec_analyst_merged_fp16/"
CALIB_JSONL     = "./outputs/self_calib_samples_c2.jsonl"
OUTPUT_PATH     = "./outputs/cyber_int4_awq_c2/"
N_CALIB_AWQ     = 128    # match C1 sample count; change domain, not quantity
MAX_CALIB_CHARS = 512    # cap sequence length to ~128 tokens (matches WikiText-2 passage lengths)

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2B — INT4 AWQ  |  Calibration: C2 (Self-generated)")
    print("=" * 60)

    if not os.path.exists(CALIB_JSONL):
        print(f"ERROR: Self-calibration file not found: {CALIB_JSONL}")
        print("  Run generate_self_calib.py first.")
        raise FileNotFoundError(CALIB_JSONL)

    # Load self-calibration data
    print(f"[1/3] Loading self-calibration data from {CALIB_JSONL}...")
    with open(CALIB_JSONL) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    calib_texts = [t[:MAX_CALIB_CHARS] for t in calib_texts[:N_CALIB_AWQ]]
    print(f"  Using {len(calib_texts)} self-generated samples, each capped at {MAX_CALIB_CHARS} chars")
    print(f"  Sample preview: {calib_texts[0][:120]!r}")
    print(f"  Note: These are MODEL-GENERATED texts — domain-influenced but NOT SFT corpus.")

    # Load model
    print(f"\n[2/3] Loading merged FP16 model...")
    model = AutoAWQForCausalLM.from_pretrained(MERGED_MODEL, safetensors=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Quantize — ONLY change from C1: calib_data is self-generated texts
    print(f"\n[3/3] Running AWQ calibration + INT4 quantization (C2 data)...")
    model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=calib_texts)

    model.save_quantized(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    size_gb = sum(os.path.getsize(os.path.join(OUTPUT_PATH, f))
                  for f in os.listdir(OUTPUT_PATH)
                  if os.path.isfile(os.path.join(OUTPUT_PATH, f))) / 1e9

    print()
    print("=" * 60)
    print(f" Activity 2B AWQ (C2) COMPLETE  |  {OUTPUT_PATH}  |  {size_gb:.1f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
