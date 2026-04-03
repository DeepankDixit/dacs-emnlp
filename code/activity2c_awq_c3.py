"""
DACS Project — Activity 2C: INT4 AWQ, DACS Calibration (C3) — Proposed Method
===============================================================================
This is the core experiment of the paper.
AWQ calibrated on the original SFT training corpus (cybersecurity domain data).
The only code difference from C1: calib_data = DACS samples instead of WikiText-2.

Run prepare_dacs_calib.py BEFORE this script.

Input:    ./outputs/cybersec_analyst_merged_fp16/ + ./outputs/dacs_calib_512.jsonl
Output:   ./outputs/cyber_int4_awq_c3/  (~4 GB)
Runtime:  ~45 minutes on A10G
Cost:     ~$0.56 on A10G at $0.75/h
"""

import os
import json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
CALIB_JSONL  = "./outputs/dacs_calib_512.jsonl"
OUTPUT_PATH  = "./outputs/cyber_int4_awq_c3/"
# AWQ holds ALL calibration activations for each layer simultaneously in memory
# during scale search. 512 samples × Llama hidden dims exceeds the ~6 GB VRAM
# headroom. 128 is the standard AWQ calibration count (used in the original
# paper and in our C1 baseline) — we change calibration DOMAIN, not quantity.
N_CALIB_AWQ  = 128

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
    print(" Activity 2C — INT4 AWQ  |  Calibration: C3 (DACS — PROPOSED)")
    print("=" * 60)

    if not os.path.exists(CALIB_JSONL):
        print(f"ERROR: DACS calibration file not found: {CALIB_JSONL}")
        print("  Run prepare_dacs_calib.py first.")
        raise FileNotFoundError(CALIB_JSONL)

    print(f"[1/3] Loading DACS calibration data from {CALIB_JSONL}...")
    with open(CALIB_JSONL) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    calib_texts = calib_texts[:N_CALIB_AWQ]   # AWQ needs same count as C1 (128)
    print(f"  Loaded {len(calib_texts)} DACS calibration samples (truncated to {N_CALIB_AWQ} for AWQ)")
    print(f"  Sample preview: {calib_texts[0][:120]!r}")
    print(f"  Note: These are ASSISTANT-TURN texts from the SFT training corpus.")
    print(f"  They contain the exact domain vocabulary the model was fine-tuned on.")
    print(f"  AWQ will protect the channels that activate for these domain tokens.")

    print(f"\n[2/3] Loading merged FP16 model...")
    model = AutoAWQForCausalLM.from_pretrained(MERGED_MODEL, safetensors=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[3/3] Running AWQ calibration + INT4 quantization (DACS C3 data)...")
    print("  Key difference from C1: AWQ will observe activation patterns for")
    print("  cybersecurity tokens → protect the right channels for domain inference.")

    # ONLY change from C1: calib_data is DACS corpus samples
    model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=calib_texts)

    model.save_quantized(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    size_gb = sum(os.path.getsize(os.path.join(OUTPUT_PATH, f))
                  for f in os.listdir(OUTPUT_PATH)
                  if os.path.isfile(os.path.join(OUTPUT_PATH, f))) / 1e9

    print()
    print("=" * 60)
    print(f" Activity 2C AWQ (C3/DACS) COMPLETE  |  {OUTPUT_PATH}  |  {size_gb:.1f} GB")
    print()
    print("  NEXT: Evaluate C1 and C3 on CyberSecEval immediately to get early signal!")
    print("  python evaluate_all.py --models c1 c3 --benchmarks cyberseceval")
    print("=" * 60)

if __name__ == "__main__":
    main()
