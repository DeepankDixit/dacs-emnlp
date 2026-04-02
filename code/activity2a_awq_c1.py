"""
DACS Project — Activity 2A: INT4 AWQ, Generic Calibration (C1)
==============================================================
Condition 1 (C1) — Standard industry baseline.
Calibrate AWQ on WikiText-2: generic English text with no cybersecurity content.
This is "the problem" your paper is solving — quantization calibrated on the wrong domain.

What this does:
  1. Loads 128 samples from WikiText-2 (generic English Wikipedia text)
  2. Runs AWQ calibration: forward pass → observe activation distributions
     → identify high-activation channels → compute per-channel protection scales
  3. Quantizes all linear layers to INT4 with group_size=128
  4. Saves the quantized model

Input:    ./outputs/cybersec_analyst_merged_fp16/  (from merge_adapter.py)
Output:   ./outputs/cyber_int4_awq_c1/             (~4 GB)
Runtime:  ~45 minutes on A10G
Cost:     ~$0.56 on A10G at $0.75/h
"""

import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
OUTPUT_PATH  = "./outputs/cyber_int4_awq_c1/"
N_CALIB      = 128    # AWQ default — enough to profile activation distributions
MIN_LEN      = 50     # skip very short texts (less than 50 chars)

# AWQ quantization config
QUANT_CONFIG = {
    "zero_point": True,      # asymmetric quantization (slightly better accuracy)
    "q_group_size": 128,     # per-group quantization: 128 weights share one scale
    "w_bit": 4,              # 4-bit weights (INT4)
    "version": "GEMM",       # GEMM kernel (compatible with most hardware)
}
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    print("=" * 60)
    print(" Activity 2A — INT4 AWQ  |  Calibration: C1 (WikiText-2)")
    print("=" * 60)
    print(f"  Input model:  {MERGED_MODEL}")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Calib data:   WikiText-2 ({N_CALIB} samples — GENERIC English)")
    print(f"  Format:       INT4, group_size={QUANT_CONFIG['q_group_size']}")
    print()

    # Step 1: Load calibration data
    print("[1/3] Loading WikiText-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [
        t for t in dataset["text"]
        if len(t.strip()) > MIN_LEN
    ][:N_CALIB]
    print(f"  Loaded {len(calib_texts)} calibration samples")
    print(f"  Sample preview: {calib_texts[0][:100]!r}")
    print(f"  Note: These are generic English texts — NO cybersecurity content.")
    print(f"  This is the C1 baseline (standard industry practice).")

    # Step 2: Load model for AWQ quantization
    print(f"\n[2/3] Loading merged FP16 model for AWQ quantization...")
    print("  (Uses GPU memory — ~16 GB VRAM required)")
    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_MODEL,
        safetensors=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Run AWQ calibration + quantization
    print(f"\n[3/3] Running AWQ calibration and INT4 quantization...")
    print("  AWQ steps:")
    print("   a) Forward pass on 128 WikiText-2 samples (activation hooks active)")
    print("   b) Per-channel: compute α_i = max(|A_L[:, i]|) for each channel")
    print("   c) Identify high-α channels (important channels) → protect them")
    print("   d) Rescale weights: W_rescaled = W × α_i  (move variance to weights)")
    print("   e) Quantize W_rescaled to INT4 with group_size=128")
    print("  This takes ~45 minutes on A10G...")

    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=calib_texts,
    )

    # Save quantized model
    print(f"\n  Saving INT4 AWQ C1 model to {OUTPUT_PATH}...")
    model.save_quantized(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Verify output size
    total_bytes = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in os.listdir(OUTPUT_PATH)
        if os.path.isfile(os.path.join(OUTPUT_PATH, f))
    )
    size_gb = total_bytes / 1e9

    print()
    print("=" * 60)
    print(" Activity 2A (C1) COMPLETE")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_gb:.1f} GB  (expected: ~4 GB)")
    print()
    print("  Next steps:")
    print("  1. Run activity2c_awq_c3.py (DACS) to get early comparison signal")
    print("  2. Then evaluate both C1 and C3 on CyberSecEval to validate hypothesis")
    print("=" * 60)


if __name__ == "__main__":
    main()
