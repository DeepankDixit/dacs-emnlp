"""
DACS Project — Activity 2A: FP8, Generic Calibration (C1)
==========================================================
Condition 1 (C1) — Standard industry baseline.
FP8 quantization using ModelOpt, calibrated on WikiText-2.
FP8 uses a floating-point 8-bit format (E4M3: 4 exponent bits, 3 mantissa bits)
rather than integers — so it handles outliers more gracefully than INT8.

What this does:
  1. Loads 128 samples from WikiText-2 (generic English, same as AWQ C1)
  2. Runs calibration forward pass to determine per-layer FP8 scale factors
  3. Quantizes weights to FP8 E4M3 format
  4. Saves the quantized model

Input:    ./outputs/cybersec_analyst_merged_fp16/  (from merge_adapter.py)
Output:   ./outputs/cyber_fp8_c1/                  (~8 GB — same byte count as INT8)
Runtime:  ~30 minutes on A10G
Cost:     ~$0.38 on A10G at $0.75/h
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MERGED_MODEL = "./outputs/cybersec_analyst_merged_fp16/"
OUTPUT_PATH  = "./outputs/cyber_fp8_c1/"
N_CALIB      = 128
MAX_LENGTH   = 512
MIN_LEN      = 50
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2A — FP8  |  Calibration: C1 (WikiText-2)")
    print("=" * 60)
    print(f"  Input model:  {MERGED_MODEL}")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Calib data:   WikiText-2 ({N_CALIB} samples — GENERIC English)")
    print(f"  Format:       FP8 E4M3 (weights) — floating-point 8-bit")
    print()

    # Step 1: Load WikiText-2 calibration data
    print("[1/4] Loading WikiText-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [
        t for t in dataset["text"]
        if len(t.strip()) > MIN_LEN
    ][:N_CALIB]
    print(f"  Loaded {len(calib_texts)} calibration samples")

    # Step 2: Load model and tokenizer
    print(f"\n[2/4] Loading merged FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Step 3: Tokenize calibration data
    print(f"\n[3/4] Tokenizing calibration data...")
    inputs = tokenizer(
        calib_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Tokenized shape: {inputs['input_ids'].shape}")

    # Step 4: Apply FP8 quantization via ModelOpt
    print(f"\n[4/4] Running FP8 calibration and quantization...")
    print("  FP8 E4M3 notes:")
    print("   - FP8 has 1 sign bit + 4 exponent bits + 3 mantissa bits")
    print("   - Range: approximately [-240, +240]")
    print("   - Unlike INT8, FP8 has built-in dynamic range via the exponent")
    print("   - Outlier activations are less problematic than with INT8")
    print("   - Calibration still needed to decide E4M3 vs E5M2 per layer")

    try:
        import modelopt.torch.quantization as mtq

        # FP8 default config from ModelOpt
        quant_cfg = mtq.FP8_DEFAULT_CFG

        def forward_loop(model):
            """Run calibration forward passes to collect activation statistics."""
            with torch.no_grad():
                model(**inputs)

        print("  Using ModelOpt FP8 backend...")
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    except ImportError:
        print("  ERROR: ModelOpt not installed.")
        print("  Run: pip install nvidia-modelopt[torch]")
        sys.exit(1)

    # Save quantized model
    print(f"\n  Saving FP8 C1 model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH)
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
    print(" Activity 2A FP8 (C1) COMPLETE")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_gb:.1f} GB  (expected: ~8 GB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
