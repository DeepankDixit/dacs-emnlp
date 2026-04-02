"""
DACS Project — Activity 2A: INT8 SmoothQuant, Generic Calibration (C1)
=======================================================================
Condition 1 (C1) — Standard industry baseline.
Calibrate SmoothQuant on C4: generic English web text.
SmoothQuant quantizes BOTH weights and activations to INT8 (W8A8).

What this does:
  1. Loads 512 samples from the C4 dataset (generic web text — SmoothQuant default)
  2. Identifies outlier channels in activations (channels with very large values)
  3. Computes migration factors m_i to shift outliers from activations → weights
  4. Quantizes both weights and activations to INT8

Input:    ./outputs/cybersec_analyst_merged_fp16/  (from merge_adapter.py)
Output:   ./outputs/cyber_int8_sq_c1/              (~8 GB)
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
OUTPUT_PATH  = "./outputs/cyber_int8_sq_c1/"
N_CALIB      = 512    # SmoothQuant default: more samples than AWQ (512 vs 128)
MAX_LENGTH   = 512    # truncate long sequences
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2A — INT8 SmoothQuant  |  Calibration: C1 (C4)")
    print("=" * 60)
    print(f"  Input model:  {MERGED_MODEL}")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Calib data:   C4 dataset ({N_CALIB} samples — GENERIC web text)")
    print(f"  Format:       INT8 W8A8 (both weights AND activations quantized)")
    print()

    # Step 1: Load C4 calibration data (streaming to avoid downloading full dataset)
    print("[1/4] Loading C4 calibration data (streaming)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    calib_texts = []
    for sample in dataset:
        text = sample["text"].strip()
        if len(text) > 100:
            calib_texts.append(text[:MAX_LENGTH])
        if len(calib_texts) >= N_CALIB:
            break
    print(f"  Loaded {len(calib_texts)} calibration samples from C4 (generic English web)")
    print(f"  Sample preview: {calib_texts[0][:100]!r}")

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
    print(f"  Model loaded on: {next(model.parameters()).device}")

    # Step 3: Tokenize calibration data
    print(f"\n[3/4] Tokenizing calibration data...")
    inputs = tokenizer(
        calib_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    # Move to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Tokenized shape: {inputs['input_ids'].shape}")

    # Step 4: Apply SmoothQuant via ModelOpt
    print(f"\n[4/4] Running SmoothQuant calibration and INT8 quantization...")
    print("  SmoothQuant steps:")
    print("   a) Forward pass on 512 C4 samples — observe activation distributions")
    print("   b) Identify outlier channels: channels where max(|A_L[:,i]|) >> median")
    print("   c) Compute migration factor m_i per channel")
    print("   d) Rescale: W' = W × m_i,  A' = A / m_i  (mathematically equivalent)")
    print("   e) Both W' and A' now have smaller ranges → quantize both to INT8")

    try:
        import modelopt.torch.quantization as mtq

        # INT8 SmoothQuant config from ModelOpt
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG

        def forward_loop(model):
            """Run calibration forward passes to collect activation statistics."""
            with torch.no_grad():
                model(**inputs)

        print("  Using ModelOpt backend...")
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    except ImportError:
        print("  ModelOpt not available. Trying smoothquant fallback...")
        try:
            from smoothquant.smooth import smooth_lm
            smooth_lm(model, inputs, alpha=0.5)
            print("  SmoothQuant migration applied (alpha=0.5)")
        except ImportError:
            print("  ERROR: Neither modelopt nor smoothquant is installed.")
            print("  Run: pip install nvidia-modelopt[torch] --break-system-packages")
            sys.exit(1)

    # Save quantized model
    print(f"\n  Saving INT8 SmoothQuant C1 model to {OUTPUT_PATH}...")
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
    print(" Activity 2A SmoothQuant (C1) COMPLETE")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_gb:.1f} GB  (expected: ~8 GB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
