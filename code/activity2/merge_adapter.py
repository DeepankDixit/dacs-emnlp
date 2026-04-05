"""
DACS Project — Activity 2, Phase 1: Re-Merge the LoRA Adapter
==============================================================
The merged FP16 model from LoraForge Activity 1 was NOT persisted to disk.
This script recreates it. Run this ONCE before any of the 2A/2B/2C scripts.

What this does:
  1. Downloads Llama-3.1-8B-Instruct base model from HuggingFace
  2. Loads your cybersecurity LoRA adapter (from LoraForge Activity 0 outputs)
  3. Merges: W_merged = W_base + (alpha / r) * B @ A  for every layer
  4. Saves the merged FP16 model to ./outputs/cybersec_analyst_merged_fp16/

Input:    Base model (HuggingFace) + LoRA adapter checkpoint (from LoraForge)
Output:   ./outputs/cybersec_analyst_merged_fp16/  (~15-16 GB on disk)
Runtime:  ~15 minutes on CPU (do NOT merge on GPU — wastes VRAM, no speed gain)
Cost:     ~$0.19 on A10G at $0.75/h
"""

import os
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration — update these paths for your Lambda Cloud setup
# ---------------------------------------------------------------------------
BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"

# Path to the LoRA adapter saved by LoraForge Activity 0
# On Lambda Cloud, this is wherever you copied your LoraForge outputs
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    "../LoraForge/outputs/cybersec_analyst_lora/"
)

OUTPUT_PATH  = "./outputs/cybersec_analyst_merged_fp16/"

# Your HuggingFace token (needed for Llama gated model)
# Set via: export HF_TOKEN="hf_your_token_here"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("  Run: export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)

    if not os.path.isdir(ADAPTER_PATH):
        print(f"ERROR: Adapter path not found: {ADAPTER_PATH}")
        print("  Update ADAPTER_PATH in this script, or set via:")
        print("  export ADAPTER_PATH=/path/to/your/lora/adapter")
        sys.exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print(" DACS Activity 2 — Re-Merge LoRA Adapter")
    print("=" * 60)
    print(f"  Base model:   {BASE_MODEL}")
    print(f"  Adapter path: {ADAPTER_PATH}")
    print(f"  Output path:  {OUTPUT_PATH}")
    print()

    # Step 1: Load base model on CPU
    # Using device_map='cpu' is intentional — merging on GPU wastes VRAM
    print("[1/4] Loading base model on CPU (this downloads ~16 GB if not cached)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )
    print(f"  Base model loaded: {sum(p.numel() for p in base_model.parameters())/1e9:.2f}B parameters")

    # Step 2: Load LoRA adapter
    print(f"\n[2/4] Loading LoRA adapter from {ADAPTER_PATH}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Show adapter config for verification
    try:
        config = peft_model.peft_config["default"]
        print(f"  Adapter type:  {config.peft_type}")
        print(f"  Rank (r):      {config.r}")
        print(f"  Alpha:         {config.lora_alpha}")
        print(f"  Dropout:       {config.lora_dropout}")
        print(f"  Target modules:{config.target_modules}")
    except Exception:
        pass

    # Step 3: Merge weights
    # W_merged = W_base + (alpha / r) * B @ A for every targeted layer
    print("\n[3/4] Merging weights: W_merged = W_base + (alpha/r) * B @ A ...")
    print("  This takes ~15 minutes on CPU. No progress bar — just wait.")
    merged_model = peft_model.merge_and_unload()
    print("  Merge complete.")

    # Step 4: Save merged model
    print(f"\n[4/4] Saving merged FP16 model to {OUTPUT_PATH}...")
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
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
    print(" Merge COMPLETE")
    print(f"  Output:      {OUTPUT_PATH}")
    print(f"  Size on disk: {size_gb:.1f} GB  (expected: 15-16 GB)")
    print()
    print("  Next step: Run Activity 2A quantization scripts.")
    print("  Recommended order: 2A (C1) → 2C (DACS) → 2B (self-cal)")
    print("=" * 60)


if __name__ == "__main__":
    main()
