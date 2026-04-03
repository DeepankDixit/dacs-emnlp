"""
DACS Project — Activity 2B: INT8 SmoothQuant, Self-Calibration (C2)
====================================================================
Identical to activity2a_sq_c1.py except calibration data is self-generated.
Run generate_self_calib.py BEFORE this script.

Input:    ./outputs/cybersec_analyst_merged_fp16/ + ./outputs/self_calib_samples_c2.jsonl
Output:   ./outputs/cyber_int8_sq_c2/  (~8 GB)
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
OUTPUT_PATH  = "./outputs/cyber_int8_sq_c2/"
MAX_LENGTH   = 512
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2B — INT8 SmoothQuant  |  Calibration: C2 (Self-generated)")
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

    print(f"\n[4/4] Running SmoothQuant (C2 self-generated calibration data)...")
    backend_used = None

    try:
        import modelopt.torch.quantization as mtq
        _ = mtq.INT8_SMOOTHQUANT_CFG
        def forward_loop(m):
            with torch.no_grad():
                m(**inputs)
        print("  Using ModelOpt backend...")
        mtq.quantize(model, config=mtq.INT8_SMOOTHQUANT_CFG, forward_loop=forward_loop)
        backend_used = "modelopt"
    except (ImportError, AttributeError) as e:
        print(f"  ModelOpt not usable ({e}). Trying smoothquant fallback...")

    if backend_used is None:
        try:
            from smoothquant.calibration import get_act_scales
            from smoothquant.smooth import smooth_lm
            act_scales = get_act_scales(model, tokenizer, calib_texts,
                                        num_samples=len(calib_texts), seq_len=512)
            smooth_lm(model, act_scales, alpha=0.5)
            backend_used = "smoothquant"
            print("  SmoothQuant migration applied (alpha=0.5).")
        except (ImportError, Exception) as e:
            print(f"  smoothquant fallback failed: {e}")

    if backend_used is None:
        print(f"\n  ERROR: Could not apply SmoothQuant. torch version: {torch.__version__}")
        print("  Fix: pip install 'torch>=2.6.0' --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    size_gb = sum(os.path.getsize(os.path.join(OUTPUT_PATH, f))
                  for f in os.listdir(OUTPUT_PATH)
                  if os.path.isfile(os.path.join(OUTPUT_PATH, f))) / 1e9

    print()
    print("=" * 60)
    print(f" Activity 2B SmoothQuant (C2) COMPLETE  |  {OUTPUT_PATH}  |  {size_gb:.1f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
