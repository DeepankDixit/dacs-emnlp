"""
DACS Project — Activity 2A: INT8 SmoothQuant, Generic Calibration (C1)
       --- RE-RUN ON WIKITEXT-2 TO MATCH AWQ/FP8 C1 ---
=======================================================================
Re-run of sq_c1.py with WikiText-2 instead of C4, so that C1 is a uniform
corpus across all three quantization formats (AWQ, SQ, FP8).

Background:
  - awq_c1.py and fp8_c1.py both calibrate on WikiText-2 (the AWQ paper's
    reference corpus).
  - sq_c1.py was written following the original SmoothQuant paper, which
    used C4. This created a corpus mismatch across the C1 column of
    Table 1.
  - This script re-runs SmoothQuant C1 on WikiText-2 to eliminate the
    mismatch. Resulting model goes to a new directory so the original
    C4-calibrated model is preserved for comparison.

What changes vs sq_c1.py:
  - Calibration data: C4 (streaming)  →  WikiText-2 (load_dataset)
  - Output path: cyber_int8_sq_c1/    →  cyber_int8_sq_c1_wikitext/
  - Calibration sample selection: matches awq_c1.py exactly (same filter,
    same first-N slicing) so AWQ-C1 and SQ-C1 see literally the same
    512 calibration sequences.
  - Everything else (SmoothQuant alpha=0.5, INT8 W8A8, batch size, model
    path, ModelOpt backend with smoothquant fallback) is identical.

Input:    ./outputs/cybersec_analyst_merged_fp16/  (from merge_adapter.py)
Output:   ./outputs/cyber_int8_sq_c1_wikitext/     (~8 GB)
Runtime:  ~30 minutes on Lambda A10
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
OUTPUT_PATH  = "./outputs/cyber_int8_sq_c1_wikitext/"
N_CALIB      = 512    # match sq_c1.py
MAX_LENGTH   = 512    # match sq_c1.py
CALIB_BATCH  = 8      # match sq_c1.py — A10 (24 GB) OOMs at 512-at-once
MIN_LEN      = 100    # filter threshold from awq_c1.py (consistent across formats)
# ---------------------------------------------------------------------------

# Reduce VRAM fragmentation — must be set before any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(" Activity 2A — INT8 SmoothQuant  |  Calibration: C1 (WikiText-2)")
    print(" RE-RUN to fix C1 corpus mismatch (was C4, now WikiText-2)")
    print("=" * 60)
    print(f"  Input model:  {MERGED_MODEL}")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Calib data:   WikiText-2 ({N_CALIB} samples — same as AWQ/FP8 C1)")
    print(f"  Format:       INT8 W8A8 (both weights AND activations quantized)")
    print()

    # Step 1: Load WikiText-2 calibration data (mirrors awq_c1.py exactly)
    print("[1/4] Loading WikiText-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [
        t for t in dataset["text"]
        if len(t.strip()) > MIN_LEN
    ][:N_CALIB]
    # Truncate long sequences to MAX_LENGTH chars (rough proxy for token budget;
    # tokeniser will truncate properly downstream)
    calib_texts = [t[:MAX_LENGTH * 4] for t in calib_texts]  # ~4 chars per token
    print(f"  Loaded {len(calib_texts)} calibration samples from WikiText-2")
    print(f"  Sample preview: {calib_texts[0][:100]!r}")
    print(f"  Note: identical sample selection to awq_c1.py and fp8_c1.py.")

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
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Tokenized shape: {inputs['input_ids'].shape}")

    # Step 4: Apply SmoothQuant via ModelOpt (with smoothquant fallback)
    print(f"\n[4/4] Running SmoothQuant calibration and INT8 quantization...")
    print("  SmoothQuant alpha=0.5 (identical to sq_c1.py — only calibration corpus differs)")

    backend_used = None

    # ── Try ModelOpt ──────────────────────────────────────────────────────────
    try:
        import modelopt.torch.quantization as mtq
        _ = mtq.INT8_SMOOTHQUANT_CFG

        def forward_loop(model):
            n = inputs["input_ids"].shape[0]
            with torch.no_grad():
                for start in range(0, n, CALIB_BATCH):
                    batch = {k: v[start:start+CALIB_BATCH] for k, v in inputs.items()}
                    model(**batch)

        print("  Using ModelOpt backend (nvidia-modelopt)...")
        print(f"  Calibration: {N_CALIB} samples in mini-batches of {CALIB_BATCH}")
        mtq.quantize(model, config=mtq.INT8_SMOOTHQUANT_CFG, forward_loop=forward_loop)
        backend_used = "modelopt"
        print("  ModelOpt INT8 W8A8 quantization applied.")

    except (ImportError, AttributeError) as e:
        print(f"  ModelOpt not usable ({e}). Trying smoothquant fallback...")

    # ── Try smoothquant (MIT-Han-Lab) ─────────────────────────────────────────
    if backend_used is None:
        try:
            from smoothquant.calibration import get_act_scales
            from smoothquant.smooth import smooth_lm

            print("  Using smoothquant backend (MIT-Han-Lab)...")
            act_scales = get_act_scales(
                model, tokenizer, calib_texts,
                num_samples=N_CALIB, seq_len=MAX_LENGTH,
            )
            smooth_lm(model, act_scales, alpha=0.5)
            backend_used = "smoothquant"
            print("  SmoothQuant migration applied (alpha=0.5).")

        except (ImportError, Exception) as e:
            print(f"  smoothquant fallback failed: {e}")

    # ── Neither worked ────────────────────────────────────────────────────────
    if backend_used is None:
        print()
        print("  ERROR: Could not apply SmoothQuant — neither backend worked.")
        print()
        print("  Root cause: nvidia-modelopt requires torch>=2.6 but you have", torch.__version__)
        print()
        print("  Fix — upgrade torch FIRST, then re-run this script:")
        print("    pip install 'torch>=2.6.0' --index-url https://download.pytorch.org/whl/cu124")
        print("    python code/activity2/sq_c1_wikitext.py")
        sys.exit(1)

    # Save quantized model
    print(f"\n  Saving INT8 SmoothQuant C1 (WikiText-2) model to {OUTPUT_PATH}...")
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
    print(" Activity 2A SmoothQuant (C1 / WikiText-2) COMPLETE")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_gb:.1f} GB  (expected: ~8 GB)")
    print("=" * 60)
    print()
    print("  Next step: re-evaluate this model on WMDP-Cyber and MMLU.")
    print("  Run:")
    print("    python code/activity2/evaluate_all.py --models int8_sq_c1_wikitext")
    print("  (You'll need to add an 'int8_sq_c1_wikitext' entry to evaluate_all.py")
    print("   pointing at ./outputs/cyber_int8_sq_c1_wikitext/ — see RUN_THIS.md)")


if __name__ == "__main__":
    main()
