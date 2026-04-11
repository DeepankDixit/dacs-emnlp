"""
DACS Project — Activity 5: Unified Evaluation Harness
======================================================
Single interface to evaluate any quantized model on any benchmark.
Results are cached so re-running the same model/benchmark is instant.

Usage:
    from unified_eval import evaluate_model

    score = evaluate_model("./outputs/cyber_int4_awq_c3/", "wmdp_cyber")
    print(f"WMDP-Cyber: {score:.1f}%")

    score = evaluate_model("./outputs/cyber_int4_awq_c3/", "mmlu")
    print(f"MMLU: {score:.1f}%")
"""

import subprocess
import json
import os
import sys
import re
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — update these if your project structure differs
# ---------------------------------------------------------------------------
CYBEREVAL_PATH = "./PurpleLlama/CybersecurityBenchmarks"   # kept for reference; not used in eval
RESULTS_DIR    = "./results/eval_cache/"
MEDQA_TEST     = "./MedQA/data/questions/US/4_options/phrases_no_exclude_test.jsonl"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate_model(
    model_path: str,
    benchmark: str,          # "cyberseceval" | "mmlu" | "medqa" | "humaneval"
    num_fewshot: int = 5,
    limit: int = None,       # None = full benchmark; set a small number for testing
    force_rerun: bool = False,
) -> float:
    """
    Evaluate a model on a benchmark. Returns accuracy as a float (0–100).

    Results are cached by (model_path, benchmark) so re-evaluating the same
    model is instant. Set force_rerun=True to override the cache.

    Args:
        model_path:   Path to quantized model directory (e.g. "./outputs/cyber_int4_awq_c3/")
        benchmark:    One of: "cyberseceval", "mmlu", "medqa", "humaneval"
        num_fewshot:  Number of few-shot examples (default 5 for MMLU)
        limit:        Max questions to evaluate (None = full benchmark)
        force_rerun:  If True, ignore cache and re-evaluate

    Returns:
        Accuracy percentage (e.g. 72.4 means 72.4%)
    """
    cache_key  = model_path.replace("/", "_").strip("_") + f"_{benchmark}"
    cache_file = os.path.join(RESULTS_DIR, f"{cache_key}.json")

    # Return cached result if available
    if os.path.exists(cache_file) and not force_rerun:
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"  [cached] {Path(model_path).name} | {benchmark}: {cached['accuracy']:.2f}%")
        return cached["accuracy"]

    print(f"\n  Evaluating: {Path(model_path).name}")
    print(f"  Benchmark:  {benchmark}")
    if limit:
        print(f"  Limit:      {limit} questions (test mode)")

    # Dispatch to benchmark-specific runner
    if benchmark == "wmdp_cyber":
        accuracy = _run_wmdp_cyber_direct(model_path, num_fewshot, limit)
    elif benchmark == "mmlu":
        accuracy = _run_mmlu(model_path, num_fewshot, limit)
    elif benchmark == "medqa":
        accuracy = _run_medqa(model_path, limit or 500)
    elif benchmark == "humaneval":
        accuracy = _run_humaneval(model_path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}. "
                         f"Choose from: wmdp_cyber, mmlu, medqa, humaneval")

    # Cache the result
    with open(cache_file, "w") as f:
        json.dump({
            "model_path": model_path,
            "benchmark":  benchmark,
            "accuracy":   accuracy,
            "limit":      limit,
        }, f, indent=2)

    print(f"  Result: {accuracy:.2f}%  (saved to cache)")
    return accuracy


# ---------------------------------------------------------------------------
# WMDP-Cyber runner — direct evaluator (replaces lm-eval hf backend)
# ---------------------------------------------------------------------------
# WHY NOT lm-eval --model hf:
#
# lm-eval's hf backend calls AutoModelForCausalLM.from_pretrained(dtype=fp16).
# This breaks two of our three formats:
#   AWQ:  transformers ≥ 5.x requires gptqmodel which needs torch ≥ 2.7.1
#   SQ:   INT8 weights load without scales → garbage output (~25% random)
# FP8 happens to work (fp8→fp16 upcast native in PyTorch) but C1==C3 every
# time because FP8 precision is high enough that calibration data is not
# detectable after upcast.
#
# The direct evaluator (wmdp_eval.py) uses format-specific loading:
#   AWQ:  AutoAWQForCausalLM.from_quantized() (autoawq, no gptqmodel needed)
#   SQ:   AutoModelForCausalLM + mtq.restore() (modelopt quantization hooks)
#   FP8:  standard AutoModelForCausalLM (upcast works; documented limitation)
#
# WHY WMDP-CYBER INSTEAD OF CyberSecEval MITRE:
#
# CyberSecEval MITRE (PurpleLlama) is a 3-LLM pipeline:
#   1. Model under test answers cybersecurity prompts  → response file
#   2. "Expansion LLM" (external API) elaborates responses
#   3. "Judge LLM" (external API) scores the elaborated responses
#
# It has NO local model file loader — --llm-under-test takes an API endpoint
# (e.g. OPENAI::gpt-4o::key). Evaluating 9 local HuggingFace checkpoints
# would require: (a) vLLM serve each model, (b) external API keys for
# judge+expansion LLMs, and (c) managing 3 concurrent processes per model.
#
# WMDP-Cyber (Weapons of Mass Destruction Proxy — cybersecurity subset,
# Li et al. 2024) is a ~1987-question multiple-choice benchmark that runs
# identically to MMLU: model picks A/B/C/D, right or wrong, no judge needed.
# It directly tests whether cybersecurity domain knowledge is preserved after
# quantization — exactly what the DACS hypothesis predicts (C3 > C1).
# ---------------------------------------------------------------------------
def _run_wmdp_cyber_direct(model_path: str, num_fewshot=5, limit=None) -> float:
    """
    Evaluate WMDP-Cyber using the format-aware direct evaluator.
    Delegates to wmdp_eval.py which handles AWQ/SQ/FP8 loading correctly.
    """
    # Import from wmdp_eval.py (same code/ directory)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    try:
        from wmdp_eval import load_model_for_eval, evaluate_wmdp_cyber
    except ImportError as e:
        print(f"  ERROR: Could not import wmdp_eval: {e}")
        return 0.0

    try:
        (model, tokenizer), fmt = load_model_for_eval(model_path)
        accuracy = evaluate_wmdp_cyber(
            model, tokenizer,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        # Free GPU memory
        del model
        import torch; torch.cuda.empty_cache()
        return accuracy
    except Exception as e:
        print(f"  ERROR in WMDP-Cyber eval: {e}")
        import traceback; traceback.print_exc()
        return 0.0


# ---------------------------------------------------------------------------
# MMLU runner — direct evaluator (replaces lm-eval hf backend)
# ---------------------------------------------------------------------------
# WHY NOT lm-eval --model hf FOR MMLU:
#
# Identical problem to WMDP-Cyber: lm-eval's hf backend calls
# AutoModelForCausalLM.from_pretrained(dtype=float16).  This breaks:
#   AWQ:  gptqmodel version conflict (torch 2.6 vs ≥2.7.1 required)
#   SQ:   INT8 weights load without calibration scales → ~25% random output
#
# The format-aware evaluator in wmdp_eval.py handles all three formats
# correctly and now includes evaluate_mmlu() using the same logit-based
# MCQ scoring approach (cais/mmlu, all subjects, test split, 14K questions).
# ---------------------------------------------------------------------------
def _run_mmlu(model_path: str, num_fewshot=5, limit=None) -> float:
    """
    Evaluate MMLU using the format-aware direct evaluator via SUBPROCESS.

    WHY SUBPROCESS (not in-process import):
      In a 9-model batch, modelopt's CUDA hooks from SQ INT8 models do not
      release cleanly when an exception occurs (e.g. OOM). The GPU allocator
      retains ~21 GB after an SQ OOM, causing the next model (FP8) to fall
      back to CPU inference (~12 s/it instead of ~0.3 s/it).

      Running each model as a fresh subprocess guarantees:
        - Clean GPU state for every model
        - modelopt hooks fully unloaded between runs
        - No cascading OOM from one format to the next
    """
    code_dir = os.path.dirname(os.path.abspath(__file__))
    script   = os.path.join(code_dir, "wmdp_eval.py")
    python   = sys.executable

    cmd = [python, script, model_path, "--task", "mmlu",
           "--num_fewshot", str(num_fewshot)]
    if limit:
        cmd += ["--limit", str(limit)]

    # Stream live output to terminal via tee, capture same output to file.
    # wmdp_eval.py prints "RESULT:<float>" as the final parseable line.
    import shlex, tempfile
    outfile = tempfile.mktemp(suffix=".txt")
    tee_cmd = f"{shlex.join(cmd)} 2>&1 | tee {outfile}"
    print(f"  Spawning fresh process for clean GPU state...")
    os.system(tee_cmd)

    try:
        with open(outfile) as f:
            for line in f:
                if line.startswith("RESULT:"):
                    return float(line.strip().split(":")[1])
    except Exception as e:
        print(f"  ERROR parsing result file: {e}")

    print(f"  WARNING: Could not parse MMLU accuracy from subprocess output.")
    return 0.0


# ---------------------------------------------------------------------------
# MedQA runner — subprocess (same pattern as _run_mmlu)
# ---------------------------------------------------------------------------
# WHY SUBPROCESS (not in-process loading):
#
#   For SQ INT8 models, ModelOpt TensorQuantizer hooks create circular
#   references (module_map dict ↔ quantizer module ↔ model) that prevent
#   `del model + torch.cuda.empty_cache()` from releasing VRAM.  After an
#   inline MedQA run the parent process retains ~15.28 GB, causing the next
#   model's MMLU subprocess to OOM (only 6.54 GB free on the A10's 22 GB).
#
#   Running in a fresh subprocess guarantees:
#     - Clean GPU state for every model
#     - ModelOpt hooks fully unloaded between runs
#     - No cascading OOM from SQ c1→c2→c3
#
#   The evaluate_medqa() function in wmdp_eval.py uses logit-based MCQ
#   scoring (same as MMLU) rather than generate() for speed consistency.
# ---------------------------------------------------------------------------
def _run_medqa(model_path: str, limit=500) -> float:
    """
    Evaluate MedQA using the format-aware direct evaluator via SUBPROCESS.
    Delegates to wmdp_eval.py --task medqa for guaranteed clean GPU state.
    """
    if not os.path.exists(MEDQA_TEST):
        print(f"  WARNING: MedQA file not found at {MEDQA_TEST}. Skipping.")
        return 0.0

    code_dir = os.path.dirname(os.path.abspath(__file__))
    script   = os.path.join(code_dir, "wmdp_eval.py")
    python   = sys.executable

    cmd = [python, script, model_path, "--task", "medqa",
           "--medqa_path", MEDQA_TEST, "--limit", str(limit)]

    import shlex, tempfile
    outfile = tempfile.mktemp(suffix=".txt")
    tee_cmd = f"{shlex.join(cmd)} 2>&1 | tee {outfile}"
    print(f"  Spawning fresh process for clean GPU state (MedQA)...")
    os.system(tee_cmd)

    try:
        with open(outfile) as f:
            for line in f:
                if line.startswith("RESULT:"):
                    return float(line.strip().split(":")[1])
    except Exception as e:
        print(f"  ERROR parsing MedQA result file: {e}")

    print(f"  WARNING: Could not parse MedQA accuracy from subprocess output.")
    return 0.0


# ---------------------------------------------------------------------------
# HumanEval runner — format-aware, with explicit GPU cleanup
# ---------------------------------------------------------------------------
def _run_humaneval(model_path: str) -> float:
    """
    Evaluate HumanEval pass@1 using format-aware model loading.

    WHY FORMAT-AWARE:
      AutoModelForCausalLM.from_pretrained(dtype=fp16) breaks AWQ models
      (requires gptqmodel) and SQ models (scales not applied → garbage output).
      Same root cause as MedQA.  Fix: use the same format-specific loaders
      from wmdp_eval.py that are used for MedQA and MMLU.
    """
    try:
        from human_eval.data import read_problems, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("  human-eval not installed. Run: pip install human-eval")
        return 0.0

    import torch, gc

    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from wmdp_eval import detect_format, load_awq_model, load_sq_model, load_fp8_model, load_fp16_model
    fmt = detect_format(model_path)
    print(f"  Detected format: {fmt.upper()}")

    if fmt == "awq":
        model, tokenizer = load_awq_model(model_path)
    elif fmt == "sq_int8":
        model, tokenizer = load_sq_model(model_path)
    elif fmt == "fp8":
        model, tokenizer = load_fp8_model(model_path)
    else:
        model, tokenizer = load_fp16_model(model_path)

    device = getattr(model, "device", None) or next(model.parameters()).device

    problems = read_problems()
    samples = []
    for tid, prob in problems.items():
        inputs = tokenizer(prob["prompt"], return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 temperature=0.0, do_sample=False)
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        samples.append({"task_id": tid, "completion": completion})

    tmpf = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    write_jsonl(tmpf.name, samples)

    results = evaluate_functional_correctness(tmpf.name)

    # Explicit cleanup — critical for SQ INT8 circular references
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results["pass@1"] * 100


# ---------------------------------------------------------------------------
# CLI: quick test mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a model on a benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("benchmark", choices=["wmdp_cyber", "mmlu", "medqa", "humaneval"])
    parser.add_argument("--limit", type=int, default=None, help="Max questions (for quick testing)")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    score = evaluate_model(args.model_path, args.benchmark,
                           limit=args.limit, force_rerun=args.force_rerun)
    print(f"\nFinal score: {score:.2f}%")
