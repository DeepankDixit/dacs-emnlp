"""
DACS Project — Activity 5: Unified Evaluation Harness
======================================================
Single interface to evaluate any quantized model on any benchmark.
Results are cached so re-running the same model/benchmark is instant.

Usage:
    from unified_eval import evaluate_model

    score = evaluate_model("./outputs/cyber_int4_awq_c3/", "cyberseceval")
    print(f"CyberSecEval: {score:.1f}%")

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
CYBEREVAL_PATH = "./PurpleLlama/CybersecurityBenchmarks"
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
    if benchmark == "cyberseceval":
        accuracy = _run_cyberseceval(model_path, limit)
    elif benchmark == "mmlu":
        accuracy = _run_mmlu(model_path, num_fewshot, limit)
    elif benchmark == "medqa":
        accuracy = _run_medqa(model_path, limit or 500)
    elif benchmark == "humaneval":
        accuracy = _run_humaneval(model_path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}. "
                         f"Choose from: cyberseceval, mmlu, medqa, humaneval")

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
# CyberSecEval 4 runner
# ---------------------------------------------------------------------------
def _run_cyberseceval(model_path: str, limit=None) -> float:
    out_dir = tempfile.mkdtemp()
    cmd = [
        "python", "-m", "cyberseceval.run_benchmark",
        "--benchmark", "mitre",
        "--model-path", model_path,
        "--output-dir", out_dir,
    ]
    if limit:
        cmd += ["--num-samples", str(limit)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=CYBEREVAL_PATH,
    )

    if result.returncode != 0:
        print(f"  WARNING: CyberSecEval returned non-zero exit code")
        print(f"  stderr: {result.stderr[-500:]}")

    # Parse accuracy from stdout
    for line in result.stdout.split("\n"):
        if "accuracy" in line.lower():
            match = re.search(r"(\d+\.\d+)", line)
            if match:
                return float(match.group(1))

    # Try to parse from output JSON files
    for json_file in Path(out_dir).glob("**/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "accuracy" in data:
                return float(data["accuracy"]) * 100
        except Exception:
            pass

    print(f"  WARNING: Could not parse CyberSecEval accuracy. stdout: {result.stdout[-200:]}")
    return 0.0


# ---------------------------------------------------------------------------
# MMLU runner (via lm-eval)
# ---------------------------------------------------------------------------
def _run_mmlu(model_path: str, num_fewshot=5, limit=None) -> float:
    out_dir = tempfile.mkdtemp()
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float16",
        "--tasks", "mmlu",
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", out_dir,
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: lm-eval returned non-zero exit code")
        print(f"  stderr: {result.stderr[-500:]}")

    # lm-eval writes JSON results — parse that
    result_files = list(Path(out_dir).glob("**/*.json"))
    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            # lm-eval result format
            if "results" in data and "mmlu" in data["results"]:
                acc = data["results"]["mmlu"].get("acc,none",
                      data["results"]["mmlu"].get("acc", 0))
                return float(acc) * 100
        except Exception:
            pass

    # Fallback: parse from stdout
    for line in result.stdout.split("\n"):
        if "mmlu" in line.lower() and "acc" in line.lower():
            match = re.search(r"(\d+\.\d+)", line)
            if match:
                return float(match.group(1)) * 100 if float(match.group(1)) < 1 else float(match.group(1))

    print(f"  WARNING: Could not parse MMLU accuracy.")
    return 0.0


# ---------------------------------------------------------------------------
# MedQA runner (inline — 4-option US medical board questions)
# ---------------------------------------------------------------------------
def _run_medqa(model_path: str, limit=500) -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.exists(MEDQA_TEST):
        print(f"  WARNING: MedQA file not found at {MEDQA_TEST}. Skipping.")
        return 0.0

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(MEDQA_TEST) as f:
        questions = [json.loads(line) for line in f][:limit]

    correct = 0
    for i, q in enumerate(questions):
        if _answer_medqa(model, tokenizer, q) == q["answer_idx"]:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"    MedQA progress: {i+1}/{len(questions)}  "
                  f"({correct/(i+1)*100:.1f}% so far)")

    del model
    import torch; torch.cuda.empty_cache()

    return correct / len(questions) * 100


def _answer_medqa(model, tokenizer, q) -> str:
    import torch
    prompt = f"Question: {q['question']}\n"
    for k, v in q["options"].items():
        prompt += f"{k}. {v}\n"
    prompt += "Answer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    return tokenizer.decode(output[0][-1:]).strip().upper()


# ---------------------------------------------------------------------------
# HumanEval runner
# ---------------------------------------------------------------------------
def _run_humaneval(model_path: str) -> float:
    try:
        from human_eval.data import read_problems, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("  human-eval not installed. Run: pip install human-eval --break-system-packages")
        return 0.0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = read_problems()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    samples = []
    for tid, prob in problems.items():
        inputs = tokenizer(prob["prompt"], return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
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
    del model
    torch.cuda.empty_cache()

    return results["pass@1"] * 100


# ---------------------------------------------------------------------------
# CLI: quick test mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a model on a benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("benchmark", choices=["cyberseceval", "mmlu", "medqa", "humaneval"])
    parser.add_argument("--limit", type=int, default=None, help="Max questions (for quick testing)")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    score = evaluate_model(args.model_path, args.benchmark,
                           limit=args.limit, force_rerun=args.force_rerun)
    print(f"\nFinal score: {score:.2f}%")
