"""
DACS Project — Activity 2: Evaluate All 9 Quantized Models
===========================================================
Run CyberSecEval 4 and MMLU on all 9 models produced by Activities 2A/2B/2C.
Results are cached — re-running the same model/benchmark is instant.
All scores are saved to ./results/activity2_all_results.json
Record every result in DACS_Results_Capture_Log.docx as you go.

Usage:
    # Evaluate all 9 models on both benchmarks (full run)
    python evaluate_all.py

    # Evaluate only C1 and C3 on CyberSecEval (early hypothesis check)
    python evaluate_all.py --models c1 c3 --benchmarks cyberseceval

    # Quick test with 20-question limit (for debugging the pipeline)
    python evaluate_all.py --limit 20

    # Force re-run even if cached
    python evaluate_all.py --force-rerun
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add code directory to path so unified_eval.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_eval import evaluate_model

# ---------------------------------------------------------------------------
# All 9 models (3 quantization formats × 3 calibration conditions)
# ---------------------------------------------------------------------------
ALL_MODELS = {
    # INT4 AWQ
    "int4_awq_c1":  ("./outputs/cyber_int4_awq_c1/",  "AWQ INT4",  "C1 WikiText-2"),
    "int4_awq_c2":  ("./outputs/cyber_int4_awq_c2/",  "AWQ INT4",  "C2 Self-gen"),
    "int4_awq_c3":  ("./outputs/cyber_int4_awq_c3/",  "AWQ INT4",  "C3 DACS"),

    # INT8 SmoothQuant
    "int8_sq_c1":   ("./outputs/cyber_int8_sq_c1/",   "SQ INT8",   "C1 C4"),
    "int8_sq_c2":   ("./outputs/cyber_int8_sq_c2/",   "SQ INT8",   "C2 Self-gen"),
    "int8_sq_c3":   ("./outputs/cyber_int8_sq_c3/",   "SQ INT8",   "C3 DACS"),

    # FP8
    "fp8_c1":       ("./outputs/cyber_fp8_c1/",       "FP8",       "C1 WikiText-2"),
    "fp8_c2":       ("./outputs/cyber_fp8_c2/",       "FP8",       "C2 Self-gen"),
    "fp8_c3":       ("./outputs/cyber_fp8_c3/",       "FP8",       "C3 DACS"),
}

BENCHMARKS = ["cyberseceval", "mmlu"]
RESULTS_FILE = "./results/activity2_all_results.json"
# ---------------------------------------------------------------------------


def print_results_table(results: dict):
    """Print a formatted results table to the terminal."""
    print()
    print("=" * 80)
    print(" ACTIVITY 2 RESULTS TABLE")
    print("=" * 80)
    print(f"  {'Model Key':<20} {'Format':<12} {'Calibration':<22} {'CyberSecEval':>14} {'MMLU':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*22} {'-'*14} {'-'*8}")

    for model_key, (path, fmt, cal) in ALL_MODELS.items():
        if model_key not in results:
            continue
        r = results[model_key]
        cseval = f"{r.get('cyberseceval', 'N/A'):.1f}%" if isinstance(r.get('cyberseceval'), float) else "N/A"
        mmlu   = f"{r.get('mmlu', 'N/A'):.1f}%"        if isinstance(r.get('mmlu'), float)        else "N/A"

        # Mark C3 (DACS) rows
        marker = " ← DACS" if "c3" in model_key else ""
        print(f"  {model_key:<20} {fmt:<12} {cal:<22} {cseval:>14} {mmlu:>8}{marker}")

    print()
    print("  C1 = Generic calibration (baseline)")
    print("  C2 = Self-calibration (Williams et al. NAACL 2025)")
    print("  C3 = DACS domain calibration (proposed method)")
    print()

    # Summary: C3 vs C1 delta
    print("  DACS vs Generic Calibration (C3 - C1 delta on CyberSecEval):")
    for fmt_key, label in [("int4_awq", "AWQ INT4"), ("int8_sq", "SQ INT8"), ("fp8", "FP8")]:
        c1_key = f"{fmt_key}_c1" if fmt_key != "fp8" else "fp8_c1"
        c3_key = f"{fmt_key}_c3" if fmt_key != "fp8" else "fp8_c3"
        if c1_key in results and c3_key in results:
            c1 = results[c1_key].get("cyberseceval")
            c3 = results[c3_key].get("cyberseceval")
            if isinstance(c1, float) and isinstance(c3, float):
                delta = c3 - c1
                sign = "+" if delta > 0 else ""
                print(f"    {label:<15}: C3={c3:.1f}%  C1={c1:.1f}%  delta={sign}{delta:.1f}%"
                      f"  {'✓ DACS WINS' if delta > 0 else '✗ No improvement'}")
    print("=" * 80)


def main(args):
    os.makedirs("./results", exist_ok=True)

    # Determine which models and benchmarks to run
    if args.models == ["all"]:
        models_to_run = list(ALL_MODELS.keys())
    else:
        # Support shortcuts: c1, c2, c3 expand to all formats for that condition
        expanded = []
        for m in args.models:
            if m in ALL_MODELS:
                expanded.append(m)
            elif m in ("c1", "c2", "c3"):
                # Add all formats for this condition
                expanded.extend([k for k in ALL_MODELS if k.endswith(f"_{m}")]
                                 + [k for k in ALL_MODELS if k == f"fp8_{m}"])
            else:
                print(f"  WARNING: Unknown model key {m!r}. Valid keys: {list(ALL_MODELS.keys())}")
        models_to_run = list(dict.fromkeys(expanded))  # deduplicate, preserve order

    benchmarks_to_run = args.benchmarks if args.benchmarks != ["all"] else BENCHMARKS

    print("=" * 60)
    print(f" DACS Activity 2 — Batch Evaluation")
    print("=" * 60)
    print(f"  Models to evaluate:     {models_to_run}")
    print(f"  Benchmarks:             {benchmarks_to_run}")
    if args.limit:
        print(f"  Question limit:         {args.limit} (TEST MODE — not for paper)")
    print()

    # Load existing results (so we can accumulate)
    if os.path.exists(RESULTS_FILE) and not args.force_rerun:
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results for {len(all_results)} models from {RESULTS_FILE}")
    else:
        all_results = {}

    # Run evaluations
    total = len(models_to_run) * len(benchmarks_to_run)
    done  = 0

    for model_key in models_to_run:
        model_path, fmt, cal = ALL_MODELS[model_key]

        if not os.path.isdir(model_path):
            print(f"\n  SKIP {model_key}: model directory not found at {model_path}")
            print(f"       Run the corresponding quantization script first.")
            continue

        if model_key not in all_results:
            all_results[model_key] = {"format": fmt, "calibration": cal}

        print(f"\n  [{done+1}/{total}] {model_key}  ({fmt}, {cal})")

        for benchmark in benchmarks_to_run:
            try:
                score = evaluate_model(
                    model_path=model_path,
                    benchmark=benchmark,
                    limit=args.limit,
                    force_rerun=args.force_rerun,
                )
                all_results[model_key][benchmark] = score
                done += 1

                # Save after every evaluation (in case of crash)
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"  ERROR evaluating {model_key} on {benchmark}: {e}")
                all_results[model_key][benchmark] = f"ERROR: {e}"

    # Print final table
    print_results_table(all_results)

    # Add timestamp and save final results
    all_results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "limit": args.limit,
        "models_evaluated": models_to_run,
        "benchmarks": benchmarks_to_run,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  All results saved to: {RESULTS_FILE}")
    print(f"  Next: Record scores in DACS_Results_Capture_Log.docx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all 9 DACS Activity 2 models")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model keys to evaluate. Options: all, c1, c2, c3, int4_awq_c1, etc."
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"],
        choices=["all", "cyberseceval", "mmlu"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit questions per benchmark (e.g. 20 for quick testing)"
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore cache and re-evaluate all models"
    )
    args = parser.parse_args()
    main(args)
