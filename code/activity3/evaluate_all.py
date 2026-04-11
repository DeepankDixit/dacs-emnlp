"""
DACS Activity 3 — Evaluate All 18 Multi-Domain Models
======================================================
Runs two benchmarks on each of 18 quantized models:
   9 medical models  (AWQ/SQ/FP8 × C1/C2/C3)  →  MedQA + MMLU
   9 code    models  (AWQ/SQ/FP8 × C1/C2/C3)  →  HumanEval + MMLU

Results cached in ./results/eval_cache/.  Master table saved to
./results/activity3_all_results.json.

Usage:
    python activity3/evaluate_all.py                              # full 18-model run
    python activity3/evaluate_all.py --domain med                 # 9 med only
    python activity3/evaluate_all.py --domain med --cond c1 c3    # early hierarchy check
    python activity3/evaluate_all.py --limit 20                   # quick pipeline test
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# unified_eval.py lives at code/ root (shared between all activities)
_CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _CODE_ROOT)
from unified_eval import evaluate_model

# Per-domain benchmark pairing
DOMAIN_BENCH = {
    "med":  ["medqa",     "mmlu"],
    "code": ["humaneval", "mmlu"],
}

FORMATS = ["int4_awq", "int8_sq", "fp8"]
CONDS   = ["c1", "c2", "c3"]

RESULTS_FILE = "./results/activity3_all_results.json"


def model_key(domain, fmt, cond):
    return f"{domain}_{fmt}_{cond}"


def model_path(domain, fmt, cond):
    return f"./outputs/{domain}_{fmt}_{cond}/"


def print_table(results: dict):
    print()
    print("=" * 88)
    print(" ACTIVITY 3 — MULTI-DOMAIN SENSITIVITY TABLE")
    print("=" * 88)
    header = f"  {'Model':<24} {'Format':<10} {'Cond':<5} {'DomainBench':>14} {'MMLU':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for domain in ["med", "code"]:
        dom_bench = DOMAIN_BENCH[domain][0]
        for fmt in FORMATS:
            for cond in CONDS:
                k = model_key(domain, fmt, cond)
                if k not in results:
                    continue
                r = results[k]
                db = r.get(dom_bench, "N/A")
                mm = r.get("mmlu", "N/A")
                db_s = f"{db:.1f}%" if isinstance(db, float) else "N/A"
                mm_s = f"{mm:.1f}%" if isinstance(mm, float) else "N/A"
                print(f"  {k:<24} {fmt:<10} {cond:<5} {db_s:>14} {mm_s:>8}")
        print()

    # Hierarchy + SQ-C2 checks
    print("  Format sensitivity spread per domain (|max - min| across C1/C2/C3 on MMLU):")
    for domain in ["med", "code"]:
        for fmt in FORMATS:
            scores = [results.get(model_key(domain, fmt, c), {}).get("mmlu")
                      for c in CONDS]
            scores = [s for s in scores if isinstance(s, float)]
            if len(scores) >= 2:
                spread = max(scores) - min(scores)
                print(f"    {domain:5s} {fmt:10s}: {spread:5.2f}pp")
    print("=" * 88)


def main(args):
    os.makedirs("./results", exist_ok=True)

    domains = ["med", "code"] if args.domain == "all" else [args.domain]
    formats = FORMATS if args.format == ["all"] else args.format
    conds   = CONDS   if args.cond   == ["all"] else args.cond

    if os.path.exists(RESULTS_FILE) and not args.force_rerun:
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    todo = [(d, f, c) for d in domains for f in formats for c in conds]
    print(f" Running {len(todo)} models across {len(domains)} domain(s)")
    if args.limit:
        print(f" TEST MODE limit={args.limit}")

    for i, (d, f, c) in enumerate(todo, 1):
        k    = model_key(d, f, c)
        path = model_path(d, f, c)
        if not os.path.isdir(path):
            print(f"\n  SKIP [{i}/{len(todo)}] {k}: not found at {path}")
            continue

        print(f"\n  [{i}/{len(todo)}] {k}")
        if k not in all_results:
            all_results[k] = {"domain": d, "format": f, "cond": c}

        # Both MedQA and MMLU now run as subprocesses (wmdp_eval.py --task medqa/mmlu).
        # Each subprocess exits cleanly, releasing all GPU memory before the next one
        # starts.  The original order (domain benchmark first, then MMLU) is used for
        # all formats — no special-casing needed for SQ INT8.
        for bench in DOMAIN_BENCH[d]:
            try:
                score = evaluate_model(
                    model_path=path, benchmark=bench,
                    limit=args.limit, force_rerun=args.force_rerun,
                )
                all_results[k][bench] = score
                with open(RESULTS_FILE, "w") as fout:
                    json.dump(all_results, fout, indent=2)
            except Exception as e:
                print(f"  ERROR {k} on {bench}: {e}")
                all_results[k][bench] = f"ERROR: {e}"

    print_table(all_results)

    all_results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "limit": args.limit,
        "domains": domains,
        "formats": formats,
        "conds": conds,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="all", choices=["all", "med", "code"])
    p.add_argument("--format", nargs="+", default=["all"],
                   choices=["all", "int4_awq", "int8_sq", "fp8"])
    p.add_argument("--cond", nargs="+", default=["all"],
                   choices=["all", "c1", "c2", "c3"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    main(p.parse_args())
