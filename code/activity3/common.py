"""
DACS Project — Activity 3 Common Configuration
================================================
Shared constants for Activity 3 (multi-domain extension).
Three domains: two PUBLIC checkpoints + one local fine-tune.

  med   → meta-llama/Llama-3-8B-Instruct fine-tuned as m42-health/Llama3-Med42-8B
  code  → meta-llama/CodeLlama-7b-Instruct-hf
  cyber → ./outputs/cybersec_analyst_merged_fp16/  (local; upload from Mac)

Import these in every activity3_* script to avoid path typos.
"""

import os

# ---------------------------------------------------------------------------
# Per-domain configuration
# ---------------------------------------------------------------------------
DOMAINS = {
    "med": {
        "hf_id":            "m42-health/Llama3-Med42-8B",
        "local_fp16":       "./outputs/med_fp16/",
        "c3_corpus":        "./outputs/med_c3_calib_512.jsonl",
        "c2_selfgen":       "./outputs/med_c2_selfgen_512.jsonl",
        "domain_bench":     "medqa",     # MedQA-USMLE 4-option
        "out_prefix":       "med",
        "seed_prompts":     [
            "Explain the pathophysiology of",
            "What is the first-line treatment for",
            "Describe the differential diagnosis of",
            "A 45-year-old patient presents with",
        ],
    },
    "code": {
        "hf_id":            "meta-llama/CodeLlama-7b-Instruct-hf",
        "local_fp16":       "./outputs/code_fp16/",
        "c3_corpus":        "./outputs/code_c3_calib_512.jsonl",
        "c2_selfgen":       "./outputs/code_c2_selfgen_512.jsonl",
        "domain_bench":     "humaneval",
        "out_prefix":       "code",
        "seed_prompts":     [
            "Write a Python function that",
            "Implement in Python:",
            "Debug the following code:",
            "Refactor this to be more efficient:",
        ],
    },
    "cyber": {
        "hf_id":            None,                              # local only
        "local_fp16":       "./outputs/cybersec_analyst_merged_fp16/",
        "c3_corpus":        "./data/dacs_calib_512.jsonl",     # Activity 2 DACS domain corpus
        "c2_selfgen":       "./outputs/cyber_c2_selfgen_512.jsonl",
        "domain_bench":     "wmdp_cyber",
        "out_prefix":       "cyber",
        "seed_prompts":     [
            "Explain how to detect",
            "Describe a common attack technique involving",
            "What are the indicators of compromise for",
            "How would you respond to a security incident involving",
        ],
    },
}

# Shared generic calibration corpus (C1). Same file used by Activity 2.
C1_CORPUS_PATH = "./outputs/c1_generic_calib_512.jsonl"

# Per-format output dir naming convention:  ./outputs/{prefix}_{fmt}_{cond}/
def model_out_path(domain: str, fmt: str, cond: str) -> str:
    """e.g. ('med', 'int4_awq', 'c3') -> './outputs/med_int4_awq_c3/' """
    prefix = DOMAINS[domain]["out_prefix"]
    return f"./outputs/{prefix}_{fmt}_{cond}/"


def get_calib_path(domain: str, cond: str) -> str:
    """Resolve calibration JSONL path for this (domain, condition) pair."""
    if cond == "c1":
        return C1_CORPUS_PATH
    if cond == "c2":
        return DOMAINS[domain]["c2_selfgen"]
    if cond == "c3":
        return DOMAINS[domain]["c3_corpus"]
    raise ValueError(f"Unknown calibration condition: {cond!r}")


def require_calib(path: str) -> str:
    """Verify calibration file exists; print fix hint if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"  C1: run prepare_c1_calib.py\n"
            f"  C2: run generate_self_calib_act3.py --domain <med|code|cyber>\n"
            f"  C3: run build_med_c3_corpus.py OR build_code_c3_corpus.py"
        )
    return path
