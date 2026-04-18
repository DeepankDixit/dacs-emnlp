"""
DACS Activity 4 — Activation Distribution Analysis
===================================================
Framing A (Sensitivity Study): mechanistic evidence for WHY
SmoothQuant C2 self-calibration fails on fine-tuned LLMs.

The SQ mechanism in one line:
  SQ computes per-channel smoothing vectors  s_j = max(|X_calib|)_j^alpha
  from calibration data, then migrates difficulty from activations to weights.
  If C2 text produces narrower activations than MMLU inference text,
  s_j is underestimated → wrong smoothing → quantization error → accuracy drop.

What this script measures:
  For each transformer layer, record the per-channel maximum absolute
  INPUT activation under C1, C2, C3 calibration data, and under a
  sample of MMLU questions (the actual inference distribution).

  Key metric: range_ratio[cond] = mean_j( max(|X_cond|)_j / max(|X_MMLU|)_j )
    range_ratio = 1.0 → calibration matches inference → good quantization
    range_ratio < 1.0 → calibration UNDERESTIMATES inference range → SQ fails

  Expected finding (consistent with Activities 2+3 regression magnitudes):
    C2 has the lowest range_ratio (narrowest calibration → worst SQ outcome)
    C3 has ratio closest to 1.0 (domain-aligned → closest to inference)
    The gap C2 vs MMLU is largest for cybersecurity, smaller for medical, smallest for code
    This mirrors: SQ C2 regression 9.31pp > 2.41pp > 1.80pp

Domains and models:
  med   → m42-health/Llama3-Med42-8B              (download from HF, gated)
  code  → meta-llama/CodeLlama-7b-Instruct-hf     (download from HF)
  cyber → ./outputs/cybersec_analyst_merged_fp16/  (upload from Mac or rebuild from LoRA)

Usage:
  python code/activity4/activation_analysis.py --domain med
  python code/activity4/activation_analysis.py --domain code
  python code/activity4/activation_analysis.py --domain cyber
  python code/activity4/activation_analysis.py --domain all   # runs all three

Output:
  results/activity4_activation_results.json   — per-layer stats for all domains
  (figures generated separately by make_act4_figures.py)

Runtime: ~25 min per domain on Lambda A10G (128 samples × 4 conditions)
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Add repo root to path so we can import from activity3/common.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "activity3"))
from common import DOMAINS, C1_CORPUS_PATH, get_calib_path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CALIB_SAMPLES = 128          # samples from each calibration corpus
N_MMLU_SAMPLES  = 128          # MMLU questions for inference proxy
MAX_TOKEN_LEN   = 256          # cap to save memory
RESULTS_PATH    = "./results/activity4_activation_results.json"

# Layers where SQ applies smoothing — these are the linear layer inputs SQ calibrates
SQ_TARGET_PROJECTIONS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

CYBER_DOMAIN_CFG = {
    "hf_id":        None,                              # no HF id — local only
    "local_fp16":   "./outputs/cybersec_analyst_merged_fp16/",
    "c3_corpus":    "./data/dacs_calib_512.jsonl",     # Activity 2 DACS domain corpus
    "c2_selfgen":   "./outputs/cyber_c2_selfgen_512.jsonl",
    "out_prefix":   "cyber",
}

# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

class InputActivationRecorder:
    """
    Registers forward hooks on the INPUTS of all SQ-targeted linear layers.
    Records per-channel maximum absolute activation value across all tokens
    and all samples — this is exactly the statistic SQ uses for calibration.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks = []
        # channel_max[layer_name] = running max per channel, shape [in_features]
        self.channel_max: Dict[str, torch.Tensor] = {}

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(proj in name for proj in SQ_TARGET_PROJECTIONS):
                    h = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(h)
        print(f"  Registered {len(self._hooks)} input-activation hooks")

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            # inp is a tuple; inp[0] is shape [batch, seq_len, in_features]
            x = inp[0].detach().float()               # [B, T, C]
            per_channel = x.abs().amax(dim=(0, 1))    # [C] — max over batch+seq
            if name not in self.channel_max:
                self.channel_max[name] = per_channel.cpu()
            else:
                self.channel_max[name] = torch.maximum(
                    self.channel_max[name], per_channel.cpu()
                )
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self.channel_max.clear()

    def get_mean_max(self) -> Dict[str, float]:
        """Returns mean-over-channels of per-channel max — one scalar per layer."""
        return {k: float(v.mean()) for k, v in self.channel_max.items()}

    def get_raw(self) -> Dict[str, List[float]]:
        """Returns full per-channel vector for each layer (for detailed analysis)."""
        return {k: v.tolist() for k, v in self.channel_max.items()}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl_texts(path: str, n: int) -> List[str]:
    texts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            texts.append(obj.get("text", obj.get("content", "")))
            if len(texts) >= n:
                break
    return texts


def load_mmlu_texts(tokenizer, n: int = N_MMLU_SAMPLES) -> List[str]:
    """
    Load MMLU 5-shot questions formatted as the model would see them at inference.
    Uses cais/mmlu 'all' test split, same as the eval harness.
    """
    try:
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: could not load MMLU from HF ({e}). Using fallback.")
        # Fallback: generate simple diverse prompts as inference proxy
        return [
            f"Question {i}: What is the relationship between {topic} and {concept}?\nAnswer:"
            for i, (topic, concept) in enumerate([
                ("entropy", "information"), ("mitosis", "cell division"),
                ("photosynthesis", "chlorophyll"), ("derivatives", "calculus"),
                ("Renaissance", "humanism"), ("plate tectonics", "earthquakes"),
                ("DNA replication", "polymerase"), ("supply", "demand"),
                ("Newton's laws", "momentum"), ("oxidation", "reduction"),
            ] * (n // 10 + 1))
        ][:n]

    choices_labels = ["A", "B", "C", "D"]
    texts = []
    for item in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
        prompt = (
            f"{item['question']}\n"
            + "\n".join(f"{choices_labels[i]}. {c}"
                        for i, c in enumerate(item["choices"]))
            + "\nAnswer:"
        )
        texts.append(prompt)
    return texts


# ---------------------------------------------------------------------------
# Forward pass with activation recording
# ---------------------------------------------------------------------------

def run_forward_pass(
    model,
    tokenizer,
    texts: List[str],
    recorder: InputActivationRecorder,
    label: str,
) -> Dict[str, float]:
    """
    Run all texts through the model with hooks active.
    Returns mean-over-channels max-activation per layer.
    """
    recorder.reset()
    n_ok = 0
    for i, text in enumerate(texts):
        if i % 32 == 0:
            print(f"    [{label}] {i}/{len(texts)} samples...", end="\r")
        try:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LEN,
                padding=False,
            ).to(model.device)
            with torch.no_grad():
                model(**enc)
            n_ok += 1
        except Exception as e:
            print(f"\n    Warning: sample {i} failed ({e})")
        finally:
            if i % 16 == 0:
                torch.cuda.empty_cache()
    print(f"    [{label}] {n_ok}/{len(texts)} samples processed         ")
    return recorder.get_mean_max()


# ---------------------------------------------------------------------------
# Per-channel ratio computation
# ---------------------------------------------------------------------------

def compute_range_ratios(
    stats: Dict[str, Dict[str, float]],
    mmlu_key: str = "mmlu",
) -> Dict[str, Dict[str, float]]:
    """
    For each layer, compute range_ratio[cond] = stat[cond] / stat[mmlu].
    ratio < 1 → calibration underestimates inference range (→ SQ fails).
    """
    ratios = {}
    for layer in stats.get(mmlu_key, {}):
        mmlu_val = stats[mmlu_key].get(layer, 0)
        if mmlu_val == 0:
            continue
        ratios[layer] = {}
        for cond, layer_stats in stats.items():
            if cond == mmlu_key:
                ratios[layer][cond] = 1.0
            else:
                ratios[layer][cond] = layer_stats.get(layer, 0) / mmlu_val
    return ratios


# ---------------------------------------------------------------------------
# Main analysis for one domain
# ---------------------------------------------------------------------------

def analyse_domain(domain: str) -> Dict:
    """
    Run the full activation analysis for one domain.
    Returns a dict of results to be saved in activity4_activation_results.json.
    """
    print(f"\n{'='*60}")
    print(f" Activity 4 — domain: {domain}")
    print(f"{'='*60}")

    # Resolve model path
    if domain == "cyber":
        cfg = CYBER_DOMAIN_CFG
        model_path = cfg["local_fp16"]
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Cybersecurity FP16 model not found at {model_path}.\n"
                f"Rebuild it from the LoRA adapter first:\n"
                f"  python code/activity2/merge_adapter.py  (run on your local machine)\n"
                f"  Then upload to Lambda or skip cyber and run --domain med or --domain code."
            )
    else:
        cfg = DOMAINS[domain]
        model_path = cfg["local_fp16"]
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"FP16 model not found at {model_path}.\n"
                f"Download it first:\n"
                f"  hf download {cfg['hf_id']} --local-dir {model_path}"
            )

    # Load model
    print(f"\nLoading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")

    # Count hookable layers
    recorder = InputActivationRecorder(model)
    recorder.register_hooks()

    # Load calibration corpora
    c1_texts   = load_jsonl_texts(C1_CORPUS_PATH, N_CALIB_SAMPLES)
    c2_texts   = load_jsonl_texts(cfg["c2_selfgen"], N_CALIB_SAMPLES)
    c3_texts   = load_jsonl_texts(cfg["c3_corpus"],  N_CALIB_SAMPLES)
    mmlu_texts = load_mmlu_texts(tokenizer, N_MMLU_SAMPLES)

    print(f"\n  Corpora: C1={len(c1_texts)}, C2={len(c2_texts)}, "
          f"C3={len(c3_texts)}, MMLU={len(mmlu_texts)} samples each")

    # Run forward passes — 4 conditions
    layer_stats: Dict[str, Dict[str, float]] = {}
    for cond_label, texts in [
        ("c1", c1_texts),
        ("c2", c2_texts),
        ("c3", c3_texts),
        ("mmlu", mmlu_texts),
    ]:
        print(f"\n  Condition: {cond_label.upper()}")
        layer_stats[cond_label] = run_forward_pass(
            model, tokenizer, texts, recorder, label=cond_label
        )

    recorder.remove_hooks()

    # Compute range ratios relative to MMLU
    ratios = compute_range_ratios(layer_stats, mmlu_key="mmlu")

    # Aggregate: mean range_ratio across ALL layers for each condition
    agg = {}
    for cond in ("c1", "c2", "c3"):
        vals = [ratios[layer][cond] for layer in ratios if cond in ratios[layer]]
        agg[cond] = {
            "mean_range_ratio":   float(np.mean(vals)),
            "median_range_ratio": float(np.median(vals)),
            "min_range_ratio":    float(np.min(vals)),
            "pct_layers_under_1": float(np.mean([v < 1.0 for v in vals])) * 100,
        }

    # Aggregate by projection type (q/k/v/o vs gate/up/down)
    proj_agg: Dict[str, Dict[str, float]] = {}
    for proj in SQ_TARGET_PROJECTIONS:
        proj_layers = [l for l in ratios if proj in l]
        if not proj_layers:
            continue
        proj_agg[proj] = {}
        for cond in ("c1", "c2", "c3"):
            vals = [ratios[l][cond] for l in proj_layers if cond in ratios[l]]
            if vals:
                proj_agg[proj][cond] = float(np.mean(vals))

    print(f"\n  ── Range ratio summary (ratio to MMLU, mean across layers) ──")
    print(f"  {'Condition':<10}  {'Mean ratio':>12}  {'% layers < 1.0':>15}")
    for cond in ("c1", "c2", "c3"):
        a = agg[cond]
        print(f"  {cond.upper():<10}  {a['mean_range_ratio']:>12.4f}  "
              f"{a['pct_layers_under_1']:>14.1f}%")
    print(f"  (MMLU ratio = 1.000 by definition)")

    # Check hypothesis: C2 < C1 < C3 in mean range ratio
    c1_r = agg["c1"]["mean_range_ratio"]
    c2_r = agg["c2"]["mean_range_ratio"]
    c3_r = agg["c3"]["mean_range_ratio"]
    hypothesis_confirmed = (c2_r < c1_r) and (c1_r <= c3_r)
    print(f"\n  Hypothesis (C2 < C1 ≤ C3): {'CONFIRMED ✓' if hypothesis_confirmed else 'NOT confirmed ✗'}")

    result = {
        "domain":          domain,
        "n_calib_samples": N_CALIB_SAMPLES,
        "n_mmlu_samples":  N_MMLU_SAMPLES,
        "n_layers_hooked": len(layer_stats.get("c1", {})),
        "aggregated":      agg,
        "per_projection":  proj_agg,
        "per_layer_ratios": ratios,
        "hypothesis_confirmed": hypothesis_confirmed,
    }

    # Clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DACS Activity 4: activation analysis")
    p.add_argument(
        "--domain",
        required=True,
        choices=["med", "code", "cyber", "all"],
        help="Domain to analyse. Use 'all' for all three.",
    )
    args = p.parse_args()

    domains = ["med", "code", "cyber"] if args.domain == "all" else [args.domain]

    # Load existing results if any (append-safe)
    os.makedirs("./results", exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {RESULTS_PATH} "
              f"(domains already done: {list(all_results.keys())})")
    else:
        all_results = {}

    for domain in domains:
        if domain in all_results:
            print(f"\nSkipping {domain} — already in results file. "
                  f"Delete {RESULTS_PATH} to rerun.")
            continue
        try:
            result = analyse_domain(domain)
            all_results[domain] = result
            # Save after each domain (safe against Lambda OOM/timeout)
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  Saved to {RESULTS_PATH}")
        except FileNotFoundError as e:
            print(f"\n  SKIPPED {domain}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f" Activity 4 complete. Results: {RESULTS_PATH}")
    print(f" Next: python code/activity4/make_act4_figures.py")
    print(f"{'='*60}")
