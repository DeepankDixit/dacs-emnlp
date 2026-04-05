"""
DACS Activity 3 — FP8 quantization (parameterized by domain + calibration)
===========================================================================
Usage:
    python activity3/fp8.py --domain med  --cond c1|c2|c3
    python activity3/fp8.py --domain code --cond c1|c2|c3

Input:    ./outputs/{domain}_fp16/ + calibration JSONL
Output:   ./outputs/{domain}_fp8_{cond}/  (~8 GB)
Runtime:  ~30 minutes on A10G
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DOMAINS, get_calib_path, require_calib, model_out_path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MAX_LENGTH  = 512
CALIB_BATCH = 4


def main(domain: str, cond: str):
    cfg = DOMAINS[domain]
    fp16_path  = cfg["local_fp16"]
    calib_path = require_calib(get_calib_path(domain, cond))
    out_path   = model_out_path(domain, "fp8", cond)
    os.makedirs(out_path, exist_ok=True)

    print("=" * 60)
    print(f" Activity 3 — FP8  |  domain={domain}  |  calib={cond}")
    print("=" * 60)

    print(f"[1/4] Loading calibration data from {calib_path}...")
    with open(calib_path) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    print(f"  Loaded {len(calib_texts)} samples")

    print(f"\n[2/4] Loading FP16 model {fp16_path}...")
    tokenizer = AutoTokenizer.from_pretrained(fp16_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        fp16_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"\n[3/4] Tokenizing calibration data...")
    inputs = tokenizer(
        calib_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\n[4/4] Running FP8 quantization...")
    import modelopt.torch.quantization as mtq

    def forward_loop(m):
        n = inputs["input_ids"].shape[0]
        with torch.no_grad():
            for start in range(0, n, CALIB_BATCH):
                batch = {k: v[start:start + CALIB_BATCH] for k, v in inputs.items()}
                m(**batch)

    mtq.quantize(model, config=mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    size_gb = sum(os.path.getsize(os.path.join(out_path, f))
                  for f in os.listdir(out_path)
                  if os.path.isfile(os.path.join(out_path, f))) / 1e9
    print(f"\n DONE: {out_path}  |  {size_gb:.1f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["med", "code"])
    p.add_argument("--cond",   required=True, choices=["c1", "c2", "c3"])
    args = p.parse_args()
    main(args.domain, args.cond)
