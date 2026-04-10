"""
DACS Activity 3 — INT4 AWQ quantization (parameterized by domain + calibration)
==============================================================================
Usage:
    python activity3/quant_awq.py --domain med  --cond c1
    python activity3/quant_awq.py --domain med  --cond c2
    python activity3/quant_awq.py --domain med  --cond c3
    python activity3/quant_awq.py --domain code --cond c1
    python activity3/quant_awq.py --domain code --cond c2
    python activity3/quant_awq.py --domain code --cond c3

Input:    ./outputs/{domain}_fp16/ + calibration JSONL
Output:   ./outputs/{domain}_int4_awq_{cond}/  (~4 GB)
Runtime:  ~45 minutes on A10G
"""

import os
import json
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from common import DOMAINS, get_calib_path, require_calib, model_out_path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

N_CALIB_AWQ     = 128
MAX_CALIB_CHARS = 512

QUANT_CONFIG = {
    "zero_point":   True,
    "q_group_size": 128,
    "w_bit":        4,
    "version":      "GEMM",
}


def main(domain: str, cond: str):
    cfg = DOMAINS[domain]
    fp16_path  = cfg["local_fp16"]
    calib_path = require_calib(get_calib_path(domain, cond))
    out_path   = model_out_path(domain, "int4_awq", cond)
    os.makedirs(out_path, exist_ok=True)

    print("=" * 60)
    print(f" Activity 3 — INT4 AWQ  |  domain={domain}  |  calib={cond}")
    print("=" * 60)
    print(f"  Model:  {fp16_path}")
    print(f"  Calib:  {calib_path}")
    print(f"  Out:    {out_path}")

    print(f"\n[1/3] Loading calibration data...")
    with open(calib_path) as f:
        calib_texts = [json.loads(line)["text"] for line in f]
    calib_texts = [t[:MAX_CALIB_CHARS] for t in calib_texts[:N_CALIB_AWQ]]
    print(f"  Using {len(calib_texts)} samples capped at {MAX_CALIB_CHARS} chars")

    print(f"\n[2/3] Loading FP16 model...")
    model     = AutoAWQForCausalLM.from_pretrained(fp16_path, safetensors=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(fp16_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[3/3] Running AWQ + INT4 quantization...")
    model.quantize(tokenizer, quant_config=QUANT_CONFIG, calib_data=calib_texts)
    model.save_quantized(out_path)
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
