"""
DACS Activity 3 — Download Public Domain Checkpoints
=====================================================
Pulls Llama3-Med42-8B and CodeLlama-7B-Instruct from HuggingFace and
saves them as local FP16 directories that the activity3_* quantization
scripts expect.

Med42 is a GATED model — accept the terms on HuggingFace and run
`huggingface-cli login` BEFORE invoking this script.

Usage:
    python download_domain_models.py --domain med
    python download_domain_models.py --domain code
    python download_domain_models.py --domain all

Runtime: ~10 minutes per model on Lambda (weights are 16 GB each)
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DOMAINS


def download(domain: str):
    cfg = DOMAINS[domain]
    os.makedirs(cfg["local_fp16"], exist_ok=True)

    print("=" * 60)
    print(f" Downloading {cfg['hf_id']} → {cfg['local_fp16']}")
    print("=" * 60)

    tok   = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.save_pretrained(cfg["local_fp16"])
    tok.save_pretrained(cfg["local_fp16"])

    size_gb = sum(os.path.getsize(os.path.join(cfg["local_fp16"], f))
                  for f in os.listdir(cfg["local_fp16"])
                  if os.path.isfile(os.path.join(cfg["local_fp16"], f))) / 1e9
    print(f" DONE: {cfg['local_fp16']}  |  {size_gb:.1f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["med", "code", "all"])
    args = p.parse_args()
    if args.domain == "all":
        download("med")
        download("code")
    else:
        download(args.domain)
