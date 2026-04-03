# DACS — Domain-Aware Calibration Selection

**Private research repository. Do not share.**
EMNLP 2026 submission — Optum AI Research.

---

## Setup (Lambda Cloud GPU — A10G)

```bash
# 1. System packages + fresh virtual environment
sudo apt update
sudo apt install -y python3-pip python3-venv git
python3 -m venv venv
source venv/bin/activate

# 2. Clone repo
git clone https://github.com/DeepankDixit/dacs-emnlp.git
cd dacs-emnlp

# 3. Environment variables
export HF_TOKEN="hf_your_token_here"
export SFT_CORPUS_PATH="/path/to/cybersec_sft_train.jsonl"
export ADAPTER_PATH="/path/to/cybersec_analyst_lora/"
```

## Run Order

```bash
# 4. Install eval benchmarks (do this first, before anything else)
bash code/setup_eval_harness.sh

# 5. Install quantization libraries
pip install autoawq nvidia-modelopt[torch] transformers==4.43.0 peft==0.11.0 accelerate

# 3. Re-merge LoRA adapter → FP16 model (~15 min)
python code/merge_adapter.py

# 4. Activity 2A — Generic calibration baseline (C1)
python code/activity2a_awq_c1.py
python code/activity2a_sq_c1.py
python code/activity2a_fp8_c1.py

# 5. Activity 2C — DACS proposed method (C3) — run before 2B for early signal
python code/prepare_dacs_calib.py
python code/activity2c_awq_c3.py
python code/activity2c_sq_c3.py
python code/activity2c_fp8_c3.py

# 6. Early hypothesis check: C3 vs C1
python code/evaluate_all.py --models c1 c3 --benchmarks cyberseceval

# 7. Activity 2B — Self-calibration (C2)
python code/generate_self_calib.py
python code/activity2b_awq_c2.py
python code/activity2b_sq_c2.py
python code/activity2b_fp8_c2.py

# 8. Full evaluation of all 9 models
python code/evaluate_all.py
```

## Output Structure

```
outputs/
  cybersec_analyst_merged_fp16/   # FP16 base (~16 GB)
  cyber_int4_awq_c1/              # INT4 AWQ, generic cal
  cyber_int4_awq_c2/              # INT4 AWQ, self-cal
  cyber_int4_awq_c3/              # INT4 AWQ, DACS
  cyber_int8_sq_c1/               # INT8 SmoothQuant, generic
  cyber_int8_sq_c2/               # INT8 SmoothQuant, self-cal
  cyber_int8_sq_c3/               # INT8 SmoothQuant, DACS
  cyber_fp8_c1/                   # FP8, generic
  cyber_fp8_c2/                   # FP8, self-cal
  cyber_fp8_c3/                   # FP8, DACS
results/
  activity2_all_results.json      # all benchmark scores
```
