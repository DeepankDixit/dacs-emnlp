# DACS — Domain-Aware Calibration Selection

**Private research repository. Do not share.**
EMNLP 2026 submission — Optum AI Research.

> ⚠️ **Framing A (Sensitivity Study) — April 5, 2026**
> This is a cross-format calibration sensitivity study, not a method proposal.
> See `documentation/md/DECISION_RECORD.md` for the full framing.

---

## Repository Structure

```
code/
├── unified_eval.py           # shared eval harness (activity2 + activity3)
├── wmdp_eval.py              # format-aware loader (AWQ / SQ / FP8)
├── activity2/                # Cybersecurity domain (Activity 2)
│   ├── setup_lambda.sh       # one-shot Lambda env installer
│   ├── setup_eval_harness.sh # lm-eval + WMDP setup
│   ├── awq_c1.py  awq_c2.py  awq_c3.py
│   ├── sq_c1.py   sq_c2.py   sq_c3.py
│   ├── fp8_c1.py  fp8_c2.py  fp8_c3.py
│   ├── prepare_dacs_calib.py # build C3 calibration corpus
│   ├── generate_self_calib.py# build C2 self-generated corpus
│   ├── merge_adapter.py      # merge LoRA adapter → FP16
│   └── evaluate_all.py       # batch-eval all 9 models
└── activity3/                # Medical + Code domains (Activity 3)
    ├── setup_lambda.sh       # Activity 3 delta installer (runs after activity2/setup_lambda.sh)
    ├── requirements.txt      # incremental pins
    ├── common.py             # shared config (HF IDs, output paths)
    ├── download_models.py    # pull Med42 + CodeLlama from HuggingFace
    ├── prepare_c1.py         # C1 generic corpus (WikiText-2)
    ├── build_med_c3.py       # C3 medical corpus (PubMedQA + MedicalMeadow)
    ├── build_code_c3.py      # C3 code corpus (CodeAlpaca)
    ├── generate_self_calib.py# C2 self-cal generation (--domain med|code)
    ├── awq.py                # AWQ quantization (--domain med|code --cond c1|c2|c3)
    ├── sq.py                 # SmoothQuant quantization
    ├── fp8.py                # FP8 quantization
    └── evaluate_all.py       # batch-eval all 18 models
paper/
  main.tex
  references.bib
documentation/                # gitignored (large docx/pptx/md files)
```

---

## Activity 2 — Cybersecurity Domain

### GPU

Lambda Cloud A10G (24 GB, $0.75/h). ~9h wall-clock.

### Setup (fresh Lambda instance)

```bash
# 1. System packages + venv
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv venv && source venv/bin/activate

# 2. Clone
git clone https://github.com/DeepankDixit/dacs-emnlp.git && cd dacs-emnlp

# 3. Install all dependencies (torch, modelopt, autoawq, transformers, lm-eval)
bash code/activity2/setup_lambda.sh

# 4. Install eval benchmarks
bash code/activity2/setup_eval_harness.sh
```

### Environment variables

```bash
export HF_TOKEN="hf_your_token_here"
export SFT_CORPUS_PATH="/path/to/cybersec_sft_train.jsonl"
export ADAPTER_PATH="/path/to/cybersec_analyst_lora/"
```

### Run order

```bash
# Merge LoRA adapter → FP16 model (~15 min)
python code/activity2/merge_adapter.py

# C1 — Generic calibration baseline
python code/activity2/awq_c1.py
python code/activity2/sq_c1.py
python code/activity2/fp8_c1.py

# C3 — DACS domain calibration (run before C2 for early signal)
python code/activity2/prepare_dacs_calib.py
python code/activity2/awq_c3.py
python code/activity2/sq_c3.py
python code/activity2/fp8_c3.py

# Early hypothesis check: C3 vs C1
python code/activity2/evaluate_all.py --models c1 c3 --benchmarks wmdp_cyber

# C2 — Self-calibration
python code/activity2/generate_self_calib.py
python code/activity2/awq_c2.py
python code/activity2/sq_c2.py
python code/activity2/fp8_c2.py

# Full evaluation of all 9 models
python code/activity2/evaluate_all.py
```

### Activity 2 outputs

```
outputs/
  cybersec_analyst_merged_fp16/   # FP16 base (~16 GB)
  cyber_int4_awq_c1/  cyber_int4_awq_c2/  cyber_int4_awq_c3/
  cyber_int8_sq_c1/   cyber_int8_sq_c2/   cyber_int8_sq_c3/
  cyber_fp8_c1/       cyber_fp8_c2/       cyber_fp8_c3/
results/
  activity2_all_results.json
```

---

## Activity 3 — Medical + Code Domains

### GPU

Same Lambda A10G instance (or fresh one). ~10h wall-clock.

### Additional setup (after Activity 2 setup)

```bash
bash code/activity3/setup_lambda.sh   # installs human-eval, clones MedQA, verifies HF login
huggingface-cli login                 # required for gated Med42 checkpoint
# Accept terms at: https://huggingface.co/m42-health/Llama3-Med42-8B
```

### Run order

```bash
# Download public checkpoints (~30 min, bandwidth-bound)
python code/activity3/download_models.py --domain all

# Build calibration corpora
python code/activity3/prepare_c1.py           # C1 generic (WikiText-2)
python code/activity3/build_med_c3.py         # C3 medical
python code/activity3/build_code_c3.py        # C3 code
python code/activity3/generate_self_calib.py --domain med   # C2 medical
python code/activity3/generate_self_calib.py --domain code  # C2 code

# Medical domain: 9 quantized models
python code/activity3/awq.py --domain med --cond c1
python code/activity3/awq.py --domain med --cond c2
python code/activity3/awq.py --domain med --cond c3
python code/activity3/sq.py  --domain med --cond c1
python code/activity3/sq.py  --domain med --cond c2
python code/activity3/sq.py  --domain med --cond c3
python code/activity3/fp8.py --domain med --cond c1
python code/activity3/fp8.py --domain med --cond c2
python code/activity3/fp8.py --domain med --cond c3

# Evaluate medical domain (early hierarchy check)
python code/activity3/evaluate_all.py --domain med

# Code domain: 9 quantized models (same pattern)
python code/activity3/awq.py --domain code --cond c1
# ... (repeat for all 9 code conditions)
python code/activity3/fp8.py --domain code --cond c3

# Evaluate code domain
python code/activity3/evaluate_all.py --domain code

# Full 18-model cross-domain sensitivity table
python code/activity3/evaluate_all.py
```

### Activity 3 outputs

```
outputs/
  med_fp16/                             # Llama-3-Med42-8B FP16 (~16 GB)
  med_int4_awq_c1/  med_int4_awq_c2/  med_int4_awq_c3/
  med_int8_sq_c1/   med_int8_sq_c2/   med_int8_sq_c3/
  med_fp8_c1/       med_fp8_c2/       med_fp8_c3/
  code_fp16/                            # CodeLlama-7B-Instruct FP16 (~14 GB)
  code_int4_awq_c1/ ... code_fp8_c3/
  med_c3_calib_512.jsonl
  code_c3_calib_512.jsonl
  c1_generic_calib_512.jsonl
results/
  activity3_all_results.json
  eval_cache/
```

---

## Calibration Conditions

| ID | Name | Description |
|----|------|-------------|
| C1 | Generic | WikiText-2 passages (standard baseline) |
| C2 | Self-cal | Model-generated continuations (Williams et al. NAACL 2025) |
| C3 | Domain | Domain-specific corpus (SFT training distribution proxy) |

## Research Questions (Framing A — Sensitivity Study)

1. Does the format-sensitivity hierarchy (FP8 < AWQ < SQ) reproduce across medical and code domains?
2. Is the SmoothQuant C2 catastrophic MMLU regression a cross-domain phenomenon?

See `documentation/md/ACTIVITY_03_DESIGN.md` for the decision tree and pre-registered predictions.
