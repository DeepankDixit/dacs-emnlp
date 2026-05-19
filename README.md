# When Does Calibration Data Matter for Quantizing Fine-Tuned LLMs?

A Cross-Format Sensitivity Study.

This repository accompanies an anonymous submission. It contains the code,
calibration corpora, and evaluation harness used to produce the 27-configuration
result table reported in the paper.

---

## Repository structure

```
code/
├── unified_eval.py            # Shared evaluation harness
├── wmdp_eval.py               # Format-aware loader (AWQ / SQ / FP8)
├── activity2/                 # Cybersecurity domain
│   ├── setup_lambda.sh        # Cloud-GPU environment installer
│   ├── setup_eval_harness.sh  # lm-eval + WMDP setup
│   ├── awq_c{1,2,3}.py
│   ├── sq_c{1,2,3}.py         # plus sq_c1_wikitext.py (uniform-C1 re-run)
│   ├── fp8_c{1,2,3}.py
│   ├── prepare_dacs_calib.py  # Build C3 calibration corpus
│   ├── generate_self_calib.py # Build C2 self-generated corpus
│   ├── merge_adapter.py       # Merge LoRA adapter into FP16 base
│   ├── evaluate_all.py        # Batch-evaluate all 9 quantized models
│   └── make_paper_figures.py  # Regenerate fig2 / fig3 / fig4 from JSON
└── activity3/                 # Medical + code domains
    ├── setup_lambda.sh
    ├── requirements.txt
    ├── common.py              # Domain configuration (HF IDs, paths)
    ├── download_models.py
    ├── prepare_c1.py          # C1 generic (WikiText-2) corpus builder
    ├── build_med_c3.py        # C3 medical corpus
    ├── build_code_c3.py       # C3 code corpus
    ├── generate_self_calib.py
    ├── quant_awq.py / sq.py / fp8.py  # Per-domain quantization
    └── evaluate_all.py
code/activity4/                # Mechanism analysis (activation ranges)
├── activation_analysis.py
└── make_act4_figures.py
paper/
├── main.tex
└── references.bib
results/                       # Eval outputs (JSON)
data/                          # Released calibration corpora
```

---

## Calibration conditions

| ID | Name        | Description                                              |
|----|-------------|----------------------------------------------------------|
| C1 | Generic     | WikiText-2 passages (uniform across all three formats)   |
| C2 | Self-cal    | Model-generated continuations                            |
| C3 | DACS        | Domain-aligned calibration corpus                        |

---

## Reproduction

### Environment

Tested on a single 24 GB GPU (NVIDIA A10) running Ubuntu 22.04 with Python 3.10,
PyTorch 2.6, nvidia-modelopt 0.42, autoawq 0.2.9. Wall-clock for the full
cybersecurity sweep (9 quantizations + 18 evaluations): ~9 hours.

```bash
python3 -m venv venv && source venv/bin/activate
bash code/activity2/setup_lambda.sh        # installs all pinned deps
bash code/activity2/setup_eval_harness.sh  # registers lm-eval tasks
```

Set `HF_TOKEN` for HuggingFace gated-model access (Llama 3.1 base, Med42).

### Cybersecurity domain (Activity 2)

```bash
# 1. Build the fine-tuned FP16 base
python code/activity2/merge_adapter.py

# 2. Quantize under C1 / C2 / C3 for each format (9 models total)
python code/activity2/awq_c1.py
python code/activity2/sq_c1_wikitext.py     # SQ INT8 with WikiText-2 C1 corpus
python code/activity2/fp8_c1.py
python code/activity2/generate_self_calib.py
python code/activity2/awq_c2.py
python code/activity2/sq_c2.py
python code/activity2/fp8_c2.py
python code/activity2/prepare_dacs_calib.py
python code/activity2/awq_c3.py
python code/activity2/sq_c3.py
python code/activity2/fp8_c3.py

# 3. Evaluate all 9 on WMDP-Cyber + MMLU
python code/activity2/evaluate_all.py
```

### Medical + code domains (Activity 3)

```bash
bash code/activity3/setup_lambda.sh
python code/activity3/download_models.py --domain all
python code/activity3/prepare_c1.py
python code/activity3/build_med_c3.py
python code/activity3/build_code_c3.py
python code/activity3/generate_self_calib.py --domain med
python code/activity3/generate_self_calib.py --domain code

# Quantize and evaluate per domain
for dom in med code; do
  for fmt in quant_awq sq fp8; do
    for c in c1 c2 c3; do
      python code/activity3/${fmt}.py --domain $dom --cond $c
    done
  done
done
python code/activity3/evaluate_all.py
```

### Mechanism analysis (Activity 4)

```bash
python code/activity4/activation_analysis.py
python code/activity4/make_act4_figures.py
```

Produces per-channel activation range data
(`results/activity4_activation_results.json`) and per-layer summary statistics
(`results/activity4_per_layer_stats.json`).

---

## Released artifacts

- `data/dacs_calib_512.jsonl` — cybersecurity C3 calibration corpus
- `outputs/c1_generic_calib_512.jsonl`, `cyber_c2_selfgen_512.jsonl`,
  `med_c2_selfgen_512.jsonl`, `code_c2_selfgen_512.jsonl`, etc. —
  all generated calibration corpora
- `results/dacs_activity2_results_final.json` — cybersecurity 9-config results
- `results/activity3_all_results.json` — medical + code 18-config results
- `results/activity4_activation_results.json` — per-channel activation analysis
- `results/activity4_per_layer_stats.json` — per-layer summary stats

The 27-configuration result table in the paper (Tables 1, 2, 3) is reproducible
exactly from `evaluate_all.py` reading these JSONs.

---

## Calibration conditions, in detail

**C1 (generic).** WikiText-2 train split, length-filtered, first 512 passages.
Identical sample selection across AWQ, FP8, and SmoothQuant in all three domains.

**C2 (self-generated).** Temperature-0.8 nucleus-sampled continuations from
the fine-tuned target model, seed-fixed for reproducibility. 512 sequences
per domain.

**C3 (DACS / domain-aligned).** Per-domain corpus drawn from publicly
available training distributions:
- Cybersecurity: cybersecurity Q&A passages
- Biomedical: PubMedQA + MedicalMeadow
- Code: CodeSearchNet + OSS-Instruct

512 sequences each, 512 tokens per sequence.

---

## License

Code released under the MIT License; data passages retain their original
upstream licenses (WikiText-2: CC BY-SA 3.0; PubMedQA: MIT; etc.).
