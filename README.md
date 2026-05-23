# When Does Calibration Data Matter for Quantizing Fine-Tuned LLMs?

A Cross-Format Sensitivity Study.

This repository accompanies an anonymous submission. It contains the source,
calibration corpora, evaluation harness, mechanism-analysis instrumentation,
and per-layer measurement data used to produce the 27-configuration result
table reported in the paper.

The work characterises how the choice of post-training quantization (PTQ)
calibration corpus interacts with the choice of quantization format on
domain fine-tuned LLMs. We hold calibration corpora identical across
formats, vary one factor at a time, evaluate every configuration on MMLU
(14,042 questions, all 57 subjects), and add per-domain benchmarks
where applicable: WMDP-Cyber (1,987 questions) for cybersecurity and
MedQA (500) for biomedical. All comparisons are accompanied by Wilson
95% confidence intervals, and the proposed activation-range mechanism
is validated by instrumenting all 224 SmoothQuant-targeted linear
layers per domain (n = 672 module-level measurements) with forward-hook
activation maxima recording.

---

## Scope

| Axis | Levels | Count |
|---|---|---|
| Quantization format | AWQ INT4, SmoothQuant INT8, FP8 (E4M3) | 3 |
| Calibration corpus | C1 generic (WikiText-2), C2 self-generated, C3 domain-aligned | 3 |
| Fine-tuned domain | Cybersecurity (Llama-3.1-8B + QLoRA), Biomedical (Med42-8B), Code (CodeLlama-7B) | 3 |
| **Total configurations** | | **27** |

Each configuration is evaluated on MMLU (all 14,042 questions) and the
applicable in-domain benchmark in full (WMDP-Cyber 1,987 questions,
MedQA 500). All evaluations use logit-based MCQ scoring (robust to
instruction-format variation across quantized variants).

## Methodology design choices

The study was designed around a few non-obvious decisions worth flagging:

1. **Uniform C1 corpus across all formats.** WikiText-2 calibration is
   applied identically to AWQ, FP8, and SmoothQuant in every domain so
   any accuracy difference is attributable to the format–calibration
   interaction rather than to a calibration-corpus shift across formats.
   This required re-running the cybersecurity SmoothQuant INT8 C1
   condition with WikiText-2 (vs. C4, which is the SmoothQuant paper's
   reference) and recomputing all downstream Δ values; see
   `code/activity2/sq_c1_wikitext.py`.
2. **Wilson 95% CIs on every claim.** Reported half-widths: ±0.80 pp at
   n=14,042 (MMLU); ±2.18 pp at n=1,987 (WMDP-Cyber); ±4.34 pp at
   n=500 (MedQA). MedQA differences below the half-width are flagged as
   descriptive rather than inferential in the paper.
3. **Per-layer mechanistic validation, not aggregate r=0.999.** Forward
   hooks instrument all 224 SmoothQuant-targeted linear modules per
   domain (32 transformer layers × 7 projection types) and record
   per-channel activation maxima. Per-layer C2 underestimation gap
   distributions are cleanly non-overlapping between domains (cyber
   median 0.153, IQR [0.131, 0.268]; medical 0.015; code 0.007); the
   cyber lower quartile alone exceeds the medical and code upper
   quartiles. This replaces a fragile n=3 per-domain Pearson with a
   distribution-level result over n = 672 module-level observations.
4. **Subprocess isolation in the eval harness.** Each model evaluation
   spawns a fresh Python subprocess so ModelOpt CUDA hooks registered
   during one model's evaluation cannot interfere with the format
   detection of the next. See `code/activity2/evaluate_all.py` and
   `code/unified_eval.py`.
5. **Deterministic merge and quantization paths.** The fine-tuned base
   is produced by deterministic LoRA merging (`merge_adapter.py`); the
   merged FP16 model is byte-identical to a fresh merge given the same
   base + adapter. All scripts use fixed seeds.

---

## Repository structure

```
code/
├── unified_eval.py                  # Shared evaluation harness
├── wmdp_eval.py                     # Format-aware loader (AWQ / SQ / FP8)
├── activity2/                       # Cybersecurity domain
│   ├── setup_lambda.sh              # Cloud-GPU environment installer
│   ├── setup_eval_harness.sh        # lm-eval + WMDP setup
│   ├── awq_c{1,2,3}.py              # AWQ INT4 under each calibration
│   ├── sq_c{1,2,3}.py               # plus sq_c1_wikitext.py (uniform-C1 re-run)
│   ├── fp8_c{1,2,3}.py              # FP8 under each calibration
│   ├── prepare_dacs_calib.py        # Build C3 calibration corpus
│   ├── generate_self_calib.py       # Build C2 self-generated corpus
│   ├── merge_adapter.py             # Deterministic LoRA → FP16 merge
│   ├── evaluate_all.py              # Batch-evaluate all 9 quantized models
│   ├── make_paper_figures.py        # Regenerate fig2 / fig3 / fig4 from JSON
│   └── make_fig1_pipeline.py        # Regenerate fig1 (pipeline) from JSON
├── activity3/                       # Medical + code domains
│   ├── setup_lambda.sh
│   ├── requirements.txt
│   ├── common.py                    # Domain configuration (HF IDs, paths)
│   ├── download_models.py
│   ├── prepare_c1.py                # C1 generic (WikiText-2) corpus builder
│   ├── build_med_c3.py              # C3 medical corpus (PubMedQA + MedicalMeadow)
│   ├── build_code_c3.py             # C3 code corpus (CodeSearchNet + OSS-Instruct)
│   ├── generate_self_calib.py       # C2 per-domain
│   ├── quant_awq.py / sq.py / fp8.py  # Per-domain quantization
│   └── evaluate_all.py
└── activity4/                       # Mechanism analysis (activation ranges)
    ├── activation_analysis.py       # Forward-hook instrumentation
    └── make_act4_figures.py         # Generate fig5 + fig6 from per-layer JSON

paper/
├── main.tex
├── references.bib
└── fig{1,2,3,4,5,6}_*.pdf

data/                                # Domain-aligned calibration corpora
outputs/                             # Generic + self-generated calibration corpora
results/                             # Eval JSONs + per-layer activation data
```

---

## Calibration conditions

| ID | Name        | Corpus                                            |
|----|-------------|---------------------------------------------------|
| C1 | Generic     | WikiText-2 (uniform across all three formats)     |
| C2 | Self-cal    | Temperature-0.8 nucleus continuations from target |
| C3 | DACS        | Per-domain corpus, see below                      |

**C1 (generic).** WikiText-2 train split, length-filtered, first 512
passages. Identical sample selection across AWQ, FP8, and SmoothQuant
in all three domains.

**C2 (self-generated).** Temperature-0.8 nucleus-sampled continuations
from the fine-tuned target model, seed-fixed for reproducibility. 512
sequences per domain.

**C3 (DACS / domain-aligned).** Per-domain corpus drawn from publicly
available training distributions:
- Cybersecurity: cybersecurity Q&A passages (released as
  `data/dacs_calib_512.jsonl`)
- Biomedical: PubMedQA + MedicalMeadow
- Code: CodeSearchNet + OSS-Instruct

All conditions use 512 sequences × 512 tokens, held constant across
the study to isolate data-distribution effects from data-size effects.

---

## Reproduction

### Environment

A single 24 GB GPU (NVIDIA A10) is sufficient. Pinned versions: Python
3.10, PyTorch 2.6.0+cu124, nvidia-modelopt 0.42.0, autoawq 0.2.9, the
HuggingFace stack (transformers, accelerate, peft, datasets), and
lm-eval 0.4.11. The compute footprint is intentionally modest so the
study can be reproduced on commodity cloud hardware; full end-to-end
reproduction (3 domains × 9 quantizations + 27 × 2 evaluations + Activity
4 per-layer instrumentation) is approximately one GPU-day per domain.

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
python code/activity4/activation_analysis.py    # forward-hook instrumentation
python code/activity4/make_act4_figures.py      # regenerate fig5 + fig6
```

Produces per-channel activation range data
(`results/activity4_activation_results.json`) and per-layer summary
statistics (`results/activity4_per_layer_stats.json`).

### Regenerate paper figures from JSON

```bash
python code/activity2/make_fig1_pipeline.py     # fig1 (experimental pipeline)
python code/activity2/make_paper_figures.py     # fig2, fig3, fig4
python code/activity4/make_act4_figures.py      # fig5, fig6
```

All six figures regenerate deterministically from the released result
JSONs; the figures and the paper's Tables 1–3 are guaranteed to remain
in sync with the underlying data.

---

## Released artifacts

| File | Contents |
|---|---|
| `data/dacs_calib_512.jsonl` | Cybersecurity C3 calibration corpus (512 passages) |
| `outputs/c1_generic_calib_512.jsonl` | C1 generic calibration corpus (WikiText-2 sample) |
| `outputs/{cyber,med,code}_c2_selfgen_512.jsonl` | C2 self-generated calibration corpora |
| `outputs/{med,code}_c3_calib_512.jsonl` | C3 biomedical and code calibration corpora |
| `results/dacs_activity2_results_final.json` | Cybersecurity 9-config results (Table 1) |
| `results/activity3_all_results.json` | Medical + code 18-config results (Tables 2, 3) |
| `results/activity4_activation_results.json` | Per-channel activation range data, all 224 layers × 3 domains |
| `results/activity4_per_layer_stats.json` | Per-layer summary statistics (median, IQR, n) per domain |

Every numerical value in the paper (Tables 1–3, Figs 2–6, the headline
9.08 pp SQ-C2 regression at >11× the Wilson 95% half-width, and the
per-layer medians) is reproducible exactly by reading these JSONs.

---

## License

This work is released under the MIT License — see [LICENSE](LICENSE).
The released calibration corpora retain their upstream dataset licenses
(WikiText-2: CC BY-SA 3.0; PubMedQA: MIT; CodeSearchNet: per-source;
MedicalMeadow: per-source).
