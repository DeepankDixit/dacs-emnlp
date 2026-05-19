# Re-run SmoothQuant C1 on WikiText-2 — Lambda Runbook

**Goal:** Fix the C1 corpus mismatch in cyber Activity 2. The original
`sq_c1.py` calibrated SmoothQuant on C4 (following the SmoothQuant paper's
reference protocol), while `awq_c1.py` and `fp8_c1.py` calibrated on
WikiText-2 (following the AWQ paper's protocol). This makes the C1 column
of Table 1 heterogeneous and weakens the cross-format comparison.

Re-quantize SQ INT8 C1 on WikiText-2 so that C1 is uniform across all
three formats. The original C4-calibrated model is preserved untouched
for comparison.

**Wall-clock:** ~30 min quantization + ~30 min eval = ~1 hour
**Cost:** ~$1.29 on Lambda A10 (1× a10 at $1.29/hr)

---

## Part 0 — On the local Mac: commit + push the new script and runbook

```bash
cd ~/emnlp
git status                       # confirm runbook is untracked
git add code/activity2/sq_c1_wikitext.py code/activity2/RUN_THIS_sq_c1_wikitext.md
git commit -m "Add SQ INT8 C1 re-run on WikiText-2 (fixes C1 corpus mismatch)"
git push origin main
```

If `sq_c1_wikitext.py` was already pushed in an earlier commit, the
runbook will be the only new file in this commit.

---

## Part 1 — SSH into the Lambda instance

```bash
ssh ubuntu@<INSTANCE_IP>
```

(Instance: `<your A10 instance>`, region `us-west-1`. Verify it's `Running`
in the Lambda console before connecting — if it's been terminated, launch
a fresh A10 and update the IP.)

## Part 2 — Start a tmux session

The quantization + eval together run for ~1 hour. SSH disconnects mid-run
would kill the job without tmux.

```bash
tmux new -s sq-rerun

# Useful tmux commands:
#   Ctrl-b  d         detach without killing the session
#   tmux ls           list sessions
#   tmux attach -t sq-rerun   reattach after disconnect
#   Ctrl-b  [         scrollback mode (then arrow keys / Page-Up)
#   q                 exit scrollback mode
```

**Everything from here on happens inside the tmux session.**

## Part 3 — Locate or clone the repo

### If the repo is already on this instance from prior work

```bash
cd ~/emnlp                       # adjust path if cloned elsewhere — `ls ~` to find
git pull origin main             # pulls sq_c1_wikitext.py and this runbook
```

### If this is a fresh Lambda instance

```bash
cd ~
git clone <repo-url> emnlp
cd emnlp
```

## Part 4 — Activate (or create) the Python venv

### Venv already exists from prior work

```bash
source venv/bin/activate
python -c "import torch, modelopt.torch.quantization; print('imports OK')"
```

If the import line prints `imports OK`, **skip Part 5** and continue to Part 6.

### Venv missing or imports failed

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Then continue to Part 5.

## Part 5 — Run setup_lambda.sh to install dependencies

Only needed if Part 4's import check failed, or this is a fresh instance.

```bash
bash code/activity2/setup_lambda.sh
```

Installs torch 2.6.0+cu124, nvidia-modelopt 0.42.0, autoawq, the
HuggingFace stack, lm-eval, and verifies CUDA + the `wmdp_cyber` lm-eval
task. Takes ~8 minutes.

**HuggingFace login is not required for this re-run** — WikiText-2 is open,
and the cyber FP16 base model already exists locally at
`./outputs/cybersec_analyst_merged_fp16/`.

Sanity checks before running the quantization:

```bash
ls -lh ./outputs/cybersec_analyst_merged_fp16/ | head -5    # FP16 model present
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi | head -20
```

Expected: torch 2.6.0+cu124, `cuda True`, GPU named A10 with ~24 GB VRAM.

## Part 6 — Run the SQ INT8 re-quantization

```bash
python code/activity2/sq_c1_wikitext.py 2>&1 | tee /tmp/sq_c1_wikitext.log
```

Expected output: `./outputs/cyber_int8_sq_c1_wikitext/` (~8 GB), takes
~30 minutes. The original C4-calibrated model at
`./outputs/cyber_int8_sq_c1/` is preserved untouched.

If the script errors (ModelOpt version conflict, OOM, etc.): capture the
error, stop, and fall back to the disclose-only path (Option A —
documented separately).

## Part 7 — Edit `evaluate_all.py` to point at the new model

Find this line (around line 45 of `code/activity2/evaluate_all.py`):

```python
    "int8_sq_c1":   ("./outputs/cyber_int8_sq_c1/",   "SQ INT8",   "C1 C4"),
```

Change to:

```python
    "int8_sq_c1":   ("./outputs/cyber_int8_sq_c1_wikitext/",   "SQ INT8",   "C1 WikiText-2"),
```

One-liner with `sed`:

```bash
sed -i 's|"./outputs/cyber_int8_sq_c1/",   "SQ INT8",   "C1 C4"|"./outputs/cyber_int8_sq_c1_wikitext/",   "SQ INT8",   "C1 WikiText-2"|' code/activity2/evaluate_all.py
grep -n "int8_sq_c1" code/activity2/evaluate_all.py   # verify the change took
```

## Part 8 — Re-run eval

`--force-rerun` bypasses the eval cache (which still holds the C4 result).

```bash
python code/activity2/evaluate_all.py --models int8_sq_c1 --force-rerun 2>&1 | tee /tmp/sq_c1_eval.log
```

Runs WMDP-Cyber (1,987 questions) and MMLU (14,042 questions). Takes
~30 minutes total.

Results land in `./results/activity2_all_results.json` (or
`dacs_activity2_results_final.json` — the script writes to whichever it's
configured for).

## Part 9 — Capture the two numbers

```bash
python -c "
import json, os
for p in ('./results/activity2_all_results.json',
          './results/dacs_activity2_results_final.json'):
    if not os.path.exists(p):
        continue
    with open(p) as f:
        d = json.load(f)
    if 'int8_sq_c1' in d:
        r = d['int8_sq_c1']
        print(f'{p}:')
        print(f'  WMDP-Cyber: {r.get(\"wmdp_cyber\"):.4f}%')
        print(f'  MMLU:       {r.get(\"mmlu\"):.4f}%')
"
```

Record the two numbers. They drive the downstream paper edits:

1. Table 1 SQ INT8 / C1 row update
2. Recompute Δ for SQ C2 vs new C1, SQ C3 vs new C1
3. Propagate the new headline regression magnitude through the abstract,
   §3.4, §4, §5.2, §5.5, §6, Conclusion, and Limitations (the existing
   value is 9.31 pp)
4. Regenerate `fig2_sensitivity.pdf` and `fig4_sensitivity_summary.pdf`
5. Simplify §3.3 to say "C1 = WikiText-2 across all three formats"
6. Drop `gao2020pile` from `references.bib`, add `merity2017wikitext`

## Part 10 — Sync the new results back to the local Mac

`results/` is in `.gitignore`, so the new JSON will NOT come back via
`git pull`. Two options:

### Option A (recommended): force-add the specific JSON file

On Lambda, inside the tmux session:

```bash
# Find which JSON the eval script actually wrote to
ls -la results/*.json

# Force-add it (overrides .gitignore for just this one file)
git add -f results/activity2_all_results.json   # or dacs_activity2_results_final.json
git add code/activity2/evaluate_all.py          # the one-line label/path change
git status
git commit -m "SQ INT8 C1 re-run on WikiText-2 — updated results"
git push origin main
```

Then on the local Mac:

```bash
cd ~/emnlp
git pull origin main
```

### Option B: scp the file directly (skip git for the JSON)

From the local Mac (NOT from inside Lambda SSH):

```bash
scp ubuntu@<INSTANCE_IP>:~/emnlp/results/activity2_all_results.json \
    ~/emnlp/results/
```

Useful if `git add -f` on the Lambda box feels too risky (e.g. you have
other gitignored files in `results/` you don't want to publish).

## Part 11 — Detach + optionally terminate

Inside tmux: `Ctrl-b` then `d` to detach. Then `exit` to drop SSH.

If the box won't be used again immediately, terminate it from the Lambda
console to stop the $1.29/hr meter. Confirm the dashboard flips to
`Terminating`.

---

## Decision branches

**Re-run succeeds, new SQ C1 numbers within ±3 pp of old C4-based numbers:**
expected case. Proceed with planned paper edits.

**New SQ C1 numbers shift dramatically (>3 pp from C4 result):** also a
useful finding — demonstrates SmoothQuant is sensitive to corpus choice
even within generic-text variants. Worth a sentence in §5.2 reinforcing
the "SQ is calibration-critical" message. Adjust headline numbers
accordingly.

**Script errors and can't be fixed in ~15 minutes of debugging:** abort and
fall back to Option A (text-only disclosure of the C4/WikiText-2
heterogeneity in §3.3 + Limitations). The rewrite is queued.

**New SQ C2 regression (vs new C1) drops below 5 pp:** tone down
"catastrophic" language in abstract and intro. Conclusion still stands;
becomes a softer claim.

---

## Quick reference

- Lambda instance: `<your A10 instance>` at `ubuntu@<INSTANCE_IP>` (`us-west-1`)
- Repo path on Lambda: typically `~/emnlp` (verify with `ls ~`)
- venv: `<repo>/venv`
- New model output: `./outputs/cyber_int8_sq_c1_wikitext/`
- Old C4 model (preserved): `./outputs/cyber_int8_sq_c1/`
- Eval cache: `./results/eval_cache/` (bypassed by `--force-rerun`)
- Results JSON: `./results/activity2_all_results.json` or
  `./results/dacs_activity2_results_final.json`
