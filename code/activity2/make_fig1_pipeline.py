"""
DACS Paper — Fig 1 (Experimental Pipeline)
==========================================
Reproduces fig1_pipeline.pdf as a matplotlib-driven vector diagram so
its labels and cell values stay in sync with the JSON results.

Replaces an earlier static PDF whose "(The Pile)" label and one stale
SQ INT8 cell value drifted from the corpus + post-SQ-C1-rerun reality.

Run:
  python code/activity2/make_fig1_pipeline.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CYBER_JSON = "./results/dacs_activity2_results_final.json"
OUT_PATH   = "./paper/fig1_pipeline.pdf"

with open(CYBER_JSON) as f:
    cyber = json.load(f)


def m(fmt, calib):
    return cyber[f"{fmt}_{calib}"]["mmlu"]


# Cyber MMLU cells (rows = calibration, cols = format)
cells = {
    ("c1", "fp8"):     m("fp8", "c1"),
    ("c1", "int4_awq"): m("int4_awq", "c1"),
    ("c1", "int8_sq"):  m("int8_sq", "c1"),
    ("c2", "fp8"):     m("fp8", "c2"),
    ("c2", "int4_awq"): m("int4_awq", "c2"),
    ("c2", "int8_sq"):  m("int8_sq", "c2"),
    ("c3", "fp8"):     m("fp8", "c3"),
    ("c3", "int4_awq"): m("int4_awq", "c3"),
    ("c3", "int8_sq"):  m("int8_sq", "c3"),
}

# Colours
LEFT_FILL    = "#E7F0F9"
LEFT_EDGE    = "#3F6CB3"
MATRIX_FILL  = "#FFFFFF"
MATRIX_EDGE  = "#666666"
HIGHLIGHT_FILL = "#FBDADA"
HIGHLIGHT_EDGE = "#B23A3A"
RIGHT_FILL   = "#EFF6E6"
RIGHT_EDGE   = "#5D8A2A"
ARROW_COLOR  = "#555555"
TEXT_COLOR   = "#222222"

fig, ax = plt.subplots(figsize=(14, 5.6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.6)
ax.axis("off")
ax.set_facecolor("white")

# ---------------------------------------------------------------------------
# LEFT — Fine-tuning pipeline
# ---------------------------------------------------------------------------
def box(x, y, w, h, fill, edge, label, fontsize=9, lw=1.0):
    r = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT_COLOR)


ax.text(1.7, 5.0, "Fine-Tuning Pipeline", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#3F6CB3")

box(0.3, 3.95, 2.8, 0.85, LEFT_FILL, LEFT_EDGE,
    "Llama-3.1-8B-Instruct\n(pre-trained, FP16)", fontsize=8.5)

# Arrow down
ax.annotate("", xy=(1.7, 3.40), xytext=(1.7, 3.92),
            arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=1.2))

box(0.3, 2.50, 2.8, 0.85, LEFT_FILL, LEFT_EDGE,
    "QLoRA Fine-Tuning\n83K cyber examples\nrank-16, 2 epochs",
    fontsize=8)

# Arrow down
ax.annotate("", xy=(1.7, 1.95), xytext=(1.7, 2.47),
            arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=1.2))

box(0.3, 1.05, 2.8, 0.85, LEFT_FILL, LEFT_EDGE,
    "Fine-Tuned Checkpoint (FP16)\nWMDP 44--46% | MMLU 63--65%",
    fontsize=8)

# Arrow right (to matrix)
ax.annotate("", xy=(4.3, 1.47), xytext=(3.15, 1.47),
            arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=1.3))
ax.text(3.7, 1.65, "PTQ", ha="center", va="bottom",
        fontsize=8.5, color="#555555", style="italic")

# ---------------------------------------------------------------------------
# CENTRE — 3 x 3 Calibration x Format matrix
# ---------------------------------------------------------------------------
ax.text(8.0, 5.0, "3 $\\times$ 3 Calibration $\\times$ Format Matrix    "
        "(cyber MMLU %)", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#444444")

# Format column headers
fmt_x = {"fp8": 5.6, "int4_awq": 7.5, "int8_sq": 9.4}
fmt_labels = {"fp8": "FP8 E4M3", "int4_awq": "AWQ INT4", "int8_sq": "SQ INT8"}
for fmt, x in fmt_x.items():
    ax.text(x + 0.85, 4.45, fmt_labels[fmt], ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#3F6CB3")

# Calibration row headers
cal_y = {"c1": 3.55, "c2": 2.50, "c3": 1.45}
cal_labels = {
    "c1": "C1 Generic\n(WikiText-2)",
    "c2": "C2 Self-gen\n(T=0.8)",
    "c3": "C3 DACS\n(domain-aligned)",
}
for cal, y in cal_y.items():
    ax.text(4.85, y + 0.4, cal_labels[cal], ha="center", va="center",
            fontsize=8.5, color="#5D8A2A", fontweight="bold")

# Cells
for (cal, fmt), val in cells.items():
    x = fmt_x[fmt]
    y = cal_y[cal]
    is_anomaly = (cal == "c2" and fmt == "int8_sq")
    fill = HIGHLIGHT_FILL if is_anomaly else MATRIX_FILL
    edge = HIGHLIGHT_EDGE if is_anomaly else MATRIX_EDGE
    lw   = 1.6 if is_anomaly else 0.8
    r = patches.Rectangle((x, y), 1.7, 0.8,
                          facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(r)
    suffix = " (!)" if is_anomaly else ""
    ax.text(x + 0.85, y + 0.40, f"{val:.2f}%{suffix}",
            ha="center", va="center",
            fontsize=10, fontweight="bold" if is_anomaly else "normal",
            color=HIGHLIGHT_EDGE if is_anomaly else TEXT_COLOR)

# Annotate the anomaly (positioned above the matrix title to avoid overlap)
ax.annotate(
    "Self-cal failure (−9.08 pp)",
    xy=(10.25, 3.25), xytext=(10.4, 5.4),
    fontsize=7.5, color=HIGHLIGHT_EDGE, fontweight="bold",
    ha="center",
    arrowprops=dict(arrowstyle="->", color=HIGHLIGHT_EDGE, lw=0.9),
)

# Arrow right to eval
ax.annotate("", xy=(11.95, 2.5), xytext=(11.1, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=1.3))
ax.text(11.55, 2.7, "eval", ha="center", va="bottom",
        fontsize=8.5, color="#555555", style="italic")

# ---------------------------------------------------------------------------
# RIGHT — Evaluation
# ---------------------------------------------------------------------------
ax.text(12.95, 5.0, "Evaluation", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#5D8A2A")

box(12.05, 3.10, 1.85, 0.85, RIGHT_FILL, RIGHT_EDGE,
    "WMDP-Cyber\n1,987 MCQs", fontsize=8.5)
box(12.05, 1.95, 1.85, 0.85, RIGHT_FILL, RIGHT_EDGE,
    "MMLU\n14,042 MCQs\n(5-shot)", fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved {OUT_PATH}")
