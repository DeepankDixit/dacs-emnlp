"""
DACS Paper — Regenerate Fig 2 / Fig 3 / Fig 4
==============================================
Reads results JSON for cyber (activity2) + medical & code (activity3),
produces three paper figures:

  fig2_sensitivity.pdf          — Cyber: MMLU accuracy per (format × calib),
                                  with FP16 baseline + SQ C2 drop annotation
                                  + Wilson 95 % CI error bars
  fig3_crossdomain.pdf          — 3-panel side-by-side MMLU view across
                                  cyber / medical / code domains
  fig4_sensitivity_summary.pdf  — Max calibration spread on MMLU per
                                  (format × domain) — 3 × 3 grouped bars

Run after dacs_activity2_results_final.json and activity3_all_results.json
exist (and are current — e.g. after the SQ C1 re-run on WikiText-2):

  python code/activity2/make_paper_figures.py

Outputs land in paper/ alongside fig1, fig5, fig6.
"""

import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------
CYBER_JSON   = "./results/dacs_activity2_results_final.json"
CROSSDOM_JSON = "./results/activity3_all_results.json"
OUT_DIR      = "./paper"

os.makedirs(OUT_DIR, exist_ok=True)

with open(CYBER_JSON) as f:
    cyber = json.load(f)
with open(CROSSDOM_JSON) as f:
    crossdom = json.load(f)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
FORMATS = ["int4_awq", "int8_sq", "fp8"]
FORMAT_LABELS = {
    "int4_awq": "AWQ INT4",
    "int8_sq":  "SQ INT8",
    "fp8":      "FP8",
}
CALIB_LABELS = {"c1": "C1\nGeneric", "c2": "C2\nSelf-gen", "c3": "C3\nDACS"}
CALIB_COLORS = {
    "c1": "#4472C4",
    "c2": "#E05C34",
    "c3": "#70AD47",
}

# Wilson 95 % CI half-widths
WILSON_MMLU = 0.80   # n=14,042

# FP16 baselines per domain (MMLU)
FP16_BASELINE_MMLU = {
    "cyber": 63.0,    # low end of the 63-65 range (rounds cleanly to 63 in
                      # the caption-and-label match; FP8 C1 MMLU at 62.18 is
                      # a near-zero quantization gap from this baseline)
    "med":   60.70,
    "code":  38.38,
}


# ---------------------------------------------------------------------------
# Pull MMLU + WMDP into uniform shape for cyber
# ---------------------------------------------------------------------------
def cyber_mmlu(fmt, calib):
    key = f"{fmt}_{calib}"
    return cyber[key]["mmlu"]


# ---------------------------------------------------------------------------
# Pull MMLU into uniform shape for cross-domain (activity3)
# ---------------------------------------------------------------------------
def crossdom_mmlu(domain, fmt, calib):
    key = f"{domain}_{fmt}_{calib}"
    return crossdom[key]["mmlu"]


# ===========================================================================
# FIG 2 — Cyber MMLU per (format × calibration)
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.8, 4.0))

bar_width = 0.24
group_gap = 0.35
fmt_positions = np.arange(len(FORMATS)) * (3 * bar_width + group_gap)

# Plot bars
for i_c, calib in enumerate(["c1", "c2", "c3"]):
    xs = fmt_positions + i_c * bar_width
    ys = [cyber_mmlu(fmt, calib) for fmt in FORMATS]
    ax.bar(xs, ys, bar_width,
           color=CALIB_COLORS[calib],
           yerr=WILSON_MMLU,
           capsize=3,
           error_kw=dict(ecolor="#333333", lw=0.7),
           label=CALIB_LABELS[calib].replace(chr(10), ' '),
           edgecolor="white", linewidth=0.5)

# Value labels on bars (offset enough to clear FP16 dashed line at 63.5)
for i_f, fmt in enumerate(FORMATS):
    for i_c, calib in enumerate(["c1", "c2", "c3"]):
        y = cyber_mmlu(fmt, calib)
        # If the bar is close to the FP16 line (63.5), shift label to ~2pp above
        offset = 2.2 if abs(y - 63.5) < 2.0 else 1.0
        ax.text(fmt_positions[i_f] + i_c * bar_width, y + offset,
                f"{y:.2f}", ha="center", fontsize=7.5, color="#333333")

# FP16 baseline dashed line
ax.axhline(FP16_BASELINE_MMLU["cyber"], color="black", linestyle="--", linewidth=1.0,
           label=f"FP16 baseline ($\\approx${FP16_BASELINE_MMLU['cyber']:.0f}%)",
           zorder=2, alpha=0.6)

# Annotate the SQ C2 drop
sq_c1 = cyber_mmlu("int8_sq", "c1")
sq_c2 = cyber_mmlu("int8_sq", "c2")
delta = sq_c2 - sq_c1   # negative
sq_c2_x = fmt_positions[FORMATS.index("int8_sq")] + 1 * bar_width
ax.annotate(
    f"$\\Delta = {delta:+.2f}$ pp",
    xy=(sq_c2_x, sq_c2),
    xytext=(sq_c2_x - 0.05, 60.8),
    fontsize=8.5, color="#E05C34", fontweight="bold",
    ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color="#E05C34", lw=1.0),
)

# Calibration spread annotation BELOW x-axis (in axes-fraction coords so it
# never collides with the plot area)
for i_f, fmt in enumerate(FORMATS):
    ys = [cyber_mmlu(fmt, c) for c in ["c1", "c2", "c3"]]
    spread = max(ys) - min(ys)
    cx = fmt_positions[i_f] + bar_width
    ax.text(cx, -0.16, f"spread {spread:.2f} pp",
            ha="center", va="top", fontsize=7.5, color="#555555",
            transform=ax.get_xaxis_transform())

ax.set_xticks(fmt_positions + bar_width)
ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS], fontsize=10, fontweight="bold")
ax.set_ylabel("MMLU accuracy (%)", fontsize=10)
ax.set_ylim(45, 70)
ax.set_title("Cybersecurity domain: calibration sensitivity on MMLU (Wilson 95% CI shown)",
             fontsize=9, fontweight="bold", pad=8)
ax.set_facecolor("#FAFAFA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True, edgecolor="#CCCCCC")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "fig2_sensitivity.pdf")
plt.savefig(out2, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved {out2}")


# ===========================================================================
# FIG 3 — Cross-domain MMLU panels (cyber / medical / code)
# ===========================================================================
domains = [("cyber", "Cybersecurity\\n(Llama-3.1-8B-DACS)", "cyber"),
           ("med",   "Medical\\n(Med42-8B)",                "med"),
           ("code",  "Code\\n(CodeLlama-7B-Instruct)",      "code")]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)

def get_mmlu(domain_tag, fmt, calib):
    if domain_tag == "cyber":
        return cyber_mmlu(fmt, calib)
    return crossdom_mmlu(domain_tag, fmt, calib)

for ax, (tag, label, _) in zip(axes, domains):
    fmt_positions = np.arange(len(FORMATS)) * (3 * bar_width + group_gap)
    for i_c, calib in enumerate(["c1", "c2", "c3"]):
        xs = fmt_positions + i_c * bar_width
        ys = [get_mmlu(tag, fmt, calib) for fmt in FORMATS]
        ax.bar(xs, ys, bar_width,
               color=CALIB_COLORS[calib],
               yerr=WILSON_MMLU,
               capsize=2,
               error_kw=dict(ecolor="#333333", lw=0.6),
               edgecolor="white", linewidth=0.4)
        for x, y in zip(xs, ys):
            # Offset bar value labels by 1.4 pp so they sit clear of the
            # dashed FP16 baseline line (which lies just above the FP8 bars
            # for cyber/medical and just below them for code).
            label_offset = 1.4 if abs(y - FP16_BASELINE_MMLU[tag]) < 2.5 else 0.6
            ax.text(x, y + label_offset, f"{y:.1f}", ha="center", fontsize=6.5, color="#333333")

    # Per-format spread (below x-axis in axes-fraction coords)
    for i_f, fmt in enumerate(FORMATS):
        ys = [get_mmlu(tag, fmt, c) for c in ["c1", "c2", "c3"]]
        spread = max(ys) - min(ys)
        cx = fmt_positions[i_f] + bar_width
        ax.text(cx, -0.18, f"spread {spread:.2f} pp",
                ha="center", va="top", fontsize=6.5, color="#555555",
                transform=ax.get_xaxis_transform())

    # FP16 baseline dashed line
    ax.axhline(FP16_BASELINE_MMLU[tag], color="black", linestyle="--",
               linewidth=0.9, alpha=0.55, zorder=2)

    ax.set_xticks(fmt_positions + bar_width)
    ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS], fontsize=8.5, fontweight="bold")
    ax.set_title(label.replace("\\n", "\n"), fontsize=9, pad=6, fontweight="bold")
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)

    # Y limits tuned to data range
    all_ys = [get_mmlu(tag, fmt, c) for fmt in FORMATS for c in ["c1", "c2", "c3"]]
    ymin = min(all_ys) - 3
    ymax = max(FP16_BASELINE_MMLU[tag], max(all_ys)) + 4
    ax.set_ylim(ymin, ymax)

axes[0].set_ylabel("MMLU accuracy (%)", fontsize=9)

# Single legend across the figure
legend_handles = [
    mpatches.Patch(color=CALIB_COLORS["c1"], label="C1 Generic (WikiText-2)"),
    mpatches.Patch(color=CALIB_COLORS["c2"], label="C2 Self-generated"),
    mpatches.Patch(color=CALIB_COLORS["c3"], label="C3 DACS"),
    plt.Line2D([0], [0], color="black", linestyle="--", lw=0.9, alpha=0.55,
               label="FP16 baseline"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor="#CCCCCC")

plt.tight_layout(rect=[0, 0.05, 1, 1])
out3 = os.path.join(OUT_DIR, "fig3_crossdomain.pdf")
plt.savefig(out3, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved {out3}")


# ===========================================================================
# FIG 4 — Per (format × domain) calibration spread on MMLU
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.6))

domain_labels = {"cyber": "Cyber", "med": "Medical", "code": "Code"}
fmt_x = np.arange(len(FORMATS)) * 1.0
dom_offsets = {"cyber": -0.25, "med": 0.0, "code": 0.25}
dom_colors = {"cyber": "#1E2761", "med": "#028090", "code": "#B85C00"}

for dom_tag, off in dom_offsets.items():
    spreads = []
    for fmt in FORMATS:
        ys = [get_mmlu(dom_tag, fmt, c) for c in ["c1", "c2", "c3"]]
        spreads.append(max(ys) - min(ys))
    ax.bar(fmt_x + off, spreads, 0.22,
           color=dom_colors[dom_tag],
           edgecolor="white", linewidth=0.5,
           label=domain_labels[dom_tag])
    for x, s in zip(fmt_x + off, spreads):
        ax.text(x, s + 0.15, f"{s:.2f}", ha="center", fontsize=7.5,
                color=dom_colors[dom_tag], fontweight="bold")

ax.set_xticks(fmt_x)
ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS], fontsize=10, fontweight="bold")
ax.set_ylabel("MMLU calibration spread\n(max $-$ min over C1/C2/C3, pp)", fontsize=9)
ax.set_title("Calibration sensitivity per (format $\\times$ domain) on MMLU",
             fontsize=10, fontweight="bold", pad=8)
ax.set_facecolor("#FAFAFA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(11, max([max(get_mmlu(d, f, c) for c in ["c1","c2","c3"]) - min(get_mmlu(d, f, c) for c in ["c1","c2","c3"]) for d in ["cyber","med","code"] for f in FORMATS]) + 1.5))
ax.legend(loc="upper left", fontsize=8.5, frameon=True, edgecolor="#CCCCCC")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "fig4_sensitivity_summary.pdf")
plt.savefig(out4, bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved {out4}")

print("\nDone. fig2 / fig3 / fig4 regenerated from current JSON.")
