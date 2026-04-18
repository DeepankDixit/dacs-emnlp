"""
DACS Activity 4 — Figure Generation
=====================================
Reads results/activity4_activation_results.json and produces:

  fig5_activation_ranges.pdf  — bar chart: mean range ratio per condition per domain
  fig6_kl_vs_regression.pdf   — scatter: C2 underestimation gap vs SQ C2 regression

Run after activation_analysis.py completes:
  python code/activity4/make_act4_figures.py

Outputs saved to paper/ directory (same as figs 1–4).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

RESULTS_PATH = "./results/activity4_activation_results.json"
OUT_DIR      = "./paper"

os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

# ---------------------------------------------------------------------------
# Known SQ C2 regression magnitudes from Activities 2 + 3
# (C1 MMLU − C2 MMLU, i.e. positive = C2 is worse)
# ---------------------------------------------------------------------------
SQ_C2_REGRESSION = {
    "cyber": 9.31,
    "med":   2.41,
    "code":  1.80,
}

DOMAIN_LABELS = {
    "cyber": "Cybersecurity\n(Llama 3.1 8B)",
    "med":   "Biomedical\n(Llama-3-Med42-8B)",
    "code":  "Code\n(CodeLlama-7B)",
}

COND_COLORS = {
    "c1": "#4472C4",   # blue — generic
    "c2": "#E05C34",   # red  — self-gen (the problem)
    "c3": "#70AD47",   # green — domain
}
COND_LABELS = {
    "c1": "C1 Generic",
    "c2": "C2 Self-gen",
    "c3": "C3 Domain",
}

# ---------------------------------------------------------------------------
# Fig 5: Mean range ratio per (domain × condition)
# ---------------------------------------------------------------------------

available_domains = [d for d in ["cyber", "med", "code"] if d in data]

if not available_domains:
    print("No domains in results yet — run activation_analysis.py first.")
else:
    n_domains = len(available_domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(3.8 * n_domains, 3.8),
                              sharey=True)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, available_domains):
        agg = data[domain]["aggregated"]
        conds  = ["c1", "c2", "c3"]
        ratios = [agg[c]["mean_range_ratio"] for c in conds]
        colors = [COND_COLORS[c] for c in conds]

        bars = ax.bar(
            [COND_LABELS[c] for c in conds],
            ratios,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.55,
        )

        # MMLU reference line (ratio = 1.0)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                   label="MMLU (inference)", zorder=3)

        # Annotate the C2 underestimation gap
        c2_ratio = agg["c2"]["mean_range_ratio"]
        c3_ratio = agg["c3"]["mean_range_ratio"]
        gap = 1.0 - c2_ratio
        if gap > 0.005:
            ax.annotate(
                f"↓{gap:.3f}\ngap",
                xy=(1, c2_ratio),
                xytext=(1.55, (c2_ratio + 1.0) / 2),
                fontsize=7,
                color="#E05C34",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#E05C34", lw=0.8),
            )

        # If C3 also underestimates (e.g. code domain), annotate with a note
        if c3_ratio < c2_ratio:
            ax.annotate(
                "C3 < C2:\ncode text\nnarrow",
                xy=(2, c3_ratio),
                xytext=(2.0, c3_ratio - 0.035),
                fontsize=6.5,
                color="#70AD47",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#70AD47", lw=0.7),
            )

        ax.set_title(DOMAIN_LABELS[domain], fontsize=9, fontweight="bold", pad=6)
        ax.set_ylabel("Mean per-channel range ratio\nvs. MMLU inference" if domain == available_domains[0] else "",
                      fontsize=8)
        ax.set_ylim(0.88, 1.08)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_facecolor("#FAFAFA")

        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, ratio + 0.003,
                    f"{ratio:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c])
        for c in ["c1", "c2", "c3"]
    ] + [plt.Line2D([0], [0], color="black", linestyle="--", lw=1.2,
                    label="MMLU inference (ratio = 1.0)")]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor="#CCCCCC")

    fig.suptitle(
        "Fig. 5: Per-channel input activation range under each calibration condition\n"
        "relative to MMLU inference. C2 underestimates in all domains; C3 ≈ MMLU for cyber/med,\n"
        "but also underestimates for code (code text is too syntactically narrow to cover MMLU range).",
        fontsize=8.5, y=1.03
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out5 = os.path.join(OUT_DIR, "fig5_activation_ranges.pdf")
    plt.savefig(out5, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out5}")

# ---------------------------------------------------------------------------
# Fig 6: C2 underestimation gap vs SQ C2 regression (scatter)
# ---------------------------------------------------------------------------

# Collect (gap, regression) pairs for all domains with data
points = []
for domain in ["cyber", "med", "code"]:
    if domain not in data:
        continue
    if domain not in SQ_C2_REGRESSION:
        continue
    agg = data[domain]["aggregated"]
    c2_ratio = agg["c2"]["mean_range_ratio"]
    gap = 1.0 - c2_ratio          # underestimation gap (higher = worse)
    regression = SQ_C2_REGRESSION[domain]
    points.append((gap, regression, domain))

if len(points) >= 2:
    gaps       = np.array([p[0] for p in points])
    regressions = np.array([p[1] for p in points])

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    DOMAIN_COLORS = {
        "cyber": "#1E2761",
        "med":   "#028090",
        "code":  "#B85C00",
    }

    for gap, reg, domain in points:
        ax.scatter(gap, reg, s=120,
                   color=DOMAIN_COLORS.get(domain, "gray"),
                   edgecolors="white", linewidth=1.2, zorder=5)
        ax.annotate(
            DOMAIN_LABELS[domain].replace("\n", " "),
            (gap, reg),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
        )

    # Trend line (only if ≥ 2 points)
    if len(points) >= 2:
        z = np.polyfit(gaps, regressions, 1)
        poly = np.poly1d(z)
        x_range = np.linspace(gaps.min() * 0.9, gaps.max() * 1.1, 100)
        ax.plot(x_range, poly(x_range), "--", color="gray",
                alpha=0.7, linewidth=1.2)

        # Pearson r
        from scipy.stats import pearsonr
        r, pval = pearsonr(gaps, regressions)
        ax.text(0.05, 0.92, f"Pearson r = {r:.2f}",
                transform=ax.transAxes, fontsize=9, color="gray")

    ax.set_xlabel("C2 activation underestimation gap\n(1 − mean range ratio)", fontsize=9)
    ax.set_ylabel("SQ C2 MMLU regression (pp)", fontsize=9)
    ax.set_title("Fig. 6: Activation range underestimation predicts\nSQ C2 calibration failure magnitude",
                 fontsize=9, fontweight="bold")
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    out6 = os.path.join(OUT_DIR, "fig6_kl_vs_regression.pdf")
    plt.savefig(out6, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out6}")
else:
    print(f"Not enough domains ({len(points)}) for fig 6 — need at least 2.")

print("\nAll Activity 4 figures done.")
print("Next: add fig5 and fig6 to main.tex (Section 'Mechanism')")
