"""
Compare hydrologic metrics (NSE, NNSE, KGE, RMSE, MSE, Pearson-r)
between two datasets (e.g., upstream vs combined) using CDF plots.

Usage:
    python cdf_metric.py --upstream lstm_upstream_valbas_metrics.csv \
                               --combined trans_comb_valbas_metrics.csv \
                               --out cdf_metrics_comparison
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Plot CDFs of hydrologic metrics from two CSV files.")
parser.add_argument("--upstream", "-u", required=True, help="Path to upstream CSV file (metrics).")
parser.add_argument("--combined", "-c", required=True, help="Path to combined CSV file (metrics).")
parser.add_argument("--out", "-o", default="metric_cdf_comparison", help="Output file base name (no extension).")
args = parser.parse_args()

# -----------------------------
# Settings
# -----------------------------
metrics = ["NSE", "NNSE", "KGE", "RMSE", "MSE", "Pearson-r"]
nrows, ncols = 3, 3  # 3x3 layout (some cells unused)
up_color = "#1f77b4"    # blue
comb_color = "#d62728"  # red

# For consistent figure sizing in papers
plt.rcParams.update({
    "font.size": 3,
    "axes.titlesize": 2,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.3
})

# -----------------------------
# Load data
# -----------------------------
df_up = pd.read_csv(args.upstream)
df_comb = pd.read_csv(args.combined)

# Strip column names of spaces
df_up.columns = df_up.columns.str.strip()
df_comb.columns = df_comb.columns.str.strip()

# -----------------------------
# CDF computation helper
# -----------------------------
def get_cdf_data(x):
    """Return sorted x and corresponding cumulative probabilities."""
    x = np.asarray(pd.to_numeric(x, errors="coerce").dropna())
    if len(x) == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.linspace(0, 1, len(x))
    return x, y

# -----------------------------
# Figure setup
# -----------------------------
fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 6))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    if metric not in df_up.columns or metric not in df_comb.columns:
        ax.text(0.5, 0.5, f"{metric}\nmissing", ha="center", va="center")
        ax.axis("off")
        continue

    x_up, y_up = get_cdf_data(df_up[metric])
    x_comb, y_comb = get_cdf_data(df_comb[metric])

    # Choose x-range depending on metric
    if metric in ["NSE", "KGE"]:
        xmin, xmax = -1.0, 1.0
    elif metric in ["RMSE", "MSE"]:
        xmax = np.nanpercentile(np.concatenate([x_up, x_comb]), 99)
        xmin = 0
    else:
        xmin, xmax = np.nanmin(np.concatenate([x_up, x_comb])), np.nanmax(np.concatenate([x_up, x_comb]))

    # Plot CDFs
    ax.plot(x_up, y_up, color=up_color, label="Upstream")
    ax.plot(x_comb, y_comb, color=comb_color, label="Combined")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(metric)
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    # place subplot label below the x-axis, centered
    ax.text(0.5, -0.22, f"({chr(97+i)})",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=6)


# Remove unused subplots if fewer than 9
for j in range(len(metrics), nrows * ncols):
    axes[j].axis("off")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Save both PNG and PDF
out_base = Path(args.out)
plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=400)
plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=400)
print(f"Saved: {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")
plt.show()
