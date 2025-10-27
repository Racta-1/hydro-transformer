#!/usr/bin/env python3
"""
Compare NSE distributions between upstream and combined datasets
using side-by-side bar charts.

Usage:
    python compare_nse_distribution.py \
        --upstream lstm_upstream_valbas_metrics.csv \
        --combined trans_comb_valbas_metrics.csv \
        --out nse_distribution_comparison
    python 03_summarize/nse_dist.py -u exp/lstm/lstm_upstream_2410_135040/test/model_epoch001/test_metrics.csv -c exp/lstm/lstm_combined_2210_174624/test/model_epoch001/test_metrics.csv -o 03_summarize/output/nse_dist
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Plot NSE basin count comparison (Upstream vs Combined).")
parser.add_argument("--upstream", "-u", required=True, help="CSV file containing NSE for upstream run.")
parser.add_argument("--combined", "-c", required=True, help="CSV file containing NSE for combined run.")
parser.add_argument("--out", "-o", default="nse_distribution_comparison", help="Output file base name (no extension).")
args = parser.parse_args()

# -----------------------------
# Settings
# -----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 400,
})

# -----------------------------
# Load and prepare
# -----------------------------
def load_metrics(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "NSE" not in df.columns:
        raise ValueError(f"{path} must contain an 'NSE' column.")
    df["NSE"] = pd.to_numeric(df["NSE"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["NSE"])

df_up = load_metrics(args.upstream)
df_comb = load_metrics(args.combined)

# -----------------------------
# Bin and summarize
# -----------------------------
bins = [-np.inf, 0.0, 0.5, 0.8, np.inf]
labels = ["< 0.0", "0.0–0.5", "0.5–0.8", "> 0.8"]

def summarize_bins(df):
    df["NSE_bin"] = pd.cut(df["NSE"], bins=bins, labels=labels, include_lowest=True, right=False)
    return (
        df.groupby("NSE_bin")["NSE"]
          .agg(["count", "mean"])
          .reset_index()
          .assign(mean=lambda d: d["mean"].round(2))
    )

summary_up = summarize_bins(df_up)
summary_comb = summarize_bins(df_comb)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)

colors = ["#a7c7e7", "#b0c4de"]  # blue shades

for ax, summary, title, label in zip(
    axes,
    [summary_comb, summary_up],
    ["(a) Combined", "(b) Upstream"],
    ["Upstream", "Combined"]
):
    bars = ax.bar(summary["NSE_bin"], summary["count"],
                  color=colors[0] if "Up" in label else colors[1],
                  edgecolor="gray", width=0.6)

    # add mean NSE above bars
    for rect, mean_val in zip(bars, summary["mean"]):
        h = rect.get_height()
        if np.isfinite(mean_val):
            ax.text(rect.get_x() + rect.get_width() / 2, h + summary["count"].max() * 0.02,
                    f"{mean_val:.2f}", ha="center", fontsize=6)

    ax.set_xlabel("NSE range")
    ax.set_title(title, fontsize=10, loc="center")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Count of basins")
plt.tight_layout()

# -----------------------------
# Save outputs
# -----------------------------
out_base = Path(args.out)
plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
print(f"Saved: {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")
plt.show()
