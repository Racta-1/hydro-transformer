#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# ======================================================
# CONFIG
# ======================================================

METRICS = ["NNSE", "KGE", "RMSE", "Pearson-r"]

COLORS = { 
    "LSTM Combined": (0.37, 0.78, 0.38, 1.0), # #5EC961 - solid green
    "LSTM Upstream": (0.55, 0.75, 0.56, 1.0), # same green
    "Transformer Combined": (0.27, 0.0, 0.33, 1.0), # #440154 - solid purple
    "Transformer Upstream": (0.50, 0.35, 0.55, 1.0), # same purple
}

LINE_STYLES = {
    "LSTM Combined": "-",  # solid line
    "LSTM Upstream": ":",  # dotted line
    "Transformer Combined": "-",  # solid line
    "Transformer Upstream": ":",  # dotted line
}

# ======================================================
# EXTRACT RAW METRICS
# ======================================================

def extract_metrics(df):
    """Extract MEDIAN values for each metric from the CSV."""
    values = {}
    df_cols = {c.lower(): c for c in df.columns}

    for m in METRICS:
        key = m.lower()
        if key in df_cols:
            col = df_cols[key]
            values[m] = df[col].median()
        else:
            print(f"⚠ WARNING: Column '{m}' not found in CSV.")
            values[m] = np.nan

    return values


# ======================================================
# PREPARE VALUES FOR RADAR
# ======================================================

def prepare_for_radar(vals):
    """Convert RMSE to score; return values in radar order."""
    rmse = vals["RMSE"]
    rmse_score = 1 / (1 + rmse) if not np.isnan(rmse) else np.nan

    return [
        vals["NNSE"],
        vals["KGE"],
        rmse_score,
        vals["Pearson-r"]
    ]


# ======================================================
# BAR CHART
# ======================================================

def plot_bar_chart(models, out_path="bar_chart_metrics.png"):
    """
    models = dict:
        {model_name: [NNSE, KGE, RMSE_score, Pearson-r]}
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(models.keys())
    metric_names = [m.upper() for m in METRICS]

    # Convert models dict to matrix form
    data = np.array(list(models.values()))
    x = np.arange(len(metric_names))
    width = 0.18  # bar width

    # Loop each model and plot a bar series
    for i, model_name in enumerate(model_names):
        # Use hatching for upstream models
        hatch = '//' if 'Upstream' in model_name else None
        
        ax.bar(
            x + (i - 1.5) * width,
            data[i],
            width,
            label=model_name,
            color=COLORS[model_name],
            alpha=0.8,
            hatch=hatch,
            edgecolor='black' if hatch else None,
            linewidth=0.5 if hatch else 0
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=14)
    ax.set_ylabel("Median Score", fontsize=14)
    ax.set_title("Median Metric Comparison Across All Models", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, frameon=True, edgecolor="#aaaaaa", framealpha=0.9, loc="upper center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved bar chart to {out_path}")


# ======================================================
# RADAR PLOT
# ======================================================

def plot_radar(models, out_path="radar_models.png"):

    metrics = METRICS.copy()
    num_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#f7f7f7")

    # Grid style
    ax.grid(color="#cccccc", linestyle="--", linewidth=0.8)
    ax.spines['polar'].set_color("#aaaaaa")
    ax.spines['polar'].set_linewidth(1.2)

    # Plot each model
    for model_name, vals in models.items():
        vals = np.concatenate((vals, [vals[0]]))

        ax.plot(
            angles, vals,
            linewidth=2.5,
            label=model_name,
            color=COLORS[model_name],
            linestyle=LINE_STYLES[model_name]
        )
        ax.fill(angles, vals, alpha=0.12, color=COLORS[model_name])

    # Boxed metric labels
    for angle, label in zip(angles[:-1], [m.upper() for m in metrics]):
        ax.text(
            angle,
            1.05,
            label,
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="#999999",
                alpha=0.9
            )
        )

    ax.set_title(
        "Model Performance Across Metrics\nLSTM vs Transformer (Combined vs Upstream)",
        fontsize=16,
        pad=25
    )

    legend = ax.legend(
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="#aaaaaa",
        fontsize=11,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved radar plot to {out_path}")


# ======================================================
# MAIN
# ======================================================

def main():

    # Hard-coded CSV paths (correct, cleaner)
    lstm_comb_df = pd.read_csv('../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv')
    lstm_up_df   = pd.read_csv("../exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv")

    trans_comb_df = pd.read_csv('../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv')
    trans_up_df   = pd.read_csv("../exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv")

    # Extract medians
    lstm_comb = extract_metrics(lstm_comb_df)
    lstm_up   = extract_metrics(lstm_up_df)
    trans_comb = extract_metrics(trans_comb_df)
    trans_up   = extract_metrics(trans_up_df)

    # Prepare for radar / bar plots
    models = {
        "LSTM Combined": prepare_for_radar(lstm_comb),
        "LSTM Upstream": prepare_for_radar(lstm_up),
        "Transformer Combined": prepare_for_radar(trans_comb),
        "Transformer Upstream": prepare_for_radar(trans_up),
    }

    # ---- Radar Plot ----
    plot_radar(models, out_path="radar_models.png")

    # ---- Bar Chart ----
    plot_bar_chart(models, out_path="bar_chart_metrics.png")


if __name__ == "__main__":
    main()