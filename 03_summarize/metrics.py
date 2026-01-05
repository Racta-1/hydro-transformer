import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
lstm_df = pd.read_csv('../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv')
transformer_df = pd.read_csv('../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
lstm_comb = pd.read_csv('../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv')
lstm_up = pd.read_csv("../exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv")

trans_comb = pd.read_csv('../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv')
trans_up = pd.read_csv("../exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv")

# Metrics to consider
metrics = ["NNSE", "KGE", "Pearson-r",  "RMSE",]

# Thresholds
thresholds = {
    "NNSE": ("greater", 0.75),
    # "NSE": ("greater", 0.50),
    "KGE": ("greater", 0.50),
    "Pearson-r": ("greater", 0.70),
    "RMSE": ("less", 2.0)
}

# Colors
model_colors = {
    "LSTM": "#5EC961",          # Green
    "Transformer": "#440154"    # Purple
}

def compute_counts_threshold(df, metrics):
    """Return number of basins meeting threshold conditions per metric."""
    counts = []
    for m in metrics:
        rule, th = thresholds[m]
        if rule == "greater":
            counts.append((df[m] > th).sum())
        else:  # less
            counts.append((df[m] < th).sum())
    return counts

def compute_percentage_threshold(df, metrics):
    """Return percentage of basins meeting threshold per metric."""
    percentages = []
    n = len(df)
    for m in metrics:
        rule, th = thresholds[m]
        if rule == "greater":
            pct = 100 * (df[m] > th).sum() / n
        else:
            pct = 100 * (df[m] < th).sum() / n
        percentages.append(pct)
    return percentages

# def plot_bar(ax, combined_counts, upstream_counts, model_name):
#     x = np.arange(len(metrics))
#     width = 0.35

#     # Combined bars
#     ax.bar(x - width/2, combined_counts, width, 
#            label='Combined', color=model_colors[model_name], alpha=0.85)

#     # Upstream bars
#     ax.bar(x + width/2, upstream_counts, width, 
#            label='Upstream', color=model_colors[model_name], alpha=0.35)

#     ax.set_xticks(x)
#     ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)
#     ax.set_ylabel("Number of Basins Meeting Criterion", fontsize=13)
#     ax.set_title(f"{model_name}: Basins Meeting Performance Thresholds", 
#                  fontsize=14, fontweight='bold')
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
#     ax.legend()

def plot_bar(ax, combined_pct, upstream_pct, model_name):
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, combined_pct, width,
           label='Combined', color=model_colors[model_name], alpha=0.85)

    ax.bar(x + width/2, upstream_pct, width,
           label='Upstream', color=model_colors[model_name], alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)

    ax.set_ylabel("Percentage of Basins Meeting Thresholds (%)",
                  fontsize=13)

    ax.set_title(f"{model_name}: Basin Threshold Compliance",
                 fontsize=12)

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend()


# -------- FIGURE 1: LSTM --------
lstm_comb_counts = compute_percentage_threshold(lstm_comb, metrics)
lstm_up_counts = compute_percentage_threshold(lstm_up, metrics)

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Threshold box text
threshold_text = (
    "Performance Thresholds:\n"
    "• NNSE > 0.75\n"
    "• KGE > 0.50\n"
    "• Pearson-r > 0.70\n"
    "• RMSE < 2.0"
)

# Add threshold box
fig1.text(
    0.72, 0.20,
    threshold_text,
    fontsize=10,
    # fontweight='bold',
    ha='left', va='top',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.75)
)

plot_bar(ax1, lstm_comb_counts, lstm_up_counts, "LSTM")

plt.tight_layout()
plt.savefig("lstm_threshold_basin_counts.png", dpi=300)

# -------- FIGURE 2: Transformer --------
trans_comb_counts = compute_percentage_threshold(trans_comb, metrics)
trans_up_counts = compute_percentage_threshold(trans_up, metrics)

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Add threshold box
fig2.text(
    0.72, 0.20,
    threshold_text,
    fontsize=10,
    # fontweight='bold',
    ha='left', va='top',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.85)
)

plot_bar(ax2, trans_comb_counts, trans_up_counts, "Transformer")

plt.tight_layout()
plt.savefig("transformer_threshold_basin_counts.png", dpi=300)
