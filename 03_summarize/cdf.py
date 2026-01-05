import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# Academic Style
# ---------------------------------------------------------
def set_academic_style():
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.7,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
    })


# ---------------------------------------------------------
# Colors for LSTM + Transformer only
# ---------------------------------------------------------
COLORS = {
    'Transformer': '#440154',    # Dark gray
    'LSTM': '#5EC961',
    'Informer': '#3A528B',       # Taupe
    'CNN-1D': '#20908C',         # Light taupe
}
def get_color(model): return COLORS[model]


# ---------------------------------------------------------
# Load NNSE from CSV
# ---------------------------------------------------------
def load_nnse_data(csv_paths):
    """Return dict: {model -> NNSE array}"""
    data = {}
    for model, path in csv_paths.items():
        df = pd.read_csv(path)
        nnse_col = None
        for col in df.columns:
            if "nnse" in col.lower():
                nnse_col = col
                break
        if nnse_col is None:
            raise ValueError(f"NNSE column not found in {path}")

        data[model] = df[nnse_col].values
    return data


# ---------------------------------------------------------
# 1. Combined Violin + BoxPlot (LSTM + Transformer only)
# ---------------------------------------------------------
def plot_violin_box(nnse_combined, nnse_upstream, save_folder):

    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)

    models = ["Transformer", "LSTM", "Informer", "CNN-1D"]

    # Dynamic positions for 4 models
    base_positions = np.arange(1, len(models) + 1)
    positions_c = base_positions - 0.1   # Combined
    positions_u = base_positions + 0.1   # Upstream

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Violin: Combined ---
    vp_c = ax.violinplot(
        [nnse_combined[m] for m in models],
        positions=positions_c,
        widths=0.28,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, model in zip(vp_c["bodies"], models):
        body.set_facecolor(get_color(model))
        body.set_alpha(0.35)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    # --- Violin: Upstream ---
    vp_u = ax.violinplot(
        [nnse_upstream[m] for m in models],
        positions=positions_u,
        widths=0.28,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, model in zip(vp_u["bodies"], models):
        body.set_facecolor("white")
        body.set_alpha(0.45)
        body.set_edgecolor(get_color(model))
        body.set_linewidth(1.2)
        body.set_linestyle("--")

    # --- Boxplots: Combined ---
    bp_c = ax.boxplot(
        [nnse_combined[m] for m in models],
        positions=positions_c,
        widths=0.12,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=4),
    )

    for patch, model in zip(bp_c["boxes"], models):
        patch.set_facecolor(get_color(model))
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")

    # --- Boxplots: Upstream ---
    bp_u = ax.boxplot(
        [nnse_upstream[m] for m in models],
        positions=positions_u,
        widths=0.12,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=4),
        boxprops=dict(linestyle='--'),
        whiskerprops=dict(linestyle='--'),
        medianprops=dict(color='blue', linestyle='--'),
    )

    for patch, model in zip(bp_u["boxes"], models):
        patch.set_facecolor("white")
        patch.set_edgecolor(get_color(model))
        patch.set_linewidth(1.3)

    # Axis styling
    ax.set_xticks(base_positions)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("NNSE")
    ax.set_title("NNSE Distribution: Combined vs Upstream")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor='gray', edgecolor='black', alpha=0.6, label='Combined'),
            Patch(facecolor='white', edgecolor='black', label='Upstream', linestyle='--')
        ],
        loc="lower right",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(f"{save_folder}/violin_box_all_models.png", bbox_inches="tight")
    plt.savefig(f"{save_folder}/violin_box_all_models.pdf", bbox_inches="tight")
    plt.close()



# ---------------------------------------------------------
# 2. CDF Plot (LSTM + Transformer only)
# ---------------------------------------------------------
def plot_cdf(nnse_combined, nnse_upstream, save_folder):

    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)

    models = ["Transformer", "LSTM", "Informer", "CNN-1D"]

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Combined ---
    for model in models:
        data = nnse_combined[model]
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(sorted_data, yvals, color=get_color(model), linewidth=1.8, label=f"{model} (Combined)")

    # --- Upstream ---
    for model in models:
        data = nnse_upstream[model]
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(sorted_data, yvals, color=get_color(model), linestyle="--", linewidth=1.2,
                alpha=0.9, label=f"{model} (Upstream)")

    ax.set_xlabel("NNSE")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of NNSE Scores")
    ax.legend(frameon=True, loc="lower right", ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_folder}/cdf_all_models.png", bbox_inches="tight")
    plt.savefig(f"{save_folder}/cdf_all_models.pdf", bbox_inches="tight")
    plt.close()



# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    csv_combined = {
        "Transformer": "../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
        "LSTM": "../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv",
        "Informer": "../exp/informer1/informer_combined_2211_030811/resume_from002/test/model_epoch001/test_metrics.csv",
        "CNN-1D": "../exp/cnn1/cnn_combined_2211_191525/test/model_epoch002/test_metrics.csv"
    }

    csv_upstream = {
        "Transformer": "../exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
        "LSTM": "../exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv",
        "Informer": "../exp/informer1/informer_upstream_2211_082817/test/model_epoch002/test_metrics.csv",
        "CNN-1D": "../exp/cnn1/cnn_upstream_2211_150644/test/model_epoch002/test_metrics.csv"
    }

    nnse_comb = load_nnse_data(csv_combined)
    nnse_up = load_nnse_data(csv_upstream)

    plot_violin_box(nnse_comb, nnse_up, save_folder="plots_simplified")
    plot_cdf(nnse_comb, nnse_up, save_folder="plots_simplified")

    print("Saved: violin/boxplot and CDF plot for Transformer + LSTM only.")
