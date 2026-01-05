import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ----------------------------------------------------------------------
# Academic style configuration
# ----------------------------------------------------------------------
def set_academic_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.2
    })


# Academic color palette
COLORS = {
    'PITransformer': '#2E4057',  # Deep blue
    'Transformer': '#048A81',    # Sea green
    'FEDformer': '#54C6EB',      # Cadet blue
    'Informer': '#F18F01',       # Peru (tan/orange)
    # 'CNN-1D': '#8B4513',         # Saddle brown
    'LSTM': '#C73E1D'          # Purple slate
}


# ----------------------------------------------------------------------
# Load NSE data from CSV files
# ----------------------------------------------------------------------
def load_nnse_data(files_dict):
    """
    Load NSE scores from CSV files.
    Assumes CSV has columns: basin_id, NSE (or similar)
    """
    data = {}
    for model_name, filepath in files_dict.items():
        df = pd.read_csv(filepath)
        # Adjust column name if needed - common variations:
        nse_col = None
        for col in df.columns:
            if 'nnse' in col.lower() or 'NNSE' in col:
                nse_col = col
                break
        
        if nse_col is None:
            raise ValueError(f"NNSE column not found in {filepath}. Columns: {df.columns.tolist()}")
        
        data[model_name] = df[nse_col].values
    
    return data


# ----------------------------------------------------------------------
# 2. Box Plot: Distribution comparison
# ----------------------------------------------------------------------
def plot_boxplot(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    # Prepare data for boxplot
    df_list = []
    for model_name, nnse_values in data_dict.items():
        df_temp = pd.DataFrame({
            'Model': model_name,
            'NNSE': nnse_values
        })
        df_list.append(df_temp)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot with custom colors
    box_parts = ax.boxplot([data_dict[model] for model in data_dict.keys()],
                           labels=data_dict.keys(),
                           patch_artist=True,
                           widths=0.6,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', 
                                        markeredgecolor='black', markersize=6))
    
    # Apply colors
    for patch, model in zip(box_parts['boxes'], data_dict.keys()):
        patch.set_facecolor(COLORS[model])
        patch.set_edgecolor('black')
        patch.set_alpha(0.7)
    
    # Style whiskers, caps, and medians
    for whisker in box_parts['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for cap in box_parts['caps']:
        cap.set(color='black', linewidth=1.2)
    for median in box_parts['medians']:
        median.set(color='darkred', linewidth=2)
    
    ax.set_ylabel("NNSE Score")
    ax.set_xlabel("Model")
    ax.set_title("Distribution of NNSE Scores by Model")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    png_path = os.path.join(save_folder, "nnse_boxplot.png")
    pdf_path = os.path.join(save_folder, "nnse_boxplot.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved boxplot: {png_path}, {pdf_path}")


# ----------------------------------------------------------------------
# 3. Violin Plot: Detailed distribution
# ----------------------------------------------------------------------
def plot_violin(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    # Prepare data
    df_list = []
    for model_name, nnse_values in data_dict.items():
        df_temp = pd.DataFrame({
            'Model': model_name,
            'NSE': nnse_values
        })
        df_list.append(df_temp)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    parts = ax.violinplot([data_dict[model] for model in data_dict.keys()],
                         positions=range(len(data_dict)),
                         widths=0.7,
                         showmeans=True,
                         showmedians=True)
    
    # Color the violins
    for i, (pc, model) in enumerate(zip(parts['bodies'], data_dict.keys())):
        pc.set_facecolor(COLORS[model])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1)
    
    # Style the mean and median lines
    parts['cmeans'].set_edgecolor('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_edgecolor('darkred')
    parts['cmedians'].set_linewidth(2)
    
    ax.set_xticks(range(len(data_dict)))
    ax.set_xticklabels(data_dict.keys(), rotation=15, ha='right')
    ax.set_ylabel("NNSE Score")
    ax.set_xlabel("Model")
    ax.set_title("Distribution of NNSE Scores by Model (Violin Plot)")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    png_path = os.path.join(save_folder, "nnse_violin.png")
    pdf_path = os.path.join(save_folder, "nnse_violin.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved violin plot: {png_path}, {pdf_path}")

def plot_combined_upstream_box_violin(data_combined, data_upstream, save_folder="plots_combined_upstream"):
    """
    One figure with both Combined and Upstream distributions for each model.
    - Two violins per model
    - Two boxplots per model
    - Clear legend
    """

    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)

    models = list(data_combined.keys())
    n_models = len(models)

    # X-axis positions with slight offsets
    pos_base = np.arange(n_models)
    pos_combined = pos_base - 0.15
    pos_upstream = pos_base + 0.15

    fig, ax = plt.subplots(figsize=(7.5, 4))

    # -----------------------------------
    # 1. Violin plots
    # -----------------------------------

    # Combined violin
    vp_combined = ax.violinplot(
        [data_combined[m] for m in models],
        positions=pos_combined,
        widths=0.25,
        showmeans=False,
        showmedians=False
    )

    # Upstream violin
    vp_upstream = ax.violinplot(
        [data_upstream[m] for m in models],
        positions=pos_upstream,
        widths=0.25,
        showmeans=False,
        showmedians=False
    )

    # Style combined violins
    for body in vp_combined['bodies']:
        body.set_facecolor("#2E4057")  # Blue
        body.set_edgecolor("black")
        body.set_alpha(0.25)

    # Style upstream violins
    for body in vp_upstream['bodies']:
        body.set_facecolor("#C73E1D")  # Orange
        body.set_edgecolor("black")
        body.set_alpha(0.25)

    # -----------------------------------
    # 2. Boxplots (overlay)
    # -----------------------------------

    # Combined boxplot
    bp_combined = ax.boxplot(
        [data_combined[m] for m in models],
        positions=pos_combined,
        widths=0.18,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='white', markersize=5)
    )

    # Upstream boxplot
    bp_upstream = ax.boxplot(
        [data_upstream[m] for m in models],
        positions=pos_upstream,
        widths=0.18,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='white', markersize=5)
    )

    # Style boxes for Combined
    for patch in bp_combined['boxes']:
        patch.set_facecolor("#2E4057")
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    # Style boxes for Upstream
    for patch in bp_upstream['boxes']:
        patch.set_facecolor("#C73E1D")
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    # Style medians
    for median in bp_combined['medians']:
        median.set(color='darkblue', linewidth=2)
    for median in bp_upstream['medians']:
        median.set(color='darkred', linewidth=2)

    # -----------------------------------
    # Axes styling
    # -----------------------------------
    ax.set_xticks(pos_base)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel("NNSE Score")
    ax.set_title("NNSE Distribution by Model: Combined vs Upstream")

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="#2E4057", lw=8, label="Combined"),
        plt.Line2D([0], [0], color="#C73E1D", lw=8, label="Upstream")
    ]
    ax.legend(handles=handles, frameon=True)

    plt.tight_layout()

    # Save
    png_path = os.path.join(save_folder, "nnse_combined_upstream_box_violin.png")
    pdf_path = os.path.join(save_folder, "nnse_combined_upstream_box_violin.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Saved combined/upstream box-violin plot: {png_path}, {pdf_path}")

# ----------------------------------------------------------------------
# 4. Combined plot: Box + Violin overlay
# ----------------------------------------------------------------------
def plot_combined_box_violin(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Violin plot as background
    parts = ax.violinplot([data_dict[model] for model in data_dict.keys()],
                         positions=range(len(data_dict)),
                         widths=0.7,
                         showmeans=False,
                         showmedians=False)
    
    for i, (pc, model) in enumerate(zip(parts['bodies'], data_dict.keys())):
        pc.set_facecolor(COLORS[model])
        pc.set_edgecolor('black')
        pc.set_alpha(0.3)
        pc.set_linewidth(0.5)
    
    # Box plot overlay
    box_parts = ax.boxplot([data_dict[model] for model in data_dict.keys()],
                           positions=range(len(data_dict)),
                           widths=0.3,
                           patch_artist=True,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', 
                                        markeredgecolor='black', markersize=5))
    
    for patch, model in zip(box_parts['boxes'], data_dict.keys()):
        patch.set_facecolor(COLORS[model])
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)
        patch.set_linewidth(1.5)
    
    for median in box_parts['medians']:
        median.set(color='darkred', linewidth=2)
    
    ax.set_xticks(range(len(data_dict)))
    ax.set_xticklabels(data_dict.keys(), rotation=15, ha='right')
    ax.set_ylabel("NNSE Score")
    ax.set_xlabel("Model")
    ax.set_title("Distribution of NNSE Scores (Combined Box-Violin Plot)")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    png_path = os.path.join(save_folder, "nnse_combined_box_violin.png")
    pdf_path = os.path.join(save_folder, "nnse_combined_box_violin.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot: {png_path}, {pdf_path}")


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# -------------------------------------------------
# Cliff’s delta (effect size)
# -------------------------------------------------
def cliffs_delta(x, y):
    """
    Computes Cliff's Delta effect size for paired samples.
    Positive delta means x > y.
    """
    n = len(x)
    greater = sum([1 for i in range(n) if x[i] > y[i]])
    lesser = sum([1 for i in range(n) if x[i] < y[i]])
    return (greater - lesser) / n


# -------------------------------------------------
# Generate significance table
# -------------------------------------------------
def compute_significance_table(nnse_combined, nnse_upstream, output_path="significance_summary.txt"):
    """
    Computes Wilcoxon test, median differences, and Cliff's delta
    for each model and saves results to a TXT file.
    """
    results = []

    for model in nnse_combined.keys():
        x = nnse_combined[model]
        y = nnse_upstream[model]

        # Paired Wilcoxon test
        stat, p = wilcoxon(x, y, zero_method='wilcox', alternative='greater')

        # Effect size (Cliff's Delta)
        delta = cliffs_delta(x, y)

        # Median difference
        median_diff = np.median(x) - np.median(y)

        results.append({
            "Model": model,
            "Median_Combined": float(np.median(x)),
            "Median_Upstream": float(np.median(y)),
            "Δ Median (C - U)": float(median_diff),
            "Cliffs Delta": float(delta),
            "Wilcoxon p-value": float(p)
        })

    # ---- Format output text ----
    lines = []
    header = "="*80 + "\nSignificance Test Results (Combined vs Upstream)\n" + "="*80 + "\n"
    lines.append(header)

    for r in results:
        block = (
            f"\nModel: {r['Model']}\n"
            f"  Median (Combined): {r['Median_Combined']:.6f}\n"
            f"  Median (Upstream): {r['Median_Upstream']:.6f}\n"
            f"  Δ Median (C - U):  {r['Δ Median (C - U)']:.6f}\n"
            f"  Cliff's Delta:     {r['Cliffs Delta']:.6f}\n"
            f"  Wilcoxon p-value:  {r['Wilcoxon p-value']:.6e}\n"
        )
        lines.append(block)

    lines.append("\n" + "="*80 + "\n")

    # ---- Save to text file ----
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Significance summary successfully saved to: {output_path}")


def plot_side_by_side(data_combined, data_upstream, save_folder="plots_side_by_side"):
    """
    Create a single plot where each model has two boxplots:
    - Combined configuration
    - Upstream-only configuration
    """

    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)

    models = list(data_combined.keys())

    # Positions: every model gets two boxplots at positions i-0.2 and i+0.2
    positions_combined = np.arange(len(models)) - 0.2
    positions_upstream = np.arange(len(models)) + 0.2

    fig, ax = plt.subplots(figsize=(7.5, 4))

    # --- Combined Boxplots ---
    box_c = ax.boxplot(
        [data_combined[m] for m in models],
        positions=positions_combined,
        widths=0.35,
        patch_artist=True,
        labels=None,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='white', markersize=5)
    )

    # --- Upstream Boxplots ---
    box_u = ax.boxplot(
        [data_upstream[m] for m in models],
        positions=positions_upstream,
        widths=0.35,
        patch_artist=True,
        labels=None,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='white', markersize=5)
    )

    # Colors
    combined_color = "#2E4057"   # Blue
    upstream_color = "#C73E1D"   # Orange

    # Color combined
    for patch in box_c['boxes']:
        patch.set_facecolor(combined_color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)

    # Color upstream
    for patch in box_u['boxes']:
        patch.set_facecolor(upstream_color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)

    # Style medians
    for median in box_c['medians']:
        median.set(color='darkblue', linewidth=2)
    for median in box_u['medians']:
        median.set(color='darkred', linewidth=2)

    # X-axis labels centered between the two boxes
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')

    ax.set_ylabel("NNSE Score")
    ax.set_xlabel("Model")
    ax.set_title("NNSE Distribution: Combined vs Upstream Inputs")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=combined_color, lw=8, label="Combined"),
        plt.Line2D([0], [0], color=upstream_color, lw=8, label="Upstream"),
    ]
    ax.legend(handles=handles, frameon=True)

    plt.tight_layout()

    png_path = os.path.join(save_folder, "nnse_side_by_side.png")
    pdf_path = os.path.join(save_folder, "nnse_side_by_side.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Saved side-by-side comparison plot: {png_path}, {pdf_path}")

# ----------------------------------------------------------------------
# 5. Summary statistics
# ----------------------------------------------------------------------
def print_summary_statistics(data_dict):
    print("\n" + "="*70)
    print("NNSE Score Summary Statistics")
    print("="*70)
    
    for model_name, nnse_values in data_dict.items():
        print(f"\n{model_name}:")
        print(f"  Count:   {len(nnse_values)}")
        print(f"  Mean:    {np.mean(nnse_values):.4f}")
        print(f"  Median:  {np.median(nnse_values):.4f}")
        print(f"  Std Dev: {np.std(nnse_values):.4f}")
        print(f"  Min:     {np.min(nnse_values):.4f}")
        print(f"  Max:     {np.max(nnse_values):.4f}")
        print(f"  Q1:      {np.percentile(nnse_values, 25):.4f}")
        print(f"  Q3:      {np.percentile(nnse_values, 75):.4f}")

def save_basin_nnse_summary(
        nnse_combined, 
        nnse_upstream, 
        output_path="basin_nnse_summary.txt", 
        threshold=0.5):
    """
    Computes:
      - Basin count
      - Basins with NNSE > threshold
      - Mean NNSE (Combined and Upstream)
      - Median NNSE (Combined and Upstream)
      - % improvement in mean and median NNSE
    Saves results to TXT and prints them.
    """

    lines = []
    header = (
        f"{'='*80}\n"
        f"Basin Performance Summary (NNSE > {threshold})\n"
        f"{'='*80}\n"
    )
    print(header)
    lines.append(header)

    for model in nnse_combined.keys():
        combined_vals = nnse_combined[model]
        upstream_vals = nnse_upstream[model]

        # Basin count
        n_combined = len(combined_vals)
        n_upstream = len(upstream_vals)

        # Basins above threshold
        good_combined = np.sum(combined_vals > threshold)
        good_upstream = np.sum(upstream_vals > threshold)

        pct_combined = 100 * good_combined / n_combined
        pct_upstream = 100 * good_upstream / n_upstream

        # Mean & Median
        mean_combined = np.mean(combined_vals)
        mean_upstream = np.mean(upstream_vals)

        median_combined = np.median(combined_vals)
        median_upstream = np.median(upstream_vals)

        # Percent improvements
        pct_mean_improvement = 100 * (mean_combined - mean_upstream) / abs(mean_upstream)
        pct_median_improvement = 100 * (median_combined - median_upstream) / abs(median_upstream)

        block = (
            f"\nModel: {model}\n"
            f"  Basins (Combined): {n_combined}\n"
            f"  Basins (Upstream) : {n_upstream}\n"
            f"  NNSE > {threshold} (Combined): {good_combined} ({pct_combined:.2f}%)\n"
            f"  NNSE > {threshold} (Upstream) : {good_upstream} ({pct_upstream:.2f}%)\n"
            f"  Mean NNSE (Combined): {mean_combined:.4f}\n"
            f"  Mean NNSE (Upstream): {mean_upstream:.4f}\n"
            f"  % Mean Improvement:   {pct_mean_improvement:.2f}%\n"
            f"  Median NNSE (Combined): {median_combined:.4f}\n"
            f"  Median NNSE (Upstream): {median_upstream:.4f}\n"
            f"  % Median Improvement:   {pct_median_improvement:.2f}%\n"
        )

        print(block)
        lines.append(block)

    footer = f"{'='*80}\n"
    print(footer)
    lines.append(footer)

    # Save to TXT
    with open(output_path, "w") as f:import pandas as pd
import numpy as np
import os

# ----------------------------------------
# Helper: Automatically detect metric columns
# ----------------------------------------
def detect_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None


# ----------------------------------------
# Load metrics for one model
# ----------------------------------------
def load_metrics(filepath):
    df = pd.read_csv(filepath)

    metrics = {}
    metrics["NNSE"] = df[detect_column(df, ["nnse"])]
    metrics["KGE"] = df[detect_column(df, ["kge"])]
    metrics["PCC"] = df[detect_column(df, ["pearson-r"])]
    metrics["RMSE"] = df[detect_column(df, ["rmse"])]
    metrics["MAE"] = df[detect_column(df, ["mae"])]

    return metrics


# ----------------------------------------
# Compute statistics for upstream & combined
# ----------------------------------------
def compute_stats(metrics_up, metrics_comb, threshold=0.5):
    stats = {}

    for metric in metrics_comb.keys():
        x_up = metrics_up[metric]
        x_comb = metrics_comb[metric]

        stats[metric] = {
            "median_up": np.median(x_up),
            "median_comb": np.median(x_comb),
            "mean_up": np.mean(x_up),
            "mean_comb": np.mean(x_comb),
        }

    # NNSE-specific
    nnse_up = metrics_up["NNSE"]
    nnse_comb = metrics_comb["NNSE"]

    stats["basin_count_up"] = len(nnse_up)
    stats["basin_count_comb"] = len(nnse_comb)

    stats["nnse_pct_up"] = 100 * np.sum(nnse_up > threshold) / len(nnse_up)
    stats["nnse_pct_comb"] = 100 * np.sum(nnse_comb > threshold) / len(nnse_comb)

    return stats


# ----------------------------------------
# Produce LaTeX table rows
# ----------------------------------------
def generate_latex_table(model_name, stats):
    latex = []

    def fmt(x):
        return f"{x:.3f}"

    latex.append(f"% ------- {model_name} -------")
    latex.append(f"\\textbf{{NNSE}} & {fmt(stats['NNSE']['median_up'])} & {fmt(stats['NNSE']['median_comb'])} & {fmt(stats['NNSE']['mean_up'])} & {fmt(stats['NNSE']['mean_comb'])} \\\\")
    latex.append(f"\\textbf{{KGE}} & {fmt(stats['KGE']['median_up'])} & {fmt(stats['KGE']['median_comb'])} & {fmt(stats['KGE']['mean_up'])} & {fmt(stats['KGE']['mean_comb'])} \\\\")
    latex.append(f"\\textbf{{Pearson-$r$}} & {fmt(stats['PCC']['median_up'])} & {fmt(stats['PCC']['median_comb'])} & {fmt(stats['PCC']['mean_up'])} & {fmt(stats['PCC']['mean_comb'])} \\\\")
    latex.append(f"\\textbf{{RMSE}} & {fmt(stats['RMSE']['median_up'])} & {fmt(stats['RMSE']['median_comb'])} & {fmt(stats['RMSE']['mean_up'])} & {fmt(stats['RMSE']['mean_comb'])} \\\\")
    latex.append(f"\\textbf{{MAE}} & {fmt(stats['MAE']['median_up'])} & {fmt(stats['MAE']['median_comb'])} & {fmt(stats['MAE']['mean_up'])} & {fmt(stats['MAE']['mean_comb'])} \\\\")
    latex.append("\\midrule")
    latex.append(f"\\textbf{{Basin Count}} & {stats['basin_count_up']} & {stats['basin_count_comb']} & -- & -- \\\\")
    latex.append(f"\\textbf{{\\% Basins NNSE > 0.5}} & {stats['nnse_pct_up']:.2f}\\% & {stats['nnse_pct_comb']:.2f}\\% & -- & -- \\\\")
    latex.append("\\midrule\n")

    return "\n".join(latex)


# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------
def run_table_generator(combined_paths, upstream_paths, output_txt="latex_table_output.txt"):
    lines = []
    for model in combined_paths.keys():
        print(f"Processing {model}...")

        metrics_comb = load_metrics(combined_paths[model])
        metrics_up = load_metrics(upstream_paths[model])

        stats = compute_stats(metrics_up, metrics_comb)

        latex_rows = generate_latex_table(model, stats)
        lines.append(latex_rows)

    with open(output_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved LaTeX table rows to {output_txt}")


# ----------------------------------------
# EXAMPLE USAGE
# ----------------------------------------
# if __name__ == "__main__":
#     combined_csv = {
#         "Transformer": "./exp/transformer/combined/test_metrics.csv"
#     }

#     upstream_csv = {
#         "Transformer": "./exp/transformer/upstream/test_metrics.csv"
#     }

#     run_table_generator(combined_csv, upstream_csv)

#     f.writelines(lines)

#     print(f"Summary saved to: {output_path}")


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define your CSV file paths here
    # Each CSV should contain NSE scores for basins
    csv_files = {
        "PITransformer": "./exp/pitransformer1/transformer_combined_2411_190850/resume_from001/test/model_epoch001/test_metrics.csv",
        "Transformer": "./exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
        "FEDformer": "./exp/fedformer1/fedformer_combined_2111_203206/test/model_epoch002/test_metrics.csv",
        "Informer": "./exp/informer1/informer_combined_2211_030811/resume_from002/test/model_epoch001/test_metrics.csv",
        # "CNN-1D": "./exp/cnn1/cnn_combined_2211_191525/test/model_epoch002/test_metrics.csv",
        "LSTM": "./exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv",
    }

    csv_files_upstream = {
        "PITransformer": "./exp/pitransformer1/transformer_upstream_2411_091750/test/model_epoch002/test_metrics.csv",
        "Transformer": "./exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
        "FEDformer": "./exp/fedformer1/fedformer_upstream_2111_171235/test/model_epoch002/test_metrics.csv",
        "Informer": "./exp/informer1/informer_upstream_2211_082817/test/model_epoch002/test_metrics.csv",
        # "CNN-1D": "./exp/cnn1/cnn_upstream_2211_150644/test/model_epoch002/test_metrics.csv",
        "LSTM": "./exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv",
    }
    
    # Load NSE data
    nnse_data = load_nnse_data(csv_files)
    nnse_data_upstream = load_nnse_data(csv_files_upstream)

    
    # Generate all plots
    save_folder_combined = "plots_combined1"
    save_folder_upstream = "plots_upstream1"

    
    # Combined
    plot_boxplot(nnse_data, save_folder_combined)
    plot_violin(nnse_data, save_folder_combined)
    plot_combined_box_violin(nnse_data, save_folder_combined)

    # Upstream-only
    plot_boxplot(nnse_data_upstream, save_folder_upstream)
    plot_violin(nnse_data_upstream, save_folder_upstream)
    plot_combined_box_violin(nnse_data_upstream, save_folder_upstream)

    plot_side_by_side(nnse_data, nnse_data_upstream)
    plot_combined_upstream_box_violin(nnse_data, nnse_data_upstream)

    # significance_df = compute_significance_table(nnse_data, nnse_data_upstream)
    # print(significance_df.to_string(index=False))
    compute_significance_table(nnse_data, nnse_data_upstream, output_path="significance_summary.txt")

    save_basin_nnse_summary(nnse_data, nnse_data_upstream, output_path="basin_summary.txt")

    # ----------------------------------------
    # RUN LATEX TABLE GENERATOR FOR BOTH MODELS
    # ----------------------------------------

    combined_csv = {
        "Transformer": "./exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
        "LSTM": "./exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv"
    }

    upstream_csv = {
        "Transformer": "./exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
        "LSTM": "./exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv"
    }

    run_table_generator(combined_csv, upstream_csv, output_txt="latex_table_transformer_lstm.txt")


    # Print summary statistics
    # print_summary_statistics(nnse_data)