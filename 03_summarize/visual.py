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
        "LSTM": "./exp/lstm/lstm_combined_2210_174624/resume_from001/test/model_epoch001/test_metrics.csv",
    }

    csv_files_upstream = {
        "PITransformer": "./exp/pitransformer1/transformer_upstream_2411_091750/test/model_epoch002/test_metrics.csv",
        "Transformer": "./exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
        "FEDformer": "./exp/fedformer1/fedformer_upstream_2111_171235/test/model_epoch002/test_metrics.csv",
        "Informer": "./exp/informer1/informer_upstream_2211_082817/test/model_epoch002/test_metrics.csv",
        # "CNN-1D": "./exp/cnn1/cnn_upstream_2211_150644/test/model_epoch002/test_metrics.csv",
        "LSTM": "./exp/lstm/lstm_upstream_2410_135040/test/model_epoch001/test_metrics.csv",
    }
    
    # Load NSE data
    nnse_data = load_nnse_data(csv_files)
    nnse_data_upstream = load_nnse_data(csv_files_upstream)

    
    # Generate all plots
    save_folder_combined = "plots_combined"
    save_folder_upstream = "plots_upstream"

    
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



    
    # Print summary statistics
    # print_summary_statistics(nnse_data)