import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ----------------------------------------------------------------------
# Academic style configuration
# ----------------------------------------------------------------------
def set_academic_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3
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
# Load nNSE data from CSV files
# ----------------------------------------------------------------------
def load_nnse_data(files_dict):
    """Load nNSE scores from CSV files."""
    data = {}
    for model_name, filepath in files_dict.items():
        df = pd.read_csv(filepath)
        # Find nNSE column
        nnse_col = None
        for col in df.columns:
            if 'nnse' in col.lower():
                nnse_col = col
                break
        
        if nnse_col is None:
            raise ValueError(f"NNSE column not found in {filepath}")
        
        data[model_name] = df[nnse_col].values
    
    return data


# ----------------------------------------------------------------------
# Scatter plot with individual subplots for each model
# ----------------------------------------------------------------------
def plot_individual_scatters(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    n_models = len(data_dict)
    
    # Create subplots - adjust layout based on number of models
    if n_models <= 3:
        nrows, ncols = 1, n_models
        figsize = (5 * n_models, 4)
    elif n_models == 4:
        nrows, ncols = 2, 2
        figsize = (10, 8)
    else:  # 5 or 6 models
        nrows, ncols = 2, 3
        figsize = (14, 8)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    # Calculate global y-axis limits for consistency
    all_values = np.concatenate(list(data_dict.values()))
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_margin = (y_max - y_min) * 0.1
    
    # Plot each model in its own subplot
    for idx, (model_name, nnse_values) in enumerate(data_dict.items()):
        ax = axes[idx]
        basin_indices = np.arange(len(nnse_values))
        
        # Scatter plot
        ax.scatter(basin_indices, nnse_values,
                  color=COLORS[model_name],
                  alpha=0.6,
                  s=40,
                  edgecolors='black',
                  linewidth=0.5)
        
        # Add horizontal line at nNSE=0
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add mean line
        mean_nnse = np.mean(nnse_values)
        ax.axhline(y=mean_nnse, color=COLORS[model_name], 
                  linestyle=':', linewidth=1.5, alpha=0.8,
                  label=f'Mean: {mean_nnse:.3f}')
        
        # Styling
        ax.set_xlabel("Basin Index")
        ax.set_ylabel("NNSE Score")
        ax.set_title(f"{model_name}")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        
        # Add statistics text box
        median_nnse = np.median(nnse_values)
        std_nnse = np.std(nnse_values)
        stats_text = f'Median: {median_nnse:.3f}\nStd: {std_nnse:.3f}'
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.8, edgecolor='gray'))
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("NNSE Scores Across Basins by Model", 
                fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    png_path = os.path.join(save_folder, "nnse_scatter_subplots.png")
    pdf_path = os.path.join(save_folder, "nnse_scatter_subplots.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual scatter subplots: {png_path}, {pdf_path}")


# ----------------------------------------------------------------------
# Scatter plot with individual subplots - Vertical layout
# ----------------------------------------------------------------------
def plot_individual_scatters_vertical(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    n_models = len(data_dict)
    
    # Vertical stacking
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3.5 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    # Calculate global y-axis limits
    all_values = np.concatenate(list(data_dict.values()))
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_margin = (y_max - y_min) * 0.1
    
    # Plot each model
    for idx, (model_name, nnse_values) in enumerate(data_dict.items()):
        ax = axes[idx]
        basin_indices = np.arange(len(nnse_values))
        
        ax.scatter(basin_indices, nnse_values,
                  color=COLORS[model_name],
                  alpha=0.6,
                  s=35,
                  edgecolors='black',
                  linewidth=0.5)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        mean_nnse = np.mean(nnse_values)
        ax.axhline(y=mean_nnse, color=COLORS[model_name], 
                  linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Styling
        if idx == n_models - 1:
            ax.set_xlabel("Basin Index")
        ax.set_ylabel("NNSE")
        ax.set_title(f"{model_name} (Mean: {mean_nnse:.3f}, Median: {np.median(nnse_values):.3f})", 
                    fontweight='bold', loc='left')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.suptitle("NNSE Scores Across Basins by Model", 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    png_path = os.path.join(save_folder, "nnse_scatter_subplots_vertical.png")
    pdf_path = os.path.join(save_folder, "nnse_scatter_subplots_vertical.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved vertical scatter subplots: {png_path}, {pdf_path}")


# ----------------------------------------------------------------------
# Scatter plot with trend lines
# ----------------------------------------------------------------------
def plot_individual_scatters_with_trend(data_dict, save_folder="plots"):
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    n_models = len(data_dict)
    
    if n_models <= 3:
        nrows, ncols = 1, n_models
        figsize = (5 * n_models, 4)
    elif n_models == 4:
        nrows, ncols = 2, 2
        figsize = (10, 8)
    else:
        nrows, ncols = 2, 3
        figsize = (14, 8)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_values = np.concatenate(list(data_dict.values()))
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_margin = (y_max - y_min) * 0.1
    
    for idx, (model_name, nnse_values) in enumerate(data_dict.items()):
        ax = axes[idx]
        basin_indices = np.arange(len(nnse_values))
        
        # Scatter
        ax.scatter(basin_indices, nnse_values,
                  color=COLORS[model_name],
                  alpha=0.6,
                  s=40,
                  edgecolors='black',
                  linewidth=0.5,
                  label='Basin NNSE')
        
        # Polynomial trend line
        z = np.polyfit(basin_indices, nnse_values, 3)
        p = np.poly1d(z)
        ax.plot(basin_indices, p(basin_indices), 
               color='black', linestyle='-', linewidth=1.5, 
               alpha=0.7, label='Trend')
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel("Basin Index")
        ax.set_ylabel("NNSE Score")
        ax.set_title(f"{model_name}", fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=8)
    
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("NNSE Scores with Trend Lines", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    png_path = os.path.join(save_folder, "nnse_scatter_subplots_trend.png")
    pdf_path = os.path.join(save_folder, "nnse_scatter_subplots_trend.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter subplots with trend: {png_path}, {pdf_path}")


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    csv_files = {
        "PITransformer": "./exp/pitransformer1/transformer_combined_2411_190850/resume_from001/test/model_epoch001/test_metrics.csv",
        "Transformer": "./exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
        "FEDformer": "./exp/fedformer1/fedformer_combined_2111_203206/test/model_epoch002/test_metrics.csv",
        "Informer": "./exp/informer1/informer_combined_2211_030811/resume_from002/test/model_epoch001/test_metrics.csv",
        # "CNN-1D": "./exp/cnn1/cnn_combined_2211_191525/test/model_epoch002/test_metrics.csv",
        "LSTM": "./exp/lstm/lstm_combined_2210_174624/resume_from001/test/model_epoch001/test_metrics.csv",
    }
    
    
    # Load data
    nnse_data = load_nnse_data(csv_files)
    
    save_folder = "plots"
    
    # Generate different subplot layouts
    plot_individual_scatters(nnse_data, save_folder)           # Grid layout
    plot_individual_scatters_vertical(nnse_data, save_folder)  # Vertical stack
    plot_individual_scatters_with_trend(nnse_data, save_folder) # With trend lines