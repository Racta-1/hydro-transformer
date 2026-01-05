import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


# ----------------------------------------------------------------------
# Professional Academic Color Scheme (muted, print-friendly)
# ----------------------------------------------------------------------
# Option 1: Sophisticated grayscale with subtle tints (6 models)
COLORS = {
    # 'PITransformer': '#2E4057',  # Deep blue
    'Transformer': '#048A81',    # Sea green
    # 'FEDformer': '#54C6EB',      # Cadet blue
    'Informer': '#F18F01',       # Peru (tan/orange)
    'CNN-1D': '#8B4513',         # Saddle brown
    'LSTM': '#C73E1D'          # Purple slate
}
COLORS_GRAYSCALE = {
    # 'PITransformer': '#2C3E50',  # Dark slate blue
    'Transformer': '#34495E',     # Wet asphalt
    # 'FEDformer': '#7F8C8D',       # Concrete gray
    'Informer': '#95A5A6',        # Silver
    'CNN-1D': '#AAB7B8',          # Light silver-gray
    'LSTM': '#BDC3C7'             # Light gray
}

# Option 2: Colorblind-friendly palette (6 models)
COLORS_CB = {
    # 'PITransformer': '#0173B2',  # Blue
    'Transformer': '#DE8F05',    # Orange
    # 'FEDformer': '#029E73',      # Green
    'Informer': '#CC78BC',       # Purple
    'CNN-1D': '#CA9161',         # Tan/brown
    'LSTM': '#ECE133'            # Yellow
}

# Option 3: Elegant earth tones (6 models)
COLORS_EARTH = {
    # 'PITransformer': '#2C4251',  # Deep navy
    'Transformer': '#5B6D5B',    # Sage green
    # 'FEDformer': '#8D6E63',      # Clay brown
    'Informer': '#A1887F',       # Warm taupe
    'CNN-1D': '#B39A8C',         # Sand
    'LSTM': '#BCAAA4'            # Light beige
}

# Option 4: Professional blue-gray spectrum (6 models)
COLORS_BLUE = {
    # 'PITransformer': '#1A3A52',  # Deep ocean blue
    'Transformer': '#2E5266',    # Medium blue-gray
    # 'FEDformer': '#4A6D7C',      # Steel blue
    'Informer': '#6C8C9C',       # Powder blue
    'CNN-1D': '#8AA3AC',         # Medium blue-gray
    'LSTM': '#9FB3BC'            # Light blue-gray
}

# Option 5: Warm professional palette (6 models)
COLORS_WARM = {
    # 'PITransformer': '#3D3B3C',  # Charcoal
    'Transformer': '#5D5B5E',    # Dark gray
    # 'FEDformer': '#8B7D77',      # Warm gray
    'Informer': '#A39594',       # Taupe
    'CNN-1D': '#B4A9A8',         # Light taupe
    'LSTM': '#C4BCBB'            # Light warm gray
}

COLORS_ = {
    # 'PITransformer': '#3D3B3C',  # Charcoal
    'Transformer': '#440154',    # Dark gray
    # 'FEDformer': '#8B7D77',      # Warm gray
    'Informer': '#3A528B',       # Taupe
    'CNN-1D': '#20908C',         # Light taupe
    'LSTM': '#5EC961'            # Light warm gray
}

COLORS__ = {
    # 'PITransformer': '#3D3B3C',  # Charcoal
    'Transformer': '#5E1914',    # Dark gray
    # 'FEDformer': '#8B7D77',      # Warm gray
    'Informer': '#7C2F1F',       # Taupe
    'CNN-1D': '#5E143D',         # Light taupe
    'LSTM': '#D9C2C0'            # Light warm gray
}


# Select your preferred color scheme here:
COLORS = COLORS_  # Change to: COLORS_GRAYSCALE, COLORS_CB, COLORS_EARTH, COLORS_BLUE, or COLORS_WARM

# Quick toggle for colorblind mode
USE_COLORBLIND = False  # Set to True to override with colorblind-friendly palette


def get_color(model):
    return COLORS_CB[model] if USE_COLORBLIND else COLORS[model]


# ----------------------------------------------------------------------
# Academic style configuration
# ----------------------------------------------------------------------
def set_academic_style():
    plt.rcParams.update({
        # 'font.family': 'serif',
        # 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.6,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'grid.linestyle': '--',
        'axes.axisbelow': True
    })


# ----------------------------------------------------------------------
# 1. Delta/Improvement Plot (shows relative performance gains)
# ----------------------------------------------------------------------
def plot_performance_delta(data_combined, data_upstream, save_folder="plots"):
    """
    Visualize the improvement (delta) from upstream to combined configuration.
    Positive values = improvement with combined inputs.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    deltas_median = []
    deltas_mean = []
    
    for model in models:
        delta_median = np.median(data_combined[model]) - np.median(data_upstream[model])
        delta_mean = np.mean(data_combined[model]) - np.mean(data_upstream[model])
        deltas_median.append(delta_median)
        deltas_mean.append(delta_mean)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7.5, 4))
    
    bars1 = ax.bar(x - width/2, deltas_median, width, label='Median Δ',
                   color=[get_color(m) for m in models], alpha=0.8, 
                   edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, deltas_mean, width, label='Mean Δ',
                   color=[get_color(m) for m in models], alpha=0.5, 
                   edgecolor='black', linewidth=0.6, hatch='//')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('NNSE Improvement (Combined - Upstream)')
    ax.set_xlabel('Model')
    ax.set_title('Performance Gain: Combined vs Upstream Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(frameon=True, fancybox=False)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'performance_delta.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'performance_delta.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: performance_delta plots")


# ----------------------------------------------------------------------
# 2. Comparative Boxplot with Statistical Annotations
# ----------------------------------------------------------------------
def plot_comparative_boxplot_annotated(data_combined, data_upstream, save_folder="plots"):
    """
    Side-by-side boxplots with statistical significance indicators.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    from scipy.stats import wilcoxon
    
    models = list(data_combined.keys())
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(7.5, 4))
    
    positions_combined = np.arange(n_models) * 2.5 - 0.4
    positions_upstream = np.arange(n_models) * 2.5 + 0.4
    
    # Combined boxplots
    bp_combined = ax.boxplot(
        [data_combined[m] for m in models],
        positions=positions_combined,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=3)
    )
    
    # Upstream boxplots
    bp_upstream = ax.boxplot(
        [data_upstream[m] for m in models],
        positions=positions_upstream,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=3)
    )
    
    # Color boxes
    for patch, model in zip(bp_combined['boxes'], models):
        patch.set_facecolor(get_color(model))
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    for patch, model in zip(bp_upstream['boxes'], models):
        patch.set_facecolor('white')
        patch.set_alpha(0.7)
        patch.set_edgecolor(get_color(model))
        patch.set_linewidth(1.5)
        patch.set_linestyle('--')
    
    # Add significance stars
    max_val = max([max(data_combined[m].max(), data_upstream[m].max()) for m in models])
    for i, model in enumerate(models):
        stat, p_value = wilcoxon(data_combined[model], data_upstream[model])
        
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        x_pos = i * 2.5
        y_pos = max_val * 1.05
        ax.text(x_pos, y_pos, sig_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Styling
    ax.set_xticks(np.arange(n_models) * 2.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('NNSE Score')
    ax.set_title('Model Performance Comparison: Combined vs Upstream\n(*** p<0.001, ** p<0.01, * p<0.05, ns: not significant)')
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Combined'),
        Patch(facecolor='white', edgecolor='gray', linestyle='--', linewidth=1.5, label='Upstream')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'comparative_boxplot_annotated.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'comparative_boxplot_annotated.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: comparative_boxplot_annotated plots")


# ----------------------------------------------------------------------
# 3. Ranked Performance Heatmap
# ----------------------------------------------------------------------
def plot_performance_heatmap(data_combined, data_upstream, save_folder="plots"):
    """
    Heatmap showing percentile performance across all basins.
    Each cell shows % of basins where model achieves that performance level.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Calculate percentages for combined
    perf_matrix_comb = np.zeros((len(models), len(thresholds)))
    for i, model in enumerate(models):
        for j, thresh in enumerate(thresholds):
            perf_matrix_comb[i, j] = 100 * np.mean(data_combined[model] > thresh)
    
    # Calculate percentages for upstream
    perf_matrix_up = np.zeros((len(models), len(thresholds)))
    for i, model in enumerate(models):
        for j, thresh in enumerate(thresholds):
            perf_matrix_up[i, j] = 100 * np.mean(data_upstream[model] > thresh)
    
    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))
    
    # Combined heatmap
    im1 = ax1.imshow(perf_matrix_comb, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(thresholds)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels([f'>{t}' for t in thresholds])
    ax1.set_yticklabels(models)
    ax1.set_xlabel('NNSE Threshold')
    ax1.set_title('Combined Configuration')
    
    # Add percentage text
    for i in range(len(models)):
        for j in range(len(thresholds)):
            text = ax1.text(j, i, f'{perf_matrix_comb[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=6)
    
    # Upstream heatmap
    im2 = ax2.imshow(perf_matrix_up, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(thresholds)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels([f'>{t}' for t in thresholds])
    ax2.set_yticklabels(models)
    ax2.set_xlabel('NNSE Threshold')
    ax2.set_title('Upstream Configuration')
    
    # Add percentage text
    for i in range(len(models)):
        for j in range(len(thresholds)):
            text = ax2.text(j, i, f'{perf_matrix_up[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=6)
    
    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], label='% Basins Above Threshold', shrink=0.8)
    
    plt.suptitle('Basin Performance Distribution Across NNSE Thresholds', fontsize=10, y=0.98)
    
    # Use bbox_inches='tight' instead of tight_layout for heatmaps
    plt.savefig(os.path.join(save_folder, 'performance_heatmap.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(save_folder, 'performance_heatmap.pdf'), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved: performance_heatmap plots")


# ----------------------------------------------------------------------
# 4. Cumulative Distribution Function (CDF) Plot
# ----------------------------------------------------------------------
def plot_cdf_comparison(data_combined, data_upstream, save_folder="plots"):
    """
    CDF curves showing the distribution of NNSE scores.
    Steeper curves to the right indicate better overall performance.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))
    
    # Combined configuration
    for model in models:
        sorted_data = np.sort(data_combined[model])
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax1.plot(sorted_data, y_vals, label=model, color=get_color(model), 
                linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('NNSE Score')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('Combined Configuration')
    ax1.legend(frameon=True, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='NNSE=0.5')
    
    # Upstream configuration
    for model in models:
        sorted_data = np.sort(data_upstream[model])
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, y_vals, label=model, color=get_color(model), 
                linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax2.set_xlabel('NNSE Score')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Upstream Configuration')
    ax2.legend(frameon=True, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle('Cumulative Distribution of NNSE Scores', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'cdf_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'cdf_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: cdf_comparison plots")


# ----------------------------------------------------------------------
# 5. Pairwise Model Comparison Matrix
# ----------------------------------------------------------------------
def plot_pairwise_comparison_matrix(data_combined, save_folder="plots"):
    """
    Matrix showing win rate of row model vs column model.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    n_models = len(models)
    win_matrix = np.zeros((n_models, n_models))
    
    # Calculate win rates
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                # For each basin, check which model performs better
                wins = np.sum(data_combined[model_i] > data_combined[model_j])
                total = len(data_combined[model_i])
                win_matrix[i, j] = 100 * wins / total
            else:
                win_matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(7.5, 4))
    
    # Create masked array for diagonal
    masked_matrix = np.ma.masked_invalid(win_matrix)
    
    im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels(models)
    ax.set_xlabel('Model (Column)')
    ax.set_ylabel('Model (Row)')
    ax.set_title('Win Rate Matrix: % Basins Where Row Model > Column Model\n(Combined Configuration)')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax.text(j, i, f'{win_matrix[i, j]:.1f}',
                             ha="center", va="center", 
                             color="white" if win_matrix[i, j] > 50 else "black",
                             fontsize=7, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Win Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'pairwise_comparison_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'pairwise_comparison_matrix.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: pairwise_comparison_matrix plots")


# # ----------------------------------------------------------------------
# # 6. Combined Violin and Box Plot (Overlay)
# # ----------------------------------------------------------------------
# def plot_combined_violin_box(data_combined, data_upstream, save_folder="plots"):
#     """
#     Combined violin and box plot showing distribution comparison.
#     Violin shows full distribution, box shows quartiles and outliers.
#     """
#     set_academic_style()
#     os.makedirs(save_folder, exist_ok=True)
    
#     models = list(data_combined.keys())
#     n_models = len(models)
    
#     fig, ax = plt.subplots(figsize=(7.5, 4))
    
#     positions_combined = np.arange(n_models) * 2.5 - 0.3
#     positions_upstream = np.arange(n_models) * 2.5 + 0.3
    
#     # --- Violin plots (Combined) ---
#     vp_combined = ax.violinplot(
#         [data_combined[m] for m in models],
#         positions=positions_combined,
#         widths=0.5,
#         showmeans=False,
#         showmedians=False,
#         showextrema=False
#     )
    
#     # Style combined violins
#     for i, (body, model) in enumerate(zip(vp_combined['bodies'], models)):
#         body.set_facecolor(get_color(model))
#         body.set_edgecolor('black')
#         body.set_alpha(0.3)
#         body.set_linewidth(0.8)
    
#     # --- Violin plots (Upstream) ---
#     vp_upstream = ax.violinplot(
#         [data_upstream[m] for m in models],
#         positions=positions_upstream,
#         widths=0.5,
#         showmeans=False,
#         showmedians=False,
#         showextrema=False
#     )
    
#     # Style upstream violins
#     for body, model in zip(vp_upstream['bodies'], models):
#         body.set_facecolor('white')
#         body.set_edgecolor(get_color(model))
#         body.set_alpha(0.4)
#         body.set_linewidth=1.5
#         body.set_linestyle='--'
    
#     # --- Box plots overlay (Combined) ---
#     bp_combined = ax.boxplot(
#         [data_combined[m] for m in models],
#         positions=positions_combined,
#         widths=0.2,
#         patch_artist=True,
#         showmeans=True,
#         meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=3),
#         boxprops=dict(linewidth=1),
#         whiskerprops=dict(linewidth=0.8),
#         capprops=dict(linewidth=0.8),
#         medianprops=dict(color='darkred', linewidth=1.5)
#     )
    
#     # Color combined boxes
#     for patch, model in zip(bp_combined['boxes'], models):
#         patch.set_facecolor(get_color(model))
#         patch.set_alpha(0.85)
#         patch.set_edgecolor('black')
    
#     # --- Box plots overlay (Upstream) ---
#     bp_upstream = ax.boxplot(
#         [data_upstream[m] for m in models],
#         positions=positions_upstream,
#         widths=0.2,
#         patch_artist=True,
#         showmeans=True,
#         meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=3),
#         boxprops=dict(linewidth=1, linestyle='--'),
#         whiskerprops=dict(linewidth=0.8, linestyle='--'),
#         capprops=dict(linewidth=0.8),
#         medianprops=dict(color='darkblue', linewidth=1.5, linestyle='--')
#     )
    
#     # Color upstream boxes
#     for patch, model in zip(bp_upstream['boxes'], models):
#         patch.set_facecolor('white')
#         patch.set_alpha=0.7
#         patch.set_edgecolor(get_color(model))
#         patch.set_linewidth=1.5
    
#     # Styling
#     ax.set_xticks(np.arange(n_models) * 2.5)
#     ax.set_xticklabels(models, rotation=15, ha='right')
#     ax.set_ylabel('NNSE')
#     ax.set_title('Distribution Comparison: Combined vs Upstream (Violin + Box Plot)')
#     ax.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
#     ax.grid(True, alpha=0.3, axis='y')
    
#     # Legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Combined'),
#         Patch(facecolor='white', edgecolor='gray', linestyle='--', linewidth=1.5, label='Upstream')
#     ]
#     ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_folder, 'combined_violin_box.png'), dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(save_folder, 'combined_violin_box.pdf'), bbox_inches='tight')
#     plt.close()
    
#     print(f"Saved: combined_violin_box plots")

# ----------------------------------------------------------------------
# 6. Combined Violin and Box Plot (Overlay)
# ----------------------------------------------------------------------
def plot_combined_violin_box(data_combined, data_upstream, save_folder="plots"):
    """
    Combined violin and box plot showing distribution comparison.
    Violin shows full distribution, box shows quartiles and outliers.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(7.5, 4))
    
    # Reduce spacing between models from 2.5 to 1.5
    positions_combined = np.arange(n_models) * 1.5 - 0.35
    positions_upstream = np.arange(n_models) * 1.5 + 0.35
    
    # --- Violin plots (Combined) ---
    vp_combined = ax.violinplot(
        [data_combined[m] for m in models],
        positions=positions_combined,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Style combined violins
    for i, (body, model) in enumerate(zip(vp_combined['bodies'], models)):
        body.set_facecolor(get_color(model))
        body.set_edgecolor('black')
        body.set_alpha(0.3)
        body.set_linewidth(0.8)
    
    # --- Violin plots (Upstream) ---
    vp_upstream = ax.violinplot(
        [data_upstream[m] for m in models],
        positions=positions_upstream,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Style upstream violins
    for body, model in zip(vp_upstream['bodies'], models):
        body.set_facecolor('white')
        body.set_edgecolor(get_color(model))
        body.set_alpha(0.4)
        body.set_linewidth=1.5
        body.set_linestyle='--'
    
    # --- Box plots overlay (Combined) ---
    bp_combined = ax.boxplot(
        [data_combined[m] for m in models],
        positions=positions_combined,
        widths=0.25,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=3),
        boxprops=dict(linewidth=1),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        medianprops=dict(color='darkred', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none', alpha=0.5)
    )
    
    # Color combined boxes
    for patch, model in zip(bp_combined['boxes'], models):
        patch.set_facecolor(get_color(model))
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
    
    # --- Box plots overlay (Upstream) ---
    bp_upstream = ax.boxplot(
        [data_upstream[m] for m in models],
        positions=positions_upstream,
        widths=0.25,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=3),
        boxprops=dict(linewidth=1, linestyle='--'),
        whiskerprops=dict(linewidth=0.8, linestyle='--'),
        capprops=dict(linewidth=0.8),
        medianprops=dict(color='darkblue', linewidth=1.5, linestyle='--'),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, linestyle='none', alpha=0.5)
    )
    
    # Color upstream boxes
    for patch, model in zip(bp_upstream['boxes'], models):
        patch.set_facecolor('white')
        patch.set_alpha=0.7
        patch.set_edgecolor(get_color(model))
        patch.set_linewidth=1.5
    
    # Styling
    ax.set_xticks(np.arange(n_models) * 1.5)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.set_ylabel('NNSE')
    ax.set_title('Distribution Comparison: Combined vs Upstream', fontsize=11)
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust x-axis limits to reduce empty space
    ax.set_xlim(-0.8, (n_models - 1) * 1.5 + 0.8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Combined'),
        Patch(facecolor='white', edgecolor='gray', linestyle='--', linewidth=1.5, label='Upstream')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'combined_violin_box.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'combined_violin_box.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: combined_violin_box plots")

# ----------------------------------------------------------------------
# 8. Scatter Plot: Model Comparison (All models on one plot)
# ----------------------------------------------------------------------
def plot_scatter_model_comparison(data_combined, data_upstream, save_folder="plots"):
    """
    Single scatter plot comparing all models' performance.
    Shows which models benefit most from combined inputs.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot each model with different colors
    for model in models:
        x = data_upstream[model]
        y = data_combined[model]
        
        ax.scatter(x, y, alpha=0.4, s=20, color=get_color(model), 
                  label=model, edgecolors='black', linewidth=0.3)
    
    # 1:1 line
    min_val = min([data_upstream[m].min() for m in models] + [data_combined[m].min() for m in models])
    max_val = max([data_upstream[m].max() for m in models] + [data_combined[m].max() for m in models])
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', linewidth=1.5, alpha=0.7, label='1:1 line', zorder=10)
    
    # Reference lines
    ax.axhline(y=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.axvline(x=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
    
    # Shaded regions
    ax.fill_between([min_val, max_val], [min_val, max_val], max_val, 
                    alpha=0.1, color='green', label='Improvement zone')
    ax.fill_between([min_val, max_val], min_val, [min_val, max_val], 
                    alpha=0.1, color='red', label='Degradation zone')
    
    ax.set_xlabel('NNSE (Upstream)', fontsize=8)
    ax.set_ylabel('NNSE (Combined)', fontsize=8)
    ax.set_title('All Models: Basin-level Performance Comparison', fontsize=9)
    ax.legend(frameon=True, loc='lower right', fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'scatter_all_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'scatter_all_models_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_all_models_comparison plots")

# ----------------------------------------------------------------------
# 9. Scatter Plot with Density Contours
# ----------------------------------------------------------------------
def plot_scatter_with_density(data_combined, data_upstream, save_folder="plots"):
    """
    Scatter plot with density contours showing concentration of basins.
    Useful for identifying common performance patterns.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    from scipy.stats import gaussian_kde
    
    models = list(data_combined.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 3))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        x = data_upstream[model]
        y = data_combined[model]
        
        # Calculate point density
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            
            # Sort points by density for better visualization
            idx_sort = z.argsort()
            x_sorted, y_sorted, z_sorted = x[idx_sort], y[idx_sort], z[idx_sort]
            
            scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=15, 
                               cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.2)
            plt.colorbar(scatter, ax=ax, label='Density', shrink=0.8)
        except:
            # Fallback if KDE fails
            ax.scatter(x, y, alpha=0.5, s=15, color=get_color(model), 
                      edgecolors='black', linewidth=0.3)
        
        # 1:1 line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', linewidth=1, alpha=0.7)
        
        # Reference lines
        ax.axhline(y=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.axvline(x=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
        
        ax.set_xlabel('NNSE (Upstream)', fontsize=8)
        if idx == 0:
            ax.set_ylabel('NNSE (Combined)', fontsize=8)
        ax.set_title(model, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('Basin Performance with Density Distribution', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'scatter_with_density.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'scatter_with_density.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_with_density plots")

# ----------------------------------------------------------------------
# 10. Scatter Plot Grid: Combined Configuration Only (2x2)
# ----------------------------------------------------------------------
def plot_scatter_grid_combined_only(data_combined, save_folder="plots"):
    """
    2x2 grid of scatter plots for combined configuration only.
    Each subplot shows one model's performance across all basins.
    X-axis: Basin index, Y-axis: NNSE score
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    
    if len(models) != 4:
        print(f"Warning: This function is designed for exactly 4 models. You have {len(models)} models.")
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()
    
    for idx, model in enumerate(models[:4]):  # Only take first 4 models
        ax = axes[idx]
        
        y = data_combined[model]
        x = np.arange(len(y))  # Basin indices
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=12, color=get_color(model), 
                  edgecolors='black', linewidth=0.3)
        
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, 
                  alpha=0.5, label='NNSE = 0.5')
        ax.axhline(y=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
        
        # Calculate statistics
        median_val = np.median(y)
        mean_val = np.mean(y)
        above_05 = np.sum(y > 0.5)
        pct_above = 100 * above_05 / len(y)
        
        # Add statistics text box
        textstr = f'Median: {median_val:.3f}\nMean: {mean_val:.3f}\n>0.5: {pct_above:.1f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=6,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        ax.set_xlabel('Basin Index', fontsize=8)
        ax.set_ylabel('NNSE', fontsize=8)
        ax.set_title(model, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=6, frameon=True)
    
    plt.suptitle('Basin Performance: Combined Configuration', fontsize=10, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'scatter_grid_combined_2x2.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'scatter_grid_combined_2x2.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_grid_combined_2x2 plots")


# ----------------------------------------------------------------------
# 11. Scatter Plot Grid: Basin Ranking (2x2)
# ----------------------------------------------------------------------
def plot_scatter_grid_ranked(data_combined, save_folder="plots"):
    """
    2x2 grid showing basin performance ranked from worst to best.
    Useful for identifying consistently poor/good performing basins.
    """
    set_academic_style()
    os.makedirs(save_folder, exist_ok=True)
    
    models = list(data_combined.keys())
    
    if len(models) != 4:
        print(f"Warning: This function is designed for exactly 4 models. You have {len(models)} models.")
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.flatten()
    
    for idx, model in enumerate(models[:4]):
        ax = axes[idx]
        
        y = data_combined[model]
        # Sort basins by performance
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices]
        x_ranked = np.arange(len(y_sorted))
        
        # Color gradient from red (poor) to green (good)
        colors = plt.cm.RdYlGn((y_sorted - y_sorted.min()) / (y_sorted.max() - y_sorted.min()))
        
        ax.scatter(x_ranked, y_sorted, c=colors, s=12, 
                  edgecolors='black', linewidth=0.3, alpha=0.8)
        
        # Reference lines
        ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=0.8, 
                  alpha=0.5, label='NNSE = 0.5')
        ax.axhline(y=0, color='red', linestyle=':', linewidth=0.8, alpha=0.4)
        
        # Mark quartiles
        q1_idx = len(y_sorted) // 4
        q2_idx = len(y_sorted) // 2
        q3_idx = 3 * len(y_sorted) // 4
        
        ax.axvline(x=q1_idx, color='gray', linestyle=':', linewidth=0.6, alpha=0.3)
        ax.axvline(x=q2_idx, color='gray', linestyle=':', linewidth=0.6, alpha=0.3)
        ax.axvline(x=q3_idx, color='gray', linestyle=':', linewidth=0.6, alpha=0.3)
        
        # Statistics
        median_val = np.median(y)
        percentile_75 = np.percentile(y, 75)
        
        textstr = f'Median: {median_val:.3f}\n75th %: {percentile_75:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Basin Rank (Worst → Best)', fontsize=8)
        ax.set_ylabel('NNSE', fontsize=8)
        ax.set_title(model, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=6, frameon=True)
    
    plt.suptitle('Basin Performance Ranking: Combined Configuration', fontsize=10, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'scatter_grid_ranked_2x2.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'scatter_grid_ranked_2x2.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_grid_ranked_2x2 plots")


# ----------------------------------------------------------------------
# Helper function to load data
# ----------------------------------------------------------------------
def load_nnse_data(files_dict):
    """Load NSE scores from CSV files."""
    data = {}
    for model_name, filepath in files_dict.items():
        df = pd.read_csv(filepath)
        nse_col = None
        for col in df.columns:
            if 'nnse' in col.lower() or 'NNSE' in col:
                nse_col = col
                break
        
        if nse_col is None:
            raise ValueError(f"NNSE column not found in {filepath}")
        
        data[model_name] = df[nse_col].values
    
    return data


# ----------------------------------------------------------------------
# Main execution function
# ----------------------------------------------------------------------
def generate_all_comparison_plots(csv_files_combined, csv_files_upstream, save_folder="plots_comparison"):
    """
    Generate all comparison visualizations.
    """
    # Load data
    nnse_combined = load_nnse_data(csv_files_combined)
    nnse_upstream = load_nnse_data(csv_files_upstream)
    
    # Generate plots
    plot_performance_delta(nnse_combined, nnse_upstream, save_folder)
    plot_comparative_boxplot_annotated(nnse_combined, nnse_upstream, save_folder)
    plot_performance_heatmap(nnse_combined, nnse_upstream, save_folder)
    plot_cdf_comparison(nnse_combined, nnse_upstream, save_folder)
    plot_pairwise_comparison_matrix(nnse_combined, save_folder)
    plot_combined_violin_box(nnse_combined, nnse_upstream, save_folder)
    plot_scatter_model_comparison(nnse_combined, nnse_upstream, save_folder)
    plot_scatter_with_density(nnse_combined, nnse_upstream, save_folder)
    plot_scatter_grid_combined_only(nnse_combined, save_folder)
    plot_scatter_grid_ranked(nnse_combined, save_folder)
    
    print(f"\nAll comparison plots saved to: {save_folder}/")


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    csv_files_combined = {
        # "PITransformer": "./exp/pitransformer1/transformer_combined_2411_190850/resume_from001/test/model_epoch001/test_metrics.csv",
        "Transformer": "../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
        # "FEDformer": "./exp/fedformer1/fedformer_combined_2111_203206/test/model_epoch002/test_metrics.csv",
        "Informer": "../exp/informer1/informer_combined_2211_030811/resume_from002/test/model_epoch001/test_metrics.csv",
        "CNN-1D": "../exp/cnn1/cnn_combined_2211_191525/test/model_epoch002/test_metrics.csv",
        "LSTM": "../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv",
    }

    csv_files_upstream = {
        # "PITransformer": "./exp/pitransformer1/transformer_upstream_2411_091750/test/model_epoch002/test_metrics.csv",
        "Transformer": "../exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
        # "FEDformer": "./exp/fedformer1/fedformer_upstream_2111_171235/test/model_epoch002/test_metrics.csv",
        "Informer": "../exp/informer1/informer_upstream_2211_082817/test/model_epoch002/test_metrics.csv",
        "CNN-1D": "../exp/cnn1/cnn_upstream_2211_150644/test/model_epoch002/test_metrics.csv",
        "LSTM": "../exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv",
    }
    
    generate_all_comparison_plots(csv_files_combined, csv_files_upstream, save_folder="plots_comparison_nse")