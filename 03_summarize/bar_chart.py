import re
import numpy as np
import matplotlib.pyplot as plt
import os


# ----------------------------------------------------------------------
# Function: Extract Total Summary from .txt result file
# ----------------------------------------------------------------------
def extract_total_summary(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    pattern = r"====== Positive NSE Summary ======(.*?)(?======|$)"
    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Total Summary not found in file: {filepath}")

    block = match.group(1).strip()
    line_pattern = r"([A-Za-z\-r]+)\s+Mean=([\-0-9\.]+)\s+Median=([\-0-9\.]+)"

    metrics = {}
    for metric, mean, median in re.findall(line_pattern, block):
        metrics[metric] = {
            "Mean": float(mean),
            "Median": float(median)
        }

    return metrics


# ----------------------------------------------------------------------
# Plot: Only NNSE, KGE, Pearson-r, RMSE (save PNG + PDF)
# ----------------------------------------------------------------------
def plot_selected_metrics(models_dict, save_folder="plots"):
    # Academic paper color palette (professional, print-friendly)
    colors = [
        '#2E4057',  # Dark blue-grey (PITransformer)
        '#048A81',  # Teal (Transformer)
        '#54C6EB',  # Sky blue (FEDformer)
        '#F18F01',  # Orange (Informer)
        # '#8B4513',   # Red-orange (CNN-1D)
        '#C73E1D'
    ]
    
    # Alternative grayscale-friendly palette (uncomment to use):
    # colors = [
    #     '#1f77b4', 
    #     '#ff7f0e', 
    #     '#2ca02c', 
    #     '#d62728', 
    #     # '#9467bd'
    # ]
    
    # For grayscale printing compatibility:
    # colors = ['#000000', '#404040', '#808080', '#B0B0B0', '#D3D3D3']
    
    # metrics to include in bar chart
    selected_metrics = ["NNSE", "KGE", "Pearson-r", "RMSE"]

    os.makedirs(save_folder, exist_ok=True)

    model_names = list(models_dict.keys())

    # build median matrix in consistent metric order
    medians = np.array([
        [models_dict[m][metric]["Median"] for metric in selected_metrics]
        for m in model_names
    ])

    x = np.arange(len(selected_metrics))
    width = 0.18

    # Set academic style
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

    fig, ax = plt.subplots(figsize=(7.5, 4))

    # one bar group per metric with academic colors
    bars = []
    for i, model in enumerate(model_names):
        bar = ax.bar(x + (i - len(model_names)/2)*width,
                     medians[i], width, label=model,
                     color=colors[i], edgecolor='black', linewidth=0.5)
        bars.append(bar)

    ax.set_xticks(x)
    ax.set_xticklabels(selected_metrics)
    ax.set_ylabel("Median Values")
    ax.set_title("Performance Comparison of Model for Combined Configuration", pad=10)
    
    # Add subtle grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    ax.set_axisbelow(True)
    
    # Position legend outside plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), 
             fancybox=False, shadow=False)
    
    plt.tight_layout()

    png_path = os.path.join(save_folder, "selected_metrics_comparison_title.png")
    pdf_path = os.path.join(save_folder, "selected_metrics_comparison_title.pdf")

    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------

files = {
    "PITransformer": "./results1/pitrans_comb2411/evaluation_report_complete.txt",
    "Transformer": "./results1/trans_comb2111/evaluation_report_complete.txt",
    "FEDformer": "./results1/fedformer_comb2211/evaluation_report_complete.txt",
    "Informer": "./results1/informer_comb2211/evaluation_report_complete.txt",
    # "CNN-1D": "./results1/cnn_comb2211/evaluation_report_complete.txt"
    "LSTM": "./results/lstm_comb2210/evaluation_report_complete.txt",

}

summary_data = {name: extract_total_summary(path) for name, path in files.items()}

plot_selected_metrics(summary_data, save_folder="plots")