import pandas as pd
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
    metrics["NSE"] = df[detect_column(df, ["nse", "NSE"])]
    metrics["NNSE"] = df[detect_column(df, ["nnse", "NNSE"])]
    metrics["KGE"] = df[detect_column(df, ["kge", "KGE"])]
    metrics["PCC"] = df[detect_column(df, ["Pearson-r"])]
    metrics["RMSE"] = df[detect_column(df, ["rmse", "RMSE"])]
    # metrics["MAE"] = df[detect_column(df, ["mae"])]

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
    nnse_up = metrics_up["NSE"]
    nnse_comb = metrics_comb["NSE"]

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
    # latex.append(f"\\textbf{{MAE}} & {fmt(stats['MAE']['median_up'])} & {fmt(stats['MAE']['median_comb'])} & {fmt(stats['MAE']['mean_up'])} & {fmt(stats['MAE']['mean_comb'])} \\\\")
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
if __name__ == "__main__":
    combined_csv = {
        "Transformer": "./exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_metrics.csv",
    }

    upstream_csv = {
        "Transformer": "./exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_metrics.csv",
    }

    run_table_generator(combined_csv, upstream_csv)
