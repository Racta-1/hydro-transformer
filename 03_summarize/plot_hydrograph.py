"""
Hydrograph Comparison from Raw NeuralHydrology Pickle

This script loads raw test_results.p from NeuralHydrology and extracts:
- time coordinate
- observed flow
- simulated flow

It generates:
1. LSTM Combined vs LSTM Upstream
2. Transformer Combined vs Transformer Upstream

python plot_hydrograph

Author: Taye Akinrele
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")

TIME_KEY = "1h"
OBS_KEYS = ["streamflow_u_obs", "qobs", "obs"]
SIM_KEYS = ["streamflow_u_sim", "qsim", "sim", "pred"]


# ------------------------------------------------------
# Utility: load raw NeuralHydrology pickle
# ------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_nse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    denom = np.sum((obs - obs.mean())**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs)**2) / denom

def compute_nnse(nse):
    return np.nan if not np.isfinite(nse) else 1.0 / (2.0 - nse)


# ------------------------------------------------------
# Extract time, obs, sim from raw NH dataset
# ------------------------------------------------------
def extract_from_raw(results_dict, basin_id):
    top = results_dict[basin_id]
    ds = top[TIME_KEY]["xr"]  # xarray dataset

    # ---- Choose observed variable ----
    obs_var = None
    for k in OBS_KEYS:
        if k in ds.data_vars:
            obs_var = k
            break
    if obs_var is None:
        raise KeyError(f"No observed variable found in {basin_id}")

    # ---- Choose simulated variable ----
    sim_var = None
    for k in SIM_KEYS:
        for v in ds.data_vars:
            if k in v.lower():
                sim_var = v
                break
        if sim_var:
            break
    if sim_var is None:
        raise KeyError(f"No simulated variable found in {basin_id}")

    # ---- Handle collapse time_step=1 ----
    if "time_step" in ds.dims and ds.dims["time_step"] == 1:
        ds = ds.isel(time_step=0)

    # ---- Time coordinate ----
    time_coord = None
    for cand in ["date", "time", "Time", "datetime"]:
        if cand in ds:
            time_coord = cand
            break
    if time_coord is None:
        for c in ds.coords:
            if ds[c].ndim == 1:
                time_coord = c
                break
    if time_coord is None:
        raise KeyError(f"No usable time coordinate found in {basin_id}")

    # ---- Extract arrays ----
    tvals = np.array(pd.to_datetime(np.asarray(ds[time_coord].values)))
    obs = np.asarray(ds[obs_var].values).ravel()
    sim = np.asarray(ds[sim_var].values).ravel()

    m = np.isfinite(obs) & np.isfinite(sim)

    df = pd.DataFrame({
        "time": tvals[m],
        "obs": obs[m],
        "sim": sim[m],
    })

    # ---- Compute metrics ----
    nse = compute_nse(df["obs"].values, df["sim"].values)
    nnse = compute_nnse(nse)

    return df, nse, nnse


# ------------------------------------------------------
# Plotting function
# ------------------------------------------------------
def plot_two_panels(
    df_left, gage_left, nnse_left,
    df_right, gage_right, nnse_right,
    label_left, label_right,
    out_file
):

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    # LEFT
    ax = axes[0]
    ax.plot(df_left["time"], df_left["obs"], lw=1.2, color="#1f77b4", label="Observed")
    ax.plot(df_left["time"], df_left["sim"], lw=1.2, color="#d95f02", label="SImulated")
    # ax.text(0.03, 0.9, "Observed", transform=ax.transAxes, fontsize=10, color="#1f77b4")
    # ax.text(0.03, 0.82, "Simulated", transform=ax.transAxes, fontsize=10, color="#d95f02")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(label_left)
    fig.text(0.25, 0.03, f"Gage: {gage_left}", ha="center", fontsize=10)
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1,
        edgecolor="#333333",
        facecolor="white",
        fontsize=10
    )

    for txt in legend.get_texts():
        txt.set_color("black")
    
    ax.text(0.03, 0.70, f"NNSE = {nnse_left:.2f}", transform=ax.transAxes,
        fontsize=10, color="black")


    # RIGHT
    ax = axes[1]
    ax.plot(df_right["time"], df_right["obs"], lw=1.2, color="#1f77b4", label="Observed")
    ax.plot(df_right["time"], df_right["sim"], lw=1.2, color="#d95f02", label="Simulated")
    # ax.text(0.03, 0.9, "Observed", transform=ax.transAxes, fontsize=10, color="#1f77b4")
    # ax.text(0.03, 0.82, "Simulated", transform=ax.transAxes, fontsize=10, color="#d95f02")
    ax.set_title(label_right)
    fig.text(0.75, 0.03, f"Gage: {gage_right}", ha="center", fontsize=10)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1,
        edgecolor="#333333",
        facecolor="white",
        fontsize=10
    )

    for txt in legend.get_texts():
        txt.set_color("black")

    ax.text(0.03, 0.70, f"NNSE = {nnse_right:.2f}", transform=ax.transAxes,
        fontsize=10, color="black")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved → {out_file}")


# ------------------------------------------------------
# Load YOUR raw pickles
# ------------------------------------------------------
lstm_comb = load_pickle("../exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_results.p")
lstm_up   = load_pickle("../exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_results.p")
trans_comb = load_pickle("../exp/transformer1/transformer_combined_2111_083844/resume_from002/test/model_epoch001/test_results.p")
trans_up   = load_pickle("../exp/transformer1/transformer_upstream_2111_083900/test/model_epoch002/test_results.p")


# ------------------------------------------------------
# Choose a basin (first basin)
# ------------------------------------------------------
sample_basin = list(lstm_comb.keys())[0]


# ------------------------------------------------------
# Extract data
# ------------------------------------------------------
df_lc, nse_lc, nnse_lc = extract_from_raw(lstm_comb, sample_basin)
df_lu, nse_lu, nnse_lu = extract_from_raw(lstm_up, sample_basin)
df_tc, nse_tc, nnse_tc = extract_from_raw(trans_comb, sample_basin)
df_tu, nse_tu, nnse_tu = extract_from_raw(trans_up, sample_basin)



# ------------------------------------------------------
# Plot: LSTM Combined vs Upstream
# ------------------------------------------------------
plot_two_panels(
    df_lc, sample_basin, nnse_lc,
    df_lu, sample_basin, nnse_lu,
    label_left="LSTM (Combined)",
    label_right="LSTM (Upstream)",
    out_file="hydrograph_LSTM.png"
)
# ------------------------------------------------------
# Plot: Transformer Combined vs Upstream
# ------------------------------------------------------
plot_two_panels(
    df_tc, sample_basin, nnse_tc,
    df_tu, sample_basin, nnse_tu,
    label_left="Transformer (Combined)",
    label_right="Transformer (Upstream)",
    out_file="hydrograph_Transformer.png"
)
