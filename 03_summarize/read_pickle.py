# import pickle

# def view_pickle(file):
#     try:
#         with open(file, 'rb') as f:
#             data = pickle.load(f)
#             print(f"Contents of '{file}':")
#             print(data)
#     except FileNotFoundError:
#         print(f"Error: File '{file}' not found")


# view_pickle('exp/transformer/transformer_combined_1011_084001/test/model_epoch003/test_results.p')

# import pickle
# p = 'exp/transformer/transformer_combined_1011_120903/test/model_epoch001/test_results.p'
# with open(p, 'rb') as f:
#     results = pickle.load(f)

# # pick one station id (replace with one present in your dict)
# sid = next(iter(results.keys()))
# xr_ds = results[sid]['1h']['xr']   # your xarray dataset

# print(sid, xr_ds)
# print('dims:', xr_ds.dims)
# print('time_step coords:', xr_ds.coords['time_step'].values)
# print('streamflow_u_sim shape:', xr_ds['streamflow_u_sim'].shape)


import numpy as np
import xarray as xr
import pickle
from scipy.stats import pearsonr

def nse(obs, sim):
    denom = np.sum((obs - np.mean(obs))**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs)**2) / denom

def rmse(obs, sim):
    return np.sqrt(np.mean((sim - obs)**2))

def mae(obs, sim):
    return np.mean(np.abs(sim - obs))

def kge(obs, sim):
    # Kling-Gupta Efficiency (simplified 2009 formulation)
    # KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    # alpha = std(sim)/std(obs), beta = mean(sim)/mean(obs)
    r = pearsonr(obs.flatten(), sim.flatten())[0] if obs.size>1 else np.nan
    stdo = np.std(obs)
    stds = np.std(sim)
    if stdo == 0:
        alpha = np.nan
    else:
        alpha = stds / stdo
    mo = np.mean(obs)
    ms = np.mean(sim)
    if mo == 0:
        beta = np.nan
    else:
        beta = ms / mo
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# load
with open('exp/transformer/transformer_combined_1011_120903/test/model_epoch001/test_results.p', 'rb') as f:
    results = pickle.load(f)

# example over all stations: compute per-horizon metric then mean over horizons
metrics_per_station = {}
for sid, info in results.items():
    ds = info['1h']['xr']   # xarray Dataset with dims (date, time_step)
    obs = ds['streamflow_u_obs'].values  # shape (date, time_step)
    sim = ds['streamflow_u_sim'].values  # shape (date, time_step)
    # Ensure dims: (N_dates, horizon)
    if obs.ndim != 2:
        obs = obs.reshape(obs.shape[0], -1)
        sim = sim.reshape(sim.shape[0], -1)

    horizons = obs.shape[1]
    metrics_by_horizon = {'NSE': [], 'RMSE': [], 'MAE': [], 'KGE': [], 'Pearson-r': []}

    for h in range(horizons):
        o = obs[:, h]
        s = sim[:, h]
        # mask nans
        mask = np.isfinite(o) & np.isfinite(s)
        if mask.sum() < 2:
            # not enough data
            metrics_by_horizon['NSE'].append(np.nan)
            metrics_by_horizon['RMSE'].append(np.nan)
            metrics_by_horizon['MAE'].append(np.nan)
            metrics_by_horizon['KGE'].append(np.nan)
            metrics_by_horizon['Pearson-r'].append(np.nan)
            continue

        metrics_by_horizon['NSE'].append(nse(o[mask], s[mask]))
        metrics_by_horizon['RMSE'].append(rmse(o[mask], s[mask]))
        metrics_by_horizon['MAE'].append(mae(o[mask], s[mask]))
        metrics_by_horizon['KGE'].append(kge(o[mask], s[mask]))
        metrics_by_horizon['Pearson-r'].append(pearsonr(o[mask], s[mask])[0])

    # convert to arrays and compute mean across horizons (ignoring nan)
    mean_metrics = {k: np.nanmean(np.array(v)) for k, v in metrics_by_horizon.items()}
    metrics_per_station[sid] = {'per_horizon': metrics_by_horizon, 'mean_across_horizons': mean_metrics}

# Example: print station mean NSE
for sid, m in list(metrics_per_station.items())[:5]:
    print(sid, m['mean_across_horizons'])
