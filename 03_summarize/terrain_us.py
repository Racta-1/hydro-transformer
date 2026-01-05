#!/usr/bin/env python3
"""
Merge hydrologic metrics with basin coordinates and visualize spatial performance
with terrain basemap.

Usage:
    python visualize.py --metrics metrics.csv --coords latlon.csv \
                          --metric NSE --title "(b) Upstream-Only NSE" \
                          --out upstream_nse_binned.pdf

Additional dependencies required:
    pip install contextily
"""

import argparse
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import contextily as ctx

parser = argparse.ArgumentParser(description="Visualize basin metrics across CONUS.")
parser.add_argument("--metrics", "-m", required=True, help="CSV file with basin metrics (must have 'basin' column).")
parser.add_argument("--coords", "-c", required=True, help="CSV file with basin coordinates (must have gages, lon, lat).")
parser.add_argument("--metric", "-x", required=True, help="Metric column to visualize (e.g., NSE, KGE, RMSE).")
parser.add_argument("--title", "-t", required=True, help="Title to display on the figure.")
parser.add_argument("--out", "-o", default="basin_map.pdf", help="Output file for the map (PDF or PNG).")
parser.add_argument("--shapefile", "-s", default="us_states_shapefile/tl_2023_us_state.shp",
                    help="Path to CONUS shapefile.")
parser.add_argument("--basemap", "-b", default="terrain", 
                    choices=["terrain", "satellite", "topo", "streets"],
                    help="Basemap style: terrain (USGS), satellite, topo, or streets")
parser.add_argument("--zoom", "-z", type=int, default=6,
                    help="Zoom level for basemap (higher = more detail, typically 5-8 for CONUS)")
args = parser.parse_args()


def normalize_id(obj) -> str:
    s = str(obj)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 8:
        digits = digits.zfill(8)
    elif len(digits) > 8:
        digits = digits[-8:]
    return digits


# Load and process data
df_metrics = pd.read_csv(args.metrics)
df_coords = pd.read_csv(args.coords)

df_metrics.columns = df_metrics.columns.str.strip()
df_coords.columns = df_coords.columns.str.strip()

if "basin" not in df_metrics.columns:
    raise ValueError("Metrics file must contain a 'basin' column.")

if not any(c in df_coords.columns for c in ["gages", "gagestr"]):
    raise ValueError("Coordinates file must have a 'gages' or 'gagestr' column.")

gage_col = "gages" if "gages" in df_coords.columns else "gagestr"
df_metrics["basin_id"] = df_metrics["basin"].map(normalize_id)
df_coords["gage_id"] = df_coords[gage_col].map(normalize_id)

df_coords_dedup = df_coords.drop_duplicates(subset=["gage_id"], keep="first")
merged = df_metrics.merge(df_coords_dedup[["gage_id", "lat", "lon"]],
                          left_on="basin_id", right_on="gage_id", how="left").drop(columns=["gage_id"])

for c in ["lat", "lon"]:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

total = len(merged)
matched = merged["lat"].notna().sum()
print(f"Matched {matched} of {total} basins ({matched/total:.1%}).")

# Process metric
metric = args.metric
if metric not in merged.columns:
    raise ValueError(f"Metric '{metric}' not found in file. Available: {list(merged.columns)}")

merged[metric] = pd.to_numeric(merged[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
vals = merged[metric].dropna().to_numpy()
median_val = float(np.nanmedian(vals)) if vals.size else np.nan

# Create GeoDataFrame and convert to Web Mercator (EPSG:3857)
gdf = gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged['lon'], merged['lat']), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Load and convert shapefile to Web Mercator
conus = gpd.read_file(args.shapefile)
conus = conus.to_crs(epsg=3857)

# Define bins and colors
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['0.0-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
colors = ['#8B0000', '#D32F2F', '#A7C7E7', '#6FA8DC', '#4682B4', '#2E5C8A', '#1A3A5C']


def get_color(val):
    if pd.isna(val):
        return None
    if val < 0:
        return colors[0]
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i + 1]:
            return colors[i + 1]
    if val >= bins[-1]:
        return colors[-1]
    return colors[1]


gdf["color"] = gdf[metric].apply(get_color)

# Create figure
fig, ax = plt.subplots(figsize=(9.2, 5))
ax.set_title(args.title, fontsize=10, loc="center", y=0.99)

# Select basemap source
basemap_sources = {
    "terrain": ctx.providers.USGS.USTopo,
    "satellite": ctx.providers.Esri.WorldImagery,
    "topo": ctx.providers.OpenTopoMap,
    "streets": ctx.providers.OpenStreetMap.Mapnik
}
basemap_source = basemap_sources.get(args.basemap, ctx.providers.USGS.USTopo)

# Plot state boundaries
conus.plot(ax=ax, facecolor="none", edgecolor="#969393", linewidth=1.2, alpha=0.6)

# Add basemap
zoom_level = args.zoom if args.zoom is not None else ctx.tile._calculate_zoom(*ax.axis())
try:
    ctx.add_basemap(ax, source=basemap_source, zoom=zoom_level, attribution_size=6)
    print(f"Added {args.basemap} basemap at zoom level {zoom_level}")
except Exception as e:
    print(f"Warning: Could not load basemap: {e}")
    print("Falling back to plain background")
    ax.set_facecolor("#E8E8E8")

# Plot basin points with colored dots
for color in colors:
    mask = gdf["color"] == color
    if mask.any():
        gdf[mask].plot(ax=ax, color=color, markersize=18, 
                       edgecolor="white", linewidth=0.5, alpha=0.8)

# Set bounds (convert from lat/lon to Web Mercator)
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-125, 24)
xmax, ymax = transformer.transform(-66, 50)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")

# Add inset histogram
ax_in = inset_axes(
    ax,
    width="18%", height="18%",
    loc="lower left",
    borderpad=0.1,
    bbox_to_anchor=(0, 0.04, 1, 1),  # (x0, y0, width, height)
    bbox_transform=ax.transAxes
)
hxmin, hxmax = 0.0, 1.0
ax_in.hist(vals, bins=np.linspace(hxmin, hxmax, 15),
           color="#4682B4", edgecolor="#8aa7c4", linewidth=0.5)
ax_in.set_xlim(hxmin, hxmax)
ax_in.set_xticks([0, 0.5, 0.8, 1])
ax_in.set_yticks([])
ax_in.tick_params(labelsize=6)
ax_in.patch.set_alpha(0.6)
for s in ax_in.spines.values():
    s.set_edgecolor("#666666")
    s.set_linewidth(0.8)

if np.isfinite(median_val):
    med_color = get_color(median_val)
    ax_in.axvline(median_val, color=med_color, linestyle="--", linewidth=1.2)
    y_top = ax_in.get_ylim()[1]
    ax_in.text(median_val, y_top * 1.03, f"Median: {median_val:.2f}",
               fontsize=6, ha="center", va="bottom",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
               clip_on=False)

# Add legend
legend_elems = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=0.5)
                for c in colors]
ax.legend(legend_elems, labels, title=metric,
          loc="lower right", frameon=True, facecolor="white",
          edgecolor="#7A7A7A", fontsize=8, title_fontsize=8).get_frame().set_alpha(0.65)

plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=400)
print(f"Saved map: {args.out}")
plt.show()