#!/usr/bin/env python3
"""
Simple viewer for basin locations on a terrain map of the US.

Usage:
    python view_basins.py --coords latlon.csv --out basin_locations.png
    
    python view_basins.py --coords ../data/camels_link.csv --out basins_map.png --basemap terrain --zoom 6

    python view_basins.py --coords ../data/camels_link.csv --routelink ../data/RouteLink_CONUS.nc --out basins_with_reaches.png
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
import xarray as xr
from shapely.geometry import LineString


parser = argparse.ArgumentParser(description="Visualize basin locations on US terrain map.")
parser.add_argument(
    "--coords", "-c", required=True,
    help="CSV file with basin coordinates (must have gages/gagestr, lon, lat)."
)
parser.add_argument(
    "--out", "-o", default="basin_map.png",
    help="Output file for the map (PDF or PNG)."
)
parser.add_argument(
    "--shapefile", "-s",
    default="us_states_shapefile/tl_2023_us_state.shp",
    help="Path to CONUS shapefile."
)
parser.add_argument(
    "--basemap", "-b", default="terrain",
    choices=["terrain", "satellite", "topo", "streets", "none"],
    help="Basemap style: terrain (USGS), satellite, topo, streets, or none"
)
parser.add_argument(
    "--zoom", "-z", type=int, default=6,
    help="Zoom level for basemap (5-8 works well for CONUS)"
)
# parser.add_argument("--title", "-t", default="CO#!/usr/bin/env python3
"""
Simple viewer for basin locations on a terrain map of the US.

Usage:
    python view_basins.py --coords latlon.csv --out basin_locations.png
    
    python view_basins.py --coords ../data/camels_link.csv --out basins_map.png --basemap terrain --zoom 6

    python view_basins.py --coords ../data/camels_link.csv --routelink ../data/RouteLink_CONUS.nc --out basins_with_reaches.png
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
import xarray as xr
from shapely.geometry import LineString, Point
import numpy as np


parser = argparse.ArgumentParser(description="Visualize basin locations on US terrain map.")
parser.add_argument(
    "--coords", "-c", required=True,
    help="CSV file with basin coordinates (must have gages/gagestr, lon, lat)."
)
parser.add_argument(
    "--out", "-o", default="basin_map.png",
    help="Output file for the map (PDF or PNG)."
)
parser.add_argument(
    "--shapefile", "-s",
    default="us_states_shapefile/tl_2023_us_state.shp",
    help="Path to CONUS shapefile."
)
parser.add_argument(
    "--basemap", "-b", default="terrain",
    choices=["terrain", "satellite", "topo", "streets", "none"],
    help="Basemap style: terrain (USGS), satellite, topo, streets, or none"
)
parser.add_argument(
    "--zoom", "-z", type=int, default=6,
    help="Zoom level for basemap (5-8 works well for CONUS)"
)
parser.add_argument(
    "--marker-color", "-mc", default="#1A3A5C",
    help="Color for basin markers (hex code)"
)
parser.add_argument(
    "--marker-size", "-ms", type=int, default=30,
    help="Size of basin markers"
)
parser.add_argument(
    "--routelink", "-rl", default="../data/RouteLink_CONUS.nc",
    help="Optional NWM RouteLink NetCDF (.nc) file for river reaches."
)
parser.add_argument(
    "--min-order", "-mo", type=int, default=3,
    help="Minimum Strahler order for RouteLink reaches (1-10, default=3)"
)
parser.add_argument(
    "--reach-buffer", "-rb", type=float, default=None,
    help="Buffer distance (km) around basins to filter reaches. None = show all."
)

args = parser.parse_args()


def normalize_id(obj) -> str:
    """Normalize gage IDs to 8-digit format."""
    s = str(obj)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 8:
        digits = digits.zfill(8)
    elif len(digits) > 8:
        digits = digits[-8:]
    return digits


# --------------------------------------------------------
# Load basin coordinates
# --------------------------------------------------------
print(f"Loading basin coordinates from {args.coords}...")
df_coords = pd.read_csv(args.coords)
df_coords.columns = df_coords.columns.str.strip()

# Find the gage column
if not any(c in df_coords.columns for c in ["gages", "gagestr", "gage_id"]):
    raise ValueError("Coordinates file must have a 'gages', 'gagestr', or 'gage_id' column.")

if "gages" in df_coords.columns:
    gage_col = "gages"
elif "gagestr" in df_coords.columns:
    gage_col = "gagestr"
else:
    gage_col = "gage_id"

# Process coordinates
df_coords["gage_id"] = df_coords[gage_col].map(normalize_id)
df_coords = df_coords.drop_duplicates(subset=["gage_id"], keep="first")

# Convert to numeric lat/lon
for c in ["lat", "lon"]:
    if c in df_coords.columns:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="coerce")

# Remove invalid coordinates
df_coords = df_coords.dropna(subset=["lat", "lon"])
print(f"Loaded {len(df_coords)} basin locations")

# Create GeoDataFrame and convert to Web Mercator
gdf = gpd.GeoDataFrame(
    df_coords,
    geometry=gpd.points_from_xy(df_coords["lon"], df_coords["lat"]),
    crs="EPSG:4326",
)
gdf = gdf.to_crs(epsg=3857)

# --------------------------------------------------------
# Load CONUS shapefile
# --------------------------------------------------------
print(f"Loading CONUS shapefile from {args.shapefile}...")
conus = gpd.read_file(args.shapefile)
conus = conus.to_crs(epsg=3857)
conus_border = conus.dissolve()

# --------------------------------------------------------
# Load RouteLink NetCDF and build reach line geometries (IMPROVED)
# --------------------------------------------------------
gdf_rl = None
if args.routelink is not None:
    print(f"Loading NWM RouteLink from {args.routelink}...")
    ds = xr.open_dataset(args.routelink)

    # Extract fields
    link_ids = ds["link"].values
    lat = ds["lat"].values
    lon = ds["lon"].values
    to_ids = ds["to"].values
    order = ds["order"].values  # Strahler stream order

    print(f"RouteLink contains {len(link_ids)} total reaches")
    print(f"Stream order range: {order.min()} to {order.max()}")

    # Coordinate lookup dictionary: COMID -> (lon, lat)
    id_to_xy = {
        int(fid): (float(x), float(y))
        for fid, x, y in zip(link_ids, lon, lat)
    }

    # Filter by river order
    print(f"Filtering RouteLink by Strahler order >= {args.min_order}...")
    mask = order >= args.min_order
    n_filtered = mask.sum()
    
    # Warn if too many reaches for visualization
    if n_filtered > 100000:
        print(f"⚠ WARNING: {n_filtered:,} reaches after filtering - this may be too dense!")
        print(f"  Consider using --min-order 5 or higher for CONUS-wide views")
        print(f"  Or use --reach-buffer to limit to regions near basins")

    # Build line geometries with validation
    lines = []
    skipped = 0
    for fid, toid, keep in zip(link_ids, to_ids, mask):
        if not keep:
            continue
        if int(fid) in id_to_xy and int(toid) in id_to_xy:
            x1, y1 = id_to_xy[int(fid)]
            x2, y2 = id_to_xy[int(toid)]
            
            # Validate: skip if coordinates are invalid or too far apart
            if not (-180 <= x1 <= -60 and 20 <= y1 <= 55):
                skipped += 1
                continue
            if not (-180 <= x2 <= -60 and 20 <= y2 <= 55):
                skipped += 1
                continue
            
            # Skip unreasonably long segments (> 100km ~ 1 degree)
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > 1.0:
                skipped += 1
                continue
            
            line = LineString([(x1, y1), (x2, y2)])
            lines.append({"feature_id": int(fid), "geometry": line})
    
    if skipped > 0:
        print(f"Skipped {skipped} invalid or out-of-bounds reaches")

    print(f"Built {len(lines)} reach lines after filtering")

    if lines:
        gdf_rl = gpd.GeoDataFrame(lines, crs="EPSG:4326").to_crs(3857)
        
        # Optional: Filter reaches near basins
        if args.reach_buffer is not None:
            print(f"Filtering reaches within {args.reach_buffer} km of basins...")
            buffer_m = args.reach_buffer * 1000  # Convert km to meters
            basin_union = gdf.geometry.unary_union.buffer(buffer_m)
            gdf_rl = gdf_rl[gdf_rl.geometry.intersects(basin_union)]
            print(f"Retained {len(gdf_rl)} reaches near basins")
        
        print("RouteLink reaches loaded and reprojected.")
    else:
        print("No RouteLink lines were created (check filter or data).")

    ds.close()

# --------------------------------------------------------
# Create figure and axes
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5))

# --------------------------------------------------------
# Add basemap if requested (BEFORE other layers)
# --------------------------------------------------------
if args.basemap != "none":
    basemap_sources = {
        "terrain": ctx.providers.USGS.USTopo,
        "satellite": ctx.providers.Esri.WorldImagery,
        "topo": ctx.providers.OpenTopoMap,
        "streets": ctx.providers.OpenStreetMap.Mapnik,
    }
    basemap_source = basemap_sources.get(args.basemap, ctx.providers.USGS.USTopo)

    zoom_level = args.zoom if args.zoom is not None else ctx.tile._calculate_zoom(*ax.axis())
    try:
        ctx.add_basemap(ax, source=basemap_source, zoom=zoom_level, attribution_size=6, alpha=0.6)
        print(f"Added {args.basemap} basemap at zoom level {zoom_level}")
    except Exception as e:
        print(f"Warning: Could not load basemap: {e}")
        print("Falling back to plain background")
        ax.set_facecolor("#E8E8E8")
else:
    # Plain background if basemap is disabled
    ax.set_facecolor("#E8E8E8")
    print("Basemap disabled (basemap='none').")

# --------------------------------------------------------
# Plot state boundaries (AFTER basemap, BEFORE reaches)
# --------------------------------------------------------
conus_border.plot(ax=ax, facecolor="none", edgecolor="#2d2d2d", linewidth=1.5, alpha=0.8, zorder=2)

# --------------------------------------------------------
# Plot RouteLink reaches (IMPROVED STYLING)
# --------------------------------------------------------
if gdf_rl is not None and len(gdf_rl) > 0:
    # Style reaches by creating width variation if desired
    gdf_rl.plot(
        ax=ax,
        color="#0066cc",      # Brighter blue
        linewidth=1.2,        # Thicker lines
        alpha=0.75,           # More opaque
        zorder=3,             # Above basemap and states
    )
    print(f"Added {len(gdf_rl)} RouteLink reaches to map.")
else:
    print("No reaches to display on map.")

# --------------------------------------------------------
# Plot basin points (ON TOP)
# --------------------------------------------------------
gdf.plot(
    ax=ax,
    color=args.marker_color,
    markersize=args.marker_size,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.9,
    zorder=4,              # Above everything
)

# --------------------------------------------------------
# Set bounds for CONUS (convert from lat/lon to Web Mercator)
# --------------------------------------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-125, 24)
xmax, ymax = transformer.transform(-66, 50)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")

# --------------------------------------------------------
# Legend with better visibility
# --------------------------------------------------------
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=args.marker_color, markersize=8, 
               label=f'Basin Locations (n={len(gdf)})', linewidth=0)
]

if gdf_rl is not None and len(gdf_rl) > 0:
    legend_elements.append(
        plt.Line2D([0], [0], color="#0066cc", linewidth=2, 
                   label=f'Stream Reaches (order≥{args.min_order}, n={len(gdf_rl)})')
    )

ax.legend(handles=legend_elements, loc="lower left", fontsize=9, 
          framealpha=0.9, edgecolor='black')

# --------------------------------------------------------
# Add statistics text box
# --------------------------------------------------------
if gdf_rl is not None and len(gdf_rl) > 0:
    stats_text = f"Reaches: {len(gdf_rl):,}\nBasins: {len(gdf):,}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --------------------------------------------------------
# Save figure
# --------------------------------------------------------
plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=500)
print(f"\n✓ Map saved to: {args.out}")
print(f"\nVisualization Summary:")
print(f"  • Basins plotted: {len(gdf)}")
if gdf_rl is not None:
    print(f"  • Stream reaches plotted: {len(gdf_rl)}")
    print(f"  • Minimum stream order: {args.min_order}")
print(f"  • Output file: {args.out}")
plt.show()NUS Basin Locations",
#                     help="Title for the map")
parser.add_argument(
    "--marker-color", "-mc", default="#1A3A5C",
    help="Color for basin markers (hex code)"
)
parser.add_argument(
    "--marker-size", "-ms", type=int, default=30,
    help="Size of basin markers"
)
parser.add_argument(
    "--routelink", "-rl", default="../data/RouteLink_CONUS.nc",
    help="Optional NWM RouteLink NetCDF (.nc) file for river reaches."
)

args = parser.parse_args()


def normalize_id(obj) -> str:
    """Normalize gage IDs to 8-digit format."""
    s = str(obj)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 8:
        digits = digits.zfill(8)
    elif len(digits) > 8:
        digits = digits[-8:]
    return digits


# --------------------------------------------------------
# Load basin coordinates
# --------------------------------------------------------
print(f"Loading basin coordinates from {args.coords}...")
df_coords = pd.read_csv(args.coords)
df_coords.columns = df_coords.columns.str.strip()

# Find the gage column
if not any(c in df_coords.columns for c in ["gages", "gagestr", "gage_id"]):
    raise ValueError("Coordinates file must have a 'gages', 'gagestr', or 'gage_id' column.")

if "gages" in df_coords.columns:
    gage_col = "gages"
elif "gagestr" in df_coords.columns:
    gage_col = "gagestr"
else:
    gage_col = "gage_id"

# Process coordinates
df_coords["gage_id"] = df_coords[gage_col].map(normalize_id)
df_coords = df_coords.drop_duplicates(subset=["gage_id"], keep="first")

# Convert to numeric lat/lon
for c in ["lat", "lon"]:
    if c in df_coords.columns:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="coerce")

# Remove invalid coordinates
df_coords = df_coords.dropna(subset=["lat", "lon"])
print(f"Loaded {len(df_coords)} basin locations")

# Create GeoDataFrame and convert to Web Mercator
gdf = gpd.GeoDataFrame(
    df_coords,
    geometry=gpd.points_from_xy(df_coords["lon"], df_coords["lat"]),
    crs="EPSG:4326",
)
gdf = gdf.to_crs(epsg=3857)

# --------------------------------------------------------
# Load CONUS shapefile
# --------------------------------------------------------
print(f"Loading CONUS shapefile from {args.shapefile}...")
conus = gpd.read_file(args.shapefile)
conus = conus.to_crs(epsg=3857)
conus_border = conus.dissolve()

# --------------------------------------------------------
# Load RouteLink NetCDF and build reach line geometries (optional)
# --------------------------------------------------------
gdf_rl = None
if args.routelink is not None:
    print(f"Loading NWM RouteLink from {args.routelink}...")
    ds = xr.open_dataset(args.routelink)

    # Extract fields
    link_ids = ds["link"].values
    lat = ds["lat"].values
    lon = ds["lon"].values
    to_ids = ds["to"].values
    order = ds["order"].values  # Strahler stream order

    # Coordinate lookup dictionary: COMID -> (lon, lat)
    id_to_xy = {
        int(fid): (float(x), float(y))
        for fid, x, y in zip(link_ids, lon, lat)
    }

    # Filter by river order if desired (recommended for CONUS)
    print("Filtering RouteLink by Strahler order >= 4...")
    mask = order >= 5  # you can change to >= 3 or remove filter

    lines = []
    for fid, toid, keep in zip(link_ids, to_ids, mask):
        if not keep:
            continue
        if toid in id_to_xy:
            x1, y1 = id_to_xy[int(fid)]
            x2, y2 = id_to_xy[int(toid)]
            line = LineString([(x1, y1), (x2, y2)])
            lines.append({"feature_id": int(fid), "geometry": line})

    print(f"Built {len(lines)} reach lines")

    if lines:
        gdf_rl = gpd.GeoDataFrame(lines, crs="EPSG:4326").to_crs(3857)
        print("RouteLink reaches loaded and reprojected.")
    else:
        print("No RouteLink lines were created (check filter or data).")

    ds.close()

# --------------------------------------------------------
# Create figure and axes
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5))
# ax.set_title(args.title, fontsize=10, loc="center", y=0.99)

# --------------------------------------------------------
# Plot state boundaries
# --------------------------------------------------------
conus_border.plot(ax=ax, facecolor="none", edgecolor="#001f01", linewidth=1.2, alpha=1.0, zorder=1)

# --------------------------------------------------------
# Plot RouteLink reaches (under basin points)
# --------------------------------------------------------
if gdf_rl is not None:
    gdf_rl.plot(
        ax=ax,
        color="#1f78b4",   # hydrologic blue
        linewidth=0.4,
        alpha=0.45,
        zorder=2,
    )
    print("Added RouteLink reaches to map.")

# --------------------------------------------------------
# Add basemap if requested
# --------------------------------------------------------
if args.basemap != "none":
    basemap_sources = {
        "terrain": ctx.providers.USGS.USTopo,
        "satellite": ctx.providers.Esri.WorldImagery,
        "topo": ctx.providers.OpenTopoMap,
        "streets": ctx.providers.OpenStreetMap.Mapnik,
    }
    basemap_source = basemap_sources.get(args.basemap, ctx.providers.USGS.USTopo)

    zoom_level = args.zoom if args.zoom is not None else ctx.tile._calculate_zoom(*ax.axis())
    try:
        ctx.add_basemap(ax, source=basemap_source, zoom=zoom_level, attribution_size=6)
        print(f"Added {args.basemap} basemap at zoom level {zoom_level}")
    except Exception as e:
        print(f"Warning: Could not load basemap: {e}")
        print("Falling back to plain background")
        ax.set_facecolor("#E8E8E8")
else:
    # Plain background if basemap is disabled
    ax.set_facecolor("#E8E8E8")
    print("Basemap disabled (basemap='none').")

# --------------------------------------------------------
# Plot basin points
# --------------------------------------------------------
gdf.plot(
    ax=ax,
    color=args.marker_color,
    markersize=args.marker_size,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
)

# --------------------------------------------------------
# Set bounds for CONUS (convert from lat/lon to Web Mercator)
# --------------------------------------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-125, 24)
xmax, ymax = transformer.transform(-66, 50)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")

# --------------------------------------------------------
# Legend
# --------------------------------------------------------
# Dummy handles for legend
ax.scatter([], [], color=args.marker_color, s=40, label="Basin Locations")
if gdf_rl is not None:
    ax.plot([], [], color="#1f78b4", linewidth=1.2, label="RouteLink Reaches")

ax.legend(loc="lower left", fontsize=9)

# --------------------------------------------------------
# Save figure
# --------------------------------------------------------
plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=500)
print(f"\n✓ Map saved to: {args.out}")
plt.show()
