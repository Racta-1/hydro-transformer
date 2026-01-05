#!/usr/bin/env python3
"""
Simple script to visualize basin locations with RouteLink stream reaches.

Usage:
    python view_basins_with_reaches.py --coords data/camels_link.csv \
                                        --routelink data/RouteLink.nc \
                                        --out basins_with_streams.png \
                                        --basemap terrain
"""

import argparse
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx
from pyproj import Transformer
import numpy as np

parser = argparse.ArgumentParser(description="Visualize basins with stream reaches.")
parser.add_argument("--coords", "-c", required=True, 
                    help="CSV file with basin coordinates")
parser.add_argument("--routelink", "-r", required=True,
                    help="RouteLink NetCDF or shapefile with stream reaches")
parser.add_argument("--out", "-o", default="basins_streams.png",
                    help="Output file")
parser.add_argument("--shapefile", "-s", default="us_states_shapefile/tl_2023_us_state.shp",
                    help="Path to CONUS shapefile")
parser.add_argument("--basemap", "-b", default="terrain",
                    choices=["terrain", "satellite", "topo", "streets", "none"],
                    help="Basemap style")
parser.add_argument("--zoom", "-z", type=int, default=6,
                    help="Zoom level for basemap")
parser.add_argument("--stream-color", "-sc", default="#4A90E2",
                    help="Color for stream reaches")
parser.add_argument("--basin-color", "-bc", default="#1A3A5C",
                    help="Color for basin markers")
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


# Load basin coordinates
print(f"Loading basin coordinates from {args.coords}...")
df_coords = pd.read_csv(args.coords)
df_coords.columns = df_coords.columns.str.strip()

gage_col = None
for col in ["gages", "gagestr", "gage_id"]:
    if col in df_coords.columns:
        gage_col = col
        break

if gage_col is None:
    raise ValueError("Coordinates file must have 'gages', 'gagestr', or 'gage_id' column")

df_coords["gage_id"] = df_coords[gage_col].map(normalize_id)
df_coords = df_coords.drop_duplicates(subset=["gage_id"], keep="first")

for c in ["lat", "lon"]:
    if c in df_coords.columns:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="coerce")

df_coords = df_coords.dropna(subset=["lat", "lon"])
print(f"Loaded {len(df_coords)} basin locations")

# Create basin GeoDataFrame
gdf_basins = gpd.GeoDataFrame(
    df_coords,
    geometry=gpd.points_from_xy(df_coords['lon'], df_coords['lat']),
    crs="EPSG:4326"
).to_crs(epsg=3857)


# Load RouteLink reaches
print(f"Loading RouteLink reaches from {args.routelink}...")
routelink_path = Path(args.routelink)

if routelink_path.suffix == '.nc':
    # Load from NetCDF
    import xarray as xr
    
    ds = xr.open_dataset(args.routelink)
    
    # Extract coordinates - typical RouteLink structure
    # Adjust these field names based on your actual NetCDF structure
    if 'lon' in ds.variables and 'lat' in ds.variables:
        lons = ds['lon'].values
        lats = ds['lat'].values
    elif 'longitude' in ds.variables and 'latitude' in ds.variables:
        lons = ds['longitude'].values
        lats = ds['latitude'].values
    else:
        # Try to find coordinate variables
        coord_vars = [v for v in ds.variables if 'lon' in v.lower() or 'lat' in v.lower()]
        print(f"Available variables: {list(ds.variables.keys())}")
        raise ValueError(f"Could not find lon/lat coordinates. Available: {coord_vars}")
    
    # Create line segments from coordinates
    from shapely.geometry import LineString, Point
    
    # If there are from/to coordinates, create lines
    # Otherwise create points
    if len(lons.shape) == 2 and lons.shape[1] == 2:
        # Has from/to coordinates
        geometries = [LineString([(lons[i,0], lats[i,0]), (lons[i,1], lats[i,1])]) 
                     for i in range(len(lons))]
    else:
        # Single point per reach - we'll need to connect them
        # For now, just plot as points
        geometries = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    
    gdf_reaches = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    ds.close()
    
elif routelink_path.suffix in ['.shp', '.geojson', '.gpkg']:
    # Load from vector file
    gdf_reaches = gpd.read_file(args.routelink)
    
else:
    raise ValueError(f"Unsupported RouteLink format: {routelink_path.suffix}")

# Convert to Web Mercator
gdf_reaches = gdf_reaches.to_crs(epsg=3857)
print(f"Loaded {len(gdf_reaches)} stream reaches")


# Load CONUS boundary
conus = gpd.read_file(args.shapefile)
conus = conus.to_crs(epsg=3857)
conus_border = conus.dissolve()

# Setup basemap
basemap_sources = {
    "terrain": ctx.providers.USGS.USTopo,
    "satellite": ctx.providers.Esri.WorldImagery,
    "topo": ctx.providers.OpenTopoMap,
    "streets": ctx.providers.OpenStreetMap.Mapnik
}
basemap_source = basemap_sources.get(args.basemap, ctx.providers.USGS.USTopo)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Add basemap
if args.basemap != "none":
    try:
        ctx.add_basemap(ax, source=basemap_source, zoom=args.zoom, attribution_size=6)
        print(f"Added {args.basemap} basemap")
    except Exception as e:
        print(f"Warning: Could not load basemap: {e}")
        ax.set_facecolor("#E8E8E8")
else:
    ax.set_facecolor("#E8E8E8")

# Plot CONUS border
conus_border.plot(ax=ax, facecolor="none", edgecolor="#2A2A2A",
                  linewidth=2.5, alpha=1.0, zorder=2)

# Plot stream reaches
if gdf_reaches.geometry.type[0] == 'LineString' or \
   (hasattr(gdf_reaches.geometry.type, '__iter__') and 'LineString' in gdf_reaches.geometry.type.values):
    gdf_reaches.plot(ax=ax, color=args.stream_color, linewidth=0.8,
                     alpha=0.6, zorder=3)
else:
    gdf_reaches.plot(ax=ax, color=args.stream_color, markersize=1,
                     alpha=0.5, zorder=3)

# Plot basin points
gdf_basins.plot(ax=ax, color=args.basin_color, markersize=25,
                edgecolor="white", linewidth=1.0, alpha=0.9, zorder=4)

# Set bounds
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-125, 24)
xmax, ymax = transformer.transform(-66, 50)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis("off")

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=args.basin_color,
           markersize=10, label='Basin Locations', markeredgecolor='white'),
    Line2D([0], [0], color=args.stream_color, linewidth=2,
           label='Stream Reaches', alpha=0.7)
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
          framealpha=0.9, edgecolor='#666666')

# Add title
ax.set_title("CONUS Basins with Stream Network", 
             fontsize=14, fontweight='bold', pad=15)

# Save
plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=400)
print(f"\n✓ Map saved to: {args.out}")
plt.show()