import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

combined_df = pd.read_csv('../../data/step2-aggregate/combined_environmental.csv')

dem_df = pd.read_csv('../../data/step2-aggregate/SanBernardino_DEM_1km_Points.csv')

# Convert each DataFrame to a GeoDataFrame
def to_geodataframe(df, lat_col='latitude', lon_col='longitude'):
    """Convert a DataFrame with latitude and longitude to a GeoDataFrame."""
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))

# Define a function to find the nearest neighbor based on latitude and longitude using KDTree
def kdtree_nearest(source_gdf, target_gdf, source_suffix, target_suffix):
    """Find nearest points based on latitude and longitude using KDTree."""
    # Extract coordinates of source and target GeoDataFrames for KDTree
    source_coords = np.array(list(zip(source_gdf.geometry.x, source_gdf.geometry.y)))
    target_coords = np.array(list(zip(target_gdf.geometry.x, target_gdf.geometry.y)))

    # Build a KDTree using target coordinates
    target_tree = cKDTree(target_coords)

    # Find the nearest target points for each source point
    distances, indices = target_tree.query(source_coords, k=1)  # k=1 means find the closest point

    # Create a DataFrame with the nearest neighbors and distances
    nearest_neighbors = target_gdf.iloc[indices].reset_index(drop=True)
    nearest_neighbors['distance'] = distances

    # Drop the 'geometry' column from nearest_neighbors to prevent duplication
    nearest_neighbors = nearest_neighbors.drop(columns='geometry')

    # Concatenate the source data with the nearest neighbor data
    merged = pd.concat([source_gdf.reset_index(drop=True), nearest_neighbors.reset_index(drop=True)], axis=1)

    # Rename columns to distinguish between source and target attributes
    merged = merged.rename(columns={col: f"{col}_{target_suffix}" for col in target_gdf.columns if col not in ['geometry']})

    return merged
combined_df = combined_df.rename(columns={
    'latitude_ndwi': 'latitude',
    'longitude_ndwi': 'longitude',
    'year_ndwi': 'year'
})
# Convert LST and DEM DataFrames to GeoDataFrames
gdf_combined = to_geodataframe(combined_df)
gdf_dem = to_geodataframe(dem_df)

# Perform KDTree nearest neighbor search for LST and DEM DataFrames based on latitude and longitude
merged_gdf = kdtree_nearest(gdf_combined, gdf_dem, 'lst', 'dem')

print("Merged GDF Columns:", merged_gdf.columns)
# Convert back to a Pandas DataFrame and drop the 'geometry' column if not needed
merged_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))
print(merged_df)

merged_df.to_csv('../../data/step2-aggregate/combined_data.csv', index=False)