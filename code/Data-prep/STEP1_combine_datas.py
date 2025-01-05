import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

lst_df = pd.read_csv('../../data/step2-aggregate/LST_Data_Final.csv')

impervious_df = pd.read_csv('../../data/step2-aggregate/impervious_data_final.csv')

pv_df = pd.read_csv('../../data/step2-aggregate/Final_Pv_Data.csv')

ndwi_df = pd.read_csv('../../data/step2-aggregate/Final_NDWI_data.csv')

elev_df = pd.read_csv('../../data/step2-aggregate/SanBernardino_DEM_1km_Points.csv')

# Convert each DataFrame to a GeoDataFrame
def to_geodataframe(df, lat_col='latitude', lon_col='longitude'):
    """Convert a DataFrame with latitude and longitude to a GeoDataFrame."""
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))


# Define a function to find the nearest neighbor within the same year using KDTree
def kdtree_nearest_by_year(source_gdf, target_gdf, source_suffix, target_suffix):
    """Find nearest points within the same year using KDTree."""
    # List to store results for each year
    results = []

    # Iterate through each unique year in the source GeoDataFrame
    for year in source_gdf['year'].unique():
        # Filter source and target DataFrames by the current year
        source_year = source_gdf[source_gdf['year'] == year]
        target_year = target_gdf[target_gdf['year'] == year]

        # Skip if target DataFrame is empty for this year
        if target_year.empty:
            continue

        # Extract coordinates of source and target GeoDataFrames for KDTree
        source_coords = np.array(list(zip(source_year.geometry.x, source_year.geometry.y)))
        target_coords = np.array(list(zip(target_year.geometry.x, target_year.geometry.y)))

        # Build a KDTree using target coordinates
        target_tree = cKDTree(target_coords)

        # Find the nearest target points for each source point
        distances, indices = target_tree.query(source_coords, k=1)  # k=1 means find the closest point

        # Create a DataFrame with the nearest neighbors and distances
        nearest_neighbors = target_year.iloc[indices].reset_index(drop=True)
        nearest_neighbors['distance'] = distances

        # Drop the 'geometry' column from nearest_neighbors to prevent duplication
        nearest_neighbors = nearest_neighbors.drop(columns='geometry')

        # Concatenate the source data with the nearest neighbor data
        merged = pd.concat([source_year.reset_index(drop=True), nearest_neighbors.reset_index(drop=True)], axis=1)

        # Rename columns to distinguish between source and target attributes
        merged = merged.rename(
            columns={col: f"{col}_{target_suffix}" for col in target_gdf.columns if col not in ['geometry']})

        # Append results for the current year
        results.append(merged)

    # Concatenate all results into a single DataFrame
    return pd.concat(results, ignore_index=True)


# Convert LST and Impervious DataFrames to GeoDataFrames
gdf_impervious = to_geodataframe(impervious_df)
gdf_lst = to_geodataframe(lst_df)

# Perform KDTree nearest neighbor search by year for LST and Impervious DataFrames
merged_gdf = kdtree_nearest_by_year(gdf_lst, gdf_impervious, 'lst', 'impervious')

print("Merged GDF Columns:", merged_gdf.columns)
# Convert back to a Pandas DataFrame and drop the 'geometry' column if not needed
merged_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))
print(merged_df)

merged_df.to_csv('../../data/step2-aggregate/Lst_impervious_combined.csv', index=False)