# this script cleans up the data and merges it with block group and climate zones.

import pandas as pd
import geopandas as gpd

df = pd.read_csv('../../data/step2-aggregate/combined_data.csv')

print(df.columns)

# drop unnecessary columns
df.drop(columns=['year_pv.1', 'year_ndwi.1', 'system:index_impervious.1', 'year_impervious.1', 'latitude_impervious.1',
                 'longitude_impervious.1', 'distance', 'longitude_pv.1', 'latitude_pv.1', 'year_pv.1', 'distance.1',
                 'longitude_ndwi.1', 'latitude_ndwi.1', 'year_ndwi.1', 'distance.2', 'longitude_dem.1', 'latitude_dem.1'
    , 'distance.3'], inplace=True)

print(df.columns)

# rename columns to be more concise and accurate
df = df.rename(columns={
    'system:index_impervious': 'index',
    'latitude_dem': 'latitude',
    'longitude_dem': 'longitude',
    'impervious_impervious': 'impervious',
    'Pv_pv': 'Pv',
    'NDWI_ndwi': 'NDWI',
    'elevation_dem': 'elev',
    'LST_Day_1km': 'LST_Celsius'
})

print(df.columns)

# import block group shapefile
shp = gpd.read_file(r'../../data/step1-prep-process/Block_Group_Shapefile/tl_2023_06_bg.shp')

# convert df into geodf
gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')

# Ensure both GeoDataFrames use the same CRS
gdf = gdf.to_crs(shp.crs)

# join them together
merged_gdf = gpd.sjoin(gdf, shp, how='left', predicate='intersects')

merged_gdf.drop(columns='geometry').to_csv('../../data/step2-aggregate/points_with_block_groups.csv', index=False)  # Save as CSV without geometry

print(merged_gdf.head())

