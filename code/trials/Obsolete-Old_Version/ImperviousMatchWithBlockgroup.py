import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

# 1. Read the impervious surface data CSV
impervious_df = pd.read_csv('C:\\Users\\aarav\\ScienceProject24-25\\LST Datas\\Impervious_PerPixel_SanBernardino.csv')

# 2. Parse the 'geo' field to extract longitude and latitude
def extract_coordinates(geo_str):
    geo_dict = json.loads(geo_str)
    coords = geo_dict['coordinates']
    return coords[0], coords[1]

impervious_df[['longitude', 'latitude']] = impervious_df['.geo'].apply(
    lambda x: pd.Series(extract_coordinates(x))
)

# 3. Create geometry and convert to GeoDataFrame
impervious_df['geometry'] = impervious_df.apply(
    lambda row: Point(row['longitude'], row['latitude']), axis=1
)
impervious_gdf = gpd.GeoDataFrame(impervious_df, geometry='geometry')

# 4. Read the block group shapefile
block_group_gdf = gpd.read_file('C:\\Users\\aarav\\ScienceProject24-25\\BlockGroupCoordinates\\tl_2023_06_bg\\tl_2023_06_bg.shp')

# 5. Ensure both GeoDataFrames have the same CRS
impervious_gdf.set_crs(block_group_gdf.crs, inplace=True)
print(block_group_gdf.columns)

# 6. Perform spatial join
joined_gdf = gpd.sjoin(
    impervious_gdf,
    block_group_gdf[['GEOID', 'geometry']],
    how='left',
    predicate='within'
)

# 7. Aggregate imperviousness by block group
grouped_imperviousness = joined_gdf.groupby('GEOID').agg(
    {'impervious': 'mean'}
).reset_index()

# 8. Merge aggregated data back to block group GeoDataFrame
block_group_gdf = block_group_gdf.merge(grouped_imperviousness, on='GEOID', how='left')

# 9. Handle missing values (optional)
block_group_gdf['impervious'] = block_group_gdf['impervious'].fillna(0)

# 10. Export results to CSV
export_df = block_group_gdf[['GEOID', 'impervious']]
export_df.to_csv('block_group_imperviousness.csv', index=False)

# 11. Save updated block group shapefile
block_group_gdf.to_file('block_groups_with_imperviousness.shp')