import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Paths to your files
CSV_INPUT_PATH = r"../../data/step2-aggregate/data_without_urban.csv"  # Input CSV path
CSV_OUTPUT_PATH = r"../../data/step2-aggregate/Final_sbcounty_environ_data.csv"  # Desired output CSV path
URBAN_RURAL_SHAPEFILE_PATH = r"../../data/step1-pre-process/Urban/tl_2020_us_uac20/tl_2020_us_uac20.shp"  # Urban/Rural classification shapefile path

# Load CSV and create GeoDataFrame
df = pd.read_csv(CSV_INPUT_PATH)
gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')

# Load urban/rural classification shapefile
urban_rural_zones = gpd.read_file(URBAN_RURAL_SHAPEFILE_PATH)

# Reproject points to match urban/rural shapefile CRS if they differ
gdf_points = gdf_points.to_crs(urban_rural_zones.crs)

print(urban_rural_zones.columns)
# Perform spatial join to assign urban/rural classification to each point
joined_urban_rural = gpd.sjoin(gdf_points, urban_rural_zones[['UATYP20', 'geometry']], how='left', predicate='within')
# Rename 'UATYP20' to 'urban_rural_classification' for clarity
joined_urban_rural = joined_urban_rural.rename(columns={'UATYP20': 'urban_rural_classification'})

# Drop unnecessary columns
joined_urban_rural.drop(columns=['geometry', 'index_right'], inplace=True, errors='ignore')

# Save the results to CSV
joined_urban_rural.to_csv(CSV_OUTPUT_PATH, index=False)
print("Urban/Rural classification assigned and saved successfully.")