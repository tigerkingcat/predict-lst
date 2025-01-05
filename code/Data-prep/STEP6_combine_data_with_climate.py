import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Paths to your files using raw string literals to handle backslashes
CSV_INPUT_PATH = r"../../data/step2-aggregate/points_with_block_groups.csv"  # Input CSV path
CSV_OUTPUT_PATH = r"../../data/step2-aggregate/data_without_urban.csv"  # Desired output CSV path
CLIMATE_SHAPEFILE_PATH = r"../../data/step1-pre-process/Climate_Zones_Shapefile\c1976_2000.shp"  # Climate shapefile path

# Load CSV and create GeoDataFrame
df = pd.read_csv(CSV_INPUT_PATH)
gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')

# Load climate classification shapefile
climate_zones = gpd.read_file(CLIMATE_SHAPEFILE_PATH)

# Reproject points to match climate shapefile CRS if they differ
gdf_points = gdf_points.to_crs(climate_zones.crs)

gdf_points.drop(columns='index_right', inplace=True)
# Perform spatial join to assign climate classification to each point
joined_climate = gpd.sjoin(gdf_points, climate_zones[['GRIDCODE', 'geometry']], how='left', predicate='within')

# Map GRIDCODE to Köppen-Geiger codes
gridcode_to_koppen = {
    11: 'Af', 12: 'Am', 13: 'As', 14: 'Aw', 21: 'BWk', 22: 'BWh', 26: 'BSk', 27: 'BSh',
    31: 'Cfa', 32: 'Cfb', 33: 'Cfc', 34: 'Csa', 35: 'Csb', 36: 'Csc', 37: 'Cwa', 38: 'Cwb',
    39: 'Cwc', 41: 'Dfa', 42: 'Dfb', 43: 'Dfc', 44: 'Dfd', 45: 'Dsa', 46: 'Dsb', 47: 'Dsc',
    48: 'Dsd', 49: 'Dwa', 50: 'Dwb', 51: 'Dwc', 52: 'Dwd', 61: 'EF', 62: 'ET'
}
joined_climate['koppen_code'] = joined_climate['GRIDCODE'].map(gridcode_to_koppen)

# Map Köppen codes to climate categories
koppen_to_category = {
    'Af': 'Tropical Rainforest',
    'Am': 'Tropical Monsoon',
    'As': 'Tropical Savanna',
    'Aw': 'Tropical Savanna',
    'BWk': 'Arid (Cold)',
    'BWh': 'Arid (Hot)',
    'BSk': 'Semi-Arid (Cold)',
    'BSh': 'Semi-Arid (Hot)',
    'Cfa': 'Humid Subtropical',
    'Cfb': 'Oceanic',
    'Cfc': 'Subpolar Oceanic',
    'Csa': 'Mediterranean',
    'Csb': 'Mediterranean',
    'Csc': 'Mediterranean',
    'Cwa': 'Monsoon-influenced Humid Subtropical',
    'Cwb': 'Oceanic',
    'Cwc': 'Oceanic',
    'Dfa': 'Humid Continental',
    'Dfb': 'Humid Continental',
    'Dfc': 'Subarctic',
    'Dfd': 'Subarctic',
    'Dsa': 'Continental Mediterranean',
    'Dsb': 'Continental Mediterranean',
    'Dsc': 'Continental Mediterranean',
    'Dsd': 'Continental Mediterranean',
    'Dwa': 'Humid Continental',
    'Dwb': 'Humid Continental',
    'Dwc': 'Subarctic',
    'Dwd': 'Subarctic',
    'EF': 'Polar Ice Cap',
    'ET': 'Tundra'
}
joined_climate['climate_category'] = joined_climate['koppen_code'].map(koppen_to_category)

# Drop unnecessary columns
joined_climate.drop(columns=['geometry', 'index_right'], inplace=True, errors='ignore')

# Save the results to CSV
joined_climate.to_csv(CSV_OUTPUT_PATH, index=False)
print("Climate classification assigned and saved successfully.")