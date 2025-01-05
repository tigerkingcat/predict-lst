import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------- Configuration ---------------------------- #

# Paths to your files using raw string literals to handle backslashes
CSV_INPUT_PATH = r"C:\Users\aarav\ScienceProject24-25\heatislandeffect\Model_1\SanBernardino_LST_NDVI_Impervious_Elevation_reshaped.csv"  # Input CSV path
CSV_OUTPUT_PATH = r"C:\Users\aarav\ScienceProject24-25\heatislandeffect\Model_1\SanBernardino_WithClimate.csv"  # Desired output CSV path

# Climate classification shapefile path
CLIMATE_SHAPEFILE_PATH = r"C:\Users\aarav\Downloads\c1976_2000_0\c1976_2000.shp"  # Update with your climate shapefile path

# Column names in your CSV for coordinates
LATITUDE_COLUMN = 'latitude'   # Update if different
LONGITUDE_COLUMN = 'longitude' # Update if different

# Climate classification column name in the climate shapefile
CLIMATE_COLUMN = 'GRIDCODE'      # Update based on your climate shapefile's attribute for classification

# ---------------------------- Mapping GRIDCODE to Köppen Codes ---------------------------- #

# Define a mapping dictionary for GRIDCODE to Köppen-Geiger codes based on user-provided mapping
gridcode_to_koppen = {
    11: 'Af',
    12: 'Am',
    13: 'As',
    14: 'Aw',
    21: 'BWk',
    22: 'BWh',
    26: 'BSk',
    27: 'BSh',
    31: 'Cfa',
    32: 'Cfb',
    33: 'Cfc',
    34: 'Csa',
    35: 'Csb',
    36: 'Csc',
    37: 'Cwa',
    38: 'Cwb',
    39: 'Cwc',
    41: 'Dfa',
    42: 'Dfb',
    43: 'Dfc',
    44: 'Dfd',
    45: 'Dsa',
    46: 'Dsb',
    47: 'Dsc',
    48: 'Dsd',
    49: 'Dwa',
    50: 'Dwb',
    51: 'Dwc',
    52: 'Dwd',
    61: 'EF',
    62: 'ET'
}

# ---------------------------- Mapping Köppen Codes to Climate Categories ---------------------------- #

# Define a mapping dictionary for Köppen-Geiger codes to broader climate categories
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
    # Add more mappings as needed
}

# ---------------------------- Load CSV and Create Geometry ---------------------------- #

print("Loading CSV file...")
try:
    df = pd.read_csv(CSV_INPUT_PATH)
    print(f"CSV file loaded successfully. Number of records: {len(df)}")
except Exception as e:
    raise FileNotFoundError(f"Error loading CSV file: {e}")

# Check if coordinate columns exist
if LATITUDE_COLUMN not in df.columns or LONGITUDE_COLUMN not in df.columns:
    available_columns = ", ".join(df.columns)
    raise ValueError(f"CSV must contain '{LATITUDE_COLUMN}' and '{LONGITUDE_COLUMN}' columns. Available columns: {available_columns}")

# Handle potential missing or invalid coordinate data
if df[LATITUDE_COLUMN].isnull().any() or df[LONGITUDE_COLUMN].isnull().any():
    raise ValueError("CSV contains missing values in coordinate columns. Please clean the data before proceeding.")

print("\nCreating geometry for points...")
try:
    geometry = [Point(xy) for xy in zip(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry)
    print("Geometry column created successfully.")
except Exception as e:
    raise RuntimeError(f"Error creating geometry: {e}")

# ---------------------------- Load Climate Shapefile ---------------------------- #

print("\nLoading climate classification shapefile...")
try:
    climate_zones = gpd.read_file(CLIMATE_SHAPEFILE_PATH)
    print(f"Climate shapefile loaded successfully. Number of climate zones: {len(climate_zones)}")
except Exception as e:
    raise FileNotFoundError(f"Error loading climate classification shapefile: {e}")

# Check if CLIMATE_COLUMN exists
if CLIMATE_COLUMN not in climate_zones.columns:
    available_columns = ", ".join(climate_zones.columns)
    raise ValueError(f"'{CLIMATE_COLUMN}' column not found in climate shapefile. Available columns: {available_columns}")

# ---------------------------- Set Coordinate Reference System (CRS) ---------------------------- #

# Assuming climate shapefile CRS is authoritative
climate_crs = climate_zones.crs
print(f"\nClimate shapefile CRS: {climate_crs}")

# Set the CRS for points (assuming WGS84 if coordinates are in lat/lon)
# Adjust if your CSV uses a different CRS
gdf_points.set_crs(epsg=4326, inplace=True)  # WGS84
print("Points CRS set to WGS84 (EPSG:4326).")

# Reproject points to climate shapefile CRS if they differ
if gdf_points.crs != climate_crs:
    print("Reprojecting points to match climate shapefile CRS...")
    try:
        gdf_points = gdf_points.to_crs(climate_crs)
        print("Reprojection successful.")
    except Exception as e:
        raise RuntimeError(f"Error reprojecting points: {e}")
else:
    print("Points CRS matches climate shapefile CRS. No reprojection needed.")

# ---------------------------- Spatial Join: Assign Climate Classification ---------------------------- #

print("\nPerforming spatial join to assign climate classification...")
try:
    # Perform spatial join to assign climate classification to each point
    joined_climate = gpd.sjoin(gdf_points, climate_zones[[CLIMATE_COLUMN, 'geometry']], how='left', predicate='within')
    print("Spatial join for climate classification completed successfully.")
except Exception as e:
    raise RuntimeError(f"Error during spatial join for climate classification: {e}")

# Rename climate column appropriately
if CLIMATE_COLUMN in joined_climate.columns:
    joined_climate.rename(columns={CLIMATE_COLUMN: 'climate_code'}, inplace=True)
    print(f"Renamed '{CLIMATE_COLUMN}' column to 'climate_code'.")
else:
    raise KeyError(f"'{CLIMATE_COLUMN}' column not found after spatial join.")

# Drop unnecessary columns
columns_to_drop = []
if 'geometry' in joined_climate.columns:
    columns_to_drop.append('geometry')
if 'index_right' in joined_climate.columns:
    columns_to_drop.append('index_right')

if columns_to_drop:
    joined_climate.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns: {', '.join(columns_to_drop)}")

# ---------------------------- Map GRIDCODE to Köppen-Geiger Codes ---------------------------- #

print("\nMapping GRIDCODE to Köppen-Geiger codes...")
joined_climate['koppen_code'] = joined_climate['climate_code'].map(gridcode_to_koppen)

# Handle unmapped GRIDCODE values
unmapped_koppen = joined_climate['koppen_code'].isnull().sum()
if unmapped_koppen > 0:
    print(f"Warning: {unmapped_koppen} records have unmapped GRIDCODEs. They will be set as 'Unknown'.")
    joined_climate['koppen_code'].fillna('Unknown', inplace=True)
else:
    print("All GRIDCODEs mapped to Köppen-Geiger codes successfully.")

# ---------------------------- Map Köppen-Geiger Codes to Climate Categories ---------------------------- #

print("\nMapping Köppen-Geiger codes to broader climate categories...")
joined_climate['climate_category'] = joined_climate['koppen_code'].map(koppen_to_category)

# Handle unmapped Köppen codes
unmapped_category = joined_climate['climate_category'].isnull().sum()
if unmapped_category > 0:
    print(f"Warning: {unmapped_category} records have unmapped Köppen-Geiger codes. They will be set as 'Other'.")
    joined_climate['climate_category'].fillna('Other', inplace=True)
else:
    print("All Köppen-Geiger codes mapped to climate categories successfully.")

# ---------------------------- Save to CSV ---------------------------- #

print(f"\nSaving results to '{CSV_OUTPUT_PATH}'...")
try:
    joined_climate.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Results saved successfully to '{CSV_OUTPUT_PATH}'.")
except Exception as e:
    raise IOError(f"Error saving output CSV: {e}")

print("\nAssigning climate classification completed successfully.")
