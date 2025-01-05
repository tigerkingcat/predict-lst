import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------- Configuration ---------------------------- #

# Paths to your files using raw string literals to handle backslashes
SHAPEFILE_PATH = r"C:\Users\aarav\ScienceProject24-25\BlockGroupCoordinates\tl_2023_06_bg\tl_2023_06_bg.shp"  # Shapefile path
CSV_INPUT_PATH = r"C:\Users\aarav\ScienceProject24-25\heatislandeffect\Model_1\SanBernardino_LST_NDVI_Impervious_Elevation_reshaped.csv"  # Input CSV path
CSV_OUTPUT_PATH = r"C:\Users\aarav\ScienceProject24-25\heatislandeffect\Model_1\BlockGroups_Sanbernardino.csv"  # Desired output CSV path

# Column names in your CSV for coordinates
LATITUDE_COLUMN = 'latitude'   # Update if different
LONGITUDE_COLUMN = 'longitude' # Update if different

# Geoid column name in the shapefile
GEOID_COLUMN = 'GEOID'        # Update if different

# ---------------------------- Load Shapefile ---------------------------- #

print("Loading shapefile...")
try:
    block_groups = gpd.read_file(SHAPEFILE_PATH)
    print(f"Shapefile loaded successfully. Number of block groups: {len(block_groups)}")
except Exception as e:
    raise FileNotFoundError(f"Error loading shapefile: {e}")

# Check if GEOID_COLUMN exists
if GEOID_COLUMN not in block_groups.columns:
    available_columns = ", ".join(block_groups.columns)
    raise ValueError(f"'{GEOID_COLUMN}' column not found in shapefile. Available columns: {available_columns}")

# ---------------------------- Load CSV and Create Geometry ---------------------------- #

print("Loading CSV file...")
try:
    df = pd.read_csv(CSV_INPUT_PATH)
    print(f"CSV loaded successfully. Number of records: {len(df)}")
except Exception as e:
    raise FileNotFoundError(f"Error loading CSV file: {e}")

# Check if coordinate columns exist
if LATITUDE_COLUMN not in df.columns or LONGITUDE_COLUMN not in df.columns:
    available_columns = ", ".join(df.columns)
    raise ValueError(f"CSV must contain '{LATITUDE_COLUMN}' and '{LONGITUDE_COLUMN}' columns. Available columns: {available_columns}")

# Handle potential missing or invalid coordinate data
if df[LATITUDE_COLUMN].isnull().any() or df[LONGITUDE_COLUMN].isnull().any():
    raise ValueError("CSV contains missing values in coordinate columns. Please clean the data before proceeding.")

# Create geometry column from latitude and longitude
print("Creating geometry for points...")
try:
    geometry = [Point(xy) for xy in zip(df[LONGITUDE_COLUMN], df[LATITUDE_COLUMN])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry)
    print("Geometry column created successfully.")
except Exception as e:
    raise RuntimeError(f"Error creating geometry: {e}")

# ---------------------------- Set Coordinate Reference System (CRS) ---------------------------- #

# Assuming shapefile CRS is authoritative
shapefile_crs = block_groups.crs
print(f"Shapefile CRS: {shapefile_crs}")

# Set the CRS for points (assuming WGS84 if coordinates are in lat/lon)
# You may need to adjust this if your CSV uses a different CRS
gdf_points.set_crs(epsg=4326, inplace=True)  # WGS84
print(f"Points CRS set to WGS84 (EPSG:4326).")

# Reproject points to shapefile CRS if they differ
if gdf_points.crs != shapefile_crs:
    print("Reprojecting points to match shapefile CRS...")
    try:
        gdf_points = gdf_points.to_crs(shapefile_crs)
        print("Reprojection successful.")
    except Exception as e:
        raise RuntimeError(f"Error reprojecting points: {e}")
else:
    print("Points CRS matches shapefile CRS. No reprojection needed.")

# ---------------------------- Spatial Join ---------------------------- #

print("Performing spatial join...")
try:
    # Perform spatial join
    joined = gpd.sjoin(gdf_points, block_groups[[GEOID_COLUMN, 'geometry']], how='left', predicate='within')
    print("Spatial join completed successfully.")
except Exception as e:
    raise RuntimeError(f"Error during spatial join: {e}")

# ---------------------------- Handle Results ---------------------------- #

# Rename GEOID column appropriately
if GEOID_COLUMN in joined.columns:
    joined.rename(columns={GEOID_COLUMN: 'geoid'}, inplace=True)
    print(f"Renamed '{GEOID_COLUMN}' column to 'geoid'.")
else:
    raise KeyError(f"'{GEOID_COLUMN}' column not found after spatial join.")

# If you don't need the spatial columns anymore, you can drop them
columns_to_drop = []
if 'geometry' in joined.columns:
    columns_to_drop.append('geometry')
if 'index_right' in joined.columns:
    columns_to_drop.append('index_right')

if columns_to_drop:
    joined.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns: {', '.join(columns_to_drop)}")

# ---------------------------- Save to CSV ---------------------------- #

print(f"Saving results to {CSV_OUTPUT_PATH}...")
try:
    joined.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Results saved successfully to '{CSV_OUTPUT_PATH}'.")
except Exception as e:
    raise IOError(f"Error saving output CSV: {e}")

print("Matching completed successfully.")
