import geopandas as gpd
import pandas as pd

# Load the shapefile (update the path to your shapefile)
shapefile_path = '../../../data/step1-pre-process/Block_Group_Shapefile/tl_2023_06_bg.shp'

gdf = gpd.read_file(shapefile_path)


# Function to extract the centroid coordinates of the geometry
def extract_centroid(geometry):
    centroid = geometry.centroid
    return centroid.x, centroid.y


# Create a list to store the results
results = []

# Iterate through each row in the GeoDataFrame and extract the centroid
for index, row in gdf.iterrows():
    geoid = row['GEOID']  # The unique identifier for the Census Block Group
    latitude, longitude = extract_centroid(row.geometry)

    # Append the results to the list
    results.append({'GEOID': geoid, 'Latitude': latitude, 'Longitude': longitude})

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('cbg_centroids.csv', index=False)

print("CSV file with centroid coordinates has been generated successfully.")
