import pandas as pd
from scipy.spatial import cKDTree

# Load the CBG coordinates data with GEOID
cbg_coords_df = pd.read_csv('C://Users//aarav//ScienceProject24-25//BlockGroupCoordinates//cbg_centroids.csv')  # Replace with your actual file path
# Make sure your file has columns: 'GEOID', 'Latitude', 'Longitude'

# Load the environmental data (LST data)
lst_data_df = pd.read_csv('C://Users//aarav//ScienceProject24-25//LST Datas//summer_output_lst_coordinates.csv')  # Replace with your actual file path
# Make sure your file has columns: 'Latitude', 'Longitude', 'LST_Celsius'

# Check the order of columns (for debugging purposes)
print("CBG Coordinates DataFrame Columns:", cbg_coords_df.columns)
print("LST Data DataFrame Columns:", lst_data_df.columns)

# Build a KDTree for fast spatial lookup using Longitude first, then Latitude
tree = cKDTree(lst_data_df[['Longitude', 'Latitude']].values)

# Define a function to find the nearest LST value for each GEOID coordinate
def find_nearest(lst_df, tree, lat, lon):
    distance, index = tree.query([lon, lat])  # Ensure Longitude is first, then Latitude
    return lst_df.iloc[index]['LST_Celsius']

# Apply the function to find the nearest LST value for each GEOID
cbg_coords_df['LST_Celsius'] = cbg_coords_df.apply(
    lambda row: find_nearest(lst_data_df, tree, row['Latitude'], row['Longitude']),
    axis=1
)

# Swap Latitude and Longitude if necessary (This will correct the switch if it was not done correctly initially)
cbg_coords_df = cbg_coords_df.rename(columns={'Latitude': 'Longitude_temp', 'Longitude': 'Latitude'})
cbg_coords_df = cbg_coords_df.rename(columns={'Longitude_temp': 'Longitude'})

# Save the combined data to a new CSV file
cbg_coords_df.to_csv('C://Users//aarav//ScienceProject24-25//BlockGroupCoordinates//geoids_with_corrected_lst_data.csv', index=False)

print("Data has been combined and saved to 'geoids_with_corrected_lst_data.csv'.")
