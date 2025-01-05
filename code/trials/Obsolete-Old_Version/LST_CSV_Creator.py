import rasterio
from rasterio.mask import mask
from pyproj import Proj, transform
import numpy as np
import csv


# Function to convert MODIS LST from Kelvin to Celsius
def kelvin_to_celsius(kelvin):
    return kelvin - 0


# Function to convert row, col indices to lat, lon
def pixel_to_latlon(transform, row, col):
    lon, lat = transform * (col, row)
    return lon, lat


# Define file paths
modis_tif_path = 'C:\\Users\\aarav\\ScienceProject24-25\\MODIS_LST_Summer_Southern_California.tif'
output_csv_path = 'C:\\Users\\aarav\\ScienceProject24-25\\summer_output_lst_coordinates.csv'

# Define Southern California bounding box (in lat, lon)
# Adjust this based on your specific area of interest
south_cal_bounding_box = {
    "min_lat": 32.5,
    "max_lat": 36.5,
    "min_lon": -120.0,
    "max_lon": -114.0
}

# Read the MODIS LST data
with rasterio.open(modis_tif_path) as src:
    # Read the data as an array
    lst_data = src.read(1)  # Assuming single-band LST data in Kelvin
    transform = src.transform
    # Get the dimensions
    rows, cols = lst_data.shape

    # Prepare output CSV
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude", "LST_Celsius"])

        # Iterate over each pixel
        for row in range(rows):
            for col in range(cols):
                # Convert pixel index to lat, lon
                lon, lat = pixel_to_latlon(transform, row, col)

                # Filter by Southern California bounding box
                if (south_cal_bounding_box["min_lat"] <= lat <= south_cal_bounding_box["max_lat"]) and \
                        (south_cal_bounding_box["min_lon"] <= lon <= south_cal_bounding_box["max_lon"]):
                    # Get the LST value and convert to Celsius
                    lst_value_kelvin = lst_data[row, col]
                    lst_value_celsius = kelvin_to_celsius(lst_value_kelvin)

                    # Write lat, lon, and LST to the CSV
                    writer.writerow([lat, lon, lst_value_celsius])

print(f"Data extraction completed and written to {output_csv_path}")
