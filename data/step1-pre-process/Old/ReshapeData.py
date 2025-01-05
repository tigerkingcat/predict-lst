import pandas as pd

# Load the data from the CSV file
input_csv = 'SanBernardino_LST_NDVI_Impervious_Elevation.csv'
data = pd.read_csv(input_csv)

# Extract the columns that match LST, NDVI, Impervious, and Elevation
value_vars_lst = [col for col in data.columns if '_LST_Celsius' in col]
value_vars_ndvi = [col for col in data.columns if '_NDVI' in col]
value_vars_impervious = [col for col in data.columns if '_Impervious' in col]
value_vars_elevation = [col for col in data.columns if '_Elevation' in col]

# Melt each variable separately and add a column for year
melted_lst = pd.melt(data, id_vars=['longitude', 'latitude'], value_vars=value_vars_lst,
                     var_name='year', value_name='LST_Celsius')
melted_lst['year'] = melted_lst['year'].str.extract(r'(\d+)_').astype(int) + 2005  # Assuming 0 corresponds to 2005

melted_ndvi = pd.melt(data, id_vars=['longitude', 'latitude'], value_vars=value_vars_ndvi,
                      var_name='year', value_name='NDVI')
melted_ndvi['year'] = melted_ndvi['year'].str.extract(r'(\d+)_').astype(int) + 2005

melted_impervious = pd.melt(data, id_vars=['longitude', 'latitude'], value_vars=value_vars_impervious,
                            var_name='year', value_name='Impervious')
melted_impervious['year'] = melted_impervious['year'].str.extract(r'(\d+)_').astype(int) + 2005

melted_elevation = pd.melt(data, id_vars=['longitude', 'latitude'], value_vars=value_vars_elevation,
                           var_name='year', value_name='Elevation')
melted_elevation['year'] = melted_elevation['year'].str.extract(r'(\d+)_').astype(int) + 2005

# Merge all melted DataFrames into a single DataFrame
combined = melted_lst.merge(melted_ndvi, on=['longitude', 'latitude', 'year'])
combined = combined.merge(melted_impervious, on=['longitude', 'latitude', 'year'])
combined = combined.merge(melted_elevation, on=['longitude', 'latitude', 'year'])

# Sort the data by longitude, latitude, and year
combined = combined.sort_values(by=['longitude', 'latitude', 'year']).reset_index(drop=True)

# Save the reshaped data to a new CSV file
output_csv = 'SanBernardino_LST_NDVI_Impervious_Elevation_reshaped.csv'
combined.to_csv(output_csv, index=False)

print(f'Reshaped data saved to {output_csv}')
