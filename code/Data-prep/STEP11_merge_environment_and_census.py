import pandas as pd

census_df = pd.read_csv('../../data/step2-aggregate/san_bernardino_blockgroups_socioeconomic_data_with_geoid.csv')

environmental_df = pd.read_csv(r'../../data/step2-aggregate/Final_sbcounty_environ_data.csv')

# Ensure both GEOID columns are strings and apply zero-padding to match the Census GEOID format
# For Census data (already has zero-padded GEOID)
census_df['GEOID'] = census_df['GEOID'].astype(str)

# For Environmental data (add zero-padding to match Census format)
environmental_df['GEOID'] = environmental_df['GEOID'].astype(str)  # Assuming GEOID should be 12 characters

# Merge the datasets on 'GEOID' and 'Year' columns
merged_df = pd.merge(environmental_df, census_df, on=['GEOID', 'year'], how='inner')

# Display the merged DataFrame to verify the merge
print(merged_df.head())

# Save the merged dataset
merged_df.to_csv("merged_environmental_census_data.csv", index=False)
