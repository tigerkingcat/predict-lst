import pandas as pd

# Load the concatenated data file
df = pd.read_csv(
    "../../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_combined.csv")
# Create the full GEOID by concatenating State, County, Tract, and Block Group columns
df['GEOID'] = df['State'].astype(str).str.zfill(2) + \
              df['County'].astype(str).str.zfill(3) + \
              df['Tract'].astype(str).str.zfill(6) + \
              df['Block Group'].astype(str).str.zfill(1)

# Reorder columns to place GEOID at the beginning (optional)
df = df[['GEOID'] + [col for col in df.columns if col != 'GEOID']]

# Display the DataFrame to verify GEOID creation
print(df.head())

# Save the DataFrame with the new GEOID column
df.to_csv("../data/step2-aggregate/san_bernardino_blockgroups_socioeconomic_data_with_geoid.csv", index=False)
