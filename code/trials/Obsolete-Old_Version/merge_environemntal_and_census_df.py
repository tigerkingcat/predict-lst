import pandas as pd

census_df = pd.read_csv("../../../data/trials/Rural_or_Urban.csv")

environmental_df = pd.read_csv("../../../data/trials/lst_impervious_blockgroup.csv")

merged_df = pd.merge(environmental_df, census_df, how='inner', on='GEOID' )

print(merged_df['impervious'])

merged_df.to_csv('data.csv', index=False)