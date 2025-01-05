# This program concatenates the Ndvi files for every year and puts them into one Df

import pandas as pd

df_2021 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2021.csv')

df_2019 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2019.csv')

df_2016 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2016.csv')

df_2013 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2013.csv')

df_2011 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2011.csv')

df_2008 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2008.csv')

df_2006 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2006.csv')

df_2004 = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/SanBernardino_NDVI_2004.csv')



# year adder
dfs = [df_2004, df_2006, df_2008, df_2011, df_2013, df_2016, df_2019, df_2021]
years = [2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

for df, year in zip(dfs, years):
    df['year'] = year
    print(f'on Df {year}')


# concatenates the df
combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df)

# Assuming `combined_df` is the concatenated DataFrame with 'longitude', 'latitude', and 'year' columns
sorted_df = combined_df.sort_values(by=['longitude', 'latitude', 'year'], ascending=[True, True, True])

# Display the sorted DataFrame
print(sorted_df)

sorted_df.to_csv('../../../data/final-model-data/NDVI_Pv/Final_ndvi_data.csv', index=False)
