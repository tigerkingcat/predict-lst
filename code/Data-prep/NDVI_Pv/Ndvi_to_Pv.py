# this script converts ndvi values in to Pv values

import pandas as pd

df = pd.read_csv('../../../data/step1-pre-process/NDVI_Pv/Final_ndvi_data.csv')

# calculate min and max ndvi values.

min_NDVI = df['NDVI'].min()
max_NDVI = df['NDVI'].max()

# create new Pv column
df['Pv'] = df['NDVI'].apply(lambda x: ((x-min_NDVI)/(max_NDVI-min_NDVI)) ** 2)

print(df['Pv'])

# drop Ndvi values and export to csv

df.drop(axis=1, columns='NDVI', inplace=True)
df.to_csv('../../../data/final-model-data/NDVI_Pv/Final_Pv_Data.csv', index=False)
