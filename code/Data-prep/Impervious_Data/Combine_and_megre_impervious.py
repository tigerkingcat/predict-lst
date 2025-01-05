#this script sorts and merges impervious surface data together to create impervious surface for san bernardino from year
#2005 to 2020

import pandas as pd


df_2021 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2021.csv")

df_2019 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2019.csv")

df_2016 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2016.csv")

df_2013 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2013.csv")

df_2011 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2011.csv")

df_2008 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2008.csv")

df_2006 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2006.csv")

df_2004 = pd.read_csv(r"../../../data/step1-pre-process/Impervious_Data/Impervious_Surface_SanBernardino_1km_2004.csv")


#create year column for data frames
df_2021['Year'] = 2021

df_2019['Year'] = 2019

df_2016['Year'] = 2016

df_2013['Year'] = 2013

df_2011['Year'] = 2011

df_2008['Year'] = 2008

df_2006['Year'] = 2006

df_2004['Year'] = 2004


# List of your DataFrames
dataframes = [
    df_2004,
    df_2006,
    df_2008,
    df_2011,
    df_2013,
    df_2016,
    df_2019,
    df_2021
]

# Step 1: Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Reorder columns to match original structure

#interpolated_df = combined_df.sort_values(['.geo', 'Year']).reset_index(drop=True)

# (Optional) Display the first few rows of the final DataFrame


combined_df.to_csv('../../../data/final-model-data/Impervious_Data/impervious_data.csv', index=False)
print('the file has been outputted')
