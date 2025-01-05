import pandas as pd

df1 = pd.read_csv(
    '../../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_2013.csv')
df2 = pd.read_csv(
    '../../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_2016.csv')
df3 = pd.read_csv(
    '../../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_2019.csv')
df4 = pd.read_csv(
    '../../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_2021.csv')

df1['year'] = 2013
df2['year'] = 2016
df3['year'] = 2019
df4['year'] = 2021

dfs = [df1, df2, df3, df4]

concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df.to_csv('../data/step1-pre-process/SB_Blockgroup_Socioeconomic/san_bernardino_blockgroups_socioeconomic_data_combined.csv', index=False)
