import pandas as pd

LST_DF= pd.read_csv("../../../data/trials/lst_data_and_block_group.csv")

IMP_DF = pd.read_csv("../../../data/trials/block_group_imperviousness.csv")

MERGE_DF = pd.merge(LST_DF,IMP_DF,how='left',on='GEOID')

MERGE_DF.to_csv('../../../data/trials/lst_impervious_blockgroup.csv', index=False)