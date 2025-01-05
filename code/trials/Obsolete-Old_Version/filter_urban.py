import pandas as pd

# load df from file
df = pd.read_csv("../../../data/trials/data.csv")

# make sure all are urban and reset the index
df = df[df['PercentageUrban'] >= 100]
df.reset_index(inplace=True)
df.drop(axis=1, columns='index', inplace=True)

#export to a new csv
df.to_csv("../../../data/trials/data.csv", index=False)