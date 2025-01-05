import pandas as pd

df = pd.read_csv("C:\\Users\\aarav\\ScienceProject24-25\\CensusDatas\\DECENNIALDHC2020.P2-Data.csv")
print(df.info())
df.columns = df.iloc[0]
print(df)

df = df[1:]
print(df)

df.reset_index(drop=True, inplace=True)
print(df.info())
df.rename(columns={'Geography': 'GEOID', ' !!Total:': 'Total', ' !!Total:!!Urban': 'Total_Urban',
                   ' !!Total:!!Rural': 'Total_Rural'}, inplace=True)

df.drop(' !!Total:!!Not defined for this file', axis=1, inplace=True)

df['GEOID'] = df['GEOID'].apply(lambda id: id[10:])

print(df)

df.to_csv('Rural_or_Urban.csv', index=False)
