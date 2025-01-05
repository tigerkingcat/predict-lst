import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl




df = pd.read_csv('/data/data.csv')
print(df.info())
sns.scatterplot(data=df, x='impervious', y='LST_Celsius')
df.drop(axis=1, columns='Geographic Area Name', inplace=True)
df.drop(axis=1, columns='Unnamed: 0', inplace=True)
#test = skl.MinMaxScaler()
#test.fit(df)
#sns.heatmap(df.corr())
#print(test.corr())
plt.show()
# use k-fold model potentially