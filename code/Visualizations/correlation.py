import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../../data/step2-aggregate/Final_sbcounty_environ_data.csv')
print(df.info())
df = df[['year', 'LST_Celsius', 'impervious', 'Pv', 'NDWI', 'elev']]

#correlation = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code', 'climate_category', 'urban_rural_classification'])

correlation_corr = df.corr()

sns.scatterplot(data=df, x='LST_Celsius', y='impervious')
plt.show()
