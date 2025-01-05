from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
import geopandas as gpd
from shapely.geometry import Point

# Load data
data = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data.csv')

# Filter data for the year 2021
data_2021 = data[data['year'] == 2021]

# Handle sentinel values
data_2021['Median_Housing_Value'] = data_2021['Median_Housing_Value'].replace(-666666666, np.nan)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data_2021, geometry=gpd.points_from_xy(data_2021['longitude'], data_2021['latitude']))

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
plot = gdf[gdf['Median_Housing_Value'].notna()].plot(
    column='Median_Housing_Value', cmap='RdYlBu', legend=True, ax=ax,
    legend_kwds={
        'orientation': 'horizontal', 'pad': 0.2, 'shrink': 0.8
    }
)

# Set colorbar label font size
cbar = plot.get_figure().get_axes()[1]
cbar.set_title('Median Housing Value (in millions of dollars)', fontsize=14)

# Plot null values with label
gdf[gdf['Median_Housing_Value'].isna()].plot(
    color='#FFA07A', ax=ax, label='No Data'
)
ax.legend(loc='lower left', fontsize=12)

ax.set_title('Median Housing Value in San Bernardino County (2021)', fontsize=16)
ax.set_xlabel('Longitude (degrees)', fontsize=14, labelpad=10)
ax.set_ylabel('Latitude (degrees)', fontsize=14, labelpad=10)
fig.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
