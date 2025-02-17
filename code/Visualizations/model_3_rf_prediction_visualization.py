# this model has improved the envrionemtnal variabkes to a record r2 score of 0.96189
# Best feature set: ['index', 'impervious', 'Pv', 'NDWI', 'elev', 'GEOID', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)', 'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_U', 'urban_rural_classification_nan']


import pandas as pd
import pickle
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

# 1.  Load the data
df_prediction = pd.read_csv(r'../../data/final-model-data/predictions_output.csv')
df_prediction_with_median_housing_val_decreased = pd.read_csv(
    r'../../data/final-model-data/predictions_output_median_housing_val_decreased_by_20pct.csv')
df_prediction['New_Housing_Value_LST'] = df_prediction_with_median_housing_val_decreased['predictions']
df_prediction['difference'] = df_prediction['New_Housing_Value_LST']-df_prediction['predictions']

# 7. Save or print the results
# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df_prediction, geometry=gpd.points_from_xy(df_prediction['longitude'], df_prediction['latitude']))

# Plot Actual and Predicted LST with Shared Colorbar
fig, axes = plt.subplots(1, 1, figsize=(20, 10))
# actual_plot = gdf.plot(column='LST_Celsius', cmap='coolwarm', legend=False, ax=axes[0])
predicted_plot = gdf.plot(column='difference', cmap='coolwarm', legend=True, ax=axes)

# Create a single colorbar for both plots
# sm = plt.cm.ScalarMappable(cmap='coolwarm',
# norm=plt.Normalize(vmin=min(gdf['LST_Celsius'].min(), gdf['difference'].min()),
# vmax=max(gdf['LST_Celsius'].max(), gdf['difference'].max())))
# sm._A = []  # Dummy array for ScalarMappable
# cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.5, pad=0.1)
# cbar.set_label("Land Surface Temperature (°C)")

# axes[0].set_title('Actual LST in SB County', loc='center')
axes.set_title('Difference between Actual and Precited LST', loc='center')
axes.set_xlabel('Longitude(degrees)')
axes.set_ylabel('Latitude(degrees)')

plt.show()
