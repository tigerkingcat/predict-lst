# this model has improved the envrionemtnal variabkes to a record r2 score of 0.96189
# Best feature set: ['index', 'impervious', 'Pv', 'NDWI', 'elev', 'GEOID', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)', 'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_U', 'urban_rural_classification_nan']


import pandas as pd
import pickle
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

# 1.  Load the data
df = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data_for_prediction.csv')
df = df[~df.isin([-666666666]).any(axis=1)]
print(df.info())

# correlation matricies
df = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code'])
df.drop(columns=['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'GRIDCODE',
                 ], inplace=True)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
correlation = df.drop(columns=categorical_columns)
correlation_corr = correlation.corr()

mean_year = df['year'].mean()
df['year_centered'] = df['year'].apply(lambda year: year - mean_year)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
print(df_encoded.info())
print(df_encoded.columns)

# 5. Prepare the Data for Modeling
# Ensure all variables are in the appropriate format (no need to manually create a feature matrix). Your dataset should include all environmental variables, lag variables, and the year-centered variable.
df_encoded.drop(columns='Population_Below_Poverty', inplace=True)
df_encoded['Median_Housing_Value'] = df_encoded['Median_Housing_Value'] * 0.80

# df_encoded['Pv'] = df_encoded['Pv'] * 1.20



df_encoded = df_encoded.dropna().reset_index(drop=True)

input_data = df_encoded[
    ['impervious', 'Pv', 'NDWI', 'elev', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)',
     'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_U',
     'urban_rural_classification_nan', 'Median_Household_Income', 'High_School_Diploma_25plus',
     'Unemployment', 'Median_Housing_Value', 'Median_Gross_Rent', 'Renter_Occupied_Housing_Units', 'Total_Population',
     'Median_Age', 'Per_Capita_Income', 'Families_Below_Poverty',
     'year_centered']]

with open('model3_rf_trained_model_1.pkl', 'rb') as file:
    model = pickle.load(file)

predictions = model.predict(input_data)

df_encoded['predictions'] = predictions
df_encoded['difference'] = df_encoded['LST_Celsius'] - df_encoded['predictions']

# 7. Save or print the results
df_encoded.to_csv('predictions_output_median_housing_val_decreased_by_20pct.csv', index=False)

# 6. Add predictions to the DataFrame (optional)
print(predictions)

# 7. Save or print the results
# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df_encoded, geometry=gpd.points_from_xy(df_encoded['longitude'], df_encoded['latitude']))

# Plot Actual and Predicted LST with Shared Colorbar
fig, axes = plt.subplots(1, 1, figsize=(20, 10))
#actual_plot = gdf.plot(column='LST_Celsius', cmap='coolwarm', legend=False, ax=axes[0])
predicted_plot = gdf.plot(column='difference', cmap='coolwarm', legend=True, ax=axes)



# Create a single colorbar for both plots
#sm = plt.cm.ScalarMappable(cmap='coolwarm',
                           #norm=plt.Normalize(vmin=min(gdf['LST_Celsius'].min(), gdf['difference'].min()),
                                              #vmax=max(gdf['LST_Celsius'].max(), gdf['difference'].max())))
#sm._A = []  # Dummy array for ScalarMappable
#cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.5, pad=0.1)
#cbar.set_label("Land Surface Temperature (°C)")

#axes[0].set_title('Actual LST in SB County', loc='center')
axes.set_title('Difference between Actual and Precited LST', loc='center')
axes.set_xlabel('Longitude(degrees)')
axes.set_ylabel('Latitude(degrees)')

plt.show()
