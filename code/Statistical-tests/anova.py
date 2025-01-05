# File: use_model.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from scipy.stats import ttest_rel

# Load the saved models
model_env = RandomForestRegressor()
model_combined = RandomForestRegressor()

# 1. Load the data
df = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data.csv')
df = df[~df.isin([-666666666]).any(axis=1)]
print(df.info())
# correlation matricies
df = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code'])
df.drop(columns=['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'GRIDCODE', 'latitude', 'longitude', 'State', 'County', 'Tract', 'Block Group'], inplace=True)

sns.scatterplot(data=df, x='LST_Celsius', y='Median_Household_Income')

# 2. Create Lag Variables for ISA and NDWI
# Create lagged variables for ISA and NDWI to capture how past values of impervious surfaces and vegetation affect current LST (e.g., use a 1-year lag).
# Assuming your dataset is a pandas DataFrame called 'df'
# Ensure that your data is sorted by time (Year) before creating lag variables.
# df = df.sort_values(by='year').reset_index(drop=True)

# Create a 1-year lag for ISA
# df['ISA_lag1'] = df['impervious'].shift(1)

# Create a 1-year lag for NDWI
# df['NDWI_lag1'] = df['NDWI'].shift(1)

# Create a 1-year lag for Pv
# df['Pv_lag1'] = df['Pv'].shift(1)

# Note: The shift(1) function moves the values of the column down by one row, effectively creating a lag of 1 year.

# Drop any rows with NaN values that are created due to lagging
# df = df.dropna(subset=['ISA_lag1', 'NDWI_lag1'])

# 3. Create the Year-Centered Variable to Capture Temporal Trends
# Center the year variable by subtracting the mean year from each year value. This helps capture long-term trends and reduces multicollinearity.
mean_year = df['year'].mean()
print(mean_year)
df['year_centered'] = df['year'].apply(lambda year: year - mean_year)
print(df['year_centered'])

# 4. Convert Climate Zone and Urban/Rural Classification to Categorical Variables
# Convert Köppen Climate Classification and Urban/Rural classification into categorical variables using one-hot encoding.
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
df_encoded = df_encoded.dropna().reset_index(drop=True)

# Feature Selection using RFECV (Recursive Feature Elimination with Cross-Validation)
X_env = df_encoded[['year_centered', 'impervious', 'Pv', 'NDWI', 'elev', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)',
       'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)',
       'urban_rural_classification_U', 'urban_rural_classification_nan']]
X_combined = df_encoded[['year_centered', 'impervious', 'Pv', 'NDWI', 'elev',
        'Median_Household_Income', 'High_School_Diploma_25plus', 'Unemployment', 'Median_Housing_Value', 'Median_Gross_Rent', 'Renter_Occupied_Housing_Units', 'Total_Population', 'Median_Age', 'Per_Capita_Income', 'Families_Below_Poverty',
       'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)',
       'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)',
       'urban_rural_classification_U', 'urban_rural_classification_nan']]
y = df_encoded['LST_Celsius']

# Split into training and testing sets using the selected features
X_train_env, X_test_env, y_train, y_test = train_test_split(X_env, y, test_size=0.2, random_state=42)
X_train_combined, X_test_combined, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train the RandomForest models
model_env.fit(X_train_env, y_train)
model_combined.fit(X_train_combined, y_train)

# Test Models
# Predict using the trained models
y_pred_env = model_env.predict(X_test_env)
y_pred_combined = model_combined.predict(X_test_combined)

# Calculate metrics for both models
r2_env = r2_score(y_test, y_pred_env)
r2_combined = r2_score(y_test, y_pred_combined)
rmse_env = np.sqrt(mean_squared_error(y_test, y_pred_env))
rmse_combined = np.sqrt(mean_squared_error(y_test, y_pred_combined))

print(f"Model 1 (Environmental) - R²: {r2_env:.2f}, RMSE: {rmse_env:.2f}")
print(f"Model 3 (Combined) - R²: {r2_combined:.2f}, RMSE: {rmse_combined:.2f}")

# Paired t-test for residuals
residuals_env = y_test - y_pred_env
residuals_combined = y_test - y_pred_combined
t_stat, p_value = ttest_rel(residuals_env, residuals_combined)

print(f"Paired t-test statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.2e}")

# Interpretation
if p_value < 0.05:
    print("There is a significant difference between the models' performance (p < 0.05).")
    if np.mean(residuals_combined) < np.mean(residuals_env):
        print("Model 3 (Combined) has significantly lower residuals, indicating better performance.")
    else:
        print("Model 1 (Environmental) has significantly lower residuals, indicating better performance.")
else:
    print("There is no significant difference between the models' performance (p >= 0.05).")
