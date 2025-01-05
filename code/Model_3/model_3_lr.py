# this model has improved the envrionemtnal variabkes to a record r2 score of 0.96189
# here is the copy pasted result
# Best R² score: 0.9618901703519779
# Best feature set: ['index', 'impervious', 'Pv', 'NDWI', 'elev', 'GEOID', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)', 'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_U', 'urban_rural_classification_nan']
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
X = df_encoded[['year_centered', 'impervious', 'Pv', 'NDWI', 'elev',
        'Median_Household_Income', 'High_School_Diploma_25plus', 'Unemployment', 'Median_Housing_Value', 'Median_Gross_Rent', 'Renter_Occupied_Housing_Units', 'Total_Population', 'Median_Age', 'Per_Capita_Income', 'Families_Below_Poverty',
       'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)',
       'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)',
       'urban_rural_classification_U', 'urban_rural_classification_nan']]
y = df_encoded['LST_Celsius']

# # Use Ridge regression as the estimator
# estimator = Ridge()
# rfecv = RFECV(estimator, step=1, cv=5, scoring='r2')
# rfecv.fit(X, y)
#
# # Get the optimal features
# best_features = X.columns[rfecv.support_]
# print("Optimal number of features: %d" % rfecv.n_features_)
# print("Best feature set:", best_features)

# Split into training and testing sets using the selected features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Optional: Check explained variance
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 10. Visualize Actual vs Predicted Values (Optional)
# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Data Points')
plt.xlabel('Actual LST (Celsius)', fontsize=14)
plt.ylabel('Predicted LST (Celsius)', fontsize=14)
plt.title('Actual vs Predicted LST (Linear Regression Model)', fontsize=16)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.legend(fontsize=12)
plt.show()

# Histogram of Deviance (Residuals) vs Predicted Values
residuals = np.array(y_test) - np.array(y_pred)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, alpha=0.7, label='Residual Distribution')
plt.xlabel('Residuals (Actual - Predicted)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Residuals (Linear Regression Model)', fontsize=16)
plt.legend(fontsize=12)
plt.show()

# Print Coefficients
coefficients = model.coef_
for feature, coef in zip(X.columns, coefficients):
    print(f"Coefficient for {feature}: {coef}")


