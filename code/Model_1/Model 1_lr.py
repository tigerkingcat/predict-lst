import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

#1.  Load the data
df = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data.csv')
df = df[~df.isin([-666666666]).any(axis=1)]
print(df.info())
    #correlation matricies
    #correlation = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code', 'climate_category', 'urban_rural_classification'])

    #correlation_corr = correlation.corr()

    #sns.heatmap(correlation_corr, annot=True)
    #plt.show()
    #print(correlation_corr[(correlation_corr>0.3) | (correlation_corr<-0.3)])
df.drop(columns=['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON', 'MTFCC', 'NAMELSAD', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'GRIDCODE', 'koppen_code', 'GEOIDFQ', 'FUNCSTAT'], inplace=True)
df['urban_rural_classification'].replace(to_replace=pd.NA, value='R', inplace=True)

# 2. Create Lag Variables for ISA and NDWI
# Create lagged variables for ISA and NDWI to capture how past values of impervious surfaces and vegetation affect current LST (e.g., use a 1-year lag).
# Assuming your dataset is a pandas DataFrame called 'df'
# Ensure that your data is sorted by time (Year) before creating lag variables.
#df = df.sort_values(by='year').reset_index(drop=True)

# Create a 1-year lag for ISA
#df['ISA_lag1'] = df['impervious'].shift(1)

# Create a 1-year lag for NDWI
#df['NDWI_lag1'] = df['NDWI'].shift(1)

# Create a 1-year lag for Pv
#df['Pv_lag1'] = df['Pv'].shift(1)

# Note: The shift(1) function moves the values of the column down by one row, effectively creating a lag of 1 year.

# Drop any rows with NaN values that are created due to lagging





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
print('THIS IS THE DF')
print(df_encoded.head(5))
df_encoded = df_encoded.dropna(axis=0)
print('THIS IS THE DF -1 ')
print(df_encoded)
X = df_encoded[['impervious', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)', 'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_R', 'urban_rural_classification_U', 'NDWI', 'Pv', 'year_centered', 'elev']]
y = df_encoded['LST_Celsius']

# 6. Perform Cross-Validation Using TimeSeriesSplit
# Use TimeSeriesSplit for cross-validation, ensuring the model is trained on earlier years and tested on future years. No separate train-test split is needed.
# n_splits = 2
# tscv = TimeSeriesSplit(n_splits=n_splits)
# 7. Fit the Regression Model
# # Train the regression model (e.g., linear regression, random forest, or gradient boosting) on the training data for each fold, using Pv, NDWI, ISA, DEM, Köppen Climate Classification, Urban/Rural classification, lagged variables, and the year-centered variable.
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)
model = LinearRegression()
# mse_scores = []
# rmse_scores = []
# r2_scores = []
df_encoded = df_encoded.sort_values(by='year').reset_index(drop=True)
print(df_encoded.columns)
df_encoded = df_encoded.dropna().reset_index(drop=True)
# Drop NaN values from X and y


print(X)



# 6. Split Data into Training and Testing Sets
# Use train_test_split to create training and testing datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 7. Fit the Regression Model
# Train the regression model (e.g., linear regression, random forest, or gradient boosting) on the training data.
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Make Predictions on the Test Set
# Use the trained model to make predictions on the test set.
y_pred = model.predict(X_test)

# 9. Evaluate Model Performance
# Evaluate the model performance using metrics like R², RMSE, and MAE.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 10. Visualize Actual vs Predicted Values (Optional)
# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel('Actual LST (Celsius)')
plt.ylabel('Predicted LST (Celsius)')
plt.title('Actual vs Predicted LST')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

# Histogram of Deviance (Residuals) vs Predicted Values
residuals = np.array(y_test) - np.array(y_pred)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, alpha=0.7)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Bar plot of Coefficients
coefficients = model.coef_
coeff_mean = np.mean(coefficients)
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=coefficients)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance (Coefficients)')
plt.xticks(rotation=90)
plt.show()

joblib.dump(model, 'trained_model.pkl')

print("Model saved successfully.")