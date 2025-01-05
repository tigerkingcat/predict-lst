import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

#1.  Load the data
df = pd.read_csv('../../data/step2-aggregate/Final_sbcounty_environ_data.csv')
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
#df = df.dropna(subset=['ISA_lag1', 'NDWI_lag1'])

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

# 6. Determine the Best Number of Time Series Splits
# Test different values for n_splits to find the best performance
n_splits_range = [3, 5, 7, 10]
best_n_splits = None
best_score = float('inf')
param_grid = {'alpha': np.logspace(-4, 1, 10)}

# Normalize the features
scaler = StandardScaler()
df_encoded = df_encoded.sort_values(by='year').reset_index(drop=True)
df_encoded = df_encoded.dropna().reset_index(drop=True)
X = df_encoded[['impervious', 'climate_category_Arid (Cold)', 'climate_category_Arid (Hot)', 'climate_category_Mediterranean', 'climate_category_Semi-Arid (Cold)', 'urban_rural_classification_R', 'urban_rural_classification_U', 'NDWI', 'Pv', 'year_centered', 'elev']]
y = df_encoded['LST_Celsius']
X_scaled = scaler.fit_transform(X)

for n_splits in n_splits_range:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    mean_score = -grid_search.best_score_
    print(f"n_splits: {n_splits}, Mean MSE: {mean_score:.2f}")
    if mean_score < best_score:
        best_score = mean_score
        best_n_splits = n_splits

print(f"Best n_splits: {best_n_splits} with Mean MSE: {best_score:.2f}")

# Use the best n_splits for the final model
tscv = TimeSeriesSplit(n_splits=best_n_splits)

# Hyperparameter tuning using GridSearchCV with the best n_splits
grid_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

# Best model after grid search
best_model = grid_search.best_estimator_
print(f"Best alpha: {grid_search.best_params_['alpha']}")

mse_scores = []
rmse_scores = []
r2_scores = []

# 7. Make Predictions on the Validation Set for Each Split
# For each fold in TimeSeriesSplit, use the trained model to make predictions on the validation set.
for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):

    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"Fold {fold + 1}")
    # Fit the model
    best_model.fit(X_train, y_train)

    # Predict
    y_pred = best_model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    # Calculate RMSE
    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    # 8. Evaluate Model Performance
    # Evaluate the model performance on each fold using metrics like R², RMSE, and MAE.
    print(f"Fold {fold + 1}")
    print(f"MSE: {mse:.2f}\n")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}\n")

# Calculate overall metrics
average_mse = np.mean(mse_scores)
average_rmse = np.mean(rmse_scores)
average_r2 = np.mean(r2_scores)

print(f"Overall Mean Squared Error (MSE): {average_mse:.2f}")
print(f"Overall Root Mean Squared Error (RMSE): {average_rmse:.2f}")
print(f"Overall R² Score: {average_r2:.2f}")
# 9. Visualize Actual vs Predicted Values (Optional)
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
