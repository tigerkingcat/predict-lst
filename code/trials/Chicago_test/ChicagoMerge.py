import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Load the data
df = pd.read_csv('../../../data/step1-pre-process/Chicago_test/Chicago_LST_Impervious_Surface_1kmGrid.csv')
print(df.info())

# Drop unnecessary column
df.drop(axis=1, columns='.geo', inplace=True)
df = df[df['Impervious_mean'] > 1]

# Correlation analysis
print(df.corr())

# Scatterplot of Impervious Surface vs. Land Surface Temperature
sns.scatterplot(data=df, x='Impervious_mean', y='LST_Celsius')
plt.title('Scatterplot of Impervious Surface vs. LST')
plt.xlabel('Mean Impervious Surface (%)')
plt.ylabel('LST (Celsius)')
plt.show()

# Linear regression
X = df['Impervious_mean'].to_frame()
y = df['LST_Celsius']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Print intercept and coefficient
print(f"Intercept: {lm.intercept_}")
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Make predictions
predictions = lm.predict(X_test)

# Plot predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual LST (Celsius)')
plt.ylabel('Predicted LST (Celsius)')
plt.title('Actual vs Predicted LST')
plt.show()

# Plot residuals
diff = y_test - predictions
sns.histplot(diff, kde=True, bins=50)
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.show()

# Print error metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Calculate R^2 score
r2_score = lm.score(X_test, y_test)
print(f"R^2 Score: {r2_score}")

# Plot regression line with scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Impervious_mean', y='LST_Celsius', label='Actual Data')

# Generate the regression line points
x_range = np.linspace(df['Impervious_mean'].min(), df['Impervious_mean'].max(), 100)
y_line = lm.intercept_ + lm.coef_[0] * x_range

# Plot the regression line
plt.plot(x_range, y_line, color='red', label='Regression Line')

# Customize the plot
plt.title('Scatterplot with Regression Line')
plt.xlabel('Mean Impervious Surface (%)')
plt.ylabel('LST (Celsius)')
plt.legend()
plt.grid(True)
plt.show()