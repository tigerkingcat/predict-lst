import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data.csv')

# Remove rows with placeholder values indicating missing data
df = df[~df.isin([-666666666]).any(axis=1)]

# Display information about the dataset
print(df.info())

# Drop irrelevant columns
df = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code'])
df.drop(columns=['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'GRIDCODE', 'latitude', 'longitude', 'State', 'County', 'Tract', 'Block Group'], inplace=True)

# Drop all categorical (non-numerical) variables
df = df.select_dtypes(include=['number'])

# Drop Population_Below_Poverty column
df = df.drop(columns=['Population_Below_Poverty'])

# Rename columns to remove underscores and spell out acronyms
df.rename(columns={
    'LST_Celsius': 'Land Surface Temperature (Celsius)',
    'Median_Household_Income': 'Median Household Income',
    'High_School_Diploma_25plus': 'High School Diploma (25+)',
    'Bachelors_Degree_25plus': "Bachelor's Degree (25+)",
    'Unemployment': 'Unemployment Rate',
    'Median_Housing_Value': 'Median Housing Value',
    'Median_Gross_Rent': 'Median Gross Rent',
    'Renter_Occupied_Housing_Units': 'Renter-Occupied Housing Units',
    'Total_Population': 'Total Population',
    'Median_Age': 'Median Age',
    'Per_Capita_Income': 'Per Capita Income',
    'Families_Below_Poverty': 'Families Below Poverty',
    'year': 'Year',
    'impervious': 'Impervious Surface Area',
    'Pv': 'Proportional Vegetation',
    'NDWI': 'Normalized Difference Water Index',
    'elev': 'Elevation',
    'GEOID': 'Geographic ID'
}, inplace=True)

# Generate a correlation matrix
correlation_corr = df.corr()

# Plot the heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(12, 8))  # Reduce the width while keeping the height the same
sns.heatmap(correlation_corr, annot=True, cmap='coolwarm', fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8},
            square=False, ax=ax)  # Adjusted square=False to align axes
ax.set_box_aspect(0.5)  # Adjust the aspect ratio (smaller height)
plt.title('Correlation Matrix', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8, rotation=0)
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)  # Adjust margins to fit labels
plt.show()
