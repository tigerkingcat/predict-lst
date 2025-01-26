import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'../../data/final-model-data/merged_environmental_census_data_for_prediction.csv')

f = df[~df.isin([-666666666]).any(axis=1)]
print(df.info())

# correlation matricies
df = df.drop(columns=['GEOIDFQ', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'koppen_code'])
df.drop(columns=['ALAND', 'AWATER', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE',
                 'GRIDCODE', 'latitude', 'longitude', 'State', 'County', 'Tract', 'Block Group'], inplace=True)

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
df.drop(columns=df.select_dtypes(include=['object']).columns)

coefficients = {
    "year_centered": 0.22018911157755153,
    "impervious": -0.0671800359290599,
    "Pv": -61.36372787445643,
    "NDWI": -64.20469141898867,
    "elev": -0.00749517382804331,
    "Median_Household_Income": -3.106696253510582e-05,
    "High_School_Diploma_25plus": 0.0009470296842158684,
    "Unemployment": 0.007681715037090699,
    "Median_Housing_Value": 0.0000014953776090951859,
    "Median_Gross_Rent": 0.00035081820599695715,
    "Renter_Occupied_Housing_Units": -0.0009913633101654826,
    "Total_Population": -0.0004289942315471142,
    "Median_Age": -0.03902630119817395,
    "Per_Capita_Income": -0.000011522360793109294,
    "Families_Below_Poverty": 0.0011222404370434984,
    "climate_category_Arid (Cold)": 0.16196727462733734,
    "climate_category_Arid (Hot)": -0.1653116376478015,
    "climate_category_Mediterranean": 0.08118353287506347,
    "climate_category_Semi-Arid (Cold)": -0.07783916985452838,
    "urban_rural_classification_U": -0.22592498729949587,
    "urban_rural_classification_nan": 0.22592498729900173,
}
df_encoded.drop(columns='Population_Below_Poverty', inplace=True)
std_devs = df_encoded.std().to_dict()
print(std_devs)
for coefficent in coefficients:
    print(coefficent, abs(coefficients[coefficent]) * std_devs[coefficent])
    print(coefficent, coefficients[coefficent] * (std_devs[coefficent]/4.635272069320759))
    print("-----")
