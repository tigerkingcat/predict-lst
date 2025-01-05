from census import Census
import pandas as pd

# Your API Key here
census_api_key = '67dee7eadd6439e09e48355c965afd9ee4085828'
c = Census(census_api_key)

# Define variables
variables = [
    "B19013_001E",  # Median household income
    "B17001_002E",  # Population below poverty level
    "B15003_017E",  # High school diploma (25+)
    "B15003_022E",  # Bachelor's degree or higher (25+)
    "B23025_005E",  # Unemployed
    "B25077_001E",  # Median housing value
    "B25064_001E",  # Median gross rent
    "B25003_003E",  # Renter-occupied housing units
    "B01003_001E",  # Total population
    "B01002_001E",  # Median age
    "B19301_001E",  # Per capita income
    "B17010_002E"  # Families with children below poverty level
]

# San Bernardino County (FIPS: 06 for California, 071 for San Bernardino)
state_fips = '06'
county_fips = '071'
blockgroup_fips = '*'  # Use '*' to get data for all block groups

# Years to retrieve data for
years = [2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

# Loop through each year and fetch data
for year in years:
    try:
        # Fetch data for each block group in the county for the given year
        data = c.acs5.state_county_blockgroup(
            fields=variables,
            state_fips=state_fips,
            county_fips=county_fips,
            blockgroup=blockgroup_fips,
            year=year
        )

        # Convert to DataFrame and set column names for readability
        df = pd.DataFrame(data)
        df.columns = [
            'Median_Household_Income',
            'Population_Below_Poverty', 'High_School_Diploma_25plus', 'Bachelors_Degree_25plus',
            'Unemployment', 'Median_Housing_Value', 'Median_Gross_Rent', 'Renter_Occupied_Housing_Units',
            'Total_Population', 'Median_Age', 'Per_Capita_Income', 'Families_Below_Poverty', 'State', 'County', 'Tract', 'Block Group'
        ]


        # Display or save the data
        print(f"Data for year {year}:")
        print(df.head())

        # Save to CSV with the year in the file name
        df.to_csv(f"san_bernardino_blockgroups_socioeconomic_data_{year}.csv", index=False)
        print(f"Data for year {year} saved successfully.\n")

    except Exception as e:
        print(f"Failed to retrieve data for year {year}. Error: {e}")