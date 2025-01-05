import pandas as pd
import json
import glob
import os


def extract_coordinates(geojson_str):
    try:
        geojson_cleaned = geojson_str.replace('""', '"').strip('"')
        geojson = json.loads(geojson_cleaned)
        coordinates = geojson.get('coordinates', [0, 0])
        longitude, latitude = coordinates if len(coordinates) >= 2 else (0, 0)
        return longitude, latitude
    except json.JSONDecodeError:
        return 0, 0


def main():
    input_directory = '../../../data/final-model-data/Impervious_Data'  # Current directory
    output_file = '../../../data/step1-pre-process/Impervious_Data/combined_sorted.csv'
    file_pattern = os.path.join(input_directory, 'Impervious_Surface_SanBernardino_1km_*.csv')
    csv_files = glob.glob(file_pattern)
    column_names = ['system:index', 'impervious', '.geo']
    df_list = []

    for file in csv_files:
        filename = os.path.basename(file)
        year = filename.split('_')[-1].split('.')[0]
        df = pd.read_csv(file, header=None, names=column_names, skiprows=1)
        df[['longitude', 'latitude']] = df['.geo'].apply(lambda x: pd.Series(extract_coordinates(x)))
        df['year'] = year
        df = df.drop(columns=['.geo'])
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    sorted_df = combined_df.sort_values(by=['longitude', 'latitude']).reset_index(drop=True)
    sorted_df = sorted_df[['system:index', 'impervious', 'year', 'latitude', 'longitude']]
    sorted_df.to_csv(output_file, index=False, quoting=1)


if __name__ == "__main__":
    main()
