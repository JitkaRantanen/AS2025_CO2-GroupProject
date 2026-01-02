import pandas as pd

def read_files(file_paths: list, drop_columns: list, date_column: str, UTC=True):

    # read in the files and set year in file name as new column
    dfs = [
        pd.read_csv(file, low_memory=False).assign(Year=int(file[-8:-4])).drop(columns=drop_columns)
        for file in file_paths
    ]

    # combine the files to one file
    df = pd.concat(dfs, ignore_index=True)

    # get correct data type for datetime column
    df['Datetime'] = pd.to_datetime(df[date_column])

    # the datetime column in the pollutant file is in UTC
    # in the others in CET/CEST
    if UTC==True:
        df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Zurich')
    else:
        # keep time column as it is, just relabel as Europe/Zurich
        df['Datetime'] = pd.to_datetime(df[date_column]).dt.tz_localize(None)

    return df