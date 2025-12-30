import pandas as pd

def read_files(file_paths: list, drop_columns: list, date_column: str, UTC=True):
    dfs = [
        pd.read_csv(file).assign(Year=int(file[-8:-4])).drop(columns=drop_columns)
        for file in file_paths
    ]

    df = pd.concat(dfs, ignore_index=True)

    # get correct time zone representation
    df['Datetime'] = pd.to_datetime(df[date_column])

    if UTC==True:
        df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Zurich')
    else:
        # keep clock time exactly as-is, just relabel as Europe/Zurich
        df['Datetime'] = pd.to_datetime(df[date_column]).dt.tz_localize(None)

    return df