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

    # in the air pollution data set the datetime column is in UTC+1 for the whole year
    # to make things easier, we convert it to UTC
    if UTC==True:
        df['Datetime'] = (df['Datetime']
                          .dt.tz_localize("Etc/GMT-1")
                          .dt.tz_convert("UTC"))

    # the traffic data is in CET/CEST. therefore, we have to handle the missing and the duplicated hour at the end
    # of march and october, respectively (daylight saving time). this is done be the parameters in nonexistent and ambiguous.
    # to make things easier, we convert it to UTC as well.
    else:
        df['Datetime'] = (df['Datetime']
                          .dt.tz_localize("Europe/Zurich", nonexistent="shift_forward", ambiguous="infer")
                          .dt.tz_convert("UTC"))

    return df