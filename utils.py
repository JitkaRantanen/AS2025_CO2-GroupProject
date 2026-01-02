import pandas as pd

def read_files(file_paths: list, drop_columns: list, date_column: str, UTC=True):
    """
    Read multiple CSV files, clean them, parse datetime, and convert to UTC.

    This function:
    - Reads multiple CSV files and extracts the year from the file name.
    - Drops unwanted columns.
    - Combines all files into a single DataFrame.
    - Parses a datetime column and converts it to timezone-aware UTC.
    - Handles daylight saving time for traffic datasets (CET/CEST).
    - Renames the 'Standort' column to 'Location'.

    Args:
        file_paths (list): List of file paths to read.
        drop_columns (list): List of columns to drop from each CSV file.
        date_column (str): Name of the column containing datetime information.
        UTC (bool, optional): If True, treats the datetime column as fixed UTC+1 (air pollution data).
                              If False, treats it as CET/CEST (traffic data with DST). Default is True.

    Returns:
        pd.DataFrame: Combined and cleaned DataFrame with:
            - 'Year' column extracted from file names
            - 'Datetime' column as timezone-aware UTC
            - 'Location' column (renamed from 'Standort')
            - Original data columns minus dropped columns
    """

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
                          .dt.tz_convert("UTC"))

    # the traffic data is in CET/CEST. therefore, we have to handle the missing and the duplicated hour at the end
    # of march and october, respectively (daylight saving time). this is done be the parameters in nonexistent and ambiguous.
    # to make things easier, we convert it to UTC as well.
    else:
        df['Datetime'] = (df['Datetime']
                          .dt.tz_localize("Europe/Zurich", nonexistent="shift_forward", ambiguous="infer")
                          .dt.tz_convert("UTC"))

    df = df.rename(columns={"Standort": "Location"})

    return df


def float_to_int(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert one or more float columns to nullable integers in-place."""

    df[columns] = df[columns].astype("Int64")

    return df