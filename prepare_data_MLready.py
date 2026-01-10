import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer



# Import and Select the data
df = pd.read_csv("data/combined_h1_2018_2025_excl_nan.csv")
df.head()

# Re-arrange columns
cols = ["Datetime", "YearMonth", "Date", "Year", "Month", "Month_str", "Weekday", "Weekday_str", "Day", "Time",
        "Location",
        "NO2", "O3", "PM10",
        "Hr", "RainDur", "T", "WVv", "StrGlo",
        "meanVehicles"]

df = df[cols]
df.head()

# Select columns
cols_selected = ["Year", "Month", "Weekday", "Time", "Day",
                 "Location",
                 "NO2", "O3", "PM10",
                 "Hr", "RainDur", "T", "WVv", "StrGlo",
                 "meanVehicles"]

df_selected = df[cols_selected]

# Make sure the time features are numbers, not strings
df_ready = df_selected.copy()
df_ready["Time"] = pd.to_datetime(df_ready["Time"], format="%H:%M:%S").dt.hour.astype("int64")
df_ready["Month"] = df_selected["Month"].astype(int)
df_ready["Day"] = df_selected["Day"].astype(int)
df_ready["Weekday"] = df_selected["Weekday"].astype(int)

df_ready.head()



# Prepare transformer - to prepare data for ML
# Preprocessing - OneHotEncoder, StandardScaler, Trigonometric encoding and SplineTransformer
# Prepare data transformer to be used in pipeline - do not use on the whole dataset!
# Encodings must be computed only on train data and then only applied on test data
# Trigonometric encoding could be in theory be done on the whole dataset, but for simplicity it is combined to have only one transformer

# Prepare Trigonometric encoding for time (safe before split)
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# Prepare Spline Transformer
def periodic_spline_transformer(period, n_splines):
    degree = 3
    n_knots = n_splines + 1  # since include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

# Prepare columns to be used
numeric_variables = ["Hr", "RainDur", "T", "WVv", "StrGlo", "meanVehicles"]

preprocess = ColumnTransformer(
    transformers=[
        ("Month_sin", sin_transformer(12), ["Month"]),
        ("Month_cos", cos_transformer(12), ["Month"]),
        ("Day_sin", sin_transformer(7), ["Day"]),
        ("Day_cos", cos_transformer(7), ["Day"]),
        ("Weekday_sin", sin_transformer(7), ["Weekday"]),
        ("Weekday_cos", cos_transformer(7), ["Weekday"]),
        ("Time_sin", sin_transformer(24), ["Time"]),
        ("Time_cos", cos_transformer(24), ["Time"]),
        ("Spline_month", periodic_spline_transformer(12, n_splines=5), ["Month"]),
        ("Spline_day", periodic_spline_transformer(7, n_splines=6), ["Day"]),
        ("Spline_weekday", periodic_spline_transformer(7, n_splines=6), ["Weekday"]),
        ("Spline_time", periodic_spline_transformer(24, n_splines=24), ["Time"]),
        ("encoded", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Location"]),
        ("scaled", StandardScaler(), numeric_variables),
    ],
    remainder="passthrough"
).set_output(transform="pandas")



# Split for Train and Test - fit and transform the Preprocessing transformer
# Split manually based on year (to make sure the whole year is always included)
X_df = df_ready.drop(columns=["NO2", "O3", "PM10"]).copy()
y_df = df_ready[["NO2", "O3", "PM10"]].copy()

# Training datasets contain years 2018-2024
X_train = X_df[X_df["Year"].between(2018, 2024)].copy()
X_test = X_df[X_df["Year"] == 2025].copy()
# Testing datasets contain year 2025
y_train = y_df[X_df["Year"].between(2018, 2024)].copy()
y_test = y_df[X_df["Year"] == 2025].copy()

preprocess.fit(X_train)
X_train_transformed = preprocess.transform(X_train)
X_test_transformed = preprocess.transform(X_test)


#X_train_transformed.to_csv("data/Datasets_MLready/X_train_transformed.csv", index=False)
#X_test_transformed.to_csv("data/Datasets_MLready/X_test_transformed.csv", index=False)
#y_train.to_csv("data/Datasets_MLready/y_train.csv", index=False)
#y_test.to_csv("data/Datasets_MLready/y_test.csv", index=False)