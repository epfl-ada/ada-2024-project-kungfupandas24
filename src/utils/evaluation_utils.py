import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_roi(df, budget_col, revenue_col):
    condition = (df[budget_col].notna() & df[budget_col].ne(0) &
                 df[revenue_col].notna() & df[revenue_col].ne(0))
    df = df[condition]
    df["ROI"] = ((df[revenue_col] - df[budget_col]) / df[budget_col])
    return df

def scale_features(df):
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    df["Normalized_Rating"] = df["Average_ratings"].astype(float) / 10

    # Standardize the selected data
    df["ROI"] = scaler.fit_transform(df[["ROI"]])

    # Apply Min-Max scaling to the already standardized column
    df["ROI"] = minmax_scaler.fit_transform(df[["ROI"]])

    return df

def calculate_weighted_success(df, roi_weight):
    df["movie_success"] = df["ROI"] * roi_weight + df["Normalized_Rating"] * (1 - roi_weight)
    df["movie_success"] = MinMaxScaler().fit_transform(df[["movie_success"]])

    return df

def log_transform(df, columns):
    for col in columns:
        df[f'log_{col}'] = df[col].apply(np.log1p)
    return df
