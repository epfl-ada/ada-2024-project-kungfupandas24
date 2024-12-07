import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

def calculate_roi(df, budget_col, revenue_col):
    condition = (df[budget_col].notna() & df[budget_col].ne(0) &
                 df[revenue_col].notna() & df[revenue_col].ne(0))
    df_filtered = df[condition].copy()
    df_filtered["ROI"] = ((df_filtered[revenue_col] - df_filtered[budget_col]) / df_filtered[budget_col])
    return df_filtered

def scale_features(df):
    # Scale the ratings between 0 and 1
    df["Normalized_Rating"] = df["Average_ratings"].astype(float) / 10

    # Calculate the Log of ROI to obtain normalized data distribution
    df["Log_ROI"] = df["ROI"].apply(math.log1p)

    # Scale the logged ROI between 0 and 1
    df["Log_ROI"] = MinMaxScaler().fit_transform(df[["Log_ROI"]])

    return df

def calculate_weighted_success(df, roi_weight):
    df["Movie_success"] = df["Log_ROI"] * roi_weight + df["Normalized_Rating"] * (1 - roi_weight)
    df["Movie_success"] = MinMaxScaler().fit_transform(df[["Movie_success"]])

    return df

def log_transform(df, columns):
    for col in columns:
        df[f'log_{col}'] = df[col].apply(np.log1p)
    return df
