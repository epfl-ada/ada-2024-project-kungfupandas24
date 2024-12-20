import pandas as pd
import numpy as np
import ast
import re

def clean_string_columns(df, column_names):
    # Cleans string data in specified columns while checking if lst is a list
    def clean_string_list(lst):
        if isinstance(lst, list):
            return [s.strip().lower() if isinstance(s, str) and s.strip() != "" else np.nan for s in lst]
        elif isinstance(lst, str):
            return lst.strip().lower()
        else:
            return np.nan
    
    for column in column_names:
        df[column] = df[column].apply(clean_string_list)
    
    return df

def convert_dicts_to_strings(df, column_names):
    for column in column_names:
        df[column] = df[column].apply(ast.literal_eval)
        df[column] = df[column].apply(lambda x: ', '.join(x.values()))
    return df

def standardize_dates(df, date_column):
    full_date_pattern = r'^\d{4}-\d{2}-\d{2}$' # Matches YYYY-MM-DD
    year_month_pattern = r'^\d{4}-\d{2}$' # Matches YYYY-MM
    year_only_pattern = r'^\d{4}$' # Matches YYYY

    def identify_pattern(date):
        if pd.isna(date):
            return "Missing"
        elif re.match(full_date_pattern, date):
            return "Full Date (YYYY-MM-DD)"
        elif re.match(year_month_pattern, date):
            return "Year & Month Date (YYYY-MM)"
        elif re.match(year_only_pattern, date):
            return "Year Only (YYYY)"
        else:
            return "Other"
    
    # Apply pattern identification
    df['Pattern'] = df[date_column].apply(identify_pattern)
    pattern_summary = df['Pattern'].value_counts().reset_index(name="Count")
    pattern_summary.columns = ['Pattern', 'Count']
    
    # Standardize date to year only
    df[date_column] = df[date_column].apply(lambda x: str(x)[:4] if pd.notnull(x) else None)
    
    df.drop(columns=["Pattern"], inplace=True)

    return df, pattern_summary

def clean_currency_columns(df, columns):
    # Method to remove '$' and ',' from the financials
    for column in columns:
        df[column] = df[column].str.replace('[^\d.]', '', regex=True).astype(float)
    return df   

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove rows with values outside these bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df