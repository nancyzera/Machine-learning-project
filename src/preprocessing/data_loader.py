import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    return df
