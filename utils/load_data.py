import pandas as pd

def load_data(url, parse_cols):

    df = pd.read_csv(url, parse_dates=parse_cols)

    return df