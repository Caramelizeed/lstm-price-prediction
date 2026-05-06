import numpy as np

def compute_returns(df):
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    return df