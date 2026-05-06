import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_returns(df):
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    return df


def scale_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler