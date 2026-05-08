import pandas as pd
import numpy as np


def add_features(df):

    # Simple Moving Average
    df['sma_10'] = df['Close'].rolling(window=10).mean()

    # Exponential Moving Average
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Rolling Volatility
    df['volatility_10'] = df['returns'].rolling(window=10).std()

    # RSI
    delta = df['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df['rsi_14'] = 100 - (100 / (1 + rs))

    df = df.dropna()

    return df


def create_sequences(X_data, y_data, seq_length=60):

    X, y = [], []

    for i in range(len(X_data) - seq_length):

        X.append(X_data[i:i+seq_length])

        y.append(y_data[i+seq_length])

    return np.array(X), np.array(y)