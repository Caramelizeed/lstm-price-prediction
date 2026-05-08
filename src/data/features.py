import numpy as np


def create_sequences(X_data, y_data, seq_length=60):

    X, y = [], []

    for i in range(len(X_data) - seq_length):

        X.append(X_data[i:i+seq_length])

        y.append(y_data[i+seq_length])

    return np.array(X), np.array(y)