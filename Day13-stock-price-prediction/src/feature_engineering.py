import numpy as np

def return_pairs(column, days):
    prices = list(column)
    X, y = [], []

    for i in range(len(prices) - days):
        X.append(prices[i:i + days])
        y.append(prices[i + days])

    return np.array(X), np.array(y)


def chunk_data(df, columns, windows):
    chunked_data = {}

    for col in columns:
        for w in windows:
            X, y = return_pairs(df[col], w)
            chunked_data[f"X_{col}_{w}"] = X
            chunked_data[f"y_{col}_{w}"] = y

    return chunked_data
