import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.set_index("Id", inplace=True)
    test_df.set_index("Id", inplace=True)

    return train_df, test_df
