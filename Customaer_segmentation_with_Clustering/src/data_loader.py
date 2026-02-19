import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    return df
