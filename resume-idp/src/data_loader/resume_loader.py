import pandas as pd

def load_resumes(csv_path: str):
    df = pd.read_csv(csv_path)
    return df
