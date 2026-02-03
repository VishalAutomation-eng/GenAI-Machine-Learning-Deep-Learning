import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop non-predictive column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    return df
