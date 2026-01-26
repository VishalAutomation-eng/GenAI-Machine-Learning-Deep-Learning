import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.models import get_models
from src.train_evaluate import train_and_evaluate

DATA_PATH = "/Users/vishalpande/Downloads/Telco-Customer-Churn.csv"

def main():
    df = load_data(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(df)
    models = get_models()

    results = train_and_evaluate(
        models, preprocessor, X_train, X_test, y_train, y_test
    )

    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
