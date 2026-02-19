from data_loader import load_data
from feature_engineering import chunk_data
from model_zoo import get_ml_models
from evaluate import build_results

import joblib
import os
from pathlib import Path

# 📌 Resolve base directory safely
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_DIR = BASE_DIR / "models" / "saved_models"
RESULT_PATH = BASE_DIR / "models" / "models.csv"


def main():
    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    columns = ["Open", "High", "Low", "Close"]
    windows = [5, 10, 20, 30, 60, 90]

    print("⚙️ Feature engineering...")
    chunked_data = chunk_data(df, columns, windows)

    print("🤖 Training ML models...")
    models = get_ml_models()
    trained_models = {}

    for key in chunked_data:
        if key.startswith("X_"):
            y_key = key.replace("X_", "y_")
            X, y = chunked_data[key], chunked_data[y_key]

            for name, model in models:
                model.fit(X, y)
                trained_models[f"{name}_{key[2:]}"] = {
                    "model": model
                }

    print("💾 Saving models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, MODEL_DIR / "all_models.joblib")

    print("📊 Saving results...")
    results_df = build_results(trained_models)
    results_df.to_csv(RESULT_PATH, index=False)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
