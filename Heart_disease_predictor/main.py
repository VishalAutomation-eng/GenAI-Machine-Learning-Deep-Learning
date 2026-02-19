from src.data_loader import load_data
from eda.eda_analysis import perform_eda
from src.preprocessing import preprocess_data
from src.model_train import train_model
from src.model_evaluate import evaluate_model

# Load data
df = load_data("/Users/vishalpande/Downloads/heart_disease_uci.csv")

# EDA
perform_eda(df)

# Preprocessing
X, y = preprocess_data(df)

# Train
model, X_test, y_test = train_model(X, y)

# Evaluate
evaluate_model(model, X_test, y_test)
