import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import handle_missing_values
from src.feature_engineering import add_features, encode_features
from src.train import train_models
from src.evaluate import evaluate

# Load data
train_df, test_df = load_data("/Users/vishalpande/Downloads/train.csv", "/Users/vishalpande/Downloads/test.csv")

# Log transform target
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Combine data
all_data = pd.concat([
    train_df.drop('SalePrice', axis=1),
    test_df
])

# Preprocessing
all_data = handle_missing_values(all_data)
all_data = add_features(all_data)
all_data = encode_features(all_data)

# Split back
X = all_data.iloc[:len(train_df)]
X_test = all_data.iloc[len(train_df):]
y = train_df['SalePrice']

# Train
lr, xgbr, scaler, X_val, y_val = train_models(X, y)

# Evaluate XGBoost
y_pred_xgb = xgbr.predict(X_val)
evaluate(y_val, y_pred_xgb, "XGBoost")

# Final predictions
final_preds_log = xgbr.predict(X_test)
final_preds = np.expm1(final_preds_log)

submission = pd.DataFrame({
    "Id": test_df.index,
    "SalePrice": final_preds
})

submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv generated successfully")
