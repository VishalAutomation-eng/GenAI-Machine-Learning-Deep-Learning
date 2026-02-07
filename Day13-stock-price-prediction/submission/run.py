import joblib
import pandas as pd

MODEL_NAME = "KNN_High_90"

loaded_models = joblib.load("../models/saved_models/all_models.joblib")
model = loaded_models[MODEL_NAME]["model"]

sample_input = [100, 102, 105, 103, 104, 106, 108, 110, 109, 111]
prediction = model.predict([sample_input])

print("Prediction:", prediction)
