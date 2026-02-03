from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    }
