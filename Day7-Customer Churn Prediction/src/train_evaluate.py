from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (Churn)": f1_score(y_test, y_pred)
        })

    return results
