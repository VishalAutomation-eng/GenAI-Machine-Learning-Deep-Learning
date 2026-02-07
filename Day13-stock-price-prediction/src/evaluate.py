import pandas as pd

def build_results(trained_models):
    results_df = pd.DataFrame([
        {"Model": name, **metrics}
        for name, metrics in trained_models.items()
    ])
    return results_df
