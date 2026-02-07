import matplotlib.pyplot as plt
import pandas as pd

def plot_top_windows(results_df):
    top_50 = results_df.sort_values(by='test_mae').head(50)
    time_windows = pd.Series([i.split('_')[-1] for i in top_50['Model']])
    time_windows.value_counts().plot(kind='bar')
    plt.show()


def plot_model_types(results_df):
    top_50 = results_df.sort_values(by='test_mae').head(50)
    model_types = pd.Series([i.split('_')[0] for i in top_50['Model']])
    model_types.value_counts().plot(kind='bar')
    plt.show()
