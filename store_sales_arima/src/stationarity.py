from statsmodels.tsa.stattools import adfuller

def make_stationary(series):
    p_value = adfuller(series)[1]
    diff_series = series.copy()
    d = 0

    while p_value > 0.05:
        diff_series = diff_series.diff().dropna()
        p_value = adfuller(diff_series)[1]
        d += 1
        print(f"Differencing order {d} → ADF p-value: {p_value}")

    print("Series is stationary (p < 0.05)")
    return diff_series
