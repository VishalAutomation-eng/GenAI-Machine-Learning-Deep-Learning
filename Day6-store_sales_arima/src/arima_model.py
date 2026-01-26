from statsmodels.tsa.arima.model import ARIMA

def build_arima(series, order=(1, 0, 1), steps=5):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return model_fit, forecast
