from src.data_loader import load_data
from src.stationarity import make_stationary
from src.arima_model import build_arima

def main():
    df = load_data("/Users/vishalpande/Downloads/airline_passenger_timeseries.csv")

    # Clean column names (good practice)
    df.columns = df.columns.str.strip()

    # Convert Month to datetime and set as index (VERY IMPORTANT for time series)
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)

    # Correct target column
    series = df["Passengers"]

    stationary_series = make_stationary(series)

    model, forecast = build_arima(stationary_series)

    print(model.summary())
    print("\nForecasted Passengers:")
    print(forecast)

if __name__ == "__main__":
    import pandas as pd
    main()
