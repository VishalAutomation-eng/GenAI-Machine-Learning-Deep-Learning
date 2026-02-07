from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Bidirectional


def get_ml_models():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge()),
        ("KNN", KNeighborsRegressor()),
        ("DecisionTree", DecisionTreeRegressor())
    ]


def get_dl_model(model_type, input_shape):
    model = Sequential()

    if model_type == "LSTM":
        model.add(LSTM(64, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(64, input_shape=input_shape))
    elif model_type == "RNN":
        model.add(SimpleRNN(64, input_shape=input_shape))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(64), input_shape=input_shape))

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
