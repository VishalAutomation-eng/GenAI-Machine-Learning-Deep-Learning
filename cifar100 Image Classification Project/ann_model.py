from tensorflow import keras
from tensorflow.keras import layers

def build_ann_model():
    model = keras.Sequential([
        layers.Input(shape=(32 * 32 * 3,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(100, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    

    return model
