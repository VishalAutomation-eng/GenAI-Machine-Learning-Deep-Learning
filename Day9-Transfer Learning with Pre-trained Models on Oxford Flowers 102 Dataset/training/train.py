from utils.preprocess import get_preprocess_fn

def train_model(model, train_ds, val_ds, model_name, epochs=5):
    preprocess_fn = get_preprocess_fn(model_name)

    train_ds = train_ds.map(
        lambda x, y: (preprocess_fn(x), y)
    )
    val_ds = val_ds.map(
        lambda x, y: (preprocess_fn(x), y)
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history
