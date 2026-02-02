from utils.preprocess import get_preprocess_fn

def evaluate_model(model, test_ds, model_name):
    preprocess_fn = get_preprocess_fn(model_name)

    test_ds = test_ds.map(
        lambda x, y: (preprocess_fn(x), y)
    )

    loss, acc = model.evaluate(test_ds)
    print(f"{model_name.upper()} Accuracy: {acc:.4f}")
    return acc
